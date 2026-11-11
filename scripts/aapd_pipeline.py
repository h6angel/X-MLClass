"""Local continuation of the repository's AAPD NLI/improvement/evaluation flow."""
import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NLI = 'MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_chunks(split, limit):
    data = ROOT / 'data/AAPD'
    texts = (data / (split + '_texts_split_50.txt')).read_text(encoding='utf-8').splitlines()
    keys = (data / ('llama2_label_50.txt' if split == 'train' else 'llama2_label_test_50.txt')).read_text(encoding='utf-8').splitlines()
    texts = [line for line in texts if int(line.partition(' ')[0]) < limit]
    keys = [line for line in keys if int(line.partition(': ')[0]) < limit]
    if len(texts) != len(keys):
        raise ValueError('Text/keyphrase row counts differ for ' + split)
    rows = []
    for index, (text, key) in enumerate(zip(texts, keys)):
        tid, _, _ = text.partition(' ')
        kid, separator, _ = key.partition(': ')
        if not separator or tid != kid:
            raise ValueError('Text/keyphrase ID mismatch at row {}'.format(index))
        if int(tid) < limit:
            rows.append({'doc_id': int(tid), 'text': text.strip(), 'keyword': key.strip()})
    if not rows:
        raise ValueError('No selected rows')
    return rows


def rank_document(outputs):
    # Same rank-by-rank voting and maximum-score tie breaking as model_eval.py.
    labels = []
    for rank in range(min(len(row['labels']) for row in outputs)):
        counts = Counter(row['labels'][rank] for row in outputs)
        scores = {}
        for row in outputs:
            label, score = row['labels'][rank], row['scores'][rank]
            scores[label] = max(scores.get(label, 0), score)
        maximum = max(counts.values())
        if maximum > 1:
            selected = sorted((label for label in counts if counts[label] == maximum), key=lambda x: scores[x], reverse=True)
        else:
            selected = [max(scores, key=scores.get)]
        labels.extend(label for label in selected if label not in labels)
    return labels


def infer(rows, labels, output, model_name, channels=('text', 'keyword')):
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline
    from tqdm import tqdm
    if not labels:
        raise ValueError('Inference requires a nonempty label space')
    signature = hashlib.sha256(json.dumps([rows, labels, model_name, channels]).encode()).hexdigest()
    manifest = output / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature'] != signature:
        raise ValueError('Run inputs changed; select a fresh output directory: ' + str(output))
    dump(manifest, {'signature': signature, 'nli_model': model_name, 'labels': labels})
    pending = {}
    results = {}
    for channel in channels:
        path = output / (channel + '.jsonl')
        done = []
        if path.exists():
            lines = path.read_text(encoding='utf-8').splitlines()
            for index, line in enumerate(lines):
                try:
                    done.append(json.loads(line))
                except ValueError:
                    if index != len(lines) - 1:
                        raise
            # Recover an interrupted final write before appending.
            path.write_text(''.join(json.dumps(row) + '\n' for row in done), encoding='utf-8')
        if len(done) > len(rows) or any(record['doc_id'] != rows[i]['doc_id'] for i, record in enumerate(done)):
            raise ValueError('Invalid resumed rows: ' + str(path))
        results[channel] = done
        pending[channel] = len(done)
    if all(count == len(rows) for count in pending.values()):
        return results
    from aapd_models import load_minilm
    encoder = load_minilm(device='cpu')
    label_vectors = encoder.encode(labels)
    candidates = {}
    for channel in channels:
        remaining = rows[pending[channel]:]
        if remaining:
            vectors = encoder.encode([row[channel] for row in remaining], batch_size=32, show_progress_bar=True)
            candidates[channel] = np.argsort(vectors @ label_vectors.T, axis=1)[:, -8:]
    del encoder
    gc.collect()
    from aapd_models import load_nli_pipeline
    classifier = load_nli_pipeline(model_name)
    for channel in channels:
        start = pending[channel]
        with (output / (channel + '.jsonl')).open('a', encoding='utf-8') as handle:
            for offset, row in enumerate(tqdm(rows[start:], desc=channel)):
                answer = classifier(row[channel], [labels[int(i)] for i in candidates[channel][offset]],
                                    hypothesis_template='This example is {}', multi_label=False, batch_size=8)
                answer.pop('sequence', None)
                answer['doc_id'] = row['doc_id']
                handle.write(json.dumps(answer) + '\n')
                handle.flush()
                results[channel].append(answer)
    del classifier
    gc.collect()
    torch.cuda.empty_cache()
    return results


def improve(rows, initial, args):
    # Reuse the author's actual label deletion/addition functions. Only the
    # backward document-boundary scan is guarded against negative indexing.
    import numpy as np
    from sentence_transformers import SentenceTransformer, util
    from types import SimpleNamespace
    source = (ROOT / 'OpenWordMLTC/self_training/self_training.py').read_text(encoding='utf-8')
    source = source.replace('while int(keyword_docs[cur_index - 1].split(\': \')[0]) == label:',
                            'while cur_index > 0 and int(keyword_docs[cur_index - 1].split(\': \')[0]) == label:')
    # jsonlines is the only extra dependency of the author's implementation.
    module = {'__name__': 'aapd_author_improvement'}
    exec(compile(source, 'author_self_training', 'exec'), module)
    from aapd_models import author_sentence_transformer
    module['SentenceTransformer'] = author_sentence_transformer
    task = getattr(args, 'dataset', 'AAPD')
    working = args.output / ('author_workspace/' + task + '/llama2/result')
    working.mkdir(parents=True, exist_ok=True)
    keyfile = working.parents[1] / 'keyphrases.txt'
    keyfile.write_text('\n'.join(row['keyword'] for row in rows) + '\n', encoding='utf-8')
    config = SimpleNamespace(path=str(args.output / 'author_workspace'), task=task, llama_model='llama2',
                             keyphrase_dir='keyphrases.txt', tail_set_size=getattr(args, 'tail_size', 500), majority_num=getattr(args, 'majority_num', 350),
                             max_majority_num=getattr(args, 'max_majority_num', 5), sim_threshold=getattr(args, 'sim_threshold', 0.55), max_add_label=getattr(args, 'max_add_label', 10))
    cumulative = list(initial)
    active = list(initial)
    majority = []
    for iteration in range(args.iterations):
        result = infer(rows, active, args.output / ('train/iteration_{}'.format(iteration)), args.model, ('text',))['text']
        saved_iteration = args.output / ('improvement/iteration_{}.json'.format(iteration))
        if saved_iteration.exists():
            # infer() above validates the exact rows/active-labels/model manifest
            # and recovers incomplete JSONL before this state is reused.
            state = json.loads(saved_iteration.read_text(encoding='utf-8'))
            cumulative = state['cumulative_labels']
            active = state['next_training_labels']
            majority = state['majority_excluded_from_next_training']
            print('Skipping completed improvement round {}'.format(iteration + 1), flush=True)
            continue
        counts = {label: 0 for label in active}
        for row in result:
            counts[row['labels'][0]] += 1
        removed = [label for label in counts if counts[label] <= getattr(args, 'remove_at_most', 6) and label in cumulative]
        cumulative = [label for label in cumulative if label not in removed]
        if len(cumulative) < 3:
            raise ValueError('Too few surviving labels for author improvement; inspect training predictions')
        (working / 'update_labelspace.txt').write_text('\n'.join(cumulative) + '\n', encoding='utf-8')
        (working / ('zero_shot_text_train{}.jsonl'.format(iteration))).write_text(''.join(json.dumps(row) + '\n' for row in result), encoding='utf-8')
        addition_details = {'rule': 'released_author_code', 'threshold': config.sim_threshold + iteration / 100}
        if getattr(args, 'improvement_rule', 'author') == 'paper':
            from pipeline_alignment import select_paper_additions
            from aapd_models import load_minilm
            encoder = load_minilm(device='cpu')
            added, addition_details = select_paper_additions(rows, result, cumulative, config.tail_set_size,
                                                            config.max_add_label, encoder.encode)
            addition_details['rule'] = 'paper_median_novelty'
            del encoder
        else:
            added = module['self_training'](config, iteration)
        cumulative += added
        majority = module['label_deletion'](majority, sorted(counts.items(), key=lambda x: x[1], reverse=True), config.majority_num, config.max_majority_num)
        active = [label for label in cumulative if label not in majority]
        dump(args.output / ('improvement/iteration_{}.json'.format(iteration)),
             {'removed': removed, 'added': added, 'addition_details': addition_details, 'majority_excluded_from_next_training': majority,
              'cumulative_labels': cumulative, 'next_training_labels': active})
        print('Improvement {}: {} cumulative labels'.format(iteration + 1, len(cumulative)), flush=True)
    # The author's final cumulative space retains majority labels; those labels
    # are excluded only from iterative training to expose underrepresented topics.
    (args.output / 'update_labelspace.txt').write_text('\n'.join(cumulative) + '\n', encoding='utf-8')
    return cumulative


def evaluate(predictions, args):
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    truth = (ROOT / 'data/AAPD/test_label_full.txt').read_text(encoding='utf-8').splitlines()
    from aapd_models import load_minilm
    encoder = load_minilm(device='cpu')
    names = sorted(set(label for row in predictions for label in row['labels'][:3]) |
                   set(label.strip().split('.', 1)[-1] for row in predictions for label in truth[row['doc_id']].split(',') if label.strip()))
    vectors = dict(zip(names, encoder.encode(names)))
    del encoder
    judges = args.output / 'evaluation_judgments.json'
    cache = json.loads(judges.read_text(encoding='utf-8')) if judges.exists() else {}
    tokenizer = model = None

    def match(real, predicted):
        nonlocal tokenizer, model
        key = json.dumps([args.judge_model, real, predicted])
        if key not in cache:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            if model is None:
                tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
                model = AutoModelForCausalLM.from_pretrained(args.judge_model, torch_dtype=torch.float16,
                                                           device_map={'': 0}, use_safetensors=True)
                model.eval()
            prompt = ('In text classification, do these two labels match in meaning? Related broader/narrower topic names can match, but unrelated topics cannot. Answer only Yes or No.\nGround truth: ' + real + '\nPrediction: ' + predicted)
            text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                answer = model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            raw = tokenizer.decode(answer[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            cache[key] = {'raw': raw, 'match': raw.rstrip('.').casefold() == 'yes'}
            dump(judges, cache)
        return cache[key]['match']

    metrics = {}
    for k in (1, 3):
        values = []
        for row in predictions:
            remaining = list(dict.fromkeys(label.strip().split('.', 1)[-1] for label in truth[row['doc_id']].split(',') if label.strip()))
            denominator = min(len(remaining), k)
            hits = 0
            for predicted in row['labels'][:k]:
                if not remaining:
                    break
                scores = [float(vectors[predicted] @ vectors[real]) for real in remaining]
                index = int(np.argmax(scores))
                score = scores[index]
                if score >= 0.75 or (score >= 0.5 and match(remaining[index], predicted)):
                    hits += 1
                    remaining.pop(index)
            values.append(hits / denominator if denominator else 0)
        metrics['author_matching_score@{}'.format(k)] = float(np.mean(values))
    dump(args.output / 'metrics.json', {'documents': len(predictions), 'judge_model': args.judge_model,
          'note': 'Author semantic matching normalization hits/min(number of true labels,k); local judge replaces GPT. Not standard precision@k.', **metrics})
    print(metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['check', 'prepare', 'train', 'test', 'all'], default='check')
    parser.add_argument('--labels', type=Path, default=ROOT / 'outputs/aapd_local/init_label_space.txt')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/aapd_pipeline')
    parser.add_argument('--train-docs', type=int, default=3000)
    parser.add_argument('--test-docs', type=int, default=1000)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--model', default=NLI)
    parser.add_argument('--judge-model', default='Qwen/Qwen2.5-1.5B-Instruct')
    args = parser.parse_args()
    if min(args.train_docs, args.test_docs, args.iterations) < 1:
        parser.error('Document limits and iterations must be positive')
    args.output = args.output.resolve()
    labels = [line.strip() for line in args.labels.read_text(encoding='utf-8').splitlines() if line.strip()]
    train, test = read_chunks('train', args.train_docs), read_chunks('test', args.test_docs)
    print('{} labels; train {} chunks / {} docs; test {} chunks / {} docs'.format(len(labels), len(train), len(set(r['doc_id'] for r in train)), len(test), len(set(r['doc_id'] for r in test))), flush=True)
    if args.stage == 'check':
        return
    if args.stage == 'prepare':
        from aapd_models import load_minilm
        encoder = load_minilm(device='cpu')
        embedding = encoder.encode(['MiniLM local loading check'])
        print('MiniLM ready; test embedding shape: {}'.format(embedding.shape))
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU PyTorch required')
    args.output.mkdir(parents=True, exist_ok=True)
    run = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    # Protect improvement/evaluation state from accidental reuse with changed settings.
    settings = {k: v for k, v in run.items() if k != 'stage'}
    settings['initial_labels_sha256'] = hashlib.sha256(args.labels.read_bytes()).hexdigest()
    configuration = args.output / 'settings.json'
    if configuration.exists() and json.loads(configuration.read_text(encoding='utf-8')) != settings:
        raise ValueError('Settings changed; use a different --output directory')
    dump(configuration, settings)
    if args.stage in ('train', 'all'):
        infer(train, labels, args.output / 'train/initial', args.model)
        labels = improve(train, labels, args)
    elif (args.output / 'update_labelspace.txt').exists():
        labels = (args.output / 'update_labelspace.txt').read_text(encoding='utf-8').splitlines()
    if args.stage in ('test', 'all'):
        results = infer(test, labels, args.output / 'test', args.model)
        grouped = defaultdict(list)
        for channel in ('keyword', 'text'):
            for row in results[channel]:
                grouped[row['doc_id']].append(row)
        predictions = [{'doc_id': doc_id, 'labels': rank_document(rows)} for doc_id, rows in sorted(grouped.items())]
        dump(args.output / 'predictions.json', predictions)
        evaluate(predictions, args)


if __name__ == '__main__':
    main()
