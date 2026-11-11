"""Paper-flow operations with explicit choices where release metadata is incomplete."""
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

from aapd_pipeline import dump

CLASSIFIERS = {
    'deberta': 'MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33',
    'bart': 'facebook/bart-large-mnli',
    'xlm': 'joeddav/xlm-roberta-large-xnli',
}


def estimate_topic_count(vectors, min_cluster_size):
    """Same density clustering criterion; record any native-library fallback."""
    try:
        from hdbscan import HDBSCAN
    except ImportError as error:
        from sklearn.cluster import HDBSCAN
        # sklearn includes the point itself in min_samples; contrib excludes it.
        topics = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_cluster_size + 1,
                         metric='euclidean', cluster_selection_method='eom').fit_predict(vectors)
        backend = {'library': 'sklearn.cluster.HDBSCAN', 'min_samples': min_cluster_size + 1,
                   'fallback_reason': str(error)}
    else:
        topics = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean',
                         cluster_selection_method='eom', prediction_data=True).fit_predict(vectors)
        backend = {'library': 'hdbscan.HDBSCAN', 'min_samples': min_cluster_size}
    return len(set(topics) - {-1, -2, -3}), backend


def available_train_limit(directory):
    lines = (directory / 'llama2_label_50.txt').read_text(encoding='utf-8-sig').splitlines()
    last = max(int(line.split(':', 1)[0]) for line in lines)
    texts = (directory / 'train_texts_split_50.txt').read_text(encoding='utf-8-sig').splitlines()
    # RCV1 supplied keywords end partway through the final document. Exclude
    # only that incomplete boundary document, never silently drop interior rows.
    text_count = sum(int(line.split(' ', 1)[0]) == last for line in texts)
    key_count = sum(int(line.split(':', 1)[0]) == last for line in lines)
    return last + 1 if text_count == key_count else last


def growth_limits(maximum, step=1000):
    return list(range(step, maximum, step)) + [maximum]


def novel_labels(labels, previous, score, threshold=.8):
    """Operationalize 'new labels' by exact name + MiniLM similarity, logged in settings."""
    return [label for label in labels if not any(
        label.casefold() == old.casefold() or score(label, old) >= threshold for old in previous)]


def dynamic_discovery(rows, args, config, discover, load_labels, read_prefix=None):
    from aapd_models import load_minilm
    root = args.output
    base = root / 'discovery'
    base.mkdir(parents=True, exist_ok=True)
    final = base / 'dynamic_selection.json'
    if final.exists():
        state = json.loads(final.read_text(encoding='utf-8'))
        return state['selected_id_limit']
    previous = []
    rounds = []
    for limit in growth_limits(args.train_docs, args.growth_step):
        selected = read_prefix(limit) if read_prefix else [row for row in rows if row['doc_id'] < limit]
        repair_missing_keywords(selected, args, config)
        step_args = SimpleNamespace(**vars(args))
        step_args.output = base / ('prefix_{}'.format(limit))
        discover(selected, step_args, config)
        path = step_args.output / 'discovery/init_label_space.txt'
        labels = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if not labels:
            raise ValueError('No labels discovered at prefix ' + str(limit))
        if previous:
            encoder = load_minilm(device='cpu')
            names = sorted(set(labels + previous))
            vectors = dict(zip(names, encoder.encode(names)))
            del encoder
            new = novel_labels(labels, previous, lambda a, b: float(vectors[a] @ vectors[b]), args.novelty_threshold)
        else:
            new = labels
        previous = list(dict.fromkeys(previous + new))
        rounds.append({'id_limit': limit, 'documents': len({r['doc_id'] for r in selected}),
                       'candidate_labels': labels, 'new_labels': new, 'cumulative_labels': previous})
        dump(base / 'dynamic_progress.json', {'rounds': rounds})
        print('Discovery prefix {}: {} new labels, {} cumulative'.format(limit, len(new), len(previous)), flush=True)
        converged = len(rounds) > 1 and not new
        if converged or limit == args.train_docs:
            (base / 'init_label_space.txt').write_text('\n'.join(previous) + '\n', encoding='utf-8')
            load_labels(base / 'init_label_space.txt')
            dump(final, {'selected_id_limit': limit, 'converged': converged,
                 'stopping_reason': 'no_new_semantic_labels' if converged else 'available_keywords_or_user_cap_reached',
                 'growth_step': args.growth_step, 'novelty_threshold': args.novelty_threshold, 'rounds': rounds})
            return limit


def repair_missing_keywords(rows, args, config, generate=True):
    """Fill empty supplied keyword rows with Qwen, with a separate auditable cache."""
    path = args.output / 'discovery/repaired_keywords.json'
    cache = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    missing = [row for row in rows if not row['keyphrases'] and
               '{}:{}'.format(row['doc_id'], row['chunk_id']) not in cache]
    if missing and not generate:
        raise ValueError('Unrepaired empty keywords; run discover first')
    if missing:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from aapd_local_labels import parse_json
        tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
        model = AutoModelForCausalLM.from_pretrained(args.judge_model, torch_dtype=torch.float16,
                    device_map={'': 0}, use_safetensors=True).eval()
        for row in missing:
            prompt = ('Extract two coarse-grained and two fine-grained topic keyphrases for this text. Domain: ' +
                      config['domain'] + '. Return only JSON {"keyphrases": ["..."]}.\nText: ' + row['content'])
            text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            raw = tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            phrases = parse_json(raw).get('keyphrases')
            if not isinstance(phrases, list) or not phrases or any(not isinstance(p, str) or not p.strip() or '\n' in p for p in phrases):
                raise ValueError('Invalid keyword repair: ' + raw)
            cache['{}:{}'.format(row['doc_id'], row['chunk_id'])] = {'keyphrases': ', '.join(phrases), 'model': args.judge_model, 'raw': raw}
            dump(path, cache)
        del model
        torch.cuda.empty_cache()
    for row in rows:
        if not row['keyphrases']:
            row['keyphrases'] = cache['{}:{}'.format(row['doc_id'], row['chunk_id'])]['keyphrases']
            row['keyword'] = '{}: {}'.format(row['doc_id'], row['keyphrases'])


def select_paper_additions(rows, predictions, labels, tail_size, max_add, encode):
    """Sections 5.3/6.4: recurring rare phrases, low-score chunks, median novelty."""
    import numpy as np
    phrases = [[part.strip().strip('[]\"') for part in row['keyphrases'].split(', ') if part.strip()]
               for row in rows]
    counts = Counter(p for row in phrases for p in row)
    population = sum(counts.values())
    rare = sorted(p for p, count in counts.items() if count > 15 and count < .01 * population)
    if not rare:
        return [], {'gamma': None, 'rare_candidates': 0, 'tail_chunks': min(tail_size, len(rows))}
    label_vectors = encode(labels)
    similarities = encode(rare) @ label_vectors.T
    maximum = np.max(similarities, axis=1)
    gamma = float(np.median(maximum))
    tail = sorted(range(len(predictions)), key=lambda i: predictions[i]['scores'][0])[:tail_size]
    tail_phrases = {p for i in tail for p in phrases[i]}
    candidates = [(phrase, float(score)) for phrase, score in zip(rare, maximum)
                  if phrase in tail_phrases and score < gamma and phrase not in labels]
    candidates.sort(key=lambda pair: (pair[1], pair[0]))
    return [p for p, _ in candidates[:max_add]], {
        'gamma': gamma, 'rare_candidates': len(rare), 'tail_chunks': len(tail),
        'frequency_rule': 'count > 15 and count < 1% of all phrase occurrences',
        'selection_order': 'ascending maximum similarity, lexical tie break',
        'max_add_source': 'released code default; paper does not specify a cap'}
