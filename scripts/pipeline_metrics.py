"""Local semantic evaluation; author-normalized scores and bipartite coverage."""
import json
from aapd_pipeline import dump


def maximum_matching(edges):
    owners = {}
    def augment(real, seen):
        for predicted in edges[real]:
            if predicted in seen:
                continue
            seen.add(predicted)
            if predicted not in owners or augment(owners[predicted], seen):
                owners[predicted] = real
                return True
        return False
    for real in edges:
        augment(real, set())
    return [{'ground_truth': real, 'prediction': predicted} for predicted, real in owners.items()]


def evaluate(predictions, truth, vocabulary, spaces, args):
    import numpy as np
    import torch
    from aapd_models import load_minilm, local_first_pretrained
    names = sorted(set(vocabulary) | {p for _, labels in spaces for p in labels} |
                   {p for row in predictions for p in row['labels'][:3]} |
                   {p for row in predictions for p in truth[row['doc_id']]})
    encoder = load_minilm(device='cpu')
    vectors = dict(zip(names, encoder.encode(names, show_progress_bar=True)))
    del encoder
    path = getattr(args, 'judgment_cache', args.output / 'evaluation_judgments.json')
    cache = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    tokenizer = model = None

    def match(real, predicted):
        nonlocal tokenizer, model
        score = float(vectors[real] @ vectors[predicted])
        if score >= .75:
            return True
        if score < .5:
            return False
        key = json.dumps([args.judge_model, real, predicted])
        if key not in cache:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            if model is None:
                tokenizer = local_first_pretrained(AutoTokenizer, args.judge_model)
                model = local_first_pretrained(AutoModelForCausalLM, args.judge_model, torch_dtype=torch.float16,
                          device_map={'': 0}, use_safetensors=True).eval()
            prompt = ('In text classification, do these two labels match in meaning? Related broader/narrower topic names can match, but unrelated topics cannot. Answer only Yes or No.\nGround truth: ' + real + '\nPrediction: ' + predicted)
            text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            raw = tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            if raw.rstrip('.').casefold() not in ('yes', 'no'):
                raise ValueError('Judge did not answer Yes/No: ' + raw)
            cache[key] = {'raw': raw, 'match': raw.rstrip('.').casefold() == 'yes'}
            dump(path, cache)
        return cache[key]['match']

    metrics = {'documents': len(predictions), 'judge_model': args.judge_model,
               'note': 'Local adaptation. Greedy semantic hits/min(k,true labels); not standard precision@k.'}
    for k in (1, 3):
        values = []
        for row in predictions:
            remaining = list(truth[row['doc_id']])
            denominator = min(k, len(remaining))
            hits = 0
            for predicted in row['labels'][:k]:
                if not remaining:
                    break
                real = max(remaining, key=lambda name: float(vectors[name] @ vectors[predicted]))
                if match(real, predicted):
                    hits += 1
                    remaining.remove(real)
            values.append(hits / denominator if denominator else 0)
        metrics['author_matching_score@{}'.format(k)] = float(np.mean(values))
    coverage = []
    for stage, labels in spaces:
        labels = list(dict.fromkeys(labels))
        edges = {real: [p for p in labels if match(real, p)] for real in vocabulary}
        pairs = maximum_matching(edges)
        coverage.append({'stage': stage, 'label_count': len(labels), 'matched_count': len(pairs),
                         'coverage': len(pairs) / len(vocabulary), 'matching_pairs': pairs})
    dump(args.output / 'coverage_metrics.json', {'ground_truth_count': len(vocabulary),
         'vocabulary_source': 'union of supplied train/test truth labels; evaluation only', 'results': coverage})
    dump(args.output / 'metrics.json', metrics)
    if model is not None:
        del model
        torch.cuda.empty_cache()
    return metrics, coverage
