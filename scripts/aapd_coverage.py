"""Evaluate saved AAPD label spaces using semantic edges and maximum matching."""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from aapd_models import load_minilm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/aapd_pipeline'
JUDGE = 'Qwen/Qwen2.5-1.5B-Instruct'


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def maximum_matching(edges, labels):
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


def main():
    truth = list(dict.fromkeys(line.split(';')[1].strip() for line in
                 (ROOT / 'data/AAPD/true_labels.txt').read_text(encoding='utf-8').splitlines() if line.strip()))
    spaces = [('initial', (ROOT / 'outputs/aapd_local/init_label_space.txt').read_text(encoding='utf-8').splitlines())]
    for i in range(10):
        state = json.loads((OUT / f'improvement/iteration_{i}.json').read_text(encoding='utf-8'))
        spaces.append((f'iteration_{i+1}', state['cumulative_labels']))
    names = sorted(set(truth + [label for _, labels in spaces for label in labels]))
    encoder = load_minilm(device='cpu')
    vectors = dict(zip(names, encoder.encode(names, show_progress_bar=False)))
    del encoder
    cache_path = OUT / 'coverage_judgments.json'
    cache = json.loads((OUT / 'evaluation_judgments.json').read_text(encoding='utf-8'))
    if cache_path.exists():
        cache.update(json.loads(cache_path.read_text(encoding='utf-8')))
    pairs = {}
    similarities = {}
    for real in truth:
        for predicted in set(label for _, labels in spaces for label in labels):
            score = float(vectors[real] @ vectors[predicted])
            similarities[real, predicted] = score
            if .5 <= score <= .75:
                key = json.dumps([JUDGE, real, predicted])
                if key not in cache:
                    pairs[key] = (real, predicted)
    print(f'{len(truth)} GT names; {len(pairs)} uncached judge pairs', flush=True)
    if pairs:
        tokenizer = AutoTokenizer.from_pretrained(JUDGE, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(JUDGE, torch_dtype=torch.float16,
                  device_map={'': 0}, use_safetensors=True, local_files_only=True)
        model.eval()
        for i, (key, (real, predicted)) in enumerate(pairs.items(), 1):
            prompt = ('In text classification, do these two labels match in meaning? Related broader/narrower topic names can match, but unrelated topics cannot. Answer only Yes or No.\nGround truth: ' + real + '\nPrediction: ' + predicted)
            text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                answer = model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            raw = tokenizer.decode(answer[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            cache[key] = {'raw': raw, 'match': raw.rstrip('.').casefold() == 'yes'}
            save(cache_path, cache)
            if i % 25 == 0:
                print(f'Judged {i}/{len(pairs)}', flush=True)
    results = []
    for stage, labels in spaces:
        labels = list(dict.fromkeys(labels))
        edges = {}
        for real in truth:
            edges[real] = [p for p in labels if similarities[real, p] > .75 or
                  (.5 <= similarities[real, p] <= .75 and cache[json.dumps([JUDGE, real, p])]['match'])]
            edges[real].sort(key=lambda p: similarities[real, p], reverse=True)
        matched = maximum_matching(edges, labels)
        assert len({p['ground_truth'] for p in matched}) == len(matched)
        assert len({p['prediction'] for p in matched}) == len(matched)
        result = {'stage': stage, 'label_count': len(labels), 'matched_count': len(matched),
                  'coverage': len(matched)/len(truth), 'matching_pairs': matched,
                  'unmatched_ground_truth': [t for t in truth if t not in {p['ground_truth'] for p in matched}]}
        results.append(result)
        print(f'{stage}: {len(matched)}/{len(truth)} = {result["coverage"]:.4%}', flush=True)
    save(OUT / 'coverage_metrics.json', {'judge_model': JUDGE, 'ground_truth_count': len(truth),
         'matching': 'maximum cardinality bipartite matching', 'results': results,
         'improvement_percentage_points': 100*(results[-1]['coverage']-results[0]['coverage'])})


if __name__ == '__main__':
    main()
