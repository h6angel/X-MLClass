"""Read-only benchmark of the existing per-chunk NLI path."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from transformers import pipeline
from aapd_pipeline import ROOT, read_chunks
from aapd_models import load_minilm


def main():
    manifest = json.loads((ROOT / 'outputs/aapd_pipeline/train/iteration_2/manifest.json').read_text())
    labels = manifest['labels']
    rows = read_chunks('train', 3000)
    # Spread samples across the actual pending data, plus unusually long chunks.
    pending = rows[273:]
    indices = np.linspace(0, len(pending) - 1, 24, dtype=int).tolist()
    indices += sorted(range(len(pending)), key=lambda i: len(pending[i]['text']), reverse=True)[:4]
    samples = [pending[i] for i in dict.fromkeys(indices)]
    encoder = load_minilm(device='cpu')
    lv = encoder.encode(labels)
    jobs = []
    for channel in ('text', 'keyword'):
        vectors = encoder.encode([r[channel] for r in samples])
        candidates = np.argsort(vectors @ lv.T, axis=1)[:, -8:]
        jobs += [(channel, row[channel], [labels[int(i)] for i in cand])
                 for row, cand in zip(samples, candidates)]
    del encoder
    gc.collect()
    print('Loading cached NLI model', flush=True)
    classifier = pipeline('zero-shot-classification', model=manifest['nli_model'],
                          device=0, torch_dtype=torch.float16)
    def call(job, batch):
        return classifier(job[1], job[2], hypothesis_template='This example is {}',
                          multi_label=False, batch_size=batch, num_workers=0)
    report = {'gpu': torch.cuda.get_device_name(), 'samples_per_channel': len(samples), 'runs': []}
    baseline = None
    for batch in (1, 2, 4, 8, 16):
        try:
            for job in jobs[:2]:
                call(job, batch)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            outputs = []
            timings = {}
            for channel in ('text', 'keyword'):
                start = time.perf_counter()
                outputs += [call(job, batch) for job in jobs if job[0] == channel]
                torch.cuda.synchronize()
                timings[channel] = time.perf_counter() - start
            if baseline is None:
                baseline = outputs
            top1 = sum(a['labels'][0] != b['labels'][0] for a, b in zip(baseline, outputs))
            ranking = sum(a['labels'] != b['labels'] for a, b in zip(baseline, outputs))
            delta = max(abs(dict(zip(a['labels'], a['scores']))[label] - score)
                        for a, b in zip(baseline, outputs) for label, score in zip(b['labels'], b['scores']))
            result = {'batch_size': batch, 'seconds': timings,
                      'chunks_per_second': {c: len(samples) / t for c, t in timings.items()},
                      'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                      'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
                      'top1_changed': top1, 'ranking_changed': ranking, 'max_score_delta': delta}
        except torch.cuda.OutOfMemoryError:
            result = {'batch_size': batch, 'error': 'CUDA out of memory'}
            gc.collect()
            torch.cuda.empty_cache()
        report['runs'].append(result)
        print(json.dumps(result), flush=True)
        target = ROOT / 'outputs/aapd_batch_benchmark.json'
        target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Report: ' + str(target), flush=True)


if __name__ == '__main__':
    main()
