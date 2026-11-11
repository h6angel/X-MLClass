"""Small AAPD embedding/clustering check; does not generate final labels."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--clusters', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    if not 6 <= args.samples or not 2 <= args.clusters < args.samples:
        parser.error('Require samples >= 6 and 2 <= clusters < samples')

    import numpy as np
    import torch
    from InstructorEmbedding import INSTRUCTOR
    from sklearn.mixture import GaussianMixture
    from umap import UMAP

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Check the GPU PyTorch installation.')
    root = Path(__file__).resolve().parents[1]
    rows = []
    with (root / 'data/AAPD/llama2_label_50.txt').open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            doc_id, separator, phrase = line.strip().partition(': ')
            if not separator or not doc_id.isdigit() or not phrase:
                raise ValueError('Invalid keyphrase row at line {}'.format(line_number))
            rows.append({'document_id': int(doc_id), 'keyphrases': phrase})
            if len(rows) >= args.samples:
                break
    if len(rows) < args.samples:
        raise ValueError('Not enough keyphrase rows for requested sample size')

    print('Read {} keyphrase chunks. GPU: {}'.format(len(rows), torch.cuda.get_device_name(0)), flush=True)
    print('Loading hkunlp/instructor-large; first run downloads model files.', flush=True)
    model = INSTRUCTOR('hkunlp/instructor-large', device='cuda')
    instruction = 'Represent documents collected from computer science paper abstract for clustering: '
    embeddings = model.encode(
        [[instruction, row['keyphrases']] for row in rows],
        batch_size=args.batch_size, show_progress_bar=True,
    )
    if not np.isfinite(embeddings).all():
        raise ValueError('Embeddings contain non-finite values')
    print('Embedding shape: {}'.format(embeddings.shape), flush=True)
    reduced = UMAP(n_neighbors=5, n_components=5, min_dist=0.0,
                   metric='cosine', random_state=42).fit_transform(embeddings)
    clustering = GaussianMixture(n_components=args.clusters, random_state=42)
    assignments = clustering.fit_predict(reduced)
    report = []
    for cluster in range(args.clusters):
        indices = np.flatnonzero(assignments == cluster)
        nearest = sorted(indices, key=lambda i: float(np.linalg.norm(reduced[i] - clustering.means_[cluster])))[:3]
        examples = [rows[int(i)] for i in nearest]
        report.append({'cluster': cluster, 'chunk_count': len(indices), 'examples': examples})
        print('\nCluster {}: {} chunks'.format(cluster, len(indices)))
        for row in examples:
            print('  Document {}: {}'.format(row['document_id'], row['keyphrases']))
    output = root / 'outputs/aapd_smoke'
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / 'embeddings.npy', embeddings)
    result = {'model': 'hkunlp/instructor-large', 'samples': len(rows),
              'requested_clusters': args.clusters, 'seed': 42, 'clusters': report}
    (output / 'clusters.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nSmoke check finished. Results: {}'.format(output))


if __name__ == '__main__':
    main()
