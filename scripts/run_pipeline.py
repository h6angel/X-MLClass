"""Independent, resumable X-MLClass local experiments for five datasets."""
import argparse
from collections import defaultdict
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from importlib.metadata import version, PackageNotFoundError
from types import SimpleNamespace

from aapd_pipeline import ROOT, NLI, dump, infer, improve, rank_document
from pipeline_data import fingerprint, read_rows, read_truth, describe
from pipeline_alignment import CLASSIFIERS, available_train_limit, dynamic_discovery, repair_missing_keywords, estimate_topic_count

CONFIG_PATH = ROOT / 'scripts/dataset_configs.json'
CONFIGS = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
PROTOCOL = json.loads((ROOT / 'scripts/pipeline_protocol.json').read_text(encoding='utf-8'))
for _config in CONFIGS.values():
    _config['improvement'] = dict(PROTOCOL['improvement_defaults'])


def compatible_cached_loading_repair(previous, current, registry_name='model_loading_repair.json'):
    """Allow only this recorded IO-only fix; keep all models/inputs/settings fixed."""
    registry = ROOT / 'scripts' / registry_name
    if not registry.exists():
        return False
    repair = json.loads(registry.read_text(encoding='utf-8'))
    if {k: v for k, v in previous.items() if k != 'implementation'} != {k: v for k, v in current.items() if k != 'implementation'}:
        return False
    before, after = previous.get('implementation', {}), current.get('implementation', {})
    changed = set(repair['after'])
    return (any(all(before.get(name) == candidate[name] for name in changed)
                for candidate in repair.get('before_versions', [repair['before']]))
            and all(after.get(name) == repair['after'][name] for name in changed)
            and {k: v for k, v in before.items() if k not in changed} == {k: v for k, v in after.items() if k not in changed})


def compatible_neighbor_repair(previous, current, output):
    """Preserve only pre-NNDescent (<4096 chunk) discovery after this fix."""
    ignored = {'implementation', 'runtime_versions'}
    if {k: v for k, v in previous.items() if k not in ignored} != {k: v for k, v in current.items() if k not in ignored}:
        return False
    old_code = previous.get('implementation', {})
    new_code = current.get('implementation', {})
    if {k: v for k, v in old_code.items() if k != 'run_pipeline.py'} != {k: v for k, v in new_code.items() if k != 'run_pipeline.py'}:
        return False
    old_versions = previous.get('runtime_versions', {})
    if any(current['runtime_versions'].get(k) != v for k, v in old_versions.items()):
        return False
    if current['runtime_versions'].get('pynndescent') != '0.5.13':
        return False
    if any((output / name).exists() for name in ('train', 'test', 'classifiers', 'update_labelspace.txt',
                                               'predictions.json', 'metrics.json', 'discovery/dynamic_selection.json')):
        return False
    clusters = list((output / 'discovery').rglob('clusters.json'))
    return bool(clusters) and all(json.loads(path.read_text(encoding='utf-8'))['samples'] < 4096 for path in clusters)


def discovery(rows, args, config):
    import numpy as np
    import torch
    from InstructorEmbedding import INSTRUCTOR
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import MiniBatchKMeans
    from umap import UMAP
    folder = args.output / 'discovery'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'clusters.json'
    if not path.exists():
        if args.clusters is not None and len(rows) <= args.clusters:
            raise ValueError('Need more training chunks than clusters')
        encoder = INSTRUCTOR('hkunlp/instructor-large', device='cuda')
        embeddings = encoder.encode([[config['instruction'], row['keyphrases']] for row in rows],
                                    batch_size=4, show_progress_bar=True)
        if not np.isfinite(embeddings).all():
            raise ValueError('Non-finite clustering embeddings')
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
        reduced = UMAP(n_neighbors=5, n_components=5, min_dist=0., metric='cosine', random_state=42).fit_transform(embeddings)
        cluster_count = args.clusters
        count_source = 'explicit/local configuration'
        count_backend = None
        if cluster_count is None:
            # The author's notebooks/bertopic.ipynb estimates topics from MiniLM
            # keyword embeddings, UMAP(5,5), HDBSCAN(min_cluster_size=20).
            from aapd_models import load_minilm
            count_encoder = load_minilm(device='cpu')
            count_vectors = count_encoder.encode([row['keyphrases'] for row in rows])
            del count_encoder
            count_reduced = UMAP(n_neighbors=5, n_components=5, min_dist=0., metric='cosine', random_state=42).fit_transform(count_vectors)
            cluster_count, count_backend = estimate_topic_count(count_reduced, args.min_cluster_size)
            count_source = 'BERTopic clustering stage; notebooks/bertopic.ipynb; excludes noise topic -1'
            if cluster_count < 2:
                raise ValueError('HDBSCAN found fewer than two topics; increase training prefix or explicitly set --clusters')
        if config['clustering'] == 'gmm':
            model = GaussianMixture(n_components=cluster_count, random_state=42).fit(reduced)
            centers = model.means_
        else:
            model = MiniBatchKMeans(n_clusters=cluster_count, random_state=42, n_init=3).fit(reduced)
            centers = model.cluster_centers_
        assignments = model.predict(reduced)
        clusters = []
        for index, center in enumerate(centers):
            members = np.flatnonzero(assignments == index)
            # Author naming uses the three original chunks nearest each center.
            nearest = np.argsort(np.linalg.norm(reduced - center, axis=1))[:3]
            clusters.append({'cluster': index, 'chunk_count': len(members),
                'examples': [{'document_id': rows[int(i)]['doc_id'], 'text': rows[int(i)]['content'],
                              'keyphrases': rows[int(i)]['keyphrases']} for i in nearest]})
        dump(path, {'samples': len(rows), 'seed': 42, 'requested_clusters': args.clusters,
                    'resolved_clusters': cluster_count, 'count_source': count_source,
                    'count_backend': count_backend, 'clusters': clusters})
    subprocess.run([sys.executable, str(ROOT / 'scripts/aapd_local_labels.py'), '--input', str(path),
                    '--output', str(folder), '--model', args.judge_model, '--domain', config['domain'],
                    '--use-text', '--dedup', args.dedup], check=True)


def labels_from(path):
    labels = list(dict.fromkeys(line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()))
    if len(labels) < 3:
        raise ValueError('Author improvement requires at least three distinct labels: ' + str(path))
    return labels


def run_dataset(options, dataset):
    started = time.monotonic()
    config = CONFIGS[dataset]
    args = SimpleNamespace(**vars(options))
    args.dataset = dataset
    args.output = (options.output_root / dataset / options.run_name).resolve()
    args.dynamic = options.subset_strategy == 'dynamic'
    directory = ROOT / 'data' / dataset
    args.train_docs = options.train_docs or (available_train_limit(directory) if args.dynamic else
                      (80 if options.profile == 'smoke' else config['train_docs']))
    args.test_docs = options.test_docs or (3 if options.profile == 'smoke' else 1000)
    args.clusters = options.clusters or (None if options.profile == 'aligned' else
                                      (16 if options.profile == 'smoke' else config['clusters']))
    args.iterations = options.iterations or (1 if options.profile == 'smoke' else 10)
    args.tail_size = 30 if options.profile == 'smoke' else config['tail_size']
    # Unpublished dataset-specific mappings are not invented: explicit release
    # defaults, overridable and recorded with their provenance.
    for key in ('majority_num', 'max_majority_num', 'sim_threshold', 'max_add_label'):
        setattr(args, key, getattr(options, key) if getattr(options, key) is not None else config['improvement'][key])
    args.remove_at_most = 0 if options.profile == 'smoke' else 6
    train, train_hash = read_rows(directory, 'train', args.train_docs, allow_empty_keywords=args.dynamic)
    test, test_hash = read_rows(directory, 'test', args.test_docs)
    truth = read_truth(directory / config['truth_file'], config)
    train_truth_path = directory / ('train_label_full.txt' if dataset == 'AAPD' else 'train_label.txt')
    train_truth = read_truth(train_truth_path, config)
    for split, rows, target in [('train', train, train_truth), ('test', test, truth)]:
        if any(row['doc_id'] >= len(target) or not target[row['doc_id']] for row in rows):
            raise ValueError(dataset + ': missing/empty truth row for selected ' + split + ' document')
    report = {'dataset': dataset, 'train': describe(train, args.train_docs), 'test': describe(test, args.test_docs),
              'truth_rows': len(truth), 'train_truth_rows': len(train_truth),
              'evaluation_vocabulary_size': len({label for row in train_truth + truth for label in row}),
              'empty_training_keyword_chunks': sum(not row['keyphrases'] for row in train),
              'fingerprints': {'train': train_hash, 'test': test_hash,
                               'test_truth': fingerprint(directory / config['truth_file']),
                               'train_truth': fingerprint(train_truth_path)}}
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / '.running.lock'
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Run already locked: {}. If interrupted, confirm its process has exited before removing this lock.'.format(lock))
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(str(os.getpid()))
        settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                    if key not in ('stage',)}
        settings['config'] = config
        settings['protocol_sources'] = PROTOCOL
        settings['runtime_versions'] = {}
        for package in ('torch', 'transformers', 'sentence-transformers', 'hdbscan', 'umap-learn', 'pynndescent', 'numba', 'llvmlite', 'scikit-learn', 'numpy'):
            try:
                settings['runtime_versions'][package] = version(package)
            except PackageNotFoundError:
                settings['runtime_versions'][package] = None
        settings['inputs'] = report['fingerprints']
        settings['initial_labels_sha256'] = fingerprint(args.initial_labels) if args.initial_labels else None
        settings['implementation'] = {name: fingerprint(ROOT / 'scripts' / name) for name in
            ('run_pipeline.py', 'pipeline_data.py', 'pipeline_alignment.py', 'pipeline_metrics.py', 'aapd_pipeline.py', 'aapd_local_labels.py', 'aapd_models.py')}
        settings['author_improvement_sha256'] = fingerprint(ROOT / 'OpenWordMLTC/self_training/self_training.py')
        settings['deviations'] = ['Qwen replaces original Llama naming and GPT judging; supplied Llama keywords retained.',
            'Dynamic stopping operationalized by cumulative semantic novelty at the recorded threshold; paper gives no precise matching rule.',
            'Unpublished per-dataset tuning is not claimed: release defaults and every override are recorded.',
            'Default test ID limit 1000 follows released evaluator, not necessarily full test set.',
            'Coverage vocabulary is union of supplied train/test labels; no manual refinement.']
        previous = args.output / 'settings.json'
        # An empty discovery directory is created before the first UMAP call.
        # A compiler failure there must not make a settings-only run immutable.
        has_model_artifacts = any((args.output / name).is_file() for name in
            ('predictions.json', 'update_labelspace.txt', 'metrics.json')) or any(
                path.is_file() for name in ('discovery', 'train', 'test', 'classifiers')
                for path in (args.output / name).rglob('*'))
        if has_model_artifacts and previous.exists():
            old_settings = json.loads(previous.read_text(encoding='utf-8'))
            if old_settings != settings:
                if compatible_cached_loading_repair(old_settings, settings):
                    dump(args.output / 'model_loading_repair.json', {
                        'reason': 'Local-first model loading; model IDs, weights selection, inputs and inference settings unchanged',
                        'previous_settings': old_settings})
                    print('Preserving completed results after cached-model loading repair.', flush=True)
                elif compatible_cached_loading_repair(old_settings, settings, 'label_response_repair.json'):
                    dump(args.output / 'label_response_repair.json', {
                        'reason': 'Label response parsing and checkpoint lock retries; preserve prior validated labels and predictions',
                        'previous_settings': old_settings})
                    print('Preserving completed results after label parsing/checkpoint saving repair.', flush=True)
                elif compatible_neighbor_repair(old_settings, settings, args.output):
                    dump(args.output / 'neighbor_dependency_repair.json', {
                        'reason': 'PyNNDescent 0.7.0 / Numba 0.58.1 parfor failure; pin 0.5.13',
                        'previous_settings': old_settings,
                        'preserved': 'Discovery prefixes with fewer than 4096 chunks; no downstream predictions exist'})
                    print('Preserving completed small-prefix discovery after nearest-neighbor dependency repair.', flush=True)
                else:
                    raise ValueError('Inputs/settings/code changed: choose a fresh --run-name')
        dump(previous, settings)
        dump(args.output / 'data_report.json', report)
        print('{}: train {} chunks/{} docs; test {} chunks/{} docs; missing test IDs {}'.format(
            dataset, len(train), report['train']['documents'], len(test), report['test']['documents'], report['test']['missing_document_ids']), flush=True)
        if args.stage == 'check':
            return
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('GPU-enabled PyTorch required; activate xmlclass')
        initial_path = args.initial_labels or args.output / 'discovery/init_label_space.txt'
        if args.stage in ('discover', 'all') and args.initial_labels is None:
            if args.dynamic:
                dynamic_discovery(train, args, config, discovery, labels_from)
            else:
                discovery(train, args, config)
        if args.dynamic and args.initial_labels is None:
            selection = json.loads((args.output / 'discovery/dynamic_selection.json').read_text(encoding='utf-8'))
            train = [row for row in train if row['doc_id'] < selection['selected_id_limit']]
            repair_missing_keywords(train, args, config, generate=False)
            report['train'] = describe(train, selection['selected_id_limit'])
            report['dynamic_converged'] = selection['converged']
            dump(args.output / 'data_report.json', report)
        if args.stage == 'discover':
            labels_from(initial_path)
            return
        initial = labels_from(initial_path)
        labels = initial
        if args.stage in ('improve', 'all'):
            infer(train, initial, args.output / 'train/initial', args.model)
            labels = improve(train, initial, args)
        elif (args.output / 'update_labelspace.txt').exists():
            labels = labels_from(args.output / 'update_labelspace.txt')
        if args.stage in ('predict', 'evaluate', 'all'):
            evaluations = {}
            for classifier in args.classifiers:
                model_name = args.model if classifier == 'deberta' else CLASSIFIERS[classifier]
                branch = args.output if len(args.classifiers) == 1 else args.output / 'classifiers' / classifier
                if args.stage == 'all' and completed_classifier(branch, test, labels, model_name):
                    evaluations[classifier] = dict(json.loads((branch / 'metrics.json').read_text(encoding='utf-8')), nli_model=model_name)
                    if len(args.classifiers) > 1:
                        dump(args.output / 'coverage_metrics.json', json.loads((branch / 'coverage_metrics.json').read_text(encoding='utf-8')))
                    print('Skipping completed classifier and evaluation: ' + classifier, flush=True)
                    continue
                if args.stage in ('predict', 'all'):
                    results = infer(test, labels, branch / 'test', model_name)
                    grouped = defaultdict(list)
                    for channel in ('keyword', 'text'):
                        for row in results[channel]:
                            grouped[row['doc_id']].append(row)
                    predictions = [{'doc_id': doc_id, 'labels': rank_document(rows)} for doc_id, rows in sorted(grouped.items())]
                    dump(branch / 'predictions.json', predictions)
                if args.stage in ('evaluate', 'all'):
                    from pipeline_metrics import evaluate
                    predictions = json.loads((branch / 'predictions.json').read_text(encoding='utf-8'))
                    if [row['doc_id'] for row in predictions] != sorted({row['doc_id'] for row in test}):
                        raise ValueError('Prediction document IDs do not match selected test documents')
                    branch_args = SimpleNamespace(**vars(args))
                    branch_args.output = branch
                    branch_args.judgment_cache = args.output / 'evaluation_judgments.json'
                    vocabulary = sorted({label for row in train_truth + truth for label in row})
                    metrics, coverage = evaluate(predictions, truth, vocabulary, [('initial', initial), ('final', labels)], branch_args)
                    evaluations[classifier] = dict(metrics, nli_model=model_name)
                    if len(args.classifiers) > 1:
                        dump(args.output / 'coverage_metrics.json', {'results': coverage})
            if evaluations:
                if len(args.classifiers) > 1:
                    dump(args.output / 'metrics.json', {'classifiers': evaluations})
                dump(args.output / 'completed.json', {'dataset': dataset, 'profile': args.profile, 'classifiers': args.classifiers, 'evaluated': True})
    finally:
        try:
            history_path = args.output / 'run_history.json'
            history = json.loads(history_path.read_text(encoding='utf-8')) if history_path.exists() else []
            history.append({'stage': args.stage, 'elapsed_seconds': time.monotonic() - started,
                            'success': sys.exc_info()[0] is None})
            dump(history_path, history)
        finally:
            lock.unlink()


def completed_classifier(branch, rows, labels, model_name):
    """Skip a finished branch only when its prediction inputs still match."""
    required = ('metrics.json', 'coverage_metrics.json', 'predictions.json', 'test/manifest.json')
    if not all((branch / name).is_file() for name in required):
        return False
    signature = hashlib.sha256(json.dumps([rows, labels, model_name, ('text', 'keyword')]).encode()).hexdigest()
    manifest = json.loads((branch / 'test/manifest.json').read_text(encoding='utf-8'))
    if manifest['signature'] != signature:
        return False
    predictions = json.loads((branch / 'predictions.json').read_text(encoding='utf-8'))
    expected_ids = sorted({row['doc_id'] for row in rows})
    return ([row['doc_id'] for row in predictions] == expected_ids and
            json.loads((branch / 'metrics.json').read_text(encoding='utf-8'))['documents'] == len(expected_ids))


def summarize(options):
    rows = []
    for dataset in CONFIGS:
        folder = options.output_root / dataset / options.run_name
        if not (folder / 'completed.json').exists():
            continue
        metrics = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
        coverage = json.loads((folder / 'coverage_metrics.json').read_text(encoding='utf-8'))['results']
        report = json.loads((folder / 'data_report.json').read_text(encoding='utf-8'))
        history = json.loads((folder / 'run_history.json').read_text(encoding='utf-8')) if (folder / 'run_history.json').exists() else []
        for classifier, scores in metrics.get('classifiers', {options.classifiers[0]: metrics}).items():
            rows.append({'dataset': dataset, 'classifier': classifier, 'run': options.run_name, 'profile': options.profile,
            'train_documents': report['train']['documents'], 'test_documents': scores['documents'],
            'initial_labels': coverage[0]['label_count'], 'final_labels': coverage[-1]['label_count'],
            'initial_coverage': coverage[0]['coverage'], 'final_coverage': coverage[-1]['coverage'],
            'author_matching_score@1': scores['author_matching_score@1'],
            'author_matching_score@3': scores['author_matching_score@3'],
            'elapsed_seconds': sum(item['elapsed_seconds'] for item in history)})
    if rows:
        path = options.output_root / ('benchmark_summary_' + options.run_name + '.csv')
        with path.open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print('Summary: ' + str(path), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=list(CONFIGS) + ['all'], required=True)
    parser.add_argument('--stage', choices=['check', 'discover', 'improve', 'predict', 'evaluate', 'all'], default='check')
    parser.add_argument('--profile', choices=['smoke', 'local', 'aligned', 'paper'], default='aligned')
    parser.add_argument('--subset-strategy', choices=['dynamic', 'fixed'])
    parser.add_argument('--growth-step', type=int, default=1000)
    parser.add_argument('--novelty-threshold', type=float, default=.8)
    parser.add_argument('--min-cluster-size', type=int, default=20)
    parser.add_argument('--dedup', choices=['exact', 'llm', 'semantic'])
    parser.add_argument('--improvement-rule', choices=['paper', 'author'])
    parser.add_argument('--classifiers', choices=list(CLASSIFIERS), nargs='+')
    parser.add_argument('--majority-num', type=int)
    parser.add_argument('--max-majority-num', type=int)
    parser.add_argument('--sim-threshold', type=float)
    parser.add_argument('--max-add-label', type=int)
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/benchmark')
    parser.add_argument('--initial-labels', type=Path, help='Reuse an explicit dataset-specific initial label file')
    for name in ('train-docs', 'test-docs', 'clusters', 'iterations'):
        parser.add_argument('--' + name, type=int)
    parser.add_argument('--model', default=NLI)
    parser.add_argument('--judge-model', default='Qwen/Qwen2.5-1.5B-Instruct')
    options = parser.parse_args()
    if options.profile == 'paper':
        print('--profile paper is an alias for aligned: Qwen substitution and unresolved release details remain recorded.', flush=True)
        options.profile = 'aligned'
    options.subset_strategy = options.subset_strategy or ('dynamic' if options.profile == 'aligned' and not options.initial_labels else 'fixed')
    options.dedup = options.dedup or ('semantic' if options.profile == 'aligned' else 'exact')
    options.improvement_rule = options.improvement_rule or ('paper' if options.profile == 'aligned' else 'author')
    options.classifiers = list(dict.fromkeys(options.classifiers or (list(CLASSIFIERS) if options.profile == 'aligned' else ['deberta'])))
    if options.initial_labels and options.dataset == 'all':
        parser.error('--initial-labels is only allowed with one dataset')
    if any(value is not None and value < 1 for value in (options.train_docs, options.test_docs, options.clusters, options.iterations)):
        parser.error('All numeric overrides must be positive')
    if options.growth_step < 1 or options.min_cluster_size < 2 or not 0 < options.novelty_threshold <= 1:
        parser.error('Require growth-step > 0, min-cluster-size >= 2 and novelty-threshold in (0,1]')
    if any(value is not None and value < 1 for value in (options.majority_num, options.max_majority_num, options.max_add_label)):
        parser.error('Improvement count overrides must be positive')
    if options.sim_threshold is not None and not 0 < options.sim_threshold < 1:
        parser.error('--sim-threshold must be in (0,1)')
    options.run_name = options.run_name or options.profile + '-v1'
    if Path(options.run_name).name != options.run_name or options.run_name in ('.', '..') or '/' in options.run_name or '\\' in options.run_name:
        parser.error('--run-name must be a single directory name')
    datasets = list(CONFIGS) if options.dataset == 'all' else [options.dataset]
    for dataset in datasets:
        run_dataset(options, dataset)
    summarize(options)


if __name__ == '__main__':
    main()
