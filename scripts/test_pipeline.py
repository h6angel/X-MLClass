"""Fast regression checks; no model downloads or GPU required."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json

from aapd_pipeline import read_chunks, rank_document
from pipeline_data import read_rows, read_truth
from pipeline_metrics import maximum_matching
from run_pipeline import ROOT, CONFIGS
from pipeline_alignment import growth_limits, available_train_limit, dynamic_discovery, select_paper_additions, CLASSIFIERS
from aapd_local_labels import parse_json, parse_label_response, save_checkpoint


class PipelineTests(unittest.TestCase):
    def test_malformed_generation_retries_with_correction_and_stops(self):
        import ast
        from contextlib import nullcontext
        from unittest.mock import MagicMock
        source = ast.parse((ROOT / 'scripts/aapd_local_labels.py').read_text(encoding='utf-8'))
        ask = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == 'ask')
        tokenizer, model = MagicMock(), MagicMock()
        inputs = MagicMock()
        inputs.__getitem__.return_value.shape = (1, 10)
        tokenizer.return_value.to.return_value = inputs
        scope = dict(args=SimpleNamespace(domain='news'), tokenizer=tokenizer, model=model,
                     torch=SimpleNamespace(inference_mode=nullcontext), parse_json=parse_json)
        exec(compile(ast.Module(body=[ask], type_ignores=[]), '<ask>', 'exec'), scope)
        malformed = '{"label":"Security",""}'
        tokenizer.decode.side_effect = [malformed, '{"label":"Security"}']
        self.assertEqual(scope['ask']('Name this topic'), ({'label': 'Security'}, '{"label":"Security"}'))
        self.assertEqual(model.generate.call_count, 2)
        correction = tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual(correction[-2]['content'], malformed)
        model.reset_mock()
        tokenizer.decode.side_effect = [malformed] * 3
        with self.assertRaisesRegex(ValueError, 'after 3 attempts'):
            scope['ask']('Name this topic')
        self.assertEqual(model.generate.call_count, 3)

    def test_checkpoint_retries_locks_and_preserves_old_state_on_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            checkpoint = Path(temporary) / 'progress.json'
            checkpoint.write_text('{"named": []}', encoding='utf-8')
            replace = Path.replace
            calls = []

            def briefly_locked(source, target):
                calls.append(1)
                if len(calls) < 3:
                    raise PermissionError('File in use')
                return replace(source, target)

            with patch.object(Path, 'replace', briefly_locked), patch('aapd_local_labels.time.sleep'):
                save_checkpoint(checkpoint, {'named': ['Finance']})
            self.assertEqual(json.loads(checkpoint.read_text()), {'named': ['Finance']})
            with patch.object(Path, 'replace', side_effect=PermissionError('File in use')), patch('aapd_local_labels.time.sleep'):
                with self.assertRaises(PermissionError):
                    save_checkpoint(checkpoint, {'named': ['Business']})
            self.assertEqual(json.loads(checkpoint.read_text()), {'named': ['Finance']})

    def test_label_response_accepts_plain_category_but_not_broken_json(self):
        for reply in ('Automobile Industry', '  Automobile Industry\n',
                      '{"label": "Automobile Industry"}', '["Automobile Industry"]'):
            self.assertEqual(parse_label_response(reply), {'label': 'Automobile Industry'})
        for reply in ('', '[]', '["Finance", "Banking"]', '[123]', '[""]', '["Finance"', '{"label": "Automobile Industry"',
                      'Automobile Industry\nBanking', 'Label: Automobile Industry',
                      'The category is Automobile Industry.'):
            with self.assertRaises(ValueError):
                parse_label_response(reply)
        # Deduplication must still require the structured duplicate_of decision.
        with self.assertRaises(ValueError):
            parse_json('Automobile Industry')

    def test_aapd_inputs_and_ranking_preserved(self):
        for split, limit in [('train', 3000), ('test', 1000)]:
            old = read_chunks(split, limit)
            new, _ = read_rows(ROOT / 'data/AAPD', split, limit)
            self.assertEqual(old, [{k: row[k] for k in ('doc_id', 'text', 'keyword')} for row in new])
        rows = [{'labels': ['a', 'b', 'c'], 'scores': [.8, .15, .05]},
                {'labels': ['b', 'a', 'c'], 'scores': [.9, .08, .02]}]
        self.assertEqual(rank_document(rows), ['b', 'c'])

    def test_truth_parsers(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            path = Path(temporary) / 'truth.txt'
            for dataset, content, expected in [
                ('AAPD', 'cs.AI, stat.Machine Learning,\n', ['AI', 'Machine Learning']),
                ('RCV1', 'government/social; crime, law enforcement;\n', ['government/social', 'crime, law enforcement']),
                ('Amazon-531', 'beauty, hair_care,\n', ['beauty', 'hair_care']),
                ('Reuters-21578', 'acq earn\n', ['acq', 'earn'])]:
                path.write_text(content, encoding='utf-8')
                self.assertEqual(read_truth(path, CONFIGS[dataset]), [expected])

    def test_alignment_rejects_wrong_ids_and_missing_chunks(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            directory = Path(temporary)
            (directory / 'train_texts_split_50.txt').write_text('0 first\n0 second\n1 third\n', encoding='utf-8')
            path = directory / 'llama2_label_50.txt'
            path.write_text('0: a\n1: b\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                read_rows(directory, 'train', 1)
            path.write_text('0: a\n0: b\n', encoding='utf-8')
            rows, _ = read_rows(directory, 'train', 1)
            self.assertEqual([r['chunk_id'] for r in rows], [0, 1])
            # A requested larger selection must not silently drop unkeyed documents.
            with self.assertRaises(ValueError):
                read_rows(directory, 'train', 2)

    def test_maximum_matching_reassigns_edges(self):
        pairs = maximum_matching({'real1': ['p1', 'p2'], 'real2': ['p1']})
        self.assertEqual(len(pairs), 2)
        self.assertEqual(len({p['prediction'] for p in pairs}), 2)

    def test_pretrained_loading_uses_cache_before_network(self):
        from aapd_models import local_first_pretrained
        from unittest.mock import Mock
        factory = Mock()
        factory.from_pretrained.return_value = 'cached-model'
        self.assertEqual(local_first_pretrained(factory, 'model-id'), 'cached-model')
        factory.from_pretrained.assert_called_once_with('model-id', local_files_only=True)
        factory.reset_mock()
        factory.from_pretrained.side_effect = [OSError('missing cache'), 'downloaded-model']
        self.assertEqual(local_first_pretrained(factory, 'model-id'), 'downloaded-model')
        self.assertEqual(factory.from_pretrained.call_args_list[1].kwargs, {})
        factory.from_pretrained.side_effect = [AttributeError("'NoneType' object has no attribute 'endswith'"), 'downloaded-weights']
        self.assertEqual(local_first_pretrained(factory, 'partial-bart'), 'downloaded-weights')
        factory.from_pretrained.side_effect = AttributeError('unrelated model bug')
        with self.assertRaisesRegex(AttributeError, 'unrelated'):
            local_first_pretrained(factory, 'model-id')

    def test_completed_classifier_rejects_changed_model(self):
        import hashlib
        from aapd_pipeline import dump
        from run_pipeline import completed_classifier
        rows, labels = [{'doc_id': 0, 'text': 'text'}], ['a', 'b', 'c']
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            branch = Path(temporary)
            signature = hashlib.sha256(json.dumps([rows, labels, 'nli', ('text', 'keyword')]).encode()).hexdigest()
            dump(branch / 'test/manifest.json', {'signature': signature})
            dump(branch / 'predictions.json', [{'doc_id': 0, 'labels': labels}])
            dump(branch / 'metrics.json', {'documents': 1})
            dump(branch / 'coverage_metrics.json', {'results': []})
            self.assertTrue(completed_classifier(branch, rows, labels, 'nli'))
            self.assertFalse(completed_classifier(branch, rows, labels, 'other-model'))
            self.assertFalse(completed_classifier(branch, rows, ['new-label'], 'nli'))

    def test_completed_improvement_rounds_do_not_recompute_additions(self):
        import aapd_pipeline
        from aapd_pipeline import dump
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            output = Path(temporary)
            initial = ['a', 'b', 'c']
            states = [dict(cumulative_labels=['a', 'b', 'c', 'd'], next_training_labels=['b', 'c', 'd'],
                           majority_excluded_from_next_training=['a']),
                      dict(cumulative_labels=['a', 'b', 'c', 'd', 'e'], next_training_labels=['c', 'd', 'e'],
                           majority_excluded_from_next_training=['a', 'b'])]
            for i, state in enumerate(states):
                dump(output / 'improvement' / ('iteration_{}.json'.format(i)), state)
            args = SimpleNamespace(output=output, dataset='AAPD', model='nli', iterations=2, improvement_rule='paper')
            with patch.object(aapd_pipeline, 'infer', return_value={'text': []}) as inference, \
                 patch('aapd_models.load_minilm', side_effect=AssertionError('Completed round recomputed')):
                labels = aapd_pipeline.improve([{'doc_id': 0, 'keyword': '0: phrase'}], initial, args)
            self.assertEqual(labels, states[-1]['cumulative_labels'])
            self.assertEqual(inference.call_args_list[0].args[1], initial)
            self.assertEqual(inference.call_args_list[1].args[1], states[0]['next_training_labels'])

    def test_neighbor_fix_preserves_only_small_discovery_prefixes(self):
        from run_pipeline import compatible_neighbor_repair
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            folder = Path(temporary)
            path = folder / 'discovery/prefix_1000/discovery/clusters.json'
            path.parent.mkdir(parents=True)
            path.write_text('{"samples": 3600}', encoding='utf-8')
            old = {'model': 'unchanged', 'implementation': {'run_pipeline.py': 'old', 'pipeline_alignment.py': 'same'},
                   'runtime_versions': {'numba': '0.58.1'}}
            new = dict(old, implementation=dict(old['implementation'], **{'run_pipeline.py': 'new'}),
                       runtime_versions={'numba': '0.58.1', 'pynndescent': '0.5.13'})
            self.assertTrue(compatible_neighbor_repair(old, new, folder))
            self.assertFalse(compatible_neighbor_repair(old, dict(new, model='changed'), folder))
            path.write_text('{"samples": 4200}', encoding='utf-8')
            self.assertFalse(compatible_neighbor_repair(old, new, folder))

    def test_dynamic_growth_stops_on_no_new_labels_and_resumes(self):
        import numpy as np
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            output = Path(temporary)
            args = SimpleNamespace(output=output, train_docs=4000, growth_step=1000, novelty_threshold=.8)
            rows = [{'doc_id': i, 'chunk_id': 0, 'keyphrases': 'existing'} for i in range(4000)]
            calls = []
            def discover(selected, step_args, config):
                limit = len(selected)
                calls.append(limit)
                labels = ['label' + str(i) for i in range(8 if limit == 1000 else 9)]
                path = step_args.output / 'discovery/init_label_space.txt'
                path.parent.mkdir(parents=True)
                path.write_text('\n'.join(labels), encoding='utf-8')
            encoder = SimpleNamespace(encode=lambda names: np.array([np.eye(9)[int(name[5:])] for name in names]))
            with patch('aapd_models.load_minilm', return_value=encoder):
                result = dynamic_discovery(rows, args, {}, discover, lambda p: p.read_text())
                self.assertEqual(result, 3000)
                self.assertEqual(calls, [1000, 2000, 3000])
                self.assertTrue(json.loads((output / 'discovery/dynamic_selection.json').read_text())['converged'])
                dynamic_discovery(rows, args, {}, discover, lambda p: p.read_text())
                self.assertEqual(calls, [1000, 2000, 3000])
        self.assertEqual(growth_limits(2400), [1000, 2000, 2400])

    def test_incomplete_keyword_boundary_is_not_selected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            directory = Path(temporary)
            (directory / 'train_texts_split_50.txt').write_text('0 first\n1 second\n1 third\n', encoding='utf-8')
            (directory / 'llama2_label_50.txt').write_text('0: a\n1: b\n', encoding='utf-8')
            self.assertEqual(available_train_limit(directory), 1)

    def test_paper_addition_uses_median_instead_of_fixed_threshold(self):
        import numpy as np
        phrases = ['rare_a'] * 16 + ['rare_b'] * 16 + ['rare_c'] * 16 + ['frequent'] * 2000
        rows = [{'keyphrases': phrase} for phrase in phrases]
        predictions = [{'scores': [.1 if i < 48 else .9]} for i in range(len(rows))]
        mapping = {'old': [1., 0.], 'rare_a': [.1, 0.], 'rare_b': [.5, 0.], 'rare_c': [.9, 0.]}
        added, details = select_paper_additions(rows, predictions, ['old'], 48, 10,
                                               lambda names: np.array([mapping[name] for name in names]))
        self.assertEqual(added, ['rare_a'])
        self.assertEqual(details['gamma'], .5)

    def test_three_classifiers_share_improved_labels_and_have_separate_outputs(self):
        import run_pipeline
        from aapd_pipeline import dump
        import torch
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            calls = []
            def inference(rows, labels, output, model, *channels):
                calls.append((model, tuple(labels), str(output)))
                records = [dict(doc_id=r['doc_id'], labels=labels[:8], scores=[.9] * 8) for r in rows]
                return {'text': records, 'keyword': records}
            def evaluation(predictions, truth, vocabulary, spaces, args):
                metrics = {'documents': len(predictions), 'author_matching_score@1': .5, 'author_matching_score@3': .5}
                coverage = [{'label_count': len(labels), 'coverage': .5} for _, labels in spaces]
                dump(args.output / 'metrics.json', metrics)
                dump(args.output / 'coverage_metrics.json', {'results': coverage})
                return metrics, coverage
            command = ['run_pipeline.py', '--dataset', 'AAPD', '--profile', 'aligned', '--subset-strategy', 'fixed',
                       '--initial-labels', str(ROOT / 'outputs/aapd_local/init_label_space.txt'), '--train-docs', '2',
                       '--test-docs', '1', '--iterations', '1', '--output-root', temporary, '--stage', 'all']
            with patch('sys.argv', command), patch.object(torch.cuda, 'is_available', return_value=True), \
                 patch.object(run_pipeline, 'infer', side_effect=inference), \
                 patch.object(run_pipeline, 'improve', side_effect=lambda rows, labels, args: labels + ['improved-label']), \
                 patch('pipeline_metrics.evaluate', side_effect=evaluation):
                run_pipeline.main()
            tests = calls[1:]
            self.assertEqual([call[0] for call in tests], list(CLASSIFIERS.values()))
            self.assertTrue(all('improved-label' in call[1] for call in tests))
            self.assertEqual(len({call[2] for call in tests}), 3)
            folder = Path(temporary) / 'AAPD/aligned-v1'
            self.assertEqual(set(json.loads((folder / 'metrics.json').read_text())['classifiers']), set(CLASSIFIERS))
            self.assertEqual(len((Path(temporary) / 'benchmark_summary_aligned-v1.csv').read_text(encoding='utf-8-sig').splitlines()), 4)


if __name__ == '__main__':
    unittest.main()
