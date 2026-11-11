"""Dependency-free dataset validation. Document IDs are zero-based truth row indices."""
import hashlib
import re
from pathlib import Path


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_rows(directory, split, limit=None, allow_empty_keywords=False):
    text_path = directory / (split + '_texts_split_50.txt')
    key_path = directory / ('llama2_label_50.txt' if split == 'train' else 'llama2_label_test_50.txt')
    texts = text_path.read_text(encoding='utf-8-sig').splitlines()
    keys = key_path.read_text(encoding='utf-8-sig').splitlines()
    # Supplied keyword files cover a training subset, not the full raw corpus.
    # Validate IDs before selection, then require exact alignment within the run.
    for lines, pattern in ((texts, r'^\d+\s+.+$'), (keys, r'^\d+:.*$' if allow_empty_keywords else r'^\d+:\s*.+$')):
        if any(not re.match(pattern, line) for line in lines):
            raise ValueError(split + ': malformed source row')
    if limit is not None:
        texts = [line for line in texts if int(line.split(' ', 1)[0]) < limit]
        keys = [line for line in keys if int(line.split(':', 1)[0]) < limit]
    if len(texts) != len(keys):
        raise ValueError('{}: text/keyphrase counts differ: {} / {}'.format(split, len(texts), len(keys)))
    rows = []
    previous, chunk = -1, -1
    for index, (text, key) in enumerate(zip(texts, keys), 1):
        tm = re.match(r'^(\d+)\s+(.+)$', text)
        km = re.match(r'^(\d+):\s*(.*)$', key)
        if not tm or not km or tm[1] != km[1] or (not km[2].strip() and not allow_empty_keywords):
            raise ValueError('{} row {}: invalid or misaligned text/keyphrase'.format(split, index))
        doc_id = int(tm[1])
        if doc_id < previous:
            raise ValueError('{} row {}: document IDs must be ordered'.format(split, index))
        chunk = chunk + 1 if doc_id == previous else 0
        previous = doc_id
        if limit is None or doc_id < limit:
            # Keep original ID prefixes to preserve existing AAPD inference behavior.
            rows.append({'doc_id': doc_id, 'chunk_id': chunk, 'text': text.strip(),
                         'keyword': key.strip(), 'content': tm[2], 'keyphrases': km[2].strip()})
    if not rows:
        raise ValueError('No selected {} rows'.format(split))
    return rows, {'text': fingerprint(text_path), 'keywords': fingerprint(key_path)}


def read_truth(path, config):
    result = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        labels = []
        for label in line.split(config['separator']):
            label = label.strip()
            if config['strip_prefix']:
                label = label.split('.', 1)[-1]
            if label and label not in labels:
                labels.append(label)
        result.append(labels)
    return result


def describe(rows, expected):
    ids = {row['doc_id'] for row in rows}
    return {'chunks': len(rows), 'documents': len(ids), 'id_limit': expected,
            'missing_document_ids': sorted(set(range(expected)) - ids)}
