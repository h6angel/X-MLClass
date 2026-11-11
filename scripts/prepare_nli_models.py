"""Download NLI artifacts separately, retrying interrupted transfers in HF cache."""
import argparse
import os
from pathlib import Path
import time

# Must precede huggingface_hub import. Its standard downloader preserves and
# resumes .incomplete files; hf_transfer's path can discard those partials.
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '60')

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError, HfHubHTTPError
from requests.exceptions import RequestException

ARTIFACTS = {
    'bart': ('facebook/bart-large-mnli',
             ['config.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'tokenizer.json', 'model.safetensors']),
    'xlm': ('joeddav/xlm-roberta-large-xnli',
            ['config.json', 'tokenizer_config.json', 'sentencepiece.bpe.model', 'special_tokens_map.json', 'model.safetensors']),
}


def download_file(repo, filename, retries=12, delay=5, download=hf_hub_download, sleep=time.sleep):
    try:
        return download(repo_id=repo, filename=filename, local_files_only=True)
    except (LocalEntryNotFoundError, OSError):
        pass
    for attempt in range(1, retries + 1):
        print('{} / {}: attempt {}/{} (resume existing partial if supported)'.format(repo, filename, attempt, retries), flush=True)
        try:
            return download(repo_id=repo, filename=filename, force_download=False)
        except EntryNotFoundError:
            raise  # Do not retry a filename that the repository does not contain.
        except (RequestException, OSError) as error:
            status = getattr(getattr(error, 'response', None), 'status_code', None)
            if isinstance(error, HfHubHTTPError) and status and status < 500 and status not in (408, 429):
                raise
            if attempt == retries:
                raise RuntimeError('Download failed after {} attempts for {}/{}. Partial cache is retained; '
                                   'check proxy/network and rerun this preparation command.'.format(retries, repo, filename)) from error
            print('Interrupted ({}{}); retrying in {}s. Partial cache retained.'.format(
                type(error).__name__, ' HTTP {}'.format(status) if status else '', delay), flush=True)
            sleep(delay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=list(ARTIFACTS), default=list(ARTIFACTS))
    parser.add_argument('--retries', type=int, default=12, help='Attempts per missing file')
    parser.add_argument('--retry-delay', type=float, default=5)
    args = parser.parse_args()
    if args.retries < 1 or not 0 <= args.retry_delay <= 60:
        parser.error('Require retries >= 1 and retry-delay between 0 and 60 seconds')
    for alias in dict.fromkeys(args.models):
        repo, filenames = ARTIFACTS[alias]
        for filename in filenames:
            path = Path(download_file(repo, filename, args.retries, args.retry_delay))
            if not path.is_file() or not path.stat().st_size:
                raise RuntimeError('Downloaded file is missing or empty: ' + str(path))
            print('Ready: {} ({} bytes)'.format(filename, path.stat().st_size), flush=True)
        print('All required artifacts cached: ' + repo, flush=True)
    print('Preparation complete. No dataset experiments were run.', flush=True)


if __name__ == '__main__':
    main()
