"""Prepare only the PyTorch MiniLM files, then load SentenceTransformer locally."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
REPO = 'sentence-transformers/all-MiniLM-L6-v2'


def local_first_pretrained(factory, model_name, **kwargs):
    """Use complete cached artifacts without a remote metadata/HEAD request."""
    try:
        return factory.from_pretrained(model_name, local_files_only=True, **kwargs)
    except (OSError, AttributeError) as error:
        # Transformers 4.44 may resolve a config-only cached snapshot to a
        # None checkpoint instead of raising OSError for absent weights.
        if isinstance(error, AttributeError) and str(error) != "'NoneType' object has no attribute 'endswith'":
            raise
        print('Local cache incomplete for {}; trying Hugging Face download.'.format(model_name), flush=True)
        try:
            return factory.from_pretrained(model_name, **kwargs)
        except Exception as error:
            raise RuntimeError('Could not load {}: cache is incomplete and online loading failed. '
                               'Check the configured proxy/network, then rerun the same command. '
                               'Existing predictions are preserved.'.format(model_name)) from error


def load_nli_pipeline(model_name):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    tokenizer = local_first_pretrained(AutoTokenizer, model_name)
    model = local_first_pretrained(AutoModelForSequenceClassification, model_name, torch_dtype=torch.float16)
    # Passing actual objects avoids pipeline() probing model_name on the Hub.
    return pipeline('zero-shot-classification', model=model, tokenizer=tokenizer, device=0)


FILES = (
    'config.json', 'config_sentence_transformers.json', 'modules.json',
    'sentence_bert_config.json', 'special_tokens_map.json', 'tokenizer.json',
    'tokenizer_config.json', 'vocab.txt', '1_Pooling/config.json',
    'model.safetensors',
)


def prepare_minilm():
    from huggingface_hub import hf_hub_download
    destination = ROOT / 'outputs/models/all-MiniLM-L6-v2'
    old_cache = Path.home() / '.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2'
    destination.mkdir(parents=True, exist_ok=True)
    # The old snapshot downloader has already fetched the complete weights.
    # Reuse only complete files; leave its ONNX partial download untouched.
    for filename in FILES:
        target = destination / filename
        if target.is_file() and target.stat().st_size:
            continue
        cached = old_cache / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if cached.is_file() and cached.stat().st_size:
            print('Reusing MiniLM file: ' + filename, flush=True)
            shutil.copy2(cached, target)
        else:
            print('Downloading required MiniLM file: ' + filename, flush=True)
            downloaded = hf_hub_download(repo_id=REPO, filename=filename)
            partial = target.with_suffix(target.suffix + '.partial')
            shutil.copy2(downloaded, partial)
            partial.replace(target)
    return destination


def load_minilm(device='cpu'):
    from sentence_transformers import SentenceTransformer
    # A directory containing modules.json skips the old whole-repository fetch.
    return SentenceTransformer(str(prepare_minilm()), device=device)


def author_sentence_transformer(model_name, *args, **kwargs):
    from sentence_transformers import SentenceTransformer
    if model_name == REPO:
        model_name = str(prepare_minilm())
    return SentenceTransformer(model_name, *args, **kwargs)
