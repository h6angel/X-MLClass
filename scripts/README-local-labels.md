# AAPD local label naming

This adaptation consumes `outputs/aapd_smoke/clusters.json`. It names clusters
from their representative **keyphrases**, unlike the original script that uses
representative original text chunks. It generates one label per nonempty cluster
and by default merges only identical names (ignoring case). No API key is required.
The optional `--dedup llm` uses Qwen to merge synonyms, but the 1.5B trial
incorrectly merged unrelated topics, so semantic merging is experimental.
Changing the deduplication mode reuses saved names and rebuilds merge decisions.
This is a local feasibility experiment, not the paper's exact reproduction.

Keep INSTRUCTOR in the existing `xmlclass` environment. Qwen needs newer
Transformers, so run naming in a separate environment (PowerShell):

```powershell
conda create -n xmlclass-local python=3.9 -y
conda activate xmlclass-local
cd "C:\mlearn\code\xml class"
python -m pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu121 --timeout 1000 --retries 10
python -m pip install -r scripts/requirements-local-labels.txt
python scripts/aapd_local_labels.py --limit 3 --output outputs/aapd_local_trial
```

The first run downloads Qwen/Qwen2.5-1.5B-Instruct. Model weights use roughly
3 GB in FP16; runtime memory is higher. Batch size is one. No training occurs.
After reviewing the trial, process all clusters:

```powershell
python scripts/aapd_local_labels.py
```

Optional 3B experiment (roughly 6 GB weights, tight on an 8 GB GPU):

```powershell
python scripts/aapd_local_labels.py --model Qwen/Qwen2.5-3B-Instruct --output outputs/aapd_local_3b
```

Outputs:
- `init_labelspace.txt`: generated names before merging.
- `init_label_space.txt`: one canonical label per line, for downstream NLI.
- `progress.json`: cluster mapping, raw replies, and resumable progress.

Use the same command to resume an interrupted run. A changed input/model/limit
requires a different output directory. Invalid model replies stop the run instead
of silently becoming labels. Review naming and merge decisions in progress.json;
small-model semantic judgments are fallible. Existing paper scripts are unchanged
and still reference `data/AAPD/llama2/init_label_space.txt`; downstream code must
explicitly select the local output before classification. Label naming alone
does not complete classification, evaluation, or self-training.
