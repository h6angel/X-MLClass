# Paper-flow experiments with local Qwen

The default profile is now `aligned`. It follows the paper's dynamic subset
growth and median-threshold addition rule, and the release's three-classifier
comparison. Qwen replaces Llama naming and GPT judging. This is not a claim of
identical published scores. No five-dataset model experiment has been started.

## Commands

Run from the repository root, using the existing environment:

```powershell
conda activate xmlclass

# Data/configuration check only; no model loaded
python scripts/run_pipeline.py --dataset all --profile aligned --stage check

# Prepare large uncached BART/XLM weights with retry/resume before experiments
python scripts/prepare_nli_models.py --models bart xlm

# Five datasets serially; each performs discovery, improvement, three tests and evaluation
python scripts/run_pipeline.py --dataset all --profile aligned --stage all --run-name aligned-v1
```

Outputs: `outputs/benchmark/<dataset>/aligned-v1/`. The command stops on the first
failure. Resume with the identical command; use a new run name for changed inputs,
code, profiles or parameters. Old local/AAPD outputs remain untouched.

The CSV `outputs/benchmark/benchmark_summary_aligned-v1.csv` has one row per
dataset/classifier: up to 15 rows. Branch outputs are isolated under
`classifiers/deberta/`, `classifiers/bart/`, and `classifiers/xlm/`. The shared
discovery/improvement stage runs once per dataset, with DeBERTa as in the release;
the three classifiers then use the same final label space. Qwen semantic
judgments are shared in the dataset-level cache.

## What has been aligned

### Dynamic training subset

The pipeline starts at 1000 document IDs, adds 1000 IDs per round, reconstructs
the initial space, and stops when no additional semantic labels appear. The
paper's reported 3000/8000/14000 subset sizes are **not forced outcomes**.
Qwen-generated labels can produce different stopping points.

The release only exposes a fixed `dynamic_iter` prefix and does not publish an
executable stopping rule. Here the reproducible operational rule is:
case-insensitive identical labels or MiniLM dot similarity >= .8 count as
already seen; compare each round against the cumulative discovered space and
retain novel labels. This threshold is configurable with `--novelty-threshold`.
It is an implementation choice, not a recovered author setting.

`discovery/dynamic_progress.json` records every prefix, its candidate/new labels
and cumulative space. `dynamic_selection.json` records the selected ID limit,
whether convergence occurred, and why it stopped. Available supplied keyword
data (or explicit `--train-docs`) is a cap. Reaching it does **not** count as
convergence. Use `--subset-strategy fixed` to deliberately reproduce a given
reported prefix size instead.

The default automatic cluster count follows the paper's nonparametric topic
count option: MiniLM keyphrase embeddings, UMAP(5 neighbors, 5 dimensions), and
HDBSCAN's non-noise topic count. These are the clustering operations in the
author's `notebooks/bertopic.ipynb` (min_cluster_size=20). The separate
`example.ipynb` uses 10; the choice here is explicit and configurable through
`--min-cluster-size`. The count then sets GMM components, or MiniBatchKMeans
clusters for Amazon as in `dynamic_GMM.py`. There is no arbitrary 128/256 mapping
in the aligned profile. `--clusters` is available as an explicit, recorded override.

Each prefix caches its clustering and naming. Generated labels are deduplicated
using MiniLM candidates in the author's code range (.80,.99), followed by Qwen
judgment. Exact-name duplicates are merged first. The local JSON output format
and keep-existing/delete-new decision convention are adaptations to make Qwen
responses parseable; this is not a reproduction of manual human refinement.

### Label-space improvement

| Dataset | Low-confidence tail chunks per round | Source |
|---|---:|---|
| AAPD | 500 | Paper section 6.4 |
| Reuters-21578 | 500 | Paper section 6.4 |
| RCV1 | 500 | Paper section 6.4 |
| DBPedia-298 | 1000 | Paper section 6.4 |
| Amazon-531 | 1500 | Paper section 6.4 |

Two explicit rules are available because the paper and released code differ:

- `--improvement-rule paper` (aligned default): recurrent keyphrases (>15
  occurrences), rare (<1% of all phrase occurrences), maximum similarity to
  current labels, median of these similarities as gamma, and additions below
  gamma from low-confidence chunks. Per-round gamma and the exact count/order
  conventions are saved in `improvement/iteration_N.json`.
- `--improvement-rule author`: call the actual released `self_training`
  implementation, using `sim_threshold + iteration/100`, its recurrence tests
  and its addition cap. The negative-index boundary guard is retained.

Low-frequency deletion (<=6 top predictions), majority exclusion, final majority
restoration, and default ten rounds follow the released implementation. The
release defaults are majority_num=350, max_majority_num=5, sim_threshold=.55,
max_add_label=10. These are now **explicit source-backed defaults**, not a claim
that a dataset-specific tuning table was recovered. No complete mapping was
found in the paper, scripts or notebooks. The same applies to the exact count
of extra Amazon rounds: the paper discusses extra rounds but does not specify
a single final configuration in its text.

Overrides are available: `--majority-num`, `--max-majority-num`,
`--sim-threshold` (author rule), `--max-add-label`, `--iterations`. With the paper
rule, the addition cap and deletion/majority settings are taken from the release
because the paper does not fully specify them. Their sources are saved in
`settings.json` and `pipeline_protocol.json`.

### Three classifiers

The default aligned comparison is:

| Branch | Exact model ID |
|---|---|
| deberta | MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33 |
| bart | facebook/bart-large-mnli |
| xlm | joeddav/xlm-roberta-large-xnli |

These models are used serially; no concurrent model loading is required. BART
and XLM-R weights may need downloading on first use. Candidate retrieval,
`This example is {label}`, multi_label=False, two-channel chunk/keyword
classification and document rank voting follow the released classifier.

For a single branch, explicitly request `--classifiers deberta`. This changes
settings/output layout and therefore requires a separate run name.

## Split execution and overrides

```powershell
python scripts/run_pipeline.py --dataset RCV1 --profile aligned --stage discover --run-name aligned-v1
python scripts/run_pipeline.py --dataset RCV1 --profile aligned --stage improve --run-name aligned-v1
python scripts/run_pipeline.py --dataset RCV1 --profile aligned --stage predict --run-name aligned-v1
python scripts/run_pipeline.py --dataset RCV1 --profile aligned --stage evaluate --run-name aligned-v1
```

Repeat all overrides identically across stages. `--initial-labels` explicitly
reuses one dataset's existing initial space and defaults to fixed subset mode;
it skips discovery and is rejected with `--dataset all`. `--model` controls
improvement and the DeBERTa branch; the BART/XLM model IDs remain explicit.

`--profile local` retains the previous fixed-subset, fixed-cluster, exact-dedup,
author-improvement, DeBERTa-only behavior. `--profile smoke` retains the small
80/3 document-ID check. `--profile paper` is an alias for `aligned`, with a
message making the Qwen substitution and unresolved metadata explicit.

## Data boundaries, repairs and evaluation

The current supplied training keyword limits are AAPD IDs <6000, Reuters <7769,
RCV1 <24400, DBPedia <20000, Amazon <29487. RCV1's next document is partially
keyed and is excluded as an incomplete boundary document. Interior mismatches
still raise errors. Some AAPD rows beyond the old 3000-ID prefix have empty
keywords. If selected during dynamic growth, Qwen fills those rows using the
original text; the original files are unchanged and raw replies are recorded
in `discovery/repaired_keywords.json`. `check` reports empties without loading Qwen.

Tests still default to IDs <1000, consistent with the released evaluator's
test_size=1000. This is not necessarily the complete dataset test split. The
actual numbers and missing IDs are saved. `--test-docs` can change the range
only when both supplied text chunks and keywords cover it.

Truth labels remain evaluation-only: Reuters is whitespace-separated; RCV1 uses
semicolons so embedded commas are preserved; Amazon/DBPedia use commas; AAPD
uses comma-separated full names with prefixes stripped as in the existing
evaluator. Coverage's inventory remains the union of supplied train/test labels,
which may differ from a manually curated full inventory.

Classification scores use the paper/release denominator min(k,true label count),
and semantic matching with local Qwen; they are named author_matching_score@k
to avoid confusion with standard precision@k. Coverage uses maximum bipartite
matching. No real labels enter discovery or improvement.

## Verification and dependencies

```powershell
python -m unittest discover -s scripts -p test_pipeline.py -v
```

Twelve fast tests cover previous AAPD input/ranking behavior, truth parsers,
misalignment, matching, dynamic convergence/resume, partial boundary handling,
the median rule and three-classifier output isolation. Classifier orchestration
tests use model stubs and do not download/run all three large models.

The added package is hdbscan==0.8.40, installed in xmlclass. The author's
requirements list 0.8.33; 0.8.40 provides a compatible Windows/Python 3.9 wheel
without changing the existing numpy/sklearn versions. All existing GPU/model
dependencies remain those used by the local AAPD pipeline.

On this Windows machine, application-control policy blocks loading the new
contrib hdbscan native DLL even outside the Codex sandbox. The pipeline therefore
can use the existing sklearn HDBSCAN implementation of the same algorithm;
min_samples is increased by one to account for its documented self-inclusion
convention. The backend and original import error are recorded in clusters.json.
This implementation difference can affect ties/numerical details and is not
hidden as exact BERTopic binary equivalence.
The min_samples adjustment follows the [sklearn HDBSCAN documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html#notes).

Primary references: [paper](https://arxiv.org/html/2407.05609v1),
[author repository](https://github.com/Kaylee0501/X-MLClass).

### Numba `Cannot add edge ... convert to parfors`

The failure was reproduced in the approximate-neighbor path (over 4096 rows):
PyNNDescent 0.7.0's hybrid_forest with Numba 0.58.1 / UMAP 0.5.4. The environment
is pinned to PyNNDescent 0.5.13, leaving UMAP, Numba, numpy and model parameters
unchanged. To restore this tested pin in another environment:

```powershell
python -m pip install --no-deps pynndescent==0.5.13
```

An empty discovery directory left by a compiler failure no longer blocks
updating the settings snapshot. A narrowly scoped repair also preserves
completed prefixes with fewer than 4096 chunks, provided inputs, all other code
and previously recorded versions match, and no downstream training/test results
exist. These small prefixes did not use the failing approximate-neighbor path.
The previous settings and repair reason are saved in neighbor_dependency_repair.json.
Other incompatible changes still require a new run name. Settings now also
record PyNNDescent, Numba and llvmlite versions.

### Hugging Face proxy failure while loading cached DeBERTa

NLI loading now tries AutoTokenizer and AutoModelForSequenceClassification with
local_files_only=True first, and passes their actual objects into pipeline().
This avoids the Hub metadata request when the cache is complete. Missing local
files still trigger normal online loading, with an actionable error on failure.
The cached DeBERTa path was verified with all requests.Session HTTP requests
blocked: one small inference completed with zero HTTP requests.

The IO-only source change has an exact before/after hash compatibility record
in scripts/model_loading_repair.json. Identical model/input/settings runs can
preserve their existing results; the prior settings are recorded in the run's
model_loading_repair.json. Other source/settings changes remain incompatible.

BART and XLM-R are not yet cached on this machine: first use still needs a
working Hugging Face connection. This repair does not disable either branch or
replace it with DeBERTa.

### Config/tokenizer cached but BART weights absent

Transformers 4.44 can raise `AttributeError: 'NoneType' object has no attribute
'endswith'` when a config-only cache has no weight checkpoint. The local-first
loader now treats this specific error as a cache miss and proceeds to the
normal online download. Unrelated AttributeErrors still propagate unchanged.
The Windows symlink warning is informational and is not this failure's cause.

On `--stage all`, completed improvement rounds reuse their saved added/removed
label state after infer() validates/reuses the prediction manifest/rows; they
do not recompute label additions. Finished classifier branches and evaluations
are skipped after validating their input signatures and document IDs. Explicit
`--stage evaluate` still forces evaluation. Incomplete NLI JSONL resumes at the
first missing row. Progress now prints `Skipping completed improvement round`
and `Skipping completed classifier and evaluation` so cache reuse is visible.

The checkpoint-scheduling/cache-loader repair is registered with exact source
hash transitions so existing compatible training and DeBERTa results survive.
No formal five-dataset experiment or large weight download was started as part
of this repair; first BART/XLM loading still requires a working connection.

### Interrupted large weight downloads / ChunkedEncodingError

Use `python scripts/prepare_nli_models.py --models bart xlm` separately before
the experiment. It downloads one PyTorch safetensors variant plus the required
configuration/tokenizer files, without loading a model or dataset. A complete
local file is reused. Broken streaming connections are retried up to twelve
times per file, using the standard Hugging Face cache's .incomplete resume path;
partial files are never force-deleted by this script. Range resume still depends
on server/proxy support. Persistent proxy failures require a working connection.
Rerunning this preparation command does not restart any experiment stage.
