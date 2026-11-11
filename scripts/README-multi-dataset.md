# Five independent local X-MLClass experiments

> Updated default: `aligned` now provides dynamic subset growth, automatic topic
> counts, source-backed improvement rules, semantic deduplication with Qwen and
> all three classifiers. See [README-aligned-pipeline.md](README-aligned-pipeline.md).
> The details below describe the earlier `--profile local` behavior. Its `paper`
> gating note is superseded: `paper` is now an explicitly qualified alias for `aligned`.

Run from the repository root in the existing GPU-enabled environment:

```powershell
conda activate xmlclass
python scripts/run_pipeline.py --dataset all --profile local --stage check
```

`check` uses the standard library only, loads no models, and writes a data report
and a settings snapshot for each dataset. It validates selected text/keyphrase
rows, ordered document IDs, truth row bounds, and nonempty truth labels. The
downloaded keyword files cover subsets of the raw training files: only the
selected ID range must align exactly. IDs are zero-based truth-file row indices.
Missing documents are recorded; predictions are never fabricated for them.

## Run the five experiments sequentially

```powershell
python scripts/run_pipeline.py --dataset all --profile local --stage all --run-name local-v1
```

This runs AAPD, Reuters-21578, RCV1, DBPedia-298, and Amazon-531 **one at a time**.
Each constructs its own labels, improves them using its own training corpus,
predicts its own test corpus, and evaluates against its own truth labels.
The first failure stops execution. Rerun the identical command to resume.
Do not launch another GPU experiment concurrently.

Results: `outputs/benchmark/<dataset>/local-v1/`.
Comparison table: `outputs/benchmark/benchmark_summary_local-v1.csv`.
Existing `outputs/aapd_pipeline` and `outputs/aapd_local` are not overwritten.

## Small checks and individual stages

```powershell
# Complete small end-to-end check: 80 training IDs, 3 test IDs, 16 clusters, 1 round
python scripts/run_pipeline.py --dataset Reuters-21578 --profile smoke --stage all

# Run stages separately, keeping all settings and the run name identical
python scripts/run_pipeline.py --dataset RCV1 --stage discover --run-name local-v1
python scripts/run_pipeline.py --dataset RCV1 --stage improve --run-name local-v1
python scripts/run_pipeline.py --dataset RCV1 --stage predict --run-name local-v1
python scripts/run_pipeline.py --dataset RCV1 --stage evaluate --run-name local-v1

# Explicit reuse of an existing AAPD initial label space
python scripts/run_pipeline.py --dataset AAPD --stage all --initial-labels outputs/aapd_local/init_label_space.txt --run-name reused-aapd-v1

# Fast regression checks without GPU/models
python -m unittest discover -s scripts -p test_pipeline.py -v
```

`predict` uses improved labels when present, otherwise initial labels. `all`
always includes improvement; `evaluate` requires existing predictions. Never
provide one initial-label file to multiple datasets (`--dataset all` rejects it).

Optional overrides: `--train-docs`, `--test-docs`, `--clusters`, `--iterations`,
`--model`, `--judge-model`, `--output-root`. Document limits select **IDs less
than the limit**, not a count of existing documents or chunks. Choose a new
`--run-name` when changing any input, code, model, or setting. For split-stage
execution, repeat any overrides on every invocation.

## Profiles and reproducibility

`local` is a runnable adaptation, **not an exact published-score reproduction**.
All defaults are visible in `dataset_configs.json` and `settings.json`:

| Dataset | Training ID limit | Tail chunks | Local clusters | Clustering |
|---|---:|---:|---:|---|
| AAPD | 3000 | 500 | 84 | GMM |
| Reuters-21578 | 3000 | 500 | 128 | GMM |
| RCV1 | 3000 | 500 | 128 | GMM |
| DBPedia-298 | 8000 | 1000 | 256 | GMM |
| Amazon-531 | 14000 | 1500 | 398 | MiniBatchKMeans |

Training limits/tail sizes follow paper section 6.4. Cluster counts are explicit
**local choices**, not verified paper mappings. All datasets default to ten
improvement rounds and test IDs below 1000. Local shared improvement settings:
majority threshold 350, maximum five majority labels, similarity threshold .55,
maximum ten additions, deletion at six or fewer top predictions. The `smoke`
profile uses 30 tail chunks and deletes only zero-frequency labels; its results
are connectivity checks and should not be compared with full runs.

Discovery uses INSTRUCTOR-large, UMAP and clustering, then local Qwen naming of
the three original text chunks nearest each center and exact-name deduplication.
It uses existing Llama-2 keywords supplied with each dataset. It does not
regenerate keywords or implement the paper's dynamic subset-size search.
Fewer than three generated labels stops with an explicit error; adjust the
training subset/cluster count in a fresh run rather than inventing labels.

Inference and improvement reuse `aapd_pipeline.py`: MiniLM retrieves eight
candidates, DeBERTa scores text and keyword channels in FP16, and the author's
rank voting merges them. The author's improvement functions are reused with
the backward-index boundary guard. Other NLI model comparisons are not automatic.

`paper` now aliases `aligned`; see the updated document for the exact sources,
remaining operational choices and the Qwen substitutions.

## Evaluation

Truth files: AAPD uses `*_label_full.txt`; others use `*_label.txt`. RCV1 labels
are separated by semicolons (commas inside label names are preserved).
Reuters labels are whitespace-separated (for example, `acq earn`). AAPD
prefixes such as `cs.` are stripped, matching the existing evaluator.

`metrics.json` contains `author_matching_score@1` and `@3`, with denominator
`min(k, number of true labels)`. These are not standard precision@k. The greedy
semantic matching follows the existing AAPD implementation: MiniLM dot score
>= .75 matches, [.5,.75) is judged by local Qwen, and < .5 does not match.
Judgments are cached; invalid Yes/No responses stop the run for inspection.

`coverage_metrics.json` evaluates initial/final label spaces with maximum
cardinality bipartite matching. Its truth vocabulary is the union of labels in
the supplied train/test truth files, read only for validation/evaluation.
This vocabulary choice is recorded and may differ from the author's separately
curated ground-truth inventory. It never becomes a discovery input.

`data_report.json` records actual selected document/chunk counts and missing IDs.
At default limits: AAPD has 999 test documents (missing ID 7); Reuters has 995
(missing IDs 113,321,666,677,835) and 2979 training documents; others have 1000
test documents. Scores divide by the actual evaluated document count.

## Resume and dependencies

Each run snapshots input hashes, implementation hashes, models, configuration
and deviations. Incompatible changes require a fresh run name. NLI JSONL rows
and naming checkpoints resume; clustering resumes after `clusters.json` is
saved; improvement may repeat calculations while reusing NLI rows.

A `.running.lock` prevents simultaneous writers. After forced termination,
confirm the recorded PID has exited before removing that specific lock file.
Normal errors remove the lock automatically. Only completed evaluations enter
the comparison CSV.

Use the existing `xmlclass` environment; dependencies are those already used by
the AAPD scripts: GPU PyTorch, transformers, sentence-transformers,
InstructorEmbedding, umap-learn, scikit-learn, numpy, jsonlines and tqdm.
Initial model access may require network access. No OpenAI key is needed for
the local profile.
