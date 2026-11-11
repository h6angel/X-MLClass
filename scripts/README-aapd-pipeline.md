# AAPD local continuation

Run from the repository root with the existing GPU-enabled `xmlclass` environment:

```powershell
python -m pip install jsonlines==4.0.0
$env:HF_HUB_DOWNLOAD_TIMEOUT = '600'
$env:HF_HUB_ETAG_TIMEOUT = '60'
python scripts/aapd_pipeline.py --stage all
```

The default `check` stage validates inputs without downloading models:

```powershell
python scripts/aapd_pipeline.py --stage check
```

If an old run stalls downloading ONNX, stop its waiting runner first and its
Python process second. Do not start a second run until those processes exit.
Then prepare the encoder explicitly:

```powershell
python scripts/aapd_pipeline.py --stage prepare
```

`aapd_models.py` reuses the completed MiniLM safetensors weights from the old
SentenceTransformer cache. It downloads only a fixed list of PyTorch model,
tokenizer and pooling configuration files into `outputs/models/all-MiniLM-L6-v2`.
All three encoder uses (retrieval, improvement and evaluation) load that local
directory. No ONNX/OpenVINO/TensorFlow export is requested. Old partial exports
are left untouched. This changes downloading only, not the embedding model.

`all` performs initial training predictions, ten label-space improvement rounds,
test predictions and semantic evaluation. `train` runs training/improvement only;
`test` uses `update_labelspace.txt` when available, otherwise the initial labels.
The run uses the original DeBERTa model, MiniLM retrieval of eight candidates,
the original hypothesis template and `multi_label=False`, followed by the
repository's rank voting across keyphrase and text chunks. MiniLM retrieval runs
on CPU, and NLI runs FP16 with batch size eight (text-label pairs) on GPU.

Improvement uses the author's `self_training` and `label_deletion` functions with
500 tail chunks, 350 majority threshold, maximum five majority labels, 0.55
similarity threshold, maximum ten additions, ten iterations and removal of labels
with at most six top predictions. A negative-index scan at the first document
boundary is guarded in the in-memory source adaptation; the original file is
unchanged. It keeps the author's cumulative space separate from the labels used
for the next training round. A space with too few labels stops with an error.

Outputs are under `outputs/aapd_pipeline`:
- `train/initial/`: keyphrase and text rankings for initial labels.
- `train/iteration_N/`: training rankings during improvement, with resumable JSONL.
- `improvement/iteration_N.json`: added/removed/majority labels and updated spaces.
- `update_labelspace.txt`: cumulative improved labels for testing.
- `test/`: test chunk rankings.
- `predictions.json`: document-level ranked labels.
- `evaluation_judgments.json`: local Qwen decisions for ambiguous semantic matches.
- `metrics.json`: final scores with actual document count and metric definition.

Re-run the same command after an interruption. Completed NLI rows are reused;
partial final JSONL writes are recovered. Improvement calculations may repeat,
but prediction manifests prevent reuse across incompatible inputs. Changed run
settings require a new output directory. Do not run two processes against the
same output directory.

Important interpretation:
- Initial labels remain the user's existing 81 labels from the first 3000 chunks,
  Qwen naming and exact-name deduplication. These prior deviations are retained.
- Subsequent training selects document IDs below 3000: 11033 aligned chunks.
- Test IDs below 1000 cover only 999 documents in the supplied chunk files:
  document ID 7 has no chunks. Evaluation reports 999 rather than fabricating a
  prediction or dividing by 1000.
- Only the author's DeBERTa branch runs by default; BART/XLM comparisons are not run.
- Ground truth comes from the supplied `test_label_full.txt`, not the missing
  `test_label.txt` path in the original evaluator. It never enters label discovery.
- Evaluation follows the author's greedy semantic matching: >=0.75 embedding
  similarity matches directly; [0.5,0.75) is judged by local Qwen instead of GPT.
  The local judge prompt is adapted, decisions are cached and should be reviewed.
- The author's normalization is hits/min(number of true labels,k), averaged over
  documents, not standard precision@k. Metrics are named accordingly. These are
  local-adaptation scores, not a claim to reproduce published scores.

For an initial three-document inference check:

```powershell
python scripts/aapd_pipeline.py --stage test --test-docs 3 --output outputs/aapd_pipeline_trial
```
