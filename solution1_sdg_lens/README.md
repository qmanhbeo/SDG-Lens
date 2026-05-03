# Solution 1: SDG Text Explainability

Minimal prototype for SDGi multi-label classification with an attention-based
explanation proxy adapted from the HateXplain approach.

The script:

- loads local SDGi parquet data from `../SDGi/data`
- trains a small BERT-family multi-label classifier for 17 SDG labels
- computes micro-F1 and macro-F1
- extracts top attended tokens from last-layer CLS attention
- saves a timestamped experiment under `outputs/runs/`
- writes `outputs/results.json` as the latest-run summary

Run:

```bash
cd solution1_sdg_lens
python run.py
```

The default run is still prototype-sized: English rows only, 2000 training
samples, 300 test samples, 3 epochs, CPU device, and a cached small BERT-family
encoder (`sentence-transformers/all-MiniLM-L6-v2`). It fine-tunes only the last
2 transformer blocks by default.

To use CUDA explicitly:

```bash
python run.py --device cuda
```

Outputs:

```text
outputs/results.json
outputs/runs_index.csv
outputs/runs/YYYYMMDD_HHMMSS/
├── model.pt
├── run_config.json
├── metrics.json
├── examples.json
├── results.json
└── README_run.md
```

Reload and revalidate a saved checkpoint:

```bash
python run.py --load outputs/runs/YYYYMMDD_HHMMSS/model.pt --eval-only
```

Create report figures from a run:

```bash
python plot_results.py --results outputs/runs/YYYYMMDD_HHMMSS/results.json
```

This writes:

```text
fig_metrics.png
fig_per_label_f1.png
fig_explanation_examples.png
```

The JSON contains `run_config`, `metrics`, `examples`, and a backward-compatible
`metadata` alias. Attention scores are explanation proxies only; SDGi does not
include token-level rationale ground truth, so these explanations are not
validated rationales.
