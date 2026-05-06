# Solution 1: SDG Text Explainability

Minimal prototype for SDGi multi-label classification with an attention-based
explanation proxy adapted from the HateXplain approach.

This directory is a standalone copy of SDG Lens. It contains its own scripts,
local SDGi parquet data under `data/`, and saved outputs under `outputs/`.

`scripts/main.py` is the marker-facing entrypoint. The root `main.py` is only a
compatibility wrapper. The other Python files are kept as internal modules so
the implementation stays inspectable instead of becoming one large mixed-purpose
script.

The pipeline:

- loads local SDGi parquet data from `data/`
- trains a small BERT-family multi-label classifier for 17 SDG labels
- trains a TF-IDF linear baseline
- computes micro-F1 and macro-F1
- extracts top attended tokens from last-layer CLS attention
- saves trained artifacts and metadata under `artifacts/`
- writes aggregate evaluation summaries under `results/`

Check the local data:

```bash
cd SDG_Lens
python scripts/main.py sweep --dry-run
```

Run the full marker-facing pipeline:

```bash
python scripts/main.py sweep
```

The sweep runs:

1. `scripts/train.py` for BERT artifacts
2. `scripts/baseline.py` for TF-IDF artifacts
3. `scripts/evaluate.py` for mean/std results
4. `scripts/visualize.py` for manuscript-ready tables and charts
5. `scripts/compile_manuscript.py` for the final PDF

The default sweep trains BERT and TF-IDF sequentially for training sizes `2000`
and `4000` across training seeds `42`, `43`, and `44`, while holding the test
sampling seed fixed at `43`.

To rerun training instead of reusing complete artifacts:

```bash
python scripts/main.py sweep --force
```

Outputs:

```text
artifacts/
├── bert_train2000_seed42/
├── bert_train4000_seed44/
├── tfidf_train2000_seed42/
└── tfidf_train4000_seed44/
results/
├── evaluation_by_seed.csv
├── evaluation_summary.csv
├── evaluation_summary.json
└── evaluation_summary.md
outputs/
├── charts/
└── tables/
manuscript/
├── sdg_lens_manuscript.tex
└── sdg_lens_manuscript.pdf
artifacts/job_status/*.json
```

Each baseline artifact saves `results.json` plus `tfidf_model.joblib`. Each BERT
artifact saves a checkpoint, metrics, examples, and `artifact.json` metadata.

The JSON contains `run_config`, `metrics`, `examples`, and a backward-compatible
`metadata` alias. Attention scores are explanation proxies only; SDGi does not
include token-level rationale ground truth, so these explanations are not
validated rationales.

Install dependencies in a fresh environment:

```bash
pip install -r requirements.txt
```
