# SDG Lens

Multi-label classifier with attention-based explanation proxy for UN Sustainable
Development Goals (SDGs), adapted from the HateXplain approach.

## Quick Start

```bash
# One-shot full pipeline (skips training if artifacts already exist)
python main.py sweep

# CPU-only machines should override the default CUDA device
python main.py sweep --device cpu

# Check what the pipeline would do without running anything
python main.py sweep --dry-run
```

## Setup

Python 3.10–3.12. Tested on WSL2 (Ubuntu). Create a virtual environment and install:

```bash
pip install -r requirements.txt
```

No additional data download is required. The SDGi parquet data, all trained model
checkpoints (30 BERT + 30 TF-IDF), results, and manuscript assets are included in
the repository under `data/`, `artifacts/`, `results/`, and `manuscript/` respectively.
The pipeline will skip training and reuse the committed artifacts automatically.

If you want to retrain from scratch (e.g. to modify hyperparameters), use:

```bash
python main.py sweep --force
```

## Pipeline Stages

`main.py` orchestrates five stages:

1. **train.py** — Fine-tune a MiniLM multi-label classifier per seed/size combination.
   All checkpoints are on GitHub; this stage is a no-op when artifacts already exist.
2. **baseline.py** — Train TF-IDF + LinearSVC baseline per seed/size combination.
   All checkpoints are on GitHub; this stage is a no-op when artifacts already exist.
3. **evaluate.py** — Compute micro-F1, macro-F1, subset accuracy, and per-label F1
   from all artifacts.
4. **visualize.py** — Generate manuscript-ready tables and charts.
5. **compile_manuscript.py** — Compile `manuscript/sdg_lens_manuscript.tex` to PDF.

## Defaults

| Parameter        | Default                |
|------------------|------------------------|
| Training sizes   | 1000, 2000, 4000       |
| Seeds            | 42, 43, 44, 45, 46     |
| Test samples     | 1470 (full test set)   |
| Test seed        | 43                     |
| Device           | cuda                   |
| Threshold        | 0.3                    |

`main.py sweep` defaults to `--device cuda`. If CUDA is not available on the
marker machine, run the same pipeline with `--device cpu`.

To run a subset of seeds or sizes:

```bash
python main.py sweep --seeds 42 43 --train-sizes 4000
```

To run stages individually:

```bash
python main.py train
python main.py baseline
python main.py evaluate
python main.py visualize
python main.py compile
```

## Outputs

```
artifacts/
  bert_train{size}_seed{seed}/   (model checkpoint, metrics, examples)
  tfidf_train{size}_seed{seed}/  (model, metrics)
  job_status/                    (training status JSONs)
results/
  evaluation_by_seed.csv          (per-artifact metrics)
  evaluation_summary.csv          (mean ± std across seeds)
  evaluation_summary.json
  threshold_sweep_bert4k.json
manuscript/
  sdg_lens_manuscript.tex         (source)
  sdg_lens_manuscript.pdf         (compiled)
  visualization/
    charts/fig_per_label_comparison.png
    tables/evaluation_summary_table.tex
    tables/threshold_sweep_table.tex
```

## Notes

- Attention tokens are proxy evidence, not validated causal explanations. SDGi
  provides SDG labels but no token-level rationale ground truth.
- Both BERT and TF-IDF artifacts (all 30) are committed to the repository.
  The sweep stage skips training when artifacts are present.
- The hard word limit is 2,750. The current manuscript sits at 2,488.
