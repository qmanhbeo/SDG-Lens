# SDG Lens

Multi-label classifier with attention-based explanation proxy for UN Sustainable
Development Goals (SDGs), adapted from the HateXplain approach.

## Prerequisite

- Python 3.10–3.12
- Tested on WSL2 (Ubuntu) and Windows

The final compile stage expects these system commands on `PATH`:

- `pdflatex` to compile `manuscript/sdg_lens_manuscript.tex`

The compile step expects `manuscript/coversheet.pdf` to already exist. The
final PDF merge uses the Python `pypdf` package on all platforms, so no
`pdfunite` dependency is required.

On Ubuntu/WSL the LaTeX toolchain is typically provided by TeX Live, for example:

```bash
# General LaTex tools
sudo apt install texlive-latex-base texlive-latex-extra
# For font/margin control
sudo apt install texlive-xetex fonts-crosextra-carlito
```

On Windows, install a LaTeX distribution such as:

TeX Live:
https://tug.org/texlive/windows.html

OR MiKTeX:
https://miktex.org/download

Or you can just ignore this step because it's just manuscript compilation

## Setup

Clone the repo:

```bash
git clone https://github.com/qmanhbeo/SDG-Lens.git
cd SDG-Lens
```

Create a virtual environment with Python 3.10–3.12 and install:

```bash
pip install -r requirements.txt
```

No additional data download is required. The SDGi parquet data, all trained model
checkpoints (30 BERT + 30 TF-IDF), results, and manuscript assets are included in
the repository under `data/`, `artifacts/`, `results/`, and `manuscript/` respectively.
The pipeline will skip training and reuse the committed artifacts automatically.

## Quick Start

```bash
# One-shot full pipeline (skips training if fine-tuned checkpoint weights already exist)
# Still needs --allow-download to fetch full model from HuggingFace into local cache
# Can remove --allow-download after first sweep
python main.py sweep --allow-download

# If CUDA is not available on the your machine, run the same pipeline with `--device cpu`.
python main.py sweep --device cpu

# Check what the pipeline would do without running anything
python main.py sweep --dry-run
```

To run a subset of seeds or sizes:

```bash
python main.py sweep --seeds 42 43 --train-sizes 4000
```

If you want to retrain from scratch (e.g. to modify hyperparameters), use:

```bash
python main.py sweep --force
```


## Pipeline Stages

`main.py` orchestrates five stages:

1. **train.py** — Fine-tune a MiniLM multi-label classifier per seed/size combination.
   All checkpoints are on GitHub; this stage is a no-op when artifacts already exist.
2. **baseline.py** — Train TF-IDF + linear SVM-style baseline per seed/size combination.
   All checkpoints are on GitHub; this stage is a no-op when artifacts already exist.
3. **evaluate.py** — Compute micro-F1, macro-F1, subset accuracy, and per-label F1
   from all artifacts.
4. **visualize.py** — Generate manuscript-ready tables and charts.
5. **compile_manuscript.py** — Compile `manuscript/sdg_lens_manuscript.tex`,
   then merge the existing coversheet PDF into the final submission PDF.

## Defaults

| Parameter        | Default                |
|------------------|------------------------|
| Training sizes   | 1000, 2000, 4000       |
| Seeds            | 42, 43, 44, 45, 46     |
| Test samples     | 1470 (full test set)   |
| Test seed        | 43                     |
| Device           | cuda                   |
| Threshold        | 0.3                    |


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
  bert_train{size}_seed{seed}/   (model fine-tuned checkpoint, metrics, examples)
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
  sdg_lens_submission.pdf         (coversheet + compiled manuscript)
  visualization/
    charts/fig_per_label_comparison.png
    tables/evaluation_summary_table.tex
    tables/threshold_sweep_table.tex
```

## Notes

- Attention tokens are proxy evidence, not validated causal explanations. SDGi
  provides SDG labels but no token-level rationale ground truth.
- Both BERT and TF-IDF artifacts (all 30) are committed to the repository.
  The sweep stage skips training when fine-tuned checkpoint weights are present.
