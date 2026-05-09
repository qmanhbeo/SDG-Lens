# SDG Lens

Multi-label classifier with attention-based explanation proxy for UN Sustainable
Development Goals (SDGs), adapted from the HateXplain approach.

## Prerequisite

- Python 3.10–3.12
- Tested on WSL2 (Ubuntu) and Windows

The final compile stage expects this system command on `PATH`:

- `xelatex` to compile `manuscript/sdg_lens_manuscript.tex`

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
the repository under [data/](./data/), [artifacts/](./artifacts/), [results/](./results/), and [manuscript/](./manuscript/) respectively.
The pipeline will skip training and reuse the committed artifacts automatically.

## Quick Start

```bash
# One-shot ML pipeline (skips training if fine-tuned checkpoint weights already exist)
# Includes: train, baseline, evaluate, visualize, and export full attention examples
# Still needs --allow-download to fetch full model from HuggingFace into local cache
# Can remove --allow-download after first sweep
python main.py sweep --allow-download

# If CUDA is not available on your machine, run the same pipeline with `--device cpu`.
python main.py sweep --device cpu

# Check what the pipeline would do without running anything
python main.py sweep --dry-run
```

To export attention examples separately (or rerun with different options):
```bash
python main.py export-examples --limit 3 --device cpu
```

To compile the manuscript PDF (requires LaTeX), run separately:
```bash
python main.py compile
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

[main.py](./main.py) orchestrates the ML pipeline (compile is optional):

1. [train.py](./scripts/train.py) — Fine-tune a MiniLM multi-label classifier per seed/size combination.
   All checkpoints are on GitHub; this stage is a no-op when artifacts already exist.
2. [baseline.py](./scripts/baseline.py) — Train TF-IDF + linear SVM-style baseline per seed/size combination.
   All checkpoints are on GitHub; this stage is a no-op when artifacts already exist.
3. [evaluate.py](./scripts/evaluate.py) — Compute micro-F1, macro-F1, subset accuracy, and per-label F1
   from all artifacts.
4. [visualize.py](./scripts/visualize.py) — Generate manuscript-ready tables and charts.
5. [export_attention_examples.py](./scripts/export_attention_examples.py) — Export full 1057 test-set predictions
   with attention tokens to `examples_results/`. Run separately with `python main.py export-examples`. (Note: Added on 9th May)

**Optional (not part of default sweep):**
- [compile_manuscript.py](./scripts/compile_manuscript.py) — Compile `manuscript/sdg_lens_manuscript.tex` to PDF.
  Run separately with `python main.py compile`.

## Defaults

| Parameter        | Default                |
|------------------|------------------------|
| Training sizes   | 1000, 2000, 4000       |
| Seeds            | 42, 43, 44, 45, 46     |
| Test samples     | 1470 (full test set, but only 1057 are used after filtering English language)   |
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

## Reproducibility Note

This appendix describes the repository workflow exactly as implemented. The marker-facing command is `python main.py sweep`. It runs five stages in order: BERT training, TF-IDF baseline training, artifact evaluation, visualization generation, and full attention-example export. LaTeX manuscript compilation is optional and can be run separately with `python main.py compile`. By default, the sweep uses training sizes `1000`, `2000`, and `4000`; seeds `42`, `43`, `44`, `45`, and `46`; language `en`; test seed `43`; threshold `0.3`; and device `cuda`. If all artifact metadata and referenced model files already exist, the training stages skip retraining unless `--force` is passed.

### Data Source and Files

The local data files are copied SDGi Corpus parquet files stored as [data/train-00000-of-00001.parquet](./data/train-00000-of-00001.parquet) and [data/test-00000-of-00001.parquet](./data/test-00000-of-00001.parquet). The dataset is the UNDP SDG Integration Corpus (SDGi Corpus) from Hugging Face repository [UNDP/sdgi-corpus](https://huggingface.co/datasets/UNDP/sdgi-corpus), also described in the GitHub repository [UNDP-Data/dsc-sdgi-corpus](https://github.com/UNDP-Data/dsc-sdgi-corpus) and paper [CEUR-WS Vol-3764](https://ceur-ws.org/Vol-3764/paper3.pdf). The source texts come from [Voluntary National Reviews](https://hlpf.un.org/vnrs) and [Voluntary Local Reviews](https://sdgs.un.org/topics/voluntary-local-reviews); the local dataset card lists license `CC BY-NC-SA 4.0`. The raw split contains `5880` train rows and `1470` test rows. After filtering to English, the current repository has `4225` train rows and `1057` test rows. Each row contains `text`, `embedding`, `labels`, and `metadata`; the model pipeline uses `text`, `labels`, and `metadata`. Labels are 1-based SDG identifiers from `1` to `17`.

### Sampling and Preprocessing

Both model families filter by `metadata["language"] == "en"` and use the same coverage-aware sampling rule. If the requested sample size is less than the filtered split, the sampler first selects at least one available row for each SDG using `random.Random(seed)`, fills the remaining slots from a seeded shuffle of all other rows, truncates to the requested size, and shuffles the selected rows again. If the requested size is non-positive or at least the split size, it uses the full split after `pandas.sample(frac=1.0, random_state=seed)`. For the test set, the sweep passes `test_seed=43`; because `1470` exceeds the English test size, all `1057` English test rows are used in shuffled order.

TF-IDF text preprocessing replaces standalone numbers with `NUM`, removes punctuation, collapses whitespace, lowercases, and strips leading/trailing whitespace. BERT text preprocessing only converts missing text to an empty string, replaces the Unicode non-character `U+FFFE` with a space, collapses whitespace, and strips.

### BERT Training

The BERT-family model is `sentence-transformers/all-MiniLM-L6-v2`. The tokenizer is loaded with `use_fast=True`; by default the code uses local HuggingFace files only unless `--allow-download` is passed. Tokenization uses truncation, dynamic padding, `return_tensors="pt"`, and `max_length=128`. The model loads `AutoConfig` with `output_attentions=True` and eager attention where supported, then loads `AutoModel`. The encoder is frozen except for the final `2` transformer layers by default; passing `--unfreeze-encoder` trains all encoder layers. The classifier pools the final hidden state of the `[CLS]` token, applies dropout using the encoder hidden dropout probability or `0.1`, and maps the hidden size to `17` logits with one linear layer. The loss is `BCEWithLogitsLoss`.

For each BERT run, the script seeds Python, NumPy, and PyTorch with the run seed. Training uses `AdamW`, learning rate `5e-5`, batch size `8`, `3` epochs, no scheduler, no class weighting, and gradient clipping at norm `1.0`. The training loader shuffles batches; the test loader does not. Probabilities are `sigmoid(logits)`. Predictions use `probability > 0.3`; if a row has no positive labels after thresholding, the highest-probability SDG is forced on as a top-1 fallback. Each BERT artifact keeps top-level `artifact.json`, `results.json`, and `runs_index.csv` files under `artifacts/bert_train{size}_seed{seed}/`. The run-specific files — `model.pt`, `run_config.json`, `metrics.json`, `examples.json`, `results.json`, and `README_run.md` — are stored under `artifacts/bert_train{size}_seed{seed}/runs/<timestamp>/`.

### TF-IDF Baseline

The baseline converts labels with `MultiLabelBinarizer(classes=range(1,18))`. It uses `TfidfVectorizer(strip_accents="ascii", lowercase=False, ngram_range=(1,1), min_df=10, max_df=0.9, max_features=10000, binary=True)`. It trains one `SGDClassifier` per SDG label with `loss="hinge"`, `penalty="l2"`, `alpha=1e-4`, `shuffle=True`, `learning_rate="optimal"`, `random_state=seed`, `early_stopping=True`, `validation_fraction=0.2`, and `class_weight="balanced"`. If a label column has only one class in a training slice, the saved model list stores `None` for that SDG. During baseline training metrics, such a column predicts the constant training value; during the separate evaluation stage, `None` columns predict `0`. Each baseline artifact saves `results.json`, `tfidf_model.joblib`, and `artifact.json` under `artifacts/tfidf_train{size}_seed{seed}/`.

### Evaluation, Sweeps, and Reporting

The evaluation stage discovers all `artifact.json` files under `artifacts/`. BERT artifacts are reloaded from `model.pt`, rebuilt with the checkpoint model name, threshold, batch size, max length, trainable-layer setting, and language, then evaluated on the same English test split. TF-IDF artifacts reload the saved vectorizer, per-label classifiers, and binarizer, then recompute predictions on the same English test split. Metrics are micro-F1, macro-F1, weighted-F1, subset accuracy, and per-label F1 with `zero_division=0`. The summary groups rows by `model_type` and `train_size`, then reports means and sample standard deviations across available seeds.

The evaluation stage writes [results/evaluation_by_seed.csv](./results/evaluation_by_seed.csv), [results/evaluation_summary.csv](./results/evaluation_summary.csv), [results/evaluation_summary.json](./results/evaluation_summary.json), and [results/evaluation_summary.md](./results/evaluation_summary.md). It also writes two threshold sweeps at thresholds `0.20`, `0.25`, `0.30`, `0.35`, `0.40`, `0.45`, and `0.50`: [results/threshold_sweep.json](./results/threshold_sweep.json) pools all evaluated artifacts, while [results/threshold_sweep_bert4k.json](./results/threshold_sweep_bert4k.json) pools only BERT artifacts with `train_size=4000`. These sweeps use `probability >= threshold` and do not apply the BERT top-1 empty-prediction fallback. TF-IDF sweep scores are produced by applying a sigmoid transform to each classifier's `decision_function` margin; these are diagnostic scores, not calibrated probabilities.

The visualization stage reads the evaluation CSV/JSON files. It writes summary tables to `outputs/tables/`, plots micro-F1 and macro-F1 bar charts, plots BERT-vs-TF-IDF per-label F1 at 4000 training samples, writes the manuscript threshold table from [results/threshold_sweep_bert4k.json](./results/threshold_sweep_bert4k.json), and selects the largest seed-42 BERT result file for single-artifact metric, per-label, and explanation-example figures. Explanation figures show at most five examples. For each example, the report displays top label probabilities and the five highest-scoring non-special, non-padding attention tokens. Generated tables and charts are copied into [manuscript/visualization/](./manuscript/visualization/). The manuscript compiler checks that the required table and comparison charts exist, runs `pdflatex` twice with `-interaction=nonstopmode -halt-on-error`, writes the PDF, and removes temporary LaTeX files.
