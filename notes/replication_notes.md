# Replication Notes

Neutral documentation of repository structure and observable differences.

## Project Overview

This repository combines three individual replications from the AI for Global Challenges course. Each replication takes a different original source (paper/dataset) and implements a solution for it.

## Folder Structure

```
/
├── GSP/                      # Original code and data
├── SDGi/                    # Original dataset
├── sdgi_benchmark/          # Original benchmark code
├── HateXplain/              # Original code and data
├── r_Leo/                  # Leo's replication work
├── r_Ting/                 # Ting's replication work
├── r_Aiden/                # Aiden's replication work
└── README.md
```

---

## GSP/ vs r_Leo/

### GSP/ (Original)

- **Data**: `SDG2025.csv` (comma-separated, ISO-8859-1 encoding)
- **Code**: `Codes.py` — contains analysis code in notebook-style format
- **Structure**: Single file with cell markers (`# In[57]:`)
- **Lines**: ~1716 lines

### r_Leo/ (Replication)

- **Data**: Uses same `SDG2025.csv` (byte-for-byte identical)
- **Code files**: `main.py`, `main_full_paper.py`, `extra_stuffs/1_author_original_refactored.py`
- **Structure**: Modular functions with docstrings
- **Algorithm**: Implements KMeans clustering, silhouette scoring

### Observable Differences

- Code organization: single notebook → multiple scripts with functions
- Input format: `# In[N]:` markers → standard Python functions
- Contains `extra_stuffs/` folder with refactored original code

---

## sdgi_benchmark/ vs r_Ting/

### sdgi_benchmark/ (Original)

- **Data**: Uses `UNDP/sdgi-corpus` dataset from HuggingFace
- **Code structure**: Multiple directories (`src/preprocessing/`, `src/frameworks/`, `src/experiments/`)
- **Pipeline**: sklearn Pipeline with GridSearchCV
- **Monitoring**: MLflow tracking
- **Model**: `TfidfVectorizer` + `SGDClassifier(loss="hinge")`
- **Lines**: ~90 lines (main.py), full framework

### r_Ting/ (Replication)

- **Data**: Uses same `UNDP/sdgi-corpus` (same parquet files)
- **Code structure**: Single file `sdgi_replication.py`
- **Pipeline**: Custom `RobustMultiLabelSGD` class
- **Monitoring**: No MLflow
- **Model**: `TfidfVectorizer` + `SGDClassifier(loss="hinge")`
- **Lines**: ~589 lines (single file)

### Observable Differences

- Number of files: multiple modules → single file
- Pipeline: sklearn Pipeline vs custom class
- Hyperparameter tuning: GridSearchCV vs fixed parameters
- Monitoring: MLflow vs none

---

## HateXplain/ vs r_Aiden/

### HateXplain/ (Original)

- **Data**: `Data/dataset.json`, `Data/post_id_divisions.json`
- **Code structure**: Multiple directories (`Models/`, `Preprocess/`, `TensorDataset/`)
- **BERT**: Custom `BertPreTrainedModel` wrapper
- **Data loading**: Reads from local pickle/cache
- **Task**: 3-class classification + rationale prediction + attention
- **Lines**: Distributed across multiple files

### r_Aiden/ (Replication)

- **Data**: Uses same `dataset.json` (byte-for-byte identical)
- **Code structure**: Single `src/` directory with module files
- **BERT**: Uses `transformers.AutoModel`
- **Data loading**: Auto-downloads from URL if not present
- **Task**: Same (3-class + rationale + attention)
- **Lines**: `src/train.py` (~178), `src/model.py` (~80), etc.

### Observable Differences

- BERT implementation: custom wrapper vs transformers library
- Number of files: distributed across many directories vs unified src/
- Data loading: local cache vs auto-download

---

## Data Consistency

All replications use their respective original data sources. Hash verification:

```bash
# Verify GSP data
md5sum GSP/SDG2025.csv r_Leo/SDG2025.csv

# Verify HateXplain data
md5sum HateXplain/Data/dataset.json r_Aiden/hatexplain_aligned/data/raw/dataset.json
```

Expected output: identical hashes for each pair, confirming data consistency.

---

## Resources

- GSP Original: https://github.com/Sadullah4535/Global-Sustainability-Performance-
- SDGi Dataset: https://huggingface.co/datasets/UNDP/sdgi-corpus
- SDGi Benchmark: https://github.com/UNDP-Data/dsc-sdgi-corpus
- HateXplain: https://github.com/hate-alert/HateXplain
- HateXplain Model: https://huggingface.co/Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two

---

## Running the Replications

```bash
# Leo
cd r_Leo && python main.py

# Ting
cd r_Ting && python sdgi_replication.py

# Aiden
cd r_Aiden/hatexplain_aligned && python src/run_once.py
```