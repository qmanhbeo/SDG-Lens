# AI4GC-group

Group project for AI for Good Coding course.

## Project Overview

This project combines three individual replications into a unified AI solution for sustainable development analysis.

## Team Members

| Member | Replication | Original Paper/Dataset |
|--------|-------------|------------------------|
| Leo | `r_Leo/` | Celik et al. (2025) - GSP |
| Ting | `r_Ting/` | SDGi - UNDP (2025) |
| Aiden | `r_Aiden/` | Mathew et al. (2021) - HateXplain |

## Project Structure

```
AI4GC-group/
├── GSP/                    # [Original] Global Sustainability Performance data
├── SDGi/                   # [Original] SDGi dataset (UNDP/sdgi-corpus)
├── HateXplain/             # [Original] HateXplain code & data (github.com/hate-alert)
├── r_Leo/                 # [Replication] Leo's SDG benchmarking
├── r_Ting/                # [Replication] Ting's SDGi replication
├── r_Aiden/               # [Replication] Aiden's HateXplain reproduction
└── README.md
```

---

## Original Resources (Input Data/Code)

### GSP/ (Global Sustainability Performance)
- **Source**: Celik et al. (2025), *MDPI Sustainability 17, 7411*
- **Original Repo**: https://github.com/Sadullah4535/Global-Sustainability-Performance-
- **Data**: `SDG2025.csv` - SDG scores for countries
- **Description**: Country-level SDG index scores with regional and spillover metrics

### SDGi/ (SDGi Dataset)
- **Source**: UNDP SDGi Corpus
- **HF Dataset**: https://huggingface.co/datasets/UNDP/sdgi-corpus
- **Data**: `train.parquet`, `test.parquet` - SDG article classifications
- **Description**: Multi-label SDG classification dataset

### HateXplain/ (HateXplain Original)
- **Source**: Mathew et al. (2021), *AAAI 2021*
- **Original Repo**: https://github.com/hate-alert/HateXplain
- **HF Model**: https://huggingface.co/Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two
- **Data**: `Data/dataset.json` - Hate speech with rationales
- **Description**: Hate speech detection with explainable rationales

---

## Replication Folders (Team Work)

### r_Leo/ (Leo's Replication)
- **Task**: Replicate GSP paper - clustering analysis, ANOVA/MANOVA
- **Input**: `GSP/SDG2025.csv`
- **Output**: Tables 2-5, Figures 1-10
- **Files**: `main.py`, `main_full_paper.py`, `requirements.txt`
- **Run**: `python main.py` or `python main_full_paper.py`

### r_Ting/ (Ting's Replication)
- **Task**: Replicate SDGi paper - TF-IDF + linear classifier
- **Input**: `SDGi/data/` (parquet files)
- **Output**: Classification metrics, prediction files
- **Files**: `sdgi_replication.py`, `requirements_sdgi.txt`
- **Run**: `python sdgi_replication.py`

### r_Aiden/ (Aiden's Reproduction)
- **Task**: Reproduce HateXplain - BERT-based hate speech detection
- **Input**: `HateXplain/Data/dataset.json`
- **Output**: Trained model, evaluation metrics
- **Files**: `src/train.py`, `src/evaluate.py`, `configs/*.yaml`
- **Run**: `python src/run_once.py` (uses config from `configs/fast.yaml` or `configs/full.yaml`)

---

## Environment Setup

Each replication has its own requirements:

```bash
# Leo's environment
cd r_Leo && python -m venv .venv && pip install -r requirements.txt

# Ting's environment
cd r_Ting && python -m venv .venv && pip install -r requirements_sdgi.txt

# Aiden's environment
cd r_Aiden/hatexplain_aligned && conda env create -f environment.yml
```

---

## Next Steps

1. Review each replication code and verify outputs
2. Identify common data inputs across replications
3. Design unified AI architecture
4. Implement combined solution
5. Evaluate and compare results

---

## References

- Celik et al. (2025): https://doi.org/10.3390/su17167411
- UNDP SDGi: https://huggingface.co/datasets/UNDP/sdgi-corpus
- Mathew et al. (2021): https://arxiv.org/abs/2012.10289
- HateXplain Dataset: https://huggingface.co/datasets/hatexplain