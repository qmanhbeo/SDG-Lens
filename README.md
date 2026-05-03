# AI4GC-group

Group project for AI for Global Challenges course — combining three individual replications into a unified AI solution for sustainable development analysis.

## Project Structure

```
AI4GC-group/
├── GSP/                    # [Original] GSP code & data
├── SDGi/                  # [Original] SDGi dataset
├── sdgi_benchmark/         # [Original] SDGi benchmark code
├── HateXplain/            # [Original] HateXplain code & data
├── r_Leo/                # [Replication] Leo's work (derived from GSP)
├── r_Ting/               # [Replication] Ting's work (reimplementation)
├── r_Aiden/              # [Replication] Aiden's work (reimplementation)
└── README.md
```

---

## Original Resources (Input Data/Code)

### GSP/ — Global Sustainability Performance
- **Source**: Çelik et al. (2025), *MDPI Sustainability 17, 7411*
- **Original Repo**: https://github.com/Sadullah4535/Global-Sustainability-Performance-
- **Data**: `SDG2025.csv` — Country-level SDG index scores
- **Code**: `Codes.py` — Full analysis notebooks (1716 lines)

### SDGi/ — SDGi Dataset
- **Source**: UNDP SDGi Corpus
- **HF Dataset**: https://huggingface.co/datasets/UNDP/sdgi-corpus
- **Data**: `train.parquet`, `test.parquet` — Multi-label SDG classifications

### sdgi_benchmark/ — SDGi Benchmark Code
- **Source**: https://github.com/UNDP-Data/dsc-sdgi-corpus
- **Original Repo**: UNDP-Data/dsc-sdgi-corpus
- **Code**: `src/experiments/svm.py` — TF-IDF + SGD classifier baseline

### HateXplain/ — HateXplain Code & Data
- **Source**: Mathew et al. (2021), *AAAI 2021*
- **Original Repo**: https://github.com/hate-alert/HateXplain
- **Data**: `Data/dataset.json`, `Data/post_id_divisions.json`
- **Code**: Full training/evaluation pipeline

---

## Replication Folders (Team Work)

### r_Leo/ — Leo's Replication
- **Relationship**: Refactored from GSP/
- **Evidence**: 
  - Data is byte-for-byte identical (MD5: `618c979d3d9c17ab61074ac04880c2c8`)
  - Code uses same logic: KMeans, random_state=42, same features
  - `extra_stuffs/1_author_original_refactored.py` is direct refactor of `GSP/Codes.py`
- **Input**: `GSP/SDG2025.csv`
- **Run**: `python main.py` or `python main_full_paper.py`

### r_Ting/ — Ting's Replication
- **Relationship**: Reimplementation of SDGi benchmark
- **Evidence**:
  - Uses same dataset from `UNDP/sdgi-corpus`
  - Same fundamental approach (TF-IDF + SGD with hinge loss)
  - But different code architecture (custom `RobustMultiLabelSGD`, no MLflow)
- **Input**: `SDGi/data/` (parquet)
- **Run**: `python sdgi_replication.py`

### r_Aiden/ — Aiden's Replication
- **Relationship**: Reimplementation of HateXplain
- **Evidence**:
  - Data is byte-for-byte identical (MD5: `4f4edef96dc46c52913602fc65d49482`)
  - Same task: 3-class hate speech + rationale + attention
  - But different code: clean `BertAttentionClassifier` using modern `transformers`
- **Input**: Downloads from original URLs (or uses `r_Aiden/hatexplain_aligned/data/raw/`)
- **Run**: `python src/run_once.py`

---

## Claims & Evidence Summary

| Member | Uses Original Data | Code Relationship | Evidence |
|--------|-------------------|-----------------|----------|
| **Leo** | ✅ Same | Refactored | `r_Leo/SDG2025.csv` == `GSP/SDG2025.csv` (MD5 match) |
| **Ting** | ✅ Same | Reimplementation | Same TF-IDF+SGD approach, but independent code |
| **Aiden** | ✅ Same | Reimplementation | `r_Aiden/data/raw/dataset.json` == `HateXplain/Data/dataset.json` (MD5 match) |

### Hash Verification Commands

```bash
# Verify GSP data
md5sum GSP/SDG2025.csv r_Leo/SDG2025.csv

# Verify HateXplain data  
md5sum HateXplain/Data/dataset.json r_Aiden/hatexplain_aligned/data/raw/dataset.json
```

---

## Environment Setup

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

1. Review each replication code and outputs
2. Identify common data/patterns across replications
3. Design unified AI architecture
4. Implement combined solution
5. Evaluate and compare results

---

## References

- Çelik et al. (2025): https://doi.org/10.3390/su17167411
- UNDP SDGi: https://huggingface.co/datasets/UNDP/sdgi-corpus
- SDGi Benchmark: https://github.com/UNDP-Data/dsc-sdgi-corpus
- Mathew et al. (2021): https://arxiv.org/abs/2012.10289
- HateXplain: https://github.com/hate-alert/HateXplain