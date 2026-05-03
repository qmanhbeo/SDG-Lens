# AI4GC-group

Group project for AI for Global Challenges — combining three individual replications into a unified AI solution for sustainable development analysis.

## Project Structure

```
AI4GC-group/
├── GSP/                    # [Original] GSP code & data
├── SDGi/                  # [Original] SDGi dataset
├── sdgi_benchmark/         # [Original] SDGi benchmark code
├── HateXplain/            # [Original] HateXplain code & data
├── r_Leo/                # [Replication] Leo's work
├── r_Ting/               # [Replication] Ting's work
├── r_Aiden/              # [Replication] Aiden's work
└── README.md
```

---

## Original vs Replication: What's Actually Different

### Leo: GSP/ vs r_Leo/

| Aspect | GSP/ (Original) | r_Leo/ (Replication) |
|--------|---------------|-------------------|
| Format | Jupyter notebooks (1716 lines) | Clean Python scripts |
| Structure | `# In[57]:` cells | Functions + docstrings |
| Features | All figures in one file | Split: `main.py` (minimal), `main_full_paper.py` (full) |
| Data | SDG2025.csv | Same (byte-for-byte) |
| Algorithm | Same | Same |

**What Leo changed**: Refactored notebook code into modular, reusable scripts.

---

### Ting: sdgi_benchmark/ vs r_Ting/

| Aspect | sdgi_benchmark/ (Original) | r_Ting/ (Replication) |
|--------|----------------------|--------------------|
| Architecture | Full ML pipeline with sklearn Pipeline | Custom class `RobustMultiLabelSGD` |
| Hyperparameter tuning | GridSearchCV with MLflow | No tuning (fixed alpha) |
| Monitoring | MLflow tracking | None |
| Code size | Multiple files (preprocessing/, frameworks/, experiments/) | Single file |
| Data | Same from HF | Same |
| Algorithm | Same: TF-IDF + SGD(hinge) | Same |

**What Ting changed**: Simplified standalone solution — same approach, but independent implementation.

---

### Aiden: HateXplain/ vs r_Aiden/

| Aspect | HateXplain/ (Original) | r_Aiden/ (Replication) |
|--------|---------------------|---------------------|
| BERT model | Custom `BertPreTrainedModel` wrapper | `transformers.AutoModel` |
| Training | Complex pipeline with pickle | Dataclass + clean PyTorch Dataset |
| Code structure | Separate: Models/, Preprocess/, TensorDataset/ | Unified: `src/` |
| Data loading | Local pickle/cache | Auto-download from URL |
| Data | Same | Same (byte-for-byte) |
| Task | Same: 3-class + rationale + attention | Same |

**What Aiden changed**: Rewrote using modern `transformers` library — cleaner, more maintainable code.

---

## Claims & Evidence Summary

| Member | Uses Original Data | Code Relationship | Evidence |
|--------|-------------------|-----------------|----------|
| **Leo** | ✅ Same | Refactored | `r_Leo/SDG2025.csv` == `GSP/SDG2025.csv` (MD5: `618c979d3d9c17ab61074ac04880c2c8`) |
| **Ting** | ✅ Same | Reimplementation | Independent code, same approach (TF-IDF+SGD) |
| **Aiden** | ✅ Same | Reimplementation | `r_Aiden/data/raw/dataset.json` == `HateXplain/Data/dataset.json` (MD5: `4f4edef9`) |

### Hash Verification Commands

```bash
# Verify GSP data
md5sum GSP/SDG2025.csv r_Leo/SDG2025.csv

# Verify HateXplain data  
md5sum HateXplain/Data/dataset.json r_Aiden/hatexplain_aligned/data/raw/dataset.json
```

---

## Original Resources Details

### GSP/ — Global Sustainability Performance
- **Source**: Çelik et al. (2025), *MDPI Sustainability 17, 7411*
- **Original Repo**: https://github.com/Sadullah4535/Global-Sustainability-Performance-
- **Data**: `SDG2025.csv` — Country-level SDG index scores
- **Code**: `Codes.py` — Full analysis notebooks

### SDGi/ — SDGi Dataset
- **Source**: UNDP SDGi Corpus
- **HF Dataset**: https://huggingface.co/datasets/UNDP/sdgi-corpus
- **Data**: `train.parquet`, `test.parquet`

### sdgi_benchmark/ — SDGi Benchmark Code
- **Source**: https://github.com/UNDP-Data/dsc-sdgi-corpus
- **Code**: Full pipeline with MLflow

### HateXplain/ — HateXplain Code & Data
- **Source**: Mathew et al. (2021), *AAAI 2021*
- **Original Repo**: https://github.com/hate-alert/HateXplain
- **Data**: `dataset.json`, `post_id_divisions.json`

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

---

## References

- Çelik et al. (2025): https://doi.org/10.3390/su17167411
- UNDP SDGi: https://huggingface.co/datasets/UNDP/sdgi-corpus
- SDGi Benchmark: https://github.com/UNDP-Data/dsc-sdgi-corpus
- Mathew et al. (2021): https://arxiv.org/abs/2012.10289
- HateXplain: https://github.com/hate-alert/HateXplain