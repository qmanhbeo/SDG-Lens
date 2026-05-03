# AI4GC-group

Group project for AI for Good Coding course.

## Project Overview

This project combines three individual replications into a unified AI solution for sustainable development analysis.

## Team Members

| Member | Replication | Description |
|--------|-------------|-------------|
| Leo | `r_Leo/` | SDG benchmarking system |
| Ting | `r_Ting/` | SDGi paper replication |
| Aiden | `r_Aiden/` | HateXplain dataset analysis |

## Project Structure

```
AI4GC-group/
├── SDGi/                    # SDGi dataset (UNDP/sdgi-corpus)
├── HateXplain/              # HateXplain dataset
├── GSP/                    # Global Sustainability Performance data
├── r_Leo/                  # Leo's replication code & results
├── r_Ting/                 # Ting's replication code
├── r_Aiden/                # Aiden's replication code
└── README.md
```

## Individual Assignments

### Leo (r_Leo/)
- **Paper**: AI4GC-1 submission
- **Task**: SDG progress prediction and benchmarking
- **Files**: `main.py`, `main_full_paper.py`, `SDG2025.csv`

### Ting (r_Ting/)
- **Original**: https://github.com/TvTfuuuu/SDGi-replication
- **Paper**: SDGi - Sustainable Development Goals integration
- **Files**: `sdgi_replication.py`, `requirements_sdgi.txt`

### Aiden (r_Aiden/)
- **Dataset**: aiden6930/Assignment (HateXplain aligned)
- **Task**: Bias detection and explainability
- **Files**: `src/train.py`, `src/evaluate.py`, `src/model.py`

## Next Steps

1. Review each replication code
2. Identify common data inputs
3. Design unified AI architecture
4. Implement combined solution
5. Evaluate and compare results

## Requirements

See individual folders for requirements:
- `r_Leo/requirements.txt`
- `r_Ting/requirements_sdgi.txt`
- `r_Aiden/hatexplain_aligned/requirements.lock.txt`

## References

- SDGi Dataset: https://huggingface.co/datasets/UNDP/sdgi-corpus
- HateXplain: https://huggingface.co/datasets/aiden6930/Assignment