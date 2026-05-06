# AI4GC-group

Final assignment for the AI for Global Challenges course.

## Assignment Overview

This repository contains **SDG Lens** — a unified solution that classifies UN/UNDP articles against all 17 Sustainable Development Goals (SDGs) and provides explainable predictions.

The final solution integrates techniques from three reference implementations:

| Reference | Contribution | Key Technique |
|-----------|--------------|---------------|
| `r_Leo/` | SDG benchmarking & clustering | KMeans on country-level SDG scores |
| `r_Ting/` | Multi-label classification | TF-IDF + LinearSVC baseline |
| `r_Aiden/` | Explainability | BERT with attention-based rationales |

These reference implementations serve as the **foundation for the unified solution** in `solution1_sdg_lens/`. They are not the final deliverable.

---

## What We Have

### Datasets

| Dataset | Description | Size |
|---------|-------------|------|
| `GSP/` | Country-level SDG index scores | ~12KB |
| `SDGi/` | UNDP SDG articles with 17 labels | ~130MB |
| `HateXplain/` | Hate speech with rationales | ~12MB |

### Pipelines

Three working pipelines ready for use:

- `r_Leo/`: KMeans clustering on country-level SDG scores
- `r_Ting/`: TF-IDF + linear classifier for SDG classification
- `r_Aiden/`: BERT-based classifier with attention for explainability

### Models

Pre-trained models and trained checkpoints available in each replication folder.

---

## Final Solution

The unified solution is in `solution1_sdg_lens/`:

- `run.py` — Main pipeline: TF-IDF baseline → MiniLM fine-tuning → multi-label classification with attention explanations
- `outputs/` — Trained models, predictions, and evaluation results
- `sdg_lens_team_brief.pdf` — Team deliverable with methodology and findings

Run the solution:
```bash
cd solution1_sdg_lens && python run.py
```

---

## Reference Implementations

The three folders below are reference implementations (not the final deliverable):

| Folder | Description |
|--------|-------------|
| `r_Leo/` | Country-level SDG clustering (KMeans) |
| `r_Ting/` | SDGi multi-label classification (TF-IDF + linear) |
| `r_Aiden/` | HateXplain explainability (BERT + attention) |

Each contains its own trained models and replication notes in `notes/replication_notes.md`.