# AI4GC-group

A combined repository representing three AI solutions for sustainable development and social good.

## What This Repository Represents

This project brings together three replications from the AI for Global Challenges course. Each replication addresses a different aspect of sustainable development:

- **Leo**: SDG benchmarking and clustering analysis
- **Ting**: Multi-label SDG text classification
- **Aiden**: Hate speech detection with explainability

Together, these provide a foundation for exploring AI applications in sustainable development contexts.

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

## Possible Directions

This is a starting point. Some directions we could explore:

### Unified Pipeline
Combine the three pipelines to go from raw text → classification → benchmarking → explanation.

### Cross-Dataset Analysis
Compare how the same model performs across different SDG-related datasets.

### Benchmarking
Use the HateXplain explainability approach on SDG classification tasks.

### SDG Applications
Apply hate speech detection techniques to identify harmful content in SDG-related text.

---

## Next Steps (Open)

- Review all three pipelines
- Identify integration points
- Design combined architecture
- Extend to new tasks or datasets

The exact path forward is open — let's explore together.

---

## Team

- Leo: SDG benchmarking
- Ting: SDGi classification
- Aiden: HateXplain explainability

---

## Getting Started

```bash
# Leo's pipeline
cd r_Leo && python main.py

# Ting's pipeline
cd r_Ting && python sdgi_replication.py

# Aiden's pipeline
cd r_Aiden/hatexplain_aligned && python src/run_once.py
```

---

## Resources

- Course materials and papers in respective folders
- Original papers linked in `notes/replication_notes.md`