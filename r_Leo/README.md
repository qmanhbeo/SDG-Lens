# Replication: Global Sustainability Performance and Regional Disparities

**Paper:** Celik et al. (2025), *MDPI Sustainability 17, 7411*  
**DOI:** https://doi.org/10.3390/su17167411  
**Original repository:** https://github.com/Sadullah4535/Global-Sustainability-Performance-  
**Replication by:** Leonardo Manh Nguyen  
**Programme:** MSc AI & Sustainable Development, University of Birmingham

Tested on Linux and Windows. macOS has not been tested.

- Use `main.py` if you only want the core replication for the selected outputs: Table 2 and Figure 5.
- Use `main_full_paper.py` only if you want the larger end-to-end replication with the extra figures, tables, classifier checks, and preview diagnostics.

## Quickstart

```bash
git clone https://github.com/qmanhbeo/AI4GC-1-submission-Leo
cd AI4GC-1-submission-Leo
```

Create an environment and install dependencies:

```bash
python -m venv .venv
# activate the virtual environment in your shell
python -m pip install -r requirements.txt
```

If you use Conda instead:

```bash
conda create -n gsp-replication python=3.11 -y
conda activate gsp-replication
python -m pip install -r requirements.txt
```

## Recommended Run

Run the minimal script:

```bash
python main.py
```

This writes:

- `replication_results/table2_elbow.csv`
- `replication_results/fig5_elbow_silhouette.png`

## Optional Full Run

If you want the broader replication:

```bash
python main_full_paper.py
```

This writes the full set of figures and tables into `replication_results/`, including:

- the paper-style outputs from the full workflow
- computed ANOVA and MANOVA verification tables
- classifier evaluation outputs
- `run_summary.txt`

It also writes a preview-only diagnostic that does **not** affect the later pipeline:

- `replication_results/fig5_elbow_silhouette_sweep.png`
- `replication_results/table2_elbow_sweep_preview.csv`

That preview compares `n_init=5`, `10`, and `20` against the paper-aligned `n_init=1` setup using simple deterministic runs for visual comparison only.

And calculating ANOVA and MANOVA on top of the hard-coded tables:

- `replication_results/table3_anova_computed.csv`
- `replication_results/table4_manova_computed.csv`
