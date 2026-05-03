"""
Minimal replication script for Table 2 and Figure 5.

This keeps only the core steps from `main_full_paper.py`:
- load and rename the SDG 2025 dataset,
- standardize the five clustering features,
- evaluate KMeans for k=2..10,
- save Table 2 as CSV and Figure 5 as a PNG.
"""

# ---------------------------------------------------------------------------
# Imports and plotting backend
# ---------------------------------------------------------------------------

import time
import warnings
from pathlib import Path

# Suppress import-time warnings so the runtime log stays clean and readable.
warnings.filterwarnings("ignore")

import matplotlib

# Use a non-interactive backend because this script writes figures to disk
# rather than opening them in a desktop window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

DATA_PATH = Path("SDG2025.csv")
OUTDIR = Path("replication_results")
OUTDIR.mkdir(exist_ok=True)

# Reuse the same five features and KMeans settings as the full-paper script
# so this smaller file remains directly comparable to the larger workflow.
FEATURES = ["sdg_score", "spillover_score", "regional_score", "population", "progress"]
PAPER_KMEANS = {"init": "k-means++", "n_init": 1, "random_state": 42}
RENAME_MAP = {
    "Country": "country",
    "2025 SDG Index Score": "sdg_score",
    "International Spillovers Score (0-100)": "spillover_score",
    "Regional Score (0-100)": "regional_score",
    "Population in 2024": "population",
    "Progress on Headline SDGi (p.p.)": "progress",
    "Regions used for the SDR": "region",
}


RUN_START = time.perf_counter()


def log(message: str) -> None:
    """Print a flush-on-write message with elapsed runtime."""

    elapsed = time.perf_counter() - RUN_START
    print(f"[{elapsed:6.1f}s] {message}", flush=True)


def load_scaled_features():
    """Load the CSV, keep the clustering features, and standardize them."""

    # Read the source file with the delimiter and encoding used by the dataset.
    log(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, sep=";", encoding="ISO-8859-1", engine="python")
    log(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns.")

    # Replace the long paper column names with shorter analysis-friendly labels.
    df = df.rename(columns=RENAME_MAP)
    log("Column names normalized for analysis.")

    # Drop incomplete rows because KMeans and silhouette scoring need a fully
    # numeric matrix without missing values.
    x = df[FEATURES].dropna()
    dropped_rows = len(df) - len(x)
    log(f"Selected {len(FEATURES)} features and dropped {dropped_rows} incomplete rows.")

    # Standardization prevents population from dominating the score-based
    # variables simply because it lives on a much larger scale.
    x_scaled = StandardScaler().fit_transform(x)
    log(f"Feature scaling complete. Matrix shape: {x_scaled.shape}.")
    return x_scaled


def build_table2(x_scaled) -> pd.DataFrame:
    """Compute the Table 2 cluster diagnostics for k=2 through k=10."""

    rows = []
    log("Building Table 2 diagnostics for k=2 through k=10...")

    # Fit one KMeans model per candidate cluster count and store the two
    # diagnostics reported in the paper-aligned elbow table.
    for k in range(2, 11):
        log(f"Running KMeans for k={k}...")
        km = KMeans(n_clusters=k, **PAPER_KMEANS)
        labels = km.fit_predict(x_scaled)
        inertia = km.inertia_
        silhouette = silhouette_score(x_scaled, labels)
        rows.append(
            {
                "k": k,
                "WCSS (Inertia)": inertia,
                "Silhouette Score": silhouette,
            }
        )
        log(f"k={k} done. Inertia={inertia:.4f}, silhouette={silhouette:.4f}.")

    # Export the diagnostics so the table can be inspected without opening
    # the figure.
    table2 = pd.DataFrame(rows)
    table2.to_csv(OUTDIR / "table2_elbow.csv", index=False)
    log(f"Table 2 saved to {OUTDIR / 'table2_elbow.csv'}.")
    return table2


def plot_figure5(table2: pd.DataFrame) -> None:
    """Plot the elbow curve and silhouette trend on a shared k-axis."""

    log("Rendering Figure 5 elbow/silhouette chart...")
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Left axis: within-cluster sum of squares, the classic elbow metric.
    ax1.plot(table2["k"], table2["WCSS (Inertia)"], "o-", color="steelblue", label="WCSS (Inertia)")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("WCSS (Inertia)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    # Highlight the paper's chosen solution directly on the chart.
    ax1.axvline(x=5, color="green", linestyle="--", linewidth=1.5, label="k=5 (optimal)")
    ax1.grid(True, alpha=0.3)

    # Right axis: silhouette score so separation quality can be compared on
    # the same horizontal scale.
    ax2 = ax1.twinx()
    ax2.plot(table2["k"], table2["Silhouette Score"], "o-", color="crimson", label="Silhouette Score")
    ax2.set_ylabel("Silhouette Score", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")

    # Combine both legends into one box because the figure uses twin axes.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Elbow Method and Silhouette Score")

    # Save the paper-style diagnostic figure to the shared output directory.
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig5_elbow_silhouette.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Figure 5 saved to {OUTDIR / 'fig5_elbow_silhouette.png'}.")


def main() -> None:
    """Run the minimal Table 2 / Figure 5 replication pipeline."""

    log("Starting minimal replication pipeline.")
    # Prepare the standardized feature matrix, compute the diagnostics table,
    # and render the corresponding elbow/silhouette plot.
    x_scaled = load_scaled_features()
    table2 = build_table2(x_scaled)
    plot_figure5(table2)
    log("Minimal replication pipeline completed successfully.")


if __name__ == "__main__":
    main()
