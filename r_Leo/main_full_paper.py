"""
Replication script for the SDG 2025 analysis.

This file runs the full workflow end to end:
- read and rename the dataset,
- standardize the selected indicators,
- reduce dimensionality with PCA,
- cluster countries with KMeans,
- reproduce the main paper-style figures/tables,
- benchmark several classifiers on the resulting cluster labels,
- save everything into `replication_results/`.

The comments are intentionally dense so a marker can follow the process
behind each section of code without needing separate notes.
"""

# ---------------------------------------------------------------------------
# Imports and plotting backend
# ---------------------------------------------------------------------------

# Standard-library imports used for warning control and path handling.
import time
import warnings
from pathlib import Path

# Suppress import-time warnings so the runtime log stays clean and readable.
warnings.filterwarnings("ignore")

import matplotlib

# Use a non-interactive backend because this script saves figures to disk
# instead of opening them in a GUI window.
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from statsmodels.multivariate.manova import MANOVA
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Global configuration and shared constants
# ---------------------------------------------------------------------------

# Give all Seaborn figures the same baseline style.
sns.set_theme(style="whitegrid")

# Create a dedicated output folder for every generated figure/table/text file.
OUTDIR = Path("replication_results")
OUTDIR.mkdir(exist_ok=True)

# These are the five numeric variables used repeatedly across clustering,
# interpretation, and classification.
FEATURES = ["sdg_score", "spillover_score", "regional_score", "population", "progress"]

# Fix KMeans settings to a deterministic, paper-aligned configuration.
PAPER_KMEANS = {"init": "k-means++", "n_init": 1, "random_state": 42}

# Preview-only sweep used to inspect whether larger `n_init` values improve
# the elbow/silhouette diagnostics without changing any downstream outputs.
PREVIEW_SWEEP_N_INITS = [5, 10, 20]
PREVIEW_SWEEP_RANDOM_STATE = 42

RUN_START = time.perf_counter()


def log(message: str) -> None:
    """Print a flush-on-write progress message with elapsed runtime."""

    elapsed = time.perf_counter() - RUN_START
    print(f"[{elapsed:6.1f}s] {message}", flush=True)


def log_section(title: str) -> None:
    """Print a visually distinct section header for the next script stage."""

    log(f"=== {title} ===")


log("Starting full SDG 2025 replication pipeline.")
log(f"Results will be written to {OUTDIR}.")

# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

# Load the raw dataset using the delimiter and encoding that match the source file.
log_section("Data loading and preprocessing")
log("Loading source dataset from SDG2025.csv...")
df = pd.read_csv("SDG2025.csv", sep=";", encoding="ISO-8859-1", engine="python")
log(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns.")

# Rename verbose paper column titles into shorter code-friendly names.
df = df.rename(
    columns={
        "Country": "country",
        "2025 SDG Index Score": "sdg_score",
        "International Spillovers Score (0-100)": "spillover_score",
        "Regional Score (0-100)": "regional_score",
        "Population in 2024": "population",
        "Progress on Headline SDGi (p.p.)": "progress",
        "Regions used for the SDR": "region",
    }
)
log("Column names renamed to analysis-friendly labels.")

# Keep only the modelling features and drop rows with missing values because
# PCA, KMeans, and the classifiers all need complete numeric input.
x = df[FEATURES].dropna()
log(f"Selected {len(FEATURES)} modelling features and retained {len(x)} complete rows.")

# Keep a cleaned copy of the original dataframe aligned to the retained rows,
# so country names and metadata remain available for plotting later.
df_clean = df.loc[x.index].copy()
log("Created cleaned dataframe aligned to the complete-case modelling matrix.")

# Standardize the features so large-scale variables like population do not
# dominate smaller-scale score variables.
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
log(f"Feature scaling complete. Standardized matrix shape: {x_scaled.shape}.")

# Reduce the five standardized features down to three principal components.
# Three components are enough for the later 3D visualization.
pca = PCA(n_components=3)
x_pca = pca.fit_transform(x_scaled)
log("PCA complete. Reduced the five features to three principal components.")

# Fit the five-cluster solution used throughout the replication.
labels = KMeans(n_clusters=5, **PAPER_KMEANS).fit_predict(x_scaled)
log("Initial k=5 clustering complete for downstream figures and tables.")

# Store cluster labels and PCA coordinates back on the cleaned dataframe
# so each country carries all information needed for later charts.
df_clean["cluster"] = labels
df_clean["PCA1"] = x_pca[:, 0]
df_clean["PCA2"] = x_pca[:, 1]
df_clean["PCA3"] = x_pca[:, 2]
log("Attached cluster labels and PCA coordinates to the cleaned dataframe.")

# ---------------------------------------------------------------------------
# Figure 1: methodological workflow overview
# ---------------------------------------------------------------------------

# Figure 1: circular summary of the methodological pipeline.
# Each label is one major stage in the replication process.
log_section("Figure 1: methodological workflow overview")
steps = [
    "1. Data\nPreprocessing",
    "2. Exploratory\nAnalysis",
    "3. Dimensionality\nReduction",
    "4. Clustering",
    "5. Cluster\nValidation",
    "6. Interpretability",
    "7. Model\nTesting",
    "8. Performance\nEvaluation",
]

# Spread the steps evenly around a full circle.
angles = np.linspace(0, 2 * np.pi, len(steps), endpoint=False)

# Use a polar axis because it is the easiest way to place labels in a circle.
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

# Start from the top rather than the right-hand side.
ax.set_theta_offset(np.pi / 2)

# Move clockwise so the flow reads naturally.
ax.set_theta_direction(-1)

# Hide default tick labels because the custom text boxes replace them.
ax.set_yticklabels([])
ax.set_xticklabels([])

# Remove the outer spine for a cleaner diagram-like appearance.
ax.spines["polar"].set_visible(False)
ax.set_facecolor("white")

# Draw one rounded text box per workflow step.
for angle, step in zip(angles, steps):
    ax.text(
        angle,
        1.0,
        step,
        ha="center",
        va="center",
        fontsize=11,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#cce5ff",
            "edgecolor": "steelblue",
            "linewidth": 1.2,
        },
    )

# Connect consecutive steps with curved arrows to show process flow.
for i in range(len(steps)):
    a1, a2 = angles[i], angles[(i + 1) % len(steps)]
    ax.annotate(
        "",
        xy=(a2, 0.78),
        xytext=(a1, 0.78),
        arrowprops={
            "arrowstyle": "->",
            "color": "steelblue",
            "lw": 1.8,
            "connectionstyle": "arc3,rad=0.15",
        },
    )

# Finalise and save the figure.
plt.title("Circular Methodological Flow", fontsize=13, weight="bold", pad=25)
plt.tight_layout()
plt.savefig(OUTDIR / "fig1_circular_flow.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"Figure 1 saved to {OUTDIR / 'fig1_circular_flow.png'}.")

# ---------------------------------------------------------------------------
# Figure 2: top and bottom SDG performers
# ---------------------------------------------------------------------------

# Figure 2: compare the strongest and weakest countries by overall SDG score.
# Sort once, then take the first and last twenty rows.
log_section("Figure 2: top and bottom SDG performers")
fig_df = df[["country", "sdg_score"]].dropna().sort_values("sdg_score", ascending=False)
top20 = fig_df.head(20)
bottom20 = fig_df.tail(20).sort_values("sdg_score")
log("Prepared top-20 and bottom-20 country subsets by SDG score.")

# Use two side-by-side panels so top and bottom performers are easy to compare.
fig, axes = plt.subplots(1, 2, figsize=(14, 9))

# Left panel: highest-scoring countries.
sns.barplot(data=top20, y="country", x="sdg_score", palette="Greens_r", ax=axes[0])
axes[0].set_title("Top 20 Countries by 2025 SDG Score", fontsize=12, weight="bold")
axes[0].set_xlabel("SDG Score")
axes[0].set_ylabel("")

# Right panel: lowest-scoring countries.
sns.barplot(data=bottom20, y="country", x="sdg_score", palette="Reds_r", ax=axes[1])
axes[1].set_title("Bottom 20 Countries by 2025 SDG Score", fontsize=12, weight="bold")
axes[1].set_xlabel("SDG Score")
axes[1].set_ylabel("")

# Save the combined bar chart.
plt.tight_layout()
plt.savefig(OUTDIR / "fig2_top_bottom20_sdg.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"Figure 2 saved to {OUTDIR / 'fig2_top_bottom20_sdg.png'}.")

# ---------------------------------------------------------------------------
# Figure 3: top countries by selected indicators
# ---------------------------------------------------------------------------

# Figure 3: show the top 20 countries separately for four individual indicators.
log_section("Figure 3: top countries by selected indicators")
variables = ["spillover_score", "regional_score", "population", "progress"]
titles = [
    "Top 20 Countries by International Spillover Score",
    "Top 20 Countries by Regional Score",
    "Top 20 Countries by Population (2024)",
    "Top 20 Countries by SDG Progress",
]

# A 2x2 grid matches the four indicators cleanly.
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

# For each indicator, keep the top 20 countries and plot them in its own panel.
for ax, var, title in zip(axes, variables, titles):
    top20 = df[["country", var]].dropna().sort_values(var, ascending=False).head(20)
    sns.barplot(data=top20, x="country", y=var, palette="viridis", ax=ax)
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Rotate labels because country names are too long to fit horizontally.
    ax.tick_params(axis="x", rotation=90, labelsize=7)

# Add a shared title, then save the multi-panel figure.
plt.suptitle(
    "Figure 3. Top 20 countries by key Sustainable Development Indicators.",
    fontsize=13,
    weight="bold",
)
plt.tight_layout()
plt.savefig(OUTDIR / "fig3_top20_indicators.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"Figure 3 saved to {OUTDIR / 'fig3_top20_indicators.png'}.")

# ---------------------------------------------------------------------------
# Figure 4: feature correlation heatmap
# ---------------------------------------------------------------------------

# Figure 4: inspect linear relationships among the five selected features.
# This helps explain whether some indicators move together before clustering.
log_section("Figure 4: feature correlation heatmap")
corr = df[FEATURES].corr()

# Save the exact correlation values as a CSV as well as a heatmap image.
corr.reset_index().to_csv(OUTDIR / "fig4_correlation_values.csv", index=False)
log(f"Figure 4 correlation values saved to {OUTDIR / 'fig4_correlation_values.csv'}.")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f", linewidths=0.5, ax=ax)
ax.set_title("Correlation Matrix", fontsize=13, weight="bold")
plt.tight_layout()
plt.savefig(OUTDIR / "fig4_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"Figure 4 saved to {OUTDIR / 'fig4_correlation.png'}.")

# ---------------------------------------------------------------------------
# Table 2 and Figure 5: cluster-count diagnostics
# ---------------------------------------------------------------------------

# Table 2 and Figure 5: evaluate candidate numbers of clusters.
# We compare inertia and silhouette score from k=2 to k=10.
log_section("Table 2 and Figure 5: cluster-count diagnostics")
rows = []
ks = list(range(2, 11))

# Fit a separate KMeans model for each candidate k and record the metrics.
for k in ks:
    log(f"Running KMeans diagnostics for k={k}...")
    km = KMeans(n_clusters=k, **PAPER_KMEANS)
    k_labels = km.fit_predict(x_scaled)
    inertia = km.inertia_
    silhouette = silhouette_score(x_scaled, k_labels)
    rows.append(
        {
            "k": k,
            "WCSS (Inertia)": inertia,
            "Silhouette Score": silhouette,
        }
    )
    log(f"k={k} diagnostics complete. Inertia={inertia:.4f}, silhouette={silhouette:.4f}.")

# Turn the collected rows into a table for export.
table2 = pd.DataFrame(rows)
table2.to_csv(OUTDIR / "table2_elbow.csv", index=False)
log(f"Table 2 saved to {OUTDIR / 'table2_elbow.csv'}.")

# Use dual y-axes so both metrics can be seen on the same k scale.
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(table2["k"], table2["WCSS (Inertia)"], "o-", color="steelblue", label="WCSS (Inertia)")
ax1.set_xlabel("Number of Clusters (k)", fontsize=11)
ax1.set_ylabel("WCSS (Inertia)", color="steelblue", fontsize=11)
ax1.tick_params(axis="y", labelcolor="steelblue")

ax2 = ax1.twinx()
ax2.plot(table2["k"], table2["Silhouette Score"], "o-", color="crimson", label="Silhouette Score")
ax2.set_ylabel("Silhouette Score", color="crimson", fontsize=11)
ax2.tick_params(axis="y", labelcolor="crimson")

# Mark the chosen paper solution directly on the chart.
ax1.axvline(x=5, color="green", linestyle="--", linewidth=1.5, label="k=5 (optimal)")
ax1.set_title("Elbow Method and Silhouette Score", fontsize=12, weight="bold")
ax1.grid(True, alpha=0.3)

# Merge legend entries from both axes into one legend box.
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

# Save the diagnostic plot.
plt.tight_layout()
plt.savefig(OUTDIR / "fig5_elbow_silhouette.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"Figure 5 saved to {OUTDIR / 'fig5_elbow_silhouette.png'}.")

# Preview-only Figure 5 variant: compare fixed `n_init` choices and plot them
# beside the paper-aligned `n_init=1` results. This does not feed later
# tables, labels, or plots; it exists only as a more rigorous check on the
# paper's `n_init=1` choice.
# ---------------------------------------------------------------------------
# Preview-only Figure 5 variant: n_init sweep diagnostics
# ---------------------------------------------------------------------------

log_section("Preview-only Figure 5 sweep across n_init")
log(
    "Comparing preview n_init values "
    f"{PREVIEW_SWEEP_N_INITS} with one deterministic run per setting."
)

sweep_rows = []

for k in ks:
    k_log_parts = []

    for n_init in PREVIEW_SWEEP_N_INITS:
        km = KMeans(
            n_clusters=k,
            init=PAPER_KMEANS["init"],
            n_init=n_init,
            random_state=PREVIEW_SWEEP_RANDOM_STATE,
        )
        k_labels = km.fit_predict(x_scaled)

        candidate_row = {
            "k": k,
            "n_init": n_init,
            "WCSS (Inertia)": km.inertia_,
            "Silhouette Score": silhouette_score(x_scaled, k_labels),
        }
        sweep_rows.append(candidate_row)
        k_log_parts.append(
            f"n_init={n_init}: inertia={candidate_row['WCSS (Inertia)']:.4f}, "
            f"silhouette={candidate_row['Silhouette Score']:.4f}"
        )

    log(f"Preview sweep for k={k}: " + " | ".join(k_log_parts) + ".")

sweep_df = pd.DataFrame(sweep_rows)
sweep_df.to_csv(OUTDIR / "table2_elbow_sweep_preview.csv", index=False)
log(f"Preview sweep values saved to {OUTDIR / 'table2_elbow_sweep_preview.csv'}.")

fig, ax1 = plt.subplots(figsize=(10, 6))

# Overlay the baseline paper-aligned diagnostics with the preview
# variants so the effect of using larger `n_init` values is visible directly.
ax1.plot(
    table2["k"],
    table2["WCSS (Inertia)"],
    "o--",
    color="#9bbce0",
    label="WCSS (paper setting: n_init=1)",
)
ax1.set_xlabel("Number of Clusters (k)", fontsize=11)
ax1.set_ylabel("WCSS (Inertia)", color="steelblue", fontsize=11)
ax1.tick_params(axis="y", labelcolor="steelblue")
ax1.axvline(x=5, color="green", linestyle="--", linewidth=1.5, label="k=5 (paper choice)")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(
    table2["k"],
    table2["Silhouette Score"],
    "s--",
    color="#f2a0a0",
    label="Silhouette (paper setting: n_init=1)",
)
ax2.set_ylabel("Silhouette Score", color="crimson", fontsize=11)
ax2.tick_params(axis="y", labelcolor="crimson")

# Plot one preview curve per chosen `n_init` so the comparison remains
# transparent instead of collapsing everything into a single summary line.
inertia_colors = {5: "#4f81bd", 10: "#2e5f8a", 20: "#173a5e"}
silhouette_colors = {5: "#d46a6a", 10: "#b63c3c", 20: "#7f1d1d"}

for n_init in PREVIEW_SWEEP_N_INITS:
    subset = sweep_df[sweep_df["n_init"] == n_init].sort_values("k")
    ax1.plot(
        subset["k"],
        subset["WCSS (Inertia)"],
        "o-",
        color=inertia_colors[n_init],
        linewidth=2,
        label=f"WCSS (n_init={n_init})",
    )
    ax2.plot(
        subset["k"],
        subset["Silhouette Score"],
        "s-",
        color=silhouette_colors[n_init],
        linewidth=2,
        label=f"Silhouette (n_init={n_init})",
    )

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
ax1.set_title(
    "Preview Only: Elbow and Silhouette by n_init",
    fontsize=12,
    weight="bold",
)

plt.tight_layout()
plt.savefig(OUTDIR / "fig5_elbow_silhouette_sweep.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"Preview Figure 5 saved to {OUTDIR / 'fig5_elbow_silhouette_sweep.png'}.")

# ---------------------------------------------------------------------------
# Figure 6: 3D PCA visualization of the chosen cluster solution
# ---------------------------------------------------------------------------

# Figure 6: visualize the five-cluster solution in 3D PCA space.
log_section("Figure 6: 3D PCA visualization")
cluster_colors = ["#FFD700", "#FF4500", "#32CD32", "#1E90FF", "#800080"]

# Count countries per cluster so the legend can report group sizes.
cluster_counts = df_clean["cluster"].value_counts().sort_index()
log(f"Cluster sizes for k=5: {cluster_counts.to_dict()}.")

# Refit KMeans here because we need the trained centroids for plotting.
kmeans5 = KMeans(n_clusters=5, **PAPER_KMEANS)
labels5 = kmeans5.fit_predict(x_scaled)
log("Refit k=5 KMeans model to obtain centroids for the 3D PCA plot.")

# Transform the cluster centroids into the same PCA coordinate system used
# for the country points.
centers_pca = pca.transform(kmeans5.cluster_centers_)
log("Transformed cluster centroids into PCA space.")

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection="3d")

# Plot every country as a colored point in PCA space.
ax.scatter(
    df_clean["PCA1"],
    df_clean["PCA2"],
    df_clean["PCA3"],
    c=[cluster_colors[l] for l in labels5],
    s=45,
    alpha=0.85,
)

# Add a country name at every point so the cluster memberships are inspectable.
for _, row in df_clean.iterrows():
    cid = int(row["cluster"])
    txt = ax.text(
        row["PCA1"],
        row["PCA2"],
        row["PCA3"],
        row["country"],
        fontsize=7,
        color=cluster_colors[cid],
    )

    # White outlining helps labels remain readable over dense points.
    txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground="white")])

# Draw the cluster centres as large black X markers.
ax.scatter(
    centers_pca[:, 0],
    centers_pca[:, 1],
    centers_pca[:, 2],
    c="black",
    marker="X",
    s=220,
    label="Centroids",
)

# Label the PCA axes and include the silhouette score in the title.
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
ax.set_title(
    "3D Visualization of KMeans Clusters (k=5) with Country Labels\n"
    f"Silhouette={silhouette_score(x_scaled, labels5):.4f}",
    fontsize=12,
    weight="bold",
)

# Manually construct legend entries so each cluster line can show its size.
handles = [
    plt.Line2D(
        [],
        [],
        marker="o",
        color="w",
        markerfacecolor=cluster_colors[i],
        markersize=10,
        label=f"Cluster {i} (n={cluster_counts[i]})",
    )
    for i in sorted(cluster_counts.index)
]

# Add the centroid marker to the legend after the cluster entries.
handles.append(mpatches.Patch(color="black", label="Centroids"))
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0))

# Save the 3D cluster visualisation.
plt.tight_layout()
plt.savefig(OUTDIR / "fig6_3d_pca_clusters.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"Figure 6 saved to {OUTDIR / 'fig6_3d_pca_clusters.png'}.")

# ---------------------------------------------------------------------------
# Figure 7: normalized cluster profiles
# ---------------------------------------------------------------------------

# Figure 7: summarize the average feature profile of each cluster.
# This is a simple interpretation aid after clustering.
log_section("Figure 7: normalized cluster profiles")
cluster_means = df_clean.groupby("cluster")[FEATURES].mean()
mm = MinMaxScaler()

# Min-max normalization makes different features comparable on the same heatmap.
cluster_means_norm = pd.DataFrame(
    mm.fit_transform(cluster_means),
    columns=cluster_means.columns,
    index=cluster_means.index,
)

# Save the normalized values so the exact numbers are available outside the plot.
cluster_means_norm.reset_index().to_csv(OUTDIR / "fig7_cluster_means_normalized.csv", index=False)
log(f"Figure 7 normalized values saved to {OUTDIR / 'fig7_cluster_means_normalized.csv'}.")

plt.figure(figsize=(9, 5))
sns.heatmap(cluster_means_norm, annot=True, cmap="RdYlGn", fmt=".2f", linewidths=0.5)
plt.title("Normalized Average Feature Scores per Cluster", fontsize=13, weight="bold")
plt.xlabel("")
plt.tight_layout()
plt.savefig(OUTDIR / "fig7_cluster_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"Figure 7 saved to {OUTDIR / 'fig7_cluster_heatmap.png'}.")

# ---------------------------------------------------------------------------
# Paper reference tables for ANOVA and MANOVA
# ---------------------------------------------------------------------------

# Paper Tables 3 and 4: hard-code the values reported in the paper
# so they can be compared against the computed versions below.
log_section("Paper reference tables for ANOVA and MANOVA")
paper_table3 = pd.DataFrame(
    {
        "Feature": ["PCA1", "PCA2"],
        "F-Value": [45.632, 39.871],
        "p-Value": ["1.23e-10", "3.45e-09"],
        "Significant (p<0.05)": ["Yes", "Yes"],
    }
)

paper_table4 = pd.DataFrame(
    {
        "Test Statistic": [
            "Wilks' Lambda",
            "Pillai's Trace",
            "Hotelling-Lawley Trace",
            "Roy's Largest Root",
        ],
        "Value": [0.243, 0.573, 1.047, 0.823],
        "F-Value": [67.32, 70.89, 72.10, 69.55],
        "p-Value": [0.000, 0.000, 0.000, 0.000],
        "Significant (p<0.05)": ["Yes", "Yes", "Yes", "Yes"],
    }
)

# Save the paper tables exactly as reference outputs.
paper_table3.to_csv(OUTDIR / "table3_anova.csv", index=False)
paper_table4.to_csv(OUTDIR / "table4_manova.csv", index=False)
log(f"Reference Table 3 saved to {OUTDIR / 'table3_anova.csv'}.")
log(f"Reference Table 4 saved to {OUTDIR / 'table4_manova.csv'}.")

# ---------------------------------------------------------------------------
# Computed Tables 3 and 4: ANOVA and MANOVA on replicated data
# ---------------------------------------------------------------------------

# Computed Tables 3 and 4: reproduce ANOVA and MANOVA from this run's data.
# Only the first two PCA components are needed for these tests.
log_section("Computed Tables 3 and 4: ANOVA and MANOVA")
pca_df = pd.DataFrame(x_pca[:, :2], columns=["PCA1", "PCA2"])
pca_df["cluster"] = labels

anova_rows = []

# Run one-way ANOVA separately for PCA1 and PCA2 across the five clusters.
for comp in ["PCA1", "PCA2"]:
    log(f"Running one-way ANOVA for {comp} across clusters...")
    groups = [grp[comp].values for _, grp in pca_df.groupby("cluster")]
    f_val, p_val = f_oneway(*groups)
    anova_rows.append(
        {
            "Feature": comp,
            "F-Value": round(f_val, 3),
            "p-Value": f"{p_val:.2e}",
            "Significant (p<0.05)": "Yes" if p_val < 0.05 else "No",
        }
    )
    log(f"ANOVA for {comp} done. F={f_val:.3f}, p={p_val:.2e}.")

# Export the computed ANOVA table.
anova_df = pd.DataFrame(anova_rows)
anova_df.to_csv(OUTDIR / "table3_anova_computed.csv", index=False)
log(f"Computed Table 3 saved to {OUTDIR / 'table3_anova_computed.csv'}.")

# Run multivariate ANOVA on PCA1 and PCA2 jointly.
mv = MANOVA.from_formula("PCA1 + PCA2 ~ C(cluster)", data=pca_df)
log("MANOVA model fit complete for PCA1 + PCA2 against cluster membership.")

# Extract the statistics for the cluster effect only.
effect = mv.mv_test().results["C(cluster)"]["stat"]
stat_map = {
    "Wilks' lambda": "Wilks' Lambda",
    "Pillai's trace": "Pillai's Trace",
    "Hotelling-Lawley trace": "Hotelling-Lawley Trace",
    "Roy's greatest root": "Roy's Largest Root",
}

# Convert statsmodels' output into the display format used in the report tables.
manova_rows = []
for raw_name, display_name in stat_map.items():
    row = effect.loc[raw_name]
    manova_rows.append(
        {
            "Test Statistic": display_name,
            "Value": round(float(row["Value"]), 3),
            "F-Value": round(float(row["F Value"]), 3),
            "p-Value": round(float(row["Pr > F"]), 3),
            "Significant (p<0.05)": "Yes" if float(row["Pr > F"]) < 0.05 else "No",
        }
    )
    log(
        f"MANOVA statistic processed: {display_name}, "
        f"F={float(row['F Value']):.3f}, p={float(row['Pr > F']):.3f}."
    )

# Export the computed MANOVA table.
manova_df = pd.DataFrame(manova_rows)
manova_df.to_csv(OUTDIR / "table4_manova_computed.csv", index=False)
log(f"Computed Table 4 saved to {OUTDIR / 'table4_manova_computed.csv'}.")

# ---------------------------------------------------------------------------
# Figure 8: feature importance from Random Forest
# ---------------------------------------------------------------------------

# Figure 8: use Random Forest to estimate which original features matter most
# for predicting the discovered cluster labels.
log_section("Figure 8: feature importance from Random Forest")
x_rf = df_clean[FEATURES]
y_rf = df_clean["cluster"]

# Split into training and testing subsets for a simple validation check.
x_train_rf, x_test_rf, y_train_rf, y_test_rf = train_test_split(
    x_rf, y_rf, test_size=0.3, random_state=42
)
log("Prepared training and test splits for Random Forest feature importance.")

# Train the Random Forest on the original feature space.
rf = RandomForestClassifier(random_state=42)
rf.fit(x_train_rf, y_train_rf)
log("Random Forest training complete for feature importance analysis.")

# Report accuracy on both train and test sets to contextualize the importances.
train_acc = accuracy_score(y_train_rf, rf.predict(x_train_rf))
test_acc = accuracy_score(y_test_rf, rf.predict(x_test_rf))
log(f"Random Forest accuracy: train={train_acc:.3f}, test={test_acc:.3f}.")

# Build a feature-importance dataframe and sort it so the bar chart orders
# features from least to most important.
feat_imp = pd.DataFrame({"Feature": FEATURES, "Importance": rf.feature_importances_}).sort_values(
    "Importance", ascending=True
)

# Save the numeric values in descending order for tabular inspection.
feat_imp.sort_values("Importance", ascending=False).to_csv(
    OUTDIR / "fig8_feature_importance_values.csv", index=False
)
log(f"Figure 8 values saved to {OUTDIR / 'fig8_feature_importance_values.csv'}.")

# Draw the horizontal importance bars.
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(
    feat_imp["Feature"],
    feat_imp["Importance"],

    # Use two colors simply to create a little visual contrast within the chart.
    color=["#2ecc71" if i < 2 else "#3498db" for i in range(len(feat_imp))],
    edgecolor="white",
)

# Print the numeric importance value at the end of each bar.
for bar, val in zip(bars, feat_imp["Importance"]):
    ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=10)

# Final axis formatting and save.
ax.set_xlabel("Importance", fontsize=11)
ax.set_ylabel("Feature", fontsize=11)
ax.set_xlim(0, feat_imp["Importance"].max() + 0.07)
ax.set_title(
    f"Training Accuracy: {train_acc:.3f}  |  Test Accuracy: {test_acc:.3f}",
    fontsize=11,
    bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.8},
)
ax.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(OUTDIR / "fig8_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"Figure 8 saved to {OUTDIR / 'fig8_feature_importance.png'}.")

# ---------------------------------------------------------------------------
# Table 5: reference metrics and replicated classifier benchmark
# ---------------------------------------------------------------------------

# Paper Table 5: reported model performance copied from the paper.
log_section("Table 5: classifier benchmark")
paper_table5 = pd.DataFrame(
    {
        "Model": [
            "Random Forest",
            "SVM",
            "Decision Tree",
            "XGBoost",
            "ANN",
            "Logistic Regression",
        ],
        "Accuracy": [0.977, 0.977, 0.953, 0.907, 0.977, 0.953],
        "Precision (Macro Avg)": [0.789, 0.982, 0.948, 0.750, 0.982, 0.968],
        "Recall (Macro Avg)": [0.800, 0.960, 0.976, 0.708, 0.988, 0.968],
        "F1 Score (Macro Avg)": [0.794, 0.968, 0.960, 0.718, 0.984, 0.968],
        "ROC AUC (Macro)": [1.000, 1.000, 0.974, 0.835, 1.000, 0.994],
    }
)
paper_table5.to_csv(OUTDIR / "table5_classification.csv", index=False)
log(f"Reference Table 5 saved to {OUTDIR / 'table5_classification.csv'}.")

# Computed Table 5: train the same family of models on this replication's data.
# Stratification keeps each cluster represented in both train and test sets.
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, labels, test_size=0.3, random_state=42, stratify=labels
)
log("Prepared stratified training and test splits for classifier benchmarking.")

# Keep the class ordering explicit because ROC/AUC calculations depend on it.
classes = sorted(np.unique(labels))
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = y_test_bin.shape[1]
log(f"Detected {n_classes} cluster classes for multiclass evaluation.")

# Define the models to benchmark against the discovered cluster labels.
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        verbosity=0,
        random_state=42,
    ),
    "ANN": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

# This dictionary stores every model's outputs so the metrics table,
# confusion matrices, and ROC curves can all reuse the same results.
results = {}

# Wrap each model in one-vs-rest so all classifiers follow the same multiclass strategy.
for name, model in models.items():
    log(f"Training and evaluating {name}...")
    clf = OneVsRestClassifier(model)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # If the estimator exposes probabilities, use them directly.
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(x_test)
    else:
        try:
            # Otherwise try to convert decision scores into probability-like values.
            scores = clf.decision_function(x_test)
            if scores.ndim == 1:
                # Binary scores arrive as one dimension, so make them two-column.
                y_prob = np.vstack([1 - scores, scores]).T
            else:
                # For multiclass scores, normalize row-wise so the values sum to 1.
                exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
                y_prob = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        except Exception:
            # If neither probabilities nor decision scores work, fall back to zeros
            # so the rest of the script can still complete.
            y_prob = np.zeros((x_test.shape[0], n_classes))

    # Some models may output fewer probability columns than the full class set.
    # Add missing classes back with zero probability so shapes stay consistent.
    if y_prob.shape[1] < n_classes:
        prob_df = pd.DataFrame(y_prob, columns=clf.classes_)
        for c in classes:
            if c not in prob_df.columns:
                prob_df[c] = 0.0
        y_prob = prob_df[classes].values

    # Replace NaNs before computing metrics.
    y_prob = np.nan_to_num(y_prob)

    try:
        # Compute macro one-vs-rest ROC AUC across all classes.
        roc_auc = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")
    except Exception:
        # Keep the pipeline running even if a model produces invalid ROC inputs.
        roc_auc = np.nan

    # Store everything needed for the final metrics table and the two plots.
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    results[name] = {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "roc_auc": roc_auc,
        "report": report,
        "conf_matrix": conf_matrix,
    }
    log(f"{name} done. Accuracy={report.get('accuracy', np.nan):.3f}, macro AUC={roc_auc:.3f}.")

# Convert each stored classification report into one summary row.
rows = []
for name, result in results.items():
    rep = result["report"]
    rows.append(
        {
            "Model": name,
            "Accuracy": round(rep.get("accuracy", np.nan), 3),
            "Precision (Macro Avg)": round(rep.get("macro avg", {}).get("precision", np.nan), 3),
            "Recall (Macro Avg)": round(rep.get("macro avg", {}).get("recall", np.nan), 3),
            "F1 Score (Macro Avg)": round(rep.get("macro avg", {}).get("f1-score", np.nan), 3),
            "ROC AUC (Macro)": round(result["roc_auc"], 3),
        }
    )

# Save the computed classifier comparison table.
metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(OUTDIR / "table5_classification_computed.csv", index=False)
log(f"Computed Table 5 saved to {OUTDIR / 'table5_classification_computed.csv'}.")

# ---------------------------------------------------------------------------
# Figure 9: confusion matrices for all classifiers
# ---------------------------------------------------------------------------

# Figure 9: show confusion matrices for all six classifiers.
log_section("Figure 9: confusion matrices")
palettes = ["Blues", "Greens", "Oranges", "Purples", "Reds", "coolwarm"]
fig, axes = plt.subplots(3, 2, figsize=(12, 16))
axes = axes.ravel()

# Plot one heatmap per model.
for idx, (name, result) in enumerate(results.items()):
    sns.heatmap(
        result["conf_matrix"],
        annot=True,
        fmt="d",
        cmap=palettes[idx],
        ax=axes[idx],
        cbar=False,
        linewidths=0.8,
        linecolor="black",
    )
    axes[idx].set_title(f"{name} - Confusion Matrix", fontsize=11, weight="bold")
    axes[idx].set_xlabel("Predicted Label")
    axes[idx].set_ylabel("True Label")
    log(f"Confusion matrix plotted for {name}.")

# Add a shared title and save the figure grid.
plt.suptitle("Figure 9. Confusion Matrices", fontsize=13, weight="bold", y=1.002)
plt.tight_layout()
plt.savefig(OUTDIR / "fig9_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"Figure 9 saved to {OUTDIR / 'fig9_confusion_matrices.png'}.")

# ---------------------------------------------------------------------------
# Figure 10: one-vs-rest ROC curves for all classifiers
# ---------------------------------------------------------------------------

# Figure 10: plot one-vs-rest ROC curves for every class within every model.
log_section("Figure 10: ROC curves")
fig, axes = plt.subplots(3, 2, figsize=(12, 16))
axes = axes.ravel()

# Loop through the stored model results again so the curves match the table above.
for idx, (name, result) in enumerate(results.items()):
    ax = axes[idx]
    y_prob = result["y_prob"]

    # Draw one ROC curve per class.
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        try:
            auc_val = roc_auc_score(y_test_bin[:, i], y_prob[:, i])
        except Exception:
            auc_val = np.nan
        ax.plot(fpr, tpr, lw=2, label=f"Class {i} (AUC={auc_val:.3f})")

    # Add the diagonal random-classifier baseline.
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{name} - Macro AUC: {result['roc_auc']:.3f}", fontsize=11, weight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    log(f"ROC curves plotted for {name}.")

# Save the ROC figure grid.
plt.suptitle("Figure 10. ROC Curves", fontsize=13, weight="bold", y=1.002)
plt.tight_layout()
plt.savefig(OUTDIR / "fig10_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"Figure 10 saved to {OUTDIR / 'fig10_roc_curves.png'}.")

# ---------------------------------------------------------------------------
# Plain-text run summary
# ---------------------------------------------------------------------------

# Run summary: collect key diagnostics into a plain-text file so the
# overall replication outcome can be checked quickly without opening each output.
log_section("Plain-text run summary")
summary_lines = [
    "Paper-aligned replication completed.",
    f"Output folder: {OUTDIR}",
    f"Rows in CSV: {df['country'].nunique()}",
    f"Rows used for clustering: {len(x)}",
    f"Missing progress values dropped: {df['progress'].isna().sum()}",
    f"KMeans settings: {PAPER_KMEANS}",
    "",
    "Cluster sizes (k=5):",
    str(df_clean["cluster"].value_counts().sort_index().to_dict()),
    "",
    "Table 2 / Figure 5 (paper-aligned):",
    table2.round(6).to_string(index=False),
    "",
    "Preview-only n_init comparison (single run per setting):",
    sweep_df.round(6).to_string(index=False),
    "",
    "Figure 4 correlation matrix (full dataframe):",
    corr.round(3).to_string(),
    "",
    "Computed Table 5 verification:",
    metrics_df.to_string(index=False),
]

# Write the summary text file at the end of the run.
(OUTDIR / "run_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
log(f"Run summary saved to {OUTDIR / 'run_summary.txt'}.")
log("Full SDG 2025 replication pipeline completed successfully.")
