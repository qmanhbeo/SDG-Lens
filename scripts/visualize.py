"""Generate manuscript-ready tables and charts from evaluation outputs.

This stage is intentionally read-only with respect to model artifacts: it turns
the CSV/JSON produced by evaluate.py into publication assets and mirrors those
assets into the manuscript tree. Keeping reporting separate from evaluation
makes it safe to iterate on chart/table presentation without changing metrics.
"""

from __future__ import annotations

import numpy as np
import argparse
import csv
import json
import textwrap
import unicodedata
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline_utils import (
    ARTIFACTS_DIR,
    MANUSCRIPT_DIR,
    OUTPUTS_DIR,
    RESULTS_DIR,
    ensure_base_dirs,
    rel_path,
    write_status,
)


# Generated assets are written under outputs/ first, then copied into the
# manuscript directory. That gives users an inspectable reporting workspace while
# keeping the LaTeX source tree self-contained for compilation.
TABLES_DIR = OUTPUTS_DIR / "tables"
CHARTS_DIR = OUTPUTS_DIR / "charts"
MANUSCRIPT_VIZ_DIR = MANUSCRIPT_DIR / "visualization"
MANUSCRIPT_TABLES_DIR = MANUSCRIPT_VIZ_DIR / "tables"
MANUSCRIPT_CHARTS_DIR = MANUSCRIPT_VIZ_DIR / "charts"

SDG_LABELS = {
    1: "No Poverty",
    2: "Zero Hunger",
    3: "Good Health and Well-being",
    4: "Quality Education",
    5: "Gender Equality",
    6: "Clean Water and Sanitation",
    7: "Affordable and Clean Energy",
    8: "Decent Work and Economic Growth",
    9: "Industry, Innovation and Infrastructure",
    10: "Reduced Inequalities",
    11: "Sustainable Cities and Communities",
    12: "Responsible Consumption and Production",
    13: "Climate Action",
    14: "Life Below Water",
    15: "Life on Land",
    16: "Peace, Justice and Strong Institutions",
    17: "Partnerships for the Goals",
}

MODEL_DISPLAY_NAMES = {
    "tfidf": "TF-IDF",
    "bert": "BERT",
}


def read_summary() -> list[dict[str, Any]]:
    """Read the aggregate CSV that drives summary tables and metric plots."""
    path = RESULTS_DIR / "evaluation_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation summary: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_summary_json() -> dict[str, Any]:
    """Read nested evaluation output needed for per-label comparison figures."""
    path = RESULTS_DIR / "evaluation_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation summary JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_by_seed() -> list[dict[str, Any]]:
    """Read optional per-seed rows for future report extensions."""
    path = RESULTS_DIR / "evaluation_by_seed.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a table using the first row as the stable display schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def short_label(label: Any, max_chars: int = 34) -> str:
    """Shorten long labels so matplotlib tick labels stay legible."""
    label = str(label)
    if len(label) > max_chars:
        return label[:max_chars].rsplit(" ", 1)[0] + "..."
    return label


def sdg_label(label: int | str) -> str:
    """Return the reader-facing SDG label with its numeric prefix."""
    label_id = int(label)
    return f"SDG {label_id}: {SDG_LABELS.get(label_id, 'Unknown')}"


def wrapped_sdg_label(label: int | str, width: int = 22) -> str:
    """Wrap a full SDG label for compact chart axes."""
    return textwrap.fill(sdg_label(label), width=width)


def truncate_token(token: Any, max_chars: int = 18) -> str:
    """Shorten WordPiece tokens for explanation charts."""
    token = str(token)
    if len(token) > max_chars:
        return token[: max_chars - 3] + "..."
    return token


def clean_display_text(text: Any) -> str:
    """Remove control characters before placing free text into figures."""
    cleaned = []
    for char in str(text):
        if char in "\n\t":
            cleaned.append(" ")
        elif unicodedata.category(char).startswith("C"):
            cleaned.append(" ")
        else:
            cleaned.append(char)
    return " ".join("".join(cleaned).split())


def shorten_text(text: Any, max_chars: int = 300) -> str:
    """Trim long example text without cutting through the final word."""
    text = clean_display_text(text)
    if len(text) > max_chars:
        return text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text


def wrap_text(text: Any, width: int = 48) -> str:
    """Wrap text blocks for fixed-width matplotlib annotation areas."""
    return textwrap.fill(str(text), width=width)


def pick_best_bert_artifact() -> Path | None:
    """Pick the most relevant BERT results file for single-model diagnostic charts."""
    candidates = sorted(
        ARTIFACTS_DIR.glob("bert_train*_seed42/results.json"),
        key=lambda p: int(p.parent.name.split("_train")[1].split("_")[0]),
        reverse=True,
    )
    if candidates:
        # Prefer the largest seed-42 BERT run because it is stable across the
        # committed sweep and best represents the headline neural model.
        return candidates[0]
    all_bert = sorted(ARTIFACTS_DIR.glob("bert_train*_seed*/results.json"))
    return all_bert[0] if all_bert else None


def write_latex_table(rows: list[dict[str, Any]]) -> None:
    """Write the compact LaTeX table included by the manuscript."""
    has_timing = rows and any(row.get("training_time_seconds_mean") is not None for row in rows)
    n_cols = 5 if has_timing else 4
    align = "l" + "r" * n_cols
    lines = [
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        "Model & Train Size & Micro-F1 & Macro-F1 & Subset Acc.",
    ]
    if has_timing:
        lines[-1] += " & Train Time"
    lines[-1] += " \\\\"
    lines.append("\\midrule")
    for row in rows:
        model_display = MODEL_DISPLAY_NAMES.get(row["model_type"], row["model_type"])
        # Use the preformatted mean +/- std strings from evaluate.py so table
        # precision stays consistent across Markdown, CSV, and LaTeX views.
        parts = [
            f"{model_display} & {int(row['train_size']):,} &",
            f"{row['micro_f1_mean_pm_std']} & {row['macro_f1_mean_pm_std']} &",
            f"{row['subset_accuracy_mean_pm_std']}",
        ]
        if has_timing:
            tt = row.get("training_time_seconds_mean_pm_std", "N/A")
            parts.append(f"& {tt}")
        parts[-1] += " \\\\"
        lines.append(" ".join(parts))
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.joinpath("evaluation_summary_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_markdown_table(rows: list[dict[str, Any]]) -> None:
    """Write a quick human-readable version of the evaluation table."""
    has_timing = rows and any(row.get("training_time_seconds_mean") is not None for row in rows)
    header = "| Model | Train Size | Micro-F1 | Macro-F1 | Weighted-F1 | Subset Acc."
    sep = "|---|---:|---:|---:|---:|---:|"
    if has_timing:
        header += " | Train Time |"
        sep += "---:|"
    lines = [header, sep]
    for row in rows:
        model_display = MODEL_DISPLAY_NAMES.get(row["model_type"], row["model_type"])
        parts = [
            f"| {model_display} | {int(row['train_size']):,} |",
            f"{row['micro_f1_mean_pm_std']} | {row['macro_f1_mean_pm_std']} | "
            f"{row['weighted_f1_mean_pm_std']} | {row['subset_accuracy_mean_pm_std']} |",
        ]
        if has_timing:
            tt = row.get("training_time_seconds_mean_pm_std", "N/A")
            parts.append(f" {tt} |")
        lines.append("".join(parts))
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.joinpath("evaluation_summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv_table(rows: list[dict[str, Any]]) -> None:
    """Write a display-oriented CSV copy of the summary table."""
    display_rows = []
    for row in rows:
        r = dict(row)
        r["model_display"] = MODEL_DISPLAY_NAMES.get(r["model_type"], r["model_type"])
        display_rows.append(r)
    write_csv(TABLES_DIR / "evaluation_summary_table.csv", display_rows)


def write_threshold_sweep_table(out_dir: Path) -> None:
    """Convert threshold-sweep JSON into the LaTeX table used by the report."""
    path = RESULTS_DIR / "threshold_sweep_bert4k.json"
    if not path.exists():
        # Evaluation may be run in a partial mode. Missing sweep data should not
        # prevent other visualizations from being generated.
        return
    sweep = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        "\\begin{tabular}{rrrrr}",
        "\\toprule",
        "Threshold & Avg Predicted Labels & Avg True Labels & Micro-F1 & Zero Predictions \\% \\\\",
        "\\midrule",
    ]
    for row in sweep:
        thresh = row["threshold"]
        avg_pred = row["avg_predicted_labels"]
        avg_true = row["avg_true_labels"]
        micro = row["micro_f1"]
        zero_pct = row["fraction_zero_predictions"] * 100
        # Mark the training/evaluation default so readers can connect the sweep
        # table back to the reported BERT-4k operating point.
        marker = " *" if thresh == 0.3 else ""
        zero_pct_latex = f"{zero_pct:.1f}"
        lines.append(
            f"${thresh:.2f}$ & ${avg_pred:.3f}$ & ${avg_true:.3f}$ & ${micro:.4f}$ & {zero_pct_latex}\\%{marker} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.joinpath("threshold_sweep_table.tex").write_text("\n".join(lines), encoding="utf-8")


def plot_metric(rows: list[dict[str, Any]], metric: str, title: str, path: Path) -> None:
    """Plot mean metric values with seed-level standard-deviation error bars."""
    ordered = sorted(rows, key=lambda row: (int(row["train_size"]), row["model_type"]))
    labels = [f"{MODEL_DISPLAY_NAMES.get(row['model_type'], row['model_type'])[:18]} {int(row['train_size']) // 1000}k" for row in ordered]
    means = [float(row[f"{metric}_mean"]) for row in ordered]
    errs = [float(row[f"{metric}_std"]) for row in ordered]
    colors = ["#386cb0" if row["model_type"] == "bert" else "#7fc97f" for row in ordered]

    # Keep the figure format stable for manuscript inclusion: fixed size, fixed
    # y-axis range, and value labels for readers comparing small differences.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, means, yerr=errs, capsize=5, color=colors, edgecolor="#333333", linewidth=0.8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    for idx, value in enumerate(means):
        ax.text(idx, min(value + 0.03, 0.98), f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_metrics_figure(metrics: dict[str, Any], out_dir: Path) -> Path:
    """Create a single-artifact aggregate metric figure for diagnostics."""
    names = ["micro_f1", "macro_f1", "weighted_f1", "subset_accuracy"]
    labels = ["Micro-F1", "Macro-F1", "Weighted-F1", "Subset Acc."]
    values = [float(metrics.get(name, 0.0) or 0.0) for name in names]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, values, color=["#3465a4", "#4e9a06", "#c17d11", "#75507b"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("SDGi Lens Held-Out Test Metrics")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    path = out_dir / "fig_metrics.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_per_label_figure(metrics: dict[str, Any], out_dir: Path) -> Path:
    """Create a single-artifact per-SDG F1 figure."""
    per_label = metrics.get("per_label_f1", {})
    labels = list(range(1, 18))
    values = [float(per_label.get(str(label), 0.0) or 0.0) for label in labels]

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#4e9a06" if value >= 0.6 else "#c17d11" if value >= 0.3 else "#a40000" for value in values]
    # Color bands are only a visual diagnostic for weak/medium/strong labels;
    # the exact numeric bars remain the source of truth.
    ax.bar([str(label) for label in labels], values, color=colors)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("SDG label")
    ax.set_ylabel("F1 score")
    ax.set_title("Per-SDG F1 Scores on Held-Out Test Set")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, min(value + 0.025, 0.98), f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = out_dir / "fig_per_label_f1.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_per_label_comparison_figure(evaluation_summary: dict[str, Any], out_dir: Path) -> Path | None:
    """Compare BERT and TF-IDF per-label F1 at the 4,000-example condition."""
    summary = evaluation_summary.get("summary", [])
    bert_4k = next((r for r in summary if r["model_type"] == "bert" and r["train_size"] == 4000), None)
    tfidf_4k = next((r for r in summary if r["model_type"] == "tfidf" and r["train_size"] == 4000), None)
    if not bert_4k or not tfidf_4k:
        return None

    bert_pl = bert_4k.get("per_label_f1", {})
    tfidf_pl = tfidf_4k.get("per_label_f1", {})
    if not bert_pl or not tfidf_pl:
        # Older summary files may not include nested per-label metrics. In that
        # case the rest of the visualization stage can still complete.
        return None

    labels = list(range(1, 18))
    bert_vals = [float(bert_pl.get(str(l), 0.0) or 0.0) for l in labels]
    tfidf_vals = [float(tfidf_pl.get(str(l), 0.0) or 0.0) for l in labels]
    x = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(13, 5.5))
    b_bars = ax.bar(x - width / 2, bert_vals, width, label="BERT", color="#3a7bd5", edgecolor="white", linewidth=0.4)
    t_bars = ax.bar(x + width / 2, tfidf_vals, width, label="TF-IDF", color="#f5a623", edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in labels])
    ax.set_xlabel("SDG label")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Per-SDG F1 Scores: BERT vs TF-IDF (4,000 training samples)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    for bar in b_bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    for bar in t_bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    path = out_dir / "fig_per_label_comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_examples_figure(examples: list[dict[str, Any]], out_dir: Path, n_examples: int = 5) -> Path:
    """Render held-out examples with predicted labels and attention-token scores."""
    picked = examples[:n_examples]
    if not picked:
        raise ValueError("No examples available for explanation figure.")

    fig, axes = plt.subplots(
        nrows=len(picked),
        ncols=3,
        figsize=(15, 19),
        dpi=200,
        gridspec_kw={"width_ratios": [1.65, 1.55, 1.55]},
    )
    if len(picked) == 1:
        axes = [[axes]] if isinstance(axes, object.__class__) else [[axes]]

    row_axes_list = axes if len(picked) > 1 else [axes]
    for row_idx, (row_axes, example) in enumerate(zip(row_axes_list, picked), start=1):
        if len(row_axes) == 3:
            text_ax, scores_ax, tokens_ax = row_axes
        else:
            continue

        gold = ", ".join(str(int(l)) for l in (example.get("gold_labels", []) or [])) or "none"
        pred = ", ".join(str(int(l)) for l in (example.get("predicted_labels", []) or [])) or "none"
        quality = example.get("example_quality")
        title = f"Example {row_idx}" + (f" ({quality})" if quality else "")
        excerpt = wrap_text(shorten_text(example.get("text", ""), max_chars=180), width=36)
        # The left panel gives enough source text to interpret the labels, while
        # the two right panels expose model scores without claiming causality.
        text_ax.axis("off")
        text_ax.text(
            0.0, 1.0, f"{title}\nGold: {gold}\nPred: {pred}\n\n{excerpt}",
            ha="left", va="top", fontsize=9.5, linespacing=1.15, transform=text_ax.transAxes,
        )

        label_scores = list(example.get("top_label_scores", []))[:5]
        # Showing only the top labels keeps the figure readable; full label
        # metrics are available in the JSON artifacts.
        score_names = [short_label(wrapped_sdg_label(item["label"], width=22)) for item in label_scores]
        score_values = [float(item["score"]) for item in label_scores]
        score_y = list(range(len(score_names)))
        scores_ax.barh(score_y, score_values, color="#3465a4")
        scores_ax.set_yticks(score_y, labels=score_names)
        scores_ax.set_xlim(0, 1.0)
        scores_ax.invert_yaxis()
        if row_idx == 1:
            scores_ax.set_title("Top label scores", fontsize=10)
        scores_ax.grid(axis="x", alpha=0.25)
        scores_ax.tick_params(axis="both", labelsize=8)
        for y_pos, value in zip(score_y, score_values):
            scores_ax.text(min(value + 0.015, 0.97), y_pos, f"{value:.2f}", va="center", fontsize=7.8)

        top_tokens = list(example.get("top_tokens", []))[:5]
        # Attention is displayed as a proxy signal. It is intentionally labelled
        # as attended tokens rather than ground-truth rationales.
        token_names = [truncate_token(item["token"]) for item in top_tokens]
        token_scores = [float(item["score"]) for item in top_tokens]
        max_score = max(token_scores) if token_scores else 1.0
        token_y = list(range(len(token_names)))
        tokens_ax.barh(token_y, token_scores, color="#4e9a06")
        tokens_ax.set_yticks(token_y, labels=token_names)
        tokens_ax.set_xlim(0, max_score * 1.25)
        tokens_ax.invert_yaxis()
        if row_idx == 1:
            tokens_ax.set_title("Top attended tokens", fontsize=10)
        tokens_ax.grid(axis="x", alpha=0.25)
        tokens_ax.tick_params(axis="both", labelsize=8)
        for y_pos, value in zip(token_y, token_scores):
            tokens_ax.text(value + max_score * 0.02, y_pos, f"{value:.3f}", va="center", fontsize=7.8)

    fig.suptitle(
        "SDG Lens: Five Held-Out Explanation Examples\n"
        "Attention is a proxy explanation, not a validated rationale.",
        fontsize=14.5,
    )
    fig.subplots_adjust(
        left=0.035, right=0.985, top=0.94, bottom=0.04, wspace=0.20, hspace=0.40,
    )
    path = out_dir / "fig_explanation_examples.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def mirror_to_manuscript() -> None:
    """Copy generated report assets into the LaTeX manuscript tree."""
    MANUSCRIPT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Copy rather than symlink so the manuscript directory remains portable in
    # zip submissions or environments that do not preserve symlinks.
    for name in ["evaluation_summary_table.tex", "threshold_sweep_table.tex"]:
        src = TABLES_DIR / name
        if src.exists():
            (MANUSCRIPT_TABLES_DIR / src.name).write_bytes(src.read_bytes())

    for name in [
        "model_comparison_micro_f1.png",
        "model_comparison_macro_f1.png",
        "fig_metrics.png",
            "fig_per_label_f1.png",
            "fig_per_label_comparison.png",
            "threshold_sweep_table.tex",
        "fig_explanation_examples.png",
    ]:
        src = CHARTS_DIR / name
        if src.exists():
            (MANUSCRIPT_CHARTS_DIR / src.name).write_bytes(src.read_bytes())


def build_parser() -> argparse.ArgumentParser:
    """Build the visualization CLI used by main.py and standalone runs."""
    parser = argparse.ArgumentParser(description="Generate manuscript-ready SDG Lens tables and charts.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    """Generate all tables/charts that can be produced from current results."""
    args = build_parser().parse_args()
    ensure_base_dirs()
    if args.dry_run:
        # Dry-run verifies directory resolution but avoids touching charts or
        # manuscript assets.
        print(f"[visualize] would read {rel_path(RESULTS_DIR)} and write {rel_path(OUTPUTS_DIR)}")
        return 0

    rows = read_summary()
    if not rows:
        raise ValueError("Evaluation summary is empty; run scripts/evaluate.py first.")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Summary outputs are cheap and deterministic, so write all three table
    # formats every time visualization runs.
    write_latex_table(rows)
    write_csv_table(rows)
    write_markdown_table(rows)

    plot_metric(rows, "micro_f1", "SDG Lens Micro-F1 by Model and Train Size", CHARTS_DIR / "model_comparison_micro_f1.png")
    plot_metric(rows, "macro_f1", "SDG Lens Macro-F1 by Model and Train Size", CHARTS_DIR / "model_comparison_macro_f1.png")

    try:
        summary_json = read_summary_json()
        comp_path = save_per_label_comparison_figure(summary_json, CHARTS_DIR)
        if comp_path:
            print(f"[visualize] saved fig_per_label_comparison.png")
    except FileNotFoundError as exc:
        print(f"[visualize] warning: {exc}")

    write_threshold_sweep_table(TABLES_DIR)
    print("[visualize] saved threshold_sweep_table.tex")

    bert_results_path = pick_best_bert_artifact()
    if bert_results_path is None:
        print("[visualize] warning: no BERT artifact results.json found — skipping BERT-specific charts")
    else:
        print(f"[visualize] using BERT artifact: {rel_path(bert_results_path)}")
        try:
            # BERT-specific figures are optional diagnostics. A malformed legacy
            # result should warn, not block summary table/chart generation.
            bert_results = json.loads(bert_results_path.read_text(encoding="utf-8"))
            metrics = bert_results.get("metrics", {})
            examples = bert_results.get("examples", [])
            if metrics:
                save_metrics_figure(metrics, CHARTS_DIR)
                print(f"[visualize] saved fig_metrics.png")
            if metrics.get("per_label_f1"):
                save_per_label_figure(metrics, CHARTS_DIR)
                print(f"[visualize] saved fig_per_label_f1.png")
            if examples:
                save_examples_figure(examples, CHARTS_DIR, n_examples=min(5, len(examples)))
                print(f"[visualize] saved fig_explanation_examples.png")
        except Exception as exc:
            print(f"[visualize] warning: could not generate BERT charts from {rel_path(bert_results_path)}: {exc}")

    mirror_to_manuscript()
    write_status("visualize", "completed", "visualization", outputs_root=rel_path(OUTPUTS_DIR))
    print("[visualize] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
