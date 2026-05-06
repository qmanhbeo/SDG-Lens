from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sdg_lens_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline_utils import (
    MANUSCRIPT_DIR,
    OUTPUTS_DIR,
    RESULTS_DIR,
    ensure_base_dirs,
    rel_path,
    write_status,
)


TABLES_DIR = OUTPUTS_DIR / "tables"
CHARTS_DIR = OUTPUTS_DIR / "charts"
MANUSCRIPT_VIZ_DIR = MANUSCRIPT_DIR / "visualization"
MANUSCRIPT_TABLES_DIR = MANUSCRIPT_VIZ_DIR / "tables"
MANUSCRIPT_CHARTS_DIR = MANUSCRIPT_VIZ_DIR / "charts"


def read_summary() -> list[dict[str, Any]]:
    path = RESULTS_DIR / "evaluation_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation summary: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    has_timing = rows and any(row.get("training_time_seconds_mean") is not None for row in rows)
    header = "| Model | Train Size | Seeds | Micro-F1 | Macro-F1 | Weighted-F1 | Subset Acc."
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    if has_timing:
        header += " | Train Time (s) |"
        sep += "---:|"
    lines = [header, sep]
    for row in rows:
        parts = [
            f"| {row['model_type']} | {row['train_size']} | {row['n_seeds']} |",
            f"{row['micro_f1_mean_pm_std']} | {row['macro_f1_mean_pm_std']} | "
            f"{row['weighted_f1_mean_pm_std']} | {row['subset_accuracy_mean_pm_std']} |",
        ]
        if has_timing:
            tt = row.get("training_time_seconds_mean_pm_std", "N/A")
            parts.append(f" {tt} |")
        lines.append("".join(parts))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(path: Path, rows: list[dict[str, Any]]) -> None:
    has_timing = rows and any(row.get("training_time_seconds_mean") is not None for row in rows)
    n_cols = 6 if has_timing else 5
    align = "l" + "r" * n_cols
    lines = [
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        "Model & Train Size & Seeds & Micro-F1 & Macro-F1 & Subset Acc.",
    ]
    if has_timing:
        lines[-1] += " & Train Time (s)"
    lines[-1] += " \\\\"
    lines.append("\\midrule")
    for row in rows:
        parts = [
            f"{row['model_type']} & {row['train_size']} & {row['n_seeds']} &",
            f"{row['micro_f1_mean_pm_std']} & {row['macro_f1_mean_pm_std']} &",
            f"{row['subset_accuracy_mean_pm_std']}",
        ]
        if has_timing:
            tt = row.get("training_time_seconds_mean_pm_std", "N/A")
            parts.append(f"& {tt}")
        parts[-1] += " \\\\"
        lines.append(" ".join(parts))
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_metric(rows: list[dict[str, Any]], metric: str, title: str, path: Path) -> None:
    ordered = sorted(rows, key=lambda row: (int(row["train_size"]), row["model_type"]))
    labels = [f"{row['model_type']} {int(row['train_size']) // 1000}k" for row in ordered]
    means = [float(row[f"{metric}_mean"]) for row in ordered]
    errs = [float(row[f"{metric}_std"]) for row in ordered]
    colors = ["#386cb0" if row["model_type"] == "bert" else "#7fc97f" for row in ordered]

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


def mirror_to_manuscript() -> None:
    MANUSCRIPT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    for src in [TABLES_DIR / "evaluation_summary_table.tex"]:
        dst = MANUSCRIPT_TABLES_DIR / src.name
        dst.write_bytes(src.read_bytes())
    for src in [
        CHARTS_DIR / "model_comparison_micro_f1.png",
        CHARTS_DIR / "model_comparison_macro_f1.png",
    ]:
        dst = MANUSCRIPT_CHARTS_DIR / src.name
        dst.write_bytes(src.read_bytes())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate manuscript-ready SDG Lens tables and charts.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ensure_base_dirs()
    if args.dry_run:
        print(f"[visualize] would read {rel_path(RESULTS_DIR)} and write {rel_path(OUTPUTS_DIR)}")
        return 0

    rows = read_summary()
    if not rows:
        raise ValueError("Evaluation summary is empty; run scripts/evaluate.py first.")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(TABLES_DIR / "evaluation_summary_table.csv", rows)
    write_markdown(TABLES_DIR / "evaluation_summary_table.md", rows)
    write_latex_table(TABLES_DIR / "evaluation_summary_table.tex", rows)
    plot_metric(rows, "micro_f1", "SDG Lens Micro-F1 by Model and Train Size", CHARTS_DIR / "model_comparison_micro_f1.png")
    plot_metric(rows, "macro_f1", "SDG Lens Macro-F1 by Model and Train Size", CHARTS_DIR / "model_comparison_macro_f1.png")
    mirror_to_manuscript()
    write_status("visualize", "completed", "visualization", outputs_root=rel_path(OUTPUTS_DIR))
    print("[visualize] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
