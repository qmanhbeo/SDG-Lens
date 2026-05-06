from __future__ import annotations

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
    "tfidf": "TF-IDF + LinearSVC",
    "bert": "BERT (MiniLM)",
}


def read_summary() -> list[dict[str, Any]]:
    path = RESULTS_DIR / "evaluation_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation summary: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_by_seed() -> list[dict[str, Any]]:
    path = RESULTS_DIR / "evaluation_by_seed.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def short_label(label: Any, max_chars: int = 34) -> str:
    label = str(label)
    if len(label) > max_chars:
        return label[:max_chars].rsplit(" ", 1)[0] + "..."
    return label


def sdg_label(label: int | str) -> str:
    label_id = int(label)
    return f"SDG {label_id}: {SDG_LABELS.get(label_id, 'Unknown')}"


def wrapped_sdg_label(label: int | str, width: int = 22) -> str:
    return textwrap.fill(sdg_label(label), width=width)


def truncate_token(token: Any, max_chars: int = 18) -> str:
    token = str(token)
    if len(token) > max_chars:
        return token[: max_chars - 3] + "..."
    return token


def clean_display_text(text: Any) -> str:
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
    text = clean_display_text(text)
    if len(text) > max_chars:
        return text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text


def wrap_text(text: Any, width: int = 48) -> str:
    return textwrap.fill(str(text), width=width)


def pick_best_bert_artifact() -> Path | None:
    candidates = sorted(
        ARTIFACTS_DIR.glob("bert_train*_seed42/results.json"),
        key=lambda p: int(p.parent.name.split("_train")[1].split("_")[0]),
        reverse=True,
    )
    if candidates:
        return candidates[0]
    all_bert = sorted(ARTIFACTS_DIR.glob("bert_train*_seed*/results.json"))
    return all_bert[0] if all_bert else None


def write_latex_table(rows: list[dict[str, Any]]) -> None:
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
        model_display = MODEL_DISPLAY_NAMES.get(row["model_type"], row["model_type"])
        parts = [
            f"{model_display} & {int(row['train_size']):,} & {row['n_seeds']} &",
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
    has_timing = rows and any(row.get("training_time_seconds_mean") is not None for row in rows)
    header = "| Model | Train Size | Seeds | Micro-F1 | Macro-F1 | Weighted-F1 | Subset Acc."
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    if has_timing:
        header += " | Train Time (s) |"
        sep += "---:|"
    lines = [header, sep]
    for row in rows:
        model_display = MODEL_DISPLAY_NAMES.get(row["model_type"], row["model_type"])
        parts = [
            f"| {model_display} | {int(row['train_size']):,} | {row['n_seeds']} |",
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
    display_rows = []
    for row in rows:
        r = dict(row)
        r["model_display"] = MODEL_DISPLAY_NAMES.get(r["model_type"], r["model_type"])
        display_rows.append(r)
    write_csv(TABLES_DIR / "evaluation_summary_table.csv", display_rows)


def plot_metric(rows: list[dict[str, Any]], metric: str, title: str, path: Path) -> None:
    ordered = sorted(rows, key=lambda row: (int(row["train_size"]), row["model_type"]))
    labels = [f"{MODEL_DISPLAY_NAMES.get(row['model_type'], row['model_type'])[:18]} {int(row['train_size']) // 1000}k" for row in ordered]
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


def save_metrics_figure(metrics: dict[str, Any], out_dir: Path) -> Path:
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
    per_label = metrics.get("per_label_f1", {})
    labels = list(range(1, 18))
    values = [float(per_label.get(str(label), 0.0) or 0.0) for label in labels]

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#4e9a06" if value >= 0.6 else "#c17d11" if value >= 0.3 else "#a40000" for value in values]
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


def save_examples_figure(examples: list[dict[str, Any]], out_dir: Path, n_examples: int = 5) -> Path:
    picked = examples[:n_examples]
    if not picked:
        raise ValueError("No examples available for explanation figure.")

    fig, axes = plt.subplots(
        nrows=len(picked),
        ncols=3,
        figsize=(15, 17),
        dpi=220,
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
        excerpt = wrap_text(shorten_text(example.get("text", ""), max_chars=180), width=38)
        text_ax.axis("off")
        text_ax.text(
            0.0, 1.0, f"{title}\nGold: {gold}\nPred: {pred}\n\n{excerpt}",
            ha="left", va="top", fontsize=8.0, linespacing=1.12, transform=text_ax.transAxes,
        )

        label_scores = list(example.get("top_label_scores", []))[:5]
        score_names = [short_label(wrapped_sdg_label(item["label"], width=22)) for item in label_scores]
        score_values = [float(item["score"]) for item in label_scores]
        score_y = list(range(len(score_names)))
        scores_ax.barh(score_y, score_values, color="#3465a4")
        scores_ax.set_yticks(score_y, labels=score_names)
        scores_ax.set_xlim(0, 1.0)
        scores_ax.invert_yaxis()
        if row_idx == 1:
            scores_ax.set_title("Top label scores", fontsize=9)
        scores_ax.grid(axis="x", alpha=0.25)
        scores_ax.tick_params(axis="both", labelsize=7)
        for y_pos, value in zip(score_y, score_values):
            scores_ax.text(min(value + 0.015, 0.97), y_pos, f"{value:.2f}", va="center", fontsize=6.8)

        top_tokens = list(example.get("top_tokens", []))[:5]
        token_names = [truncate_token(item["token"]) for item in top_tokens]
        token_scores = [float(item["score"]) for item in top_tokens]
        max_score = max(token_scores) if token_scores else 1.0
        token_y = list(range(len(token_names)))
        tokens_ax.barh(token_y, token_scores, color="#4e9a06")
        tokens_ax.set_yticks(token_y, labels=token_names)
        tokens_ax.set_xlim(0, max_score * 1.25)
        tokens_ax.invert_yaxis()
        if row_idx == 1:
            tokens_ax.set_title("Top attended tokens", fontsize=9)
        tokens_ax.grid(axis="x", alpha=0.25)
        tokens_ax.tick_params(axis="both", labelsize=7)
        for y_pos, value in zip(token_y, token_scores):
            tokens_ax.text(value + max_score * 0.02, y_pos, f"{value:.3f}", va="center", fontsize=6.8)

    fig.suptitle(
        "SDG Lens: Five Held-Out Explanation Examples\n"
        "Attention is a proxy explanation, not a validated rationale.",
        fontsize=13.5,
    )
    fig.subplots_adjust(
        left=0.035, right=0.985, top=0.94, bottom=0.04, wspace=0.20, hspace=0.38,
    )
    path = out_dir / "fig_explanation_examples.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def mirror_to_manuscript() -> None:
    MANUSCRIPT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    for name in ["evaluation_summary_table.tex"]:
        src = TABLES_DIR / name
        if src.exists():
            (MANUSCRIPT_TABLES_DIR / src.name).write_bytes(src.read_bytes())

    for name in [
        "model_comparison_micro_f1.png",
        "model_comparison_macro_f1.png",
        "fig_metrics.png",
        "fig_per_label_f1.png",
        "fig_explanation_examples.png",
    ]:
        src = CHARTS_DIR / name
        if src.exists():
            (MANUSCRIPT_CHARTS_DIR / src.name).write_bytes(src.read_bytes())


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

    write_latex_table(rows)
    write_csv_table(rows)
    write_markdown_table(rows)

    plot_metric(rows, "micro_f1", "SDG Lens Micro-F1 by Model and Train Size", CHARTS_DIR / "model_comparison_micro_f1.png")
    plot_metric(rows, "macro_f1", "SDG Lens Macro-F1 by Model and Train Size", CHARTS_DIR / "model_comparison_macro_f1.png")

    bert_results_path = pick_best_bert_artifact()
    if bert_results_path is None:
        print("[visualize] warning: no BERT artifact results.json found — skipping BERT-specific charts")
    else:
        print(f"[visualize] using BERT artifact: {rel_path(bert_results_path)}")
        try:
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