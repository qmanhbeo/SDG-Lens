from __future__ import annotations

import argparse
import json
import os
import textwrap
import unicodedata
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sdg_lens_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "data"

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


def load_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sdg_label(label: int | str) -> str:
    label_id = int(label)
    return f"SDG {label_id}: {SDG_LABELS.get(label_id, 'Unknown')}"


def wrapped_sdg_label(label: int | str, width: int = 26) -> str:
    return textwrap.fill(sdg_label(label), width=width)


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


def format_label_list(labels: list[int] | list[str], width: int = 54) -> str:
    names = ", ".join(sdg_label(label) for label in labels) or "none"
    return textwrap.fill(names, width=width, subsequent_indent="  ")


def truncate_token(token: Any, max_chars: int = 18) -> str:
    token = str(token)
    if len(token) > max_chars:
        return token[: max_chars - 3] + "..."
    return token


def short_label(label: Any, max_chars: int = 34) -> str:
    label = str(label)
    if len(label) > max_chars:
        return label[:max_chars].rsplit(" ", 1)[0] + "..."
    return label


def output_dir_from_results(results: dict[str, Any], fallback: Path) -> Path:
    run_config = results.get("run_config", {}) or results.get("metadata", {}) or {}
    run_dir = run_config.get("run_dir")
    if not run_dir:
        return fallback
    path = Path(run_dir)
    if path.is_absolute():
        try:
            path.resolve().relative_to(HERE)
        except ValueError:
            return fallback
        return path
    return HERE / path


def examples_from_results(results: dict[str, Any], out_dir: Path, n_examples: int) -> list[dict[str, Any]]:
    mixed_path = out_dir / "examples_mixed.json"
    if n_examples >= 5 and mixed_path.exists():
        with mixed_path.open("r", encoding="utf-8") as f:
            print(f"Using mixed examples: {mixed_path}")
            return json.load(f)
    return results.get("examples", [])


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
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
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


def save_examples_figure(examples: list[dict[str, Any]], out_dir: Path, n_examples: int) -> Path:
    picked = examples[:n_examples]
    if not picked:
        raise ValueError("No examples available for explanation figure.")

    fig, axes = plt.subplots(
        nrows=len(picked),
        ncols=3,
        figsize=(15, 17),
        dpi=220,
        gridspec_kw={"width_ratios": [1.65, 1.55, 1.55]}
    )
    if len(picked) == 1:
        axes = [axes]

    for row_idx, (row_axes, example) in enumerate(zip(axes, picked), start=1):
        text_ax, scores_ax, tokens_ax = row_axes
        gold = ", ".join(str(int(l)) for l in (example.get("gold_labels", []) or [])) or "none"
        pred = ", ".join(str(int(l)) for l in (example.get("predicted_labels", []) or [])) or "none"
        quality = example.get("example_quality")
        title = f"Example {row_idx}" + (f" ({quality})" if quality else "")
        excerpt = wrap_text(shorten_text(example.get("text", ""), max_chars=180), width=38)
        text_ax.axis("off")
        text_ax.text(
            0.0,
            1.0,
            f"{title}\nGold: {gold}\nPred: {pred}\n\n{excerpt}",
            ha="left",
            va="top",
            fontsize=8.0,
            linespacing=1.12,
            transform=text_ax.transAxes,
        )

        label_scores = list(example.get("top_label_scores", []))
        label_scores = label_scores[:5]
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
        left=0.035,
        right=0.985,
        top=0.94,
        bottom=0.04,
        wspace=0.20,
        hspace=0.38
    )
    path = out_dir / "fig_explanation_examples.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def pick_diverse(indices: list[int], true_rows: list[set[int]], n: int) -> list[int]:
    picked: list[int] = []
    seen_labels: set[int] = set()
    for idx in indices:
        labels = true_rows[idx]
        if labels and labels.isdisjoint(seen_labels):
            picked.append(idx)
            seen_labels.update(labels)
        if len(picked) == n:
            return picked
    for idx in indices:
        if idx not in picked:
            picked.append(idx)
        if len(picked) == n:
            return picked
    return picked


def build_mixed_examples(results: dict[str, Any], checkpoint_path: Path, device_name: str) -> list[dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from run import (
        BertMultiLabelAttentionClassifier,
        DEFAULT_DATA_DIR as RUN_DEFAULT_DATA_DIR,
        SDGiDataset,
        checkpoint_data_dir,
        evaluate_model,
        explain_examples,
        load_checkpoint,
        load_sdgi_split,
        make_collate_fn,
        pick_device,
    )

    run_config = results.get("run_config", {}) or results.get("metadata", {}) or {}
    device = pick_device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)

    model_name = str(checkpoint.get("model_name", run_config.get("model_name")))
    threshold = float(checkpoint.get("threshold", run_config.get("threshold", 0.3)))
    language = str(checkpoint.get("language", run_config.get("language", "en")))
    data_dir = checkpoint_data_dir(
        checkpoint.get("data_dir", run_config.get("data_dir", RUN_DEFAULT_DATA_DIR))
    )
    test_limit = int(checkpoint.get("test_limit", run_config.get("test_limit", 300)))
    seed = int(checkpoint.get("seed", run_config.get("seed", 42)))
    batch_size = int(checkpoint.get("batch_size", run_config.get("batch_size", 8)))
    max_length = int(checkpoint.get("max_length", run_config.get("max_length", 128)))
    trainable_layers = int(checkpoint.get("trainable_encoder_layers", checkpoint.get("unfrozen_layers", 2)))
    unfreeze_encoder = bool(checkpoint.get("unfreeze_encoder", run_config.get("unfreeze_encoder", False)))

    test_frame = load_sdgi_split(data_dir, "test", language, test_limit, seed + 1)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=True)
    model = BertMultiLabelAttentionClassifier(
        model_name,
        num_labels=int(checkpoint.get("num_labels", 17)),
        local_files_only=True,
        trainable_encoder_layers=trainable_layers,
        unfreeze_encoder=unfreeze_encoder,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    loader = DataLoader(
        SDGiDataset(test_frame),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, max_length),
    )
    _, y_true, y_pred, _ = evaluate_model(model, loader, device, threshold)

    true_rows = [{idx + 1 for idx, value in enumerate(row.tolist()) if int(value) == 1} for row in y_true]
    pred_rows = [{idx + 1 for idx, value in enumerate(row.tolist()) if int(value) == 1} for row in y_pred]
    exact = [idx for idx, (gold, pred) in enumerate(zip(true_rows, pred_rows)) if gold == pred]
    partial = [idx for idx, (gold, pred) in enumerate(zip(true_rows, pred_rows)) if gold & pred and gold != pred]
    wrong = [idx for idx, (gold, pred) in enumerate(zip(true_rows, pred_rows)) if not (gold & pred)]

    selected = pick_diverse(exact, true_rows, 3)
    if partial:
        selected.extend(pick_diverse(partial, true_rows, 2))
    if wrong:
        selected.extend(pick_diverse(wrong, true_rows, 2))
    if len(selected) < 7:
        selected.extend(idx for idx in range(len(test_frame)) if idx not in selected)
    selected = selected[:7]

    qualities = ["Good", "Good", "Good", "So-so / partial", "So-so / partial", "Bad", "Bad"]
    selected_frame = test_frame.iloc[selected].reset_index(drop=True)
    examples = explain_examples(
        model=model,
        tokenizer=tokenizer,
        frame=selected_frame,
        device=device,
        max_length=max_length,
        threshold=threshold,
        n_examples=len(selected),
        top_k=12,
    )
    for example, quality in zip(examples, qualities):
        example["example_quality"] = quality
    return examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create report figures for SDGi Lens runs.")
    parser.add_argument("--results", type=Path, default=HERE / "outputs" / "results.json")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mixed-examples", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    results = load_results(args.results)
    out_dir = args.output_dir or output_dir_from_results(results, args.results.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = results.get("metrics", {})
    if args.mixed_examples:
        if args.checkpoint is None:
            raise ValueError("--mixed-examples requires --checkpoint.")
        examples = build_mixed_examples(results, args.checkpoint, args.device)
        write_path = out_dir / "examples_mixed.json"
        write_path.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved mixed examples: {write_path}")
    else:
        examples = examples_from_results(results, out_dir, args.examples)
    paths = [
        save_metrics_figure(metrics, out_dir),
        save_per_label_figure(metrics, out_dir),
        save_examples_figure(examples, out_dir, args.examples),
    ]

    print("Saved report figures:")
    for path in paths:
        print(f"  {path}")
    return 0


def main() -> int:
    return run_from_args(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit("Use `python main.py plot`.")
