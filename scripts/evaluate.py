from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from pipeline_utils import (
    DATA_DIR,
    RESULTS_DIR,
    SEEDS,
    TEST_SAMPLES,
    TEST_SEED,
    TRAIN_SIZES,
    ensure_base_dirs,
    load_artifact_metadata,
    project_path,
    read_json,
    rel_path,
    required_conditions,
    write_json,
    write_status,
)

import baseline
import train as bert_run


METRIC_NAMES = ["micro_f1", "macro_f1", "weighted_f1", "subset_accuracy"]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def require_artifacts(metas: list[dict[str, Any]], allow_missing: bool) -> None:
    found = {(meta["model_type"], int(meta["train_size"]), int(meta["seed"])) for meta in metas}
    missing = [item for item in required_conditions() if item not in found]
    if missing and not allow_missing:
        readable = ", ".join(f"{model}/train{size}/seed{seed}" for model, size, seed in missing)
        raise FileNotFoundError(f"Missing required trained artifacts in artifacts/: {readable}")


def evaluate_bert(meta: dict[str, Any], device_name: str, allow_download: bool) -> dict[str, float]:
    device = bert_run.pick_device(device_name)
    checkpoint = bert_run.load_checkpoint(project_path(meta["paths"]["checkpoint"]), device)
    run_config = checkpoint.get("run_config", {})
    model_name = str(checkpoint.get("model_name", run_config.get("model_name")))
    language = str(meta.get("language", checkpoint.get("language", run_config.get("language", "en"))))
    test_samples = int(meta.get("test_samples", checkpoint.get("test_limit", run_config.get("test_limit", TEST_SAMPLES))))
    test_seed = int(meta.get("test_seed", checkpoint.get("test_seed", run_config.get("test_seed", TEST_SEED))))
    threshold = float(checkpoint.get("threshold", run_config.get("threshold", 0.3)))
    batch_size = int(checkpoint.get("batch_size", run_config.get("batch_size", 8)))
    max_length = int(checkpoint.get("max_length", run_config.get("max_length", 128)))
    trainable_layers = int(checkpoint.get("trainable_encoder_layers", checkpoint.get("unfrozen_layers", 2)))
    unfreeze_encoder = bool(checkpoint.get("unfreeze_encoder", run_config.get("unfreeze_encoder", False)))

    test_frame = bert_run.load_sdgi_split(DATA_DIR, "test", language, test_samples, test_seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=not allow_download)
    model = bert_run.BertMultiLabelAttentionClassifier(
        model_name,
        num_labels=int(checkpoint.get("num_labels", 17)),
        local_files_only=not allow_download,
        trainable_encoder_layers=trainable_layers,
        unfreeze_encoder=unfreeze_encoder,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    loader = DataLoader(
        bert_run.SDGiDataset(test_frame),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=bert_run.make_collate_fn(tokenizer, max_length),
    )
    metrics, _, _, _ = bert_run.evaluate_model(model, loader, device, threshold)
    return metrics


def evaluate_tfidf(meta: dict[str, Any]) -> dict[str, Any]:
    model_payload = joblib.load(project_path(meta["paths"]["model"]))
    language = str(meta.get("language", "en"))
    seed = int(meta["seed"])
    test_seed = int(meta.get("test_seed", TEST_SEED))
    test_samples = int(meta.get("test_samples", TEST_SAMPLES))
    _, test_df = baseline.load_sdgi_parquet(DATA_DIR, language, seed, test_seed)
    test_df = test_df.head(test_samples)
    x_test_texts = np.asarray([baseline.preprocess_text(text) for text in test_df["text"]], dtype=object)
    y_test = model_payload["binarizer"].transform(test_df["labels"].tolist())
    x_test = model_payload["vectorizer"].transform(x_test_texts)

    y_pred = np.zeros_like(y_test)
    for idx, clf in enumerate(model_payload["models"]):
        if clf is None:
            y_pred[:, idx] = 0
        else:
            y_pred[:, idx] = clf.predict(x_test)

    metrics = baseline.evaluate_predictions(y_test, y_pred)
    metrics["per_label_f1"] = {
        str(label): float(score)
        for label, score in zip(baseline.SDG_IDS, f1_score(y_test, y_pred, average=None, zero_division=0).tolist())
    }
    return metrics


def row_from_metrics(meta: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    row = {
        "model_type": meta["model_type"],
        "train_size": int(meta["train_size"]),
        "seed": int(meta["seed"]),
        "artifact_id": meta["artifact_id"],
        "artifact_dir": meta["paths"]["artifact_dir"],
        **{name: metrics.get(name) for name in METRIC_NAMES},
    }
    timing = meta.get("timing", {})
    if timing:
        row["training_time_seconds"] = timing.get("training_time_seconds")
    return row


def write_csv(path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model_type"], int(row["train_size"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (model_type, train_size), group_rows in sorted(groups.items()):
        summary = {
            "model_type": model_type,
            "train_size": train_size,
            "condition": f"{model_type}_{train_size // 1000}k",
            "n_seeds": len(group_rows),
            "seeds": " ".join(str(row["seed"]) for row in sorted(group_rows, key=lambda item: item["seed"])),
        }
        for metric in METRIC_NAMES:
            values = [float(row[metric]) for row in group_rows if row.get(metric) is not None]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = std(values)
            summary[f"{metric}_mean_pm_std"] = f"{mean(values):.3f} +/- {std(values):.3f}"
        timing_values = [float(row["training_time_seconds"]) for row in group_rows if row.get("training_time_seconds") is not None]
        if timing_values:
            summary["training_time_seconds_mean"] = mean(timing_values)
            summary["training_time_seconds_std"] = std(timing_values)
            summary["training_time_seconds_mean_pm_std"] = f"{mean(timing_values):.1f} +/- {std(timing_values):.1f}"
        summary_rows.append(summary)
    return summary_rows


def write_markdown(path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Evaluation Summary",
        "",
        "| Model | Train Size | Seeds | Micro-F1 | Macro-F1 | Weighted-F1 | Subset Acc. |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {model} | {size} | {n} | {micro} | {macro} | {weighted} | {subset} |".format(
                model=row["model_type"],
                size=row["train_size"],
                n=row["n_seeds"],
                micro=row["micro_f1_mean_pm_std"],
                macro=row["macro_f1_mean_pm_std"],
                weighted=row["weighted_f1_mean_pm_std"],
                subset=row["subset_accuracy_mean_pm_std"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained SDG Lens artifacts.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ensure_base_dirs()
    metas = load_artifact_metadata()
    try:
        require_artifacts(metas, allow_missing=args.allow_missing or args.dry_run)
    except FileNotFoundError as exc:
        raise SystemExit(f"[evaluate] {exc}") from exc
    if args.dry_run:
        print(f"[evaluate] would evaluate {len(metas)} artifacts -> {rel_path(RESULTS_DIR)}")
        return 0

    rows: list[dict[str, Any]] = []
    write_status("evaluate", "running", "evaluation", progress={"completed": 0, "total": len(metas)})
    for idx, meta in enumerate(sorted(metas, key=lambda item: (item["model_type"], item["train_size"], item["seed"])), start=1):
        print(f"[evaluate] {meta['artifact_id']}")
        if meta["model_type"] == "bert":
            metrics = evaluate_bert(meta, args.device, args.allow_download)
        elif meta["model_type"] == "tfidf":
            metrics = evaluate_tfidf(meta)
        else:
            raise ValueError(f"Unknown model_type in artifact metadata: {meta['model_type']}")
        rows.append(row_from_metrics(meta, metrics))
        write_status("evaluate", "running", "evaluation", progress={"completed": idx, "total": len(metas)})

    summary_rows = summarize(rows)
    by_seed_columns = ["model_type", "train_size", "seed", "artifact_id", "artifact_dir", *METRIC_NAMES]
    summary_columns = ["model_type", "train_size", "condition", "n_seeds", "seeds"]
    for metric in METRIC_NAMES:
        summary_columns.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_mean_pm_std"])
    if rows and rows[0].get("training_time_seconds") is not None:
        by_seed_columns.append("training_time_seconds")
    if any(row.get("training_time_seconds_mean") is not None for row in summary_rows):
        summary_columns.extend(["training_time_seconds_mean", "training_time_seconds_std", "training_time_seconds_mean_pm_std"])
    write_csv(RESULTS_DIR / "evaluation_by_seed.csv", rows, by_seed_columns)
    write_csv(RESULTS_DIR / "evaluation_summary.csv", summary_rows, summary_columns)
    write_json(RESULTS_DIR / "evaluation_summary.json", {"by_seed": rows, "summary": summary_rows})
    write_markdown(RESULTS_DIR / "evaluation_summary.md", summary_rows)
    write_status("evaluate", "completed", "evaluation", progress={"completed": len(metas), "total": len(metas)})
    print("[evaluate] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
