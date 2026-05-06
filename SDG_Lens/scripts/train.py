from __future__ import annotations

import argparse
from typing import Any

from pipeline_utils import (
    ARTIFACTS_DIR,
    DATA_DIR,
    SEEDS,
    TEST_SAMPLES,
    TEST_SEED,
    TRAIN_SIZES,
    artifact_dir,
    artifact_id,
    artifact_metadata_path,
    ensure_base_dirs,
    now_iso,
    project_path,
    read_json,
    rel_path,
    write_json,
    write_status,
)

import run as bert_run


def artifact_complete(train_size: int, seed: int) -> bool:
    meta_path = artifact_metadata_path("bert", train_size, seed)
    if not meta_path.exists():
        return False
    meta = read_json(meta_path)
    checkpoint = project_path(meta["paths"]["checkpoint"])
    results = project_path(meta["paths"]["results"])
    return checkpoint.exists() and results.exists()


def train_one(args: argparse.Namespace, train_size: int, seed: int) -> dict[str, Any]:
    out_dir = artifact_dir("bert", train_size, seed)
    meta_path = artifact_metadata_path("bert", train_size, seed)
    if artifact_complete(train_size, seed) and not args.force:
        print(f"[train] skip existing bert train={train_size} seed={seed}")
        return read_json(meta_path)

    if args.dry_run:
        print(f"[train] would train bert train={train_size} seed={seed} -> {rel_path(out_dir)}")
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] training bert train={train_size} seed={seed}")
    bert_args = bert_run.build_parser().parse_args(
        [
            "--data_dir",
            str(DATA_DIR),
            "--output_dir",
            str(out_dir),
            "--model_name",
            args.model_name,
            "--language",
            args.language,
            "--train_samples",
            str(train_size),
            "--test_samples",
            str(args.test_samples),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--max_length",
            str(args.max_length),
            "--learning_rate",
            str(args.learning_rate),
            "--threshold",
            str(args.threshold),
            "--examples",
            str(args.examples),
            "--top_tokens",
            str(args.top_tokens),
            "--device",
            args.device,
            "--trainable_encoder_layers",
            str(args.trainable_encoder_layers),
            "--seed",
            str(seed),
            "--test_seed",
            str(args.test_seed),
        ]
    )
    if args.allow_download:
        bert_args.allow_download = True
    if args.unfreeze_encoder:
        bert_args.unfreeze_encoder = True

    bert_run.run_from_args(bert_args)
    results_path = out_dir / "results.json"
    results = read_json(results_path)
    run_config = results.get("run_config", {})
    run_dir = project_path(run_config["run_dir"])
    checkpoint = run_dir / "model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"BERT training did not create checkpoint: {checkpoint}")

    meta = {
        "schema_version": 1,
        "model_type": "bert",
        "artifact_id": artifact_id("bert", train_size, seed),
        "created_at": now_iso(),
        "seed": seed,
        "train_size": train_size,
        "test_seed": args.test_seed,
        "test_samples": args.test_samples,
        "language": args.language,
        "paths": {
            "artifact_dir": rel_path(out_dir),
            "checkpoint": rel_path(checkpoint),
            "results": rel_path(results_path),
            "run_dir": rel_path(run_dir),
        },
        "metrics": results.get("metrics", {}),
        "run_config": run_config,
    }
    write_json(meta_path, meta)
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train BERT SDG Lens artifacts.")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--train-sizes", nargs="+", type=int, default=TRAIN_SIZES)
    parser.add_argument("--language", choices=["en", "es", "fr", "all"], default="en")
    parser.add_argument("--test-samples", type=int, default=TEST_SAMPLES)
    parser.add_argument("--test-seed", type=int, default=TEST_SEED)
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--top-tokens", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--unfreeze-encoder", action="store_true")
    parser.add_argument("--trainable-encoder-layers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dry_run:
        for train_size in args.train_sizes:
            for seed in args.seeds:
                train_one(args, train_size, seed)
        print("[train] dry run complete")
        return 0

    ensure_base_dirs()
    write_status(
        "train",
        "running",
        "bert_training",
        artifacts_root=rel_path(ARTIFACTS_DIR),
        progress={"completed": 0, "total": len(args.seeds) * len(args.train_sizes)},
    )
    completed = 0
    for train_size in args.train_sizes:
        for seed in args.seeds:
            train_one(args, train_size, seed)
            completed += 1
            write_status(
                "train",
                "running",
                "bert_training",
                progress={"completed": completed, "total": len(args.seeds) * len(args.train_sizes)},
            )
    write_status("train", "completed", "bert_training", progress={"completed": completed, "total": completed})
    print("[train] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
