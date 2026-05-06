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

import tfidf_baseline


def artifact_complete(train_size: int, seed: int) -> bool:
    meta_path = artifact_metadata_path("tfidf", train_size, seed)
    if not meta_path.exists():
        return False
    meta = read_json(meta_path)
    model = project_path(meta["paths"]["model"])
    results = project_path(meta["paths"]["results"])
    return model.exists() and results.exists()


def train_one(args: argparse.Namespace, train_size: int, seed: int) -> dict[str, Any]:
    out_dir = artifact_dir("tfidf", train_size, seed)
    meta_path = artifact_metadata_path("tfidf", train_size, seed)
    if artifact_complete(train_size, seed) and not args.force:
        print(f"[baseline] skip existing tfidf train={train_size} seed={seed}")
        return read_json(meta_path)

    if args.dry_run:
        print(f"[baseline] would train tfidf train={train_size} seed={seed} -> {rel_path(out_dir)}")
        return {}

    print(f"[baseline] training tfidf train={train_size} seed={seed}")
    result = tfidf_baseline.run_tfidf_baseline(
        data_dir=DATA_DIR,
        output_dir=out_dir,
        language=args.language,
        max_features=args.max_features,
        alpha=args.alpha,
        random_state=seed,
        test_random_state=args.test_seed,
        train_limit=train_size,
        test_limit=args.test_samples,
    )
    model_path = out_dir / "tfidf_model.joblib"
    results_path = out_dir / "results.json"
    if not model_path.exists():
        raise FileNotFoundError(f"TF-IDF training did not create model artifact: {model_path}")

    meta = {
        "schema_version": 1,
        "model_type": "tfidf",
        "artifact_id": artifact_id("tfidf", train_size, seed),
        "created_at": now_iso(),
        "seed": seed,
        "train_size": train_size,
        "test_seed": args.test_seed,
        "test_samples": args.test_samples,
        "language": args.language,
        "paths": {
            "artifact_dir": rel_path(out_dir),
            "model": rel_path(model_path),
            "results": rel_path(results_path),
        },
        "metrics": result.get("metrics", {}),
        "run_config": result.get("run_config", {}),
    }
    write_json(meta_path, meta)
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train TF-IDF SDG Lens artifacts.")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--train-sizes", nargs="+", type=int, default=TRAIN_SIZES)
    parser.add_argument("--language", choices=["en", "es", "fr", "all"], default="en")
    parser.add_argument("--test-samples", type=int, default=TEST_SAMPLES)
    parser.add_argument("--test-seed", type=int, default=TEST_SEED)
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dry_run:
        for train_size in args.train_sizes:
            for seed in args.seeds:
                train_one(args, train_size, seed)
        print("[baseline] dry run complete")
        return 0

    ensure_base_dirs()
    total = len(args.seeds) * len(args.train_sizes)
    write_status("baseline", "running", "tfidf_training", artifacts_root=rel_path(ARTIFACTS_DIR), progress={"completed": 0, "total": total})
    completed = 0
    for train_size in args.train_sizes:
        for seed in args.seeds:
            train_one(args, train_size, seed)
            completed += 1
            write_status("baseline", "running", "tfidf_training", progress={"completed": completed, "total": total})
    write_status("baseline", "completed", "tfidf_training", progress={"completed": completed, "total": total})
    print("[baseline] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
