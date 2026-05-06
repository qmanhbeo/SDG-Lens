from __future__ import annotations

import argparse
import json
import random
import re
import string
import time as time_module
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore", category=ConvergenceWarning)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
SDG_IDS = list(range(1, 18))
SDG_NAMES = [f"SDG_{i}" for i in SDG_IDS]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def preprocess_text(text: str) -> str:
    if text is None:
        return ""
    text = re.sub(r"(\b\d+[\.,]?\d*\b)", "NUM", str(text))
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def load_sdgi_parquet(
    data_dir: Path,
    language: str = "en",
    random_state: int = 42,
    test_random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "train-00000-of-00001.parquet"
    test_path = data_dir / "test-00000-of-00001.parquet"
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    if language not in {"xx", "all"}:
        def get_lang(row) -> str:
            m = row.get("metadata") or {}
            return m.get("language", "")

        train_df = train_df[train_df.apply(get_lang, axis=1) == language]
        test_df = test_df[test_df.apply(get_lang, axis=1) == language]

    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(
        frac=1,
        random_state=random_state if test_random_state is None else test_random_state,
    ).reset_index(drop=True)

    return train_df, test_df


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "subset_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
    }


def run_tfidf_baseline(
    data_dir: Path,
    output_dir: Path,
    language: str = "en",
    max_features: int = 10000,
    alpha: float = 1e-4,
    random_state: int = 42,
    test_random_state: int | None = None,
    train_limit: int | None = None,
    test_limit: int | None = None,
) -> dict[str, Any]:
    set_seed(random_state)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_sdgi_parquet(data_dir, language, random_state, test_random_state)
    train_n = len(train_df)
    test_n = len(test_df)

    if train_limit:
        train_df = train_df.head(train_limit)
    if test_limit:
        test_df = test_df.head(test_limit)

    x_train_texts = np.asarray([preprocess_text(t) for t in train_df["text"]], dtype=object)
    x_test_texts = np.asarray([preprocess_text(t) for t in test_df["text"]], dtype=object)

    binarizer = MultiLabelBinarizer(classes=SDG_IDS)
    y_train = binarizer.fit_transform(train_df["labels"].tolist())
    y_test = binarizer.transform(test_df["labels"].tolist())

    vectorizer = TfidfVectorizer(
        strip_accents="ascii",
        lowercase=False,
        ngram_range=(1, 1),
        min_df=10,
        max_df=0.9,
        max_features=max_features,
        binary=True,
    )
    t0 = time_module.perf_counter()
    vectorizer.fit(x_train_texts)
    x_train = vectorizer.transform(x_train_texts)
    x_test = vectorizer.transform(x_test_texts)
    models = []
    for i in range(17):
        y_col = y_train[:, i]
        if len(np.unique(y_col)) < 2:
            models.append(None)
            continue
        clf = SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=alpha,
            shuffle=True,
            learning_rate="optimal",
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.2,
            class_weight="balanced",
        )
        clf.fit(x_train, y_col)
        models.append(clf)
    training_time_seconds = time_module.perf_counter() - t0
    training_time_minutes = training_time_seconds / 60.0

    y_train_pred = np.zeros_like(y_train)
    y_test_pred = np.zeros_like(y_test)
    for i, clf in enumerate(models):
        if clf is None:
            y_train_pred[:, i] = y_train[0, i]
            y_test_pred[:, i] = y_train[0, i]
        else:
            y_train_pred[:, i] = clf.predict(x_train)
            y_test_pred[:, i] = clf.predict(x_test)

    train_metrics = evaluate_predictions(y_train, y_train_pred)
    test_metrics = evaluate_predictions(y_test, y_test_pred)

    per_label_f1 = {}
    for i, name in enumerate(SDG_NAMES):
        per_label_f1[name] = float(f1_score(y_test[:, i], y_test_pred[:, i], zero_division=0))

    output = {
        "run_config": {
            "method": "TF-IDF + LinearSVC",
            "language": language,
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "source_train_rows": train_n,
            "source_test_rows": test_n,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_limit": train_limit,
            "test_limit": test_limit,
            "max_features": max_features,
            "alpha": alpha,
            "random_state": random_state,
            "test_random_state": test_random_state,
        },
        "metrics": test_metrics,
        "per_label_f1": per_label_f1,
        "train_metrics": train_metrics,
        "timing": {"training_time_seconds": training_time_seconds, "training_time_minutes": training_time_minutes},
    }

    (output_dir / "results.json").write_text(json.dumps(output, indent=2, ensure_ascii=False))
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "models": models,
            "binarizer": binarizer,
            "run_config": output["run_config"],
            "metrics": test_metrics,
            "per_label_f1": per_label_f1,
            "timing": {"training_time_seconds": training_time_seconds, "training_time_minutes": training_time_minutes},
        },
        output_dir / "tfidf_model.joblib",
    )

    print(f"TF-IDF baseline: micro_f1={test_metrics['micro_f1']:.4f}, macro_f1={test_metrics['macro_f1']:.4f}")
    return output


# Marker-facing baseline stage orchestration.
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
    result = run_tfidf_baseline(
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
        "timing": result.get("timing", {}),
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
