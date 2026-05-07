"""TF-IDF baseline training for SDG Lens.

This script provides a deliberately simple comparison point for the BERT model:
bag-of-words style features, one linear classifier per SDG label, and the same
train/test sampling contract used by the neural pipeline. The goal is not to
maximize baseline complexity; it is to make the neural model's gain measurable
against a transparent and reproducible reference.
"""

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

# Linear SVM-style optimization can emit convergence warnings on small or
# imbalanced label slices. The persisted metrics are the contract here, so the
# warning is suppressed to keep marker-facing logs readable.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

# SDGi labels are stored as 1-based SDG identifiers. Internal matrices keep one
# column per SDG in the same order so metrics can be mapped back to label names.
SDG_IDS = list(range(1, 18))
SDG_NAMES = [f"SDG_{i}" for i in SDG_IDS]


def set_seed(seed: int) -> None:
    """Seed Python and NumPy so text shuffling and model initialization repeat."""
    random.seed(seed)
    np.random.seed(seed)


def preprocess_text(text: str) -> str:
    """Normalize text for sparse lexical features while preserving word signal."""
    if text is None:
        return ""
    # Numbers can be topical but exact values are usually too sparse; replacing
    # them gives the linear baseline a reusable numeric signal.
    text = re.sub(r"(\b\d+[\.,]?\d*\b)", "NUM", str(text))
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def coverage_sample(frame: pd.DataFrame, n_rows: int | None, seed: int) -> pd.DataFrame:
    """Sample rows with the same SDG coverage rule used by the BERT pipeline."""
    limit = 0 if n_rows is None else int(n_rows)
    if limit <= 0 or limit >= len(frame):
        # A non-positive or oversized limit means "use the full split"; still
        # shuffle so saved examples and evaluation order remain seed-controlled.
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    rng = random.Random(seed)
    label_sets = {
        idx: {int(label) for label in labels}
        for idx, labels in zip(frame.index.tolist(), frame["labels"].tolist())
    }
    selected: list[int] = []
    seen: set[int] = set()

    for sdg_id in SDG_IDS:
        # Reserve at least one row for each SDG that exists in the filtered
        # split, preventing small TF-IDF slices from silently losing rare goals.
        candidates = [idx for idx, labels in label_sets.items() if sdg_id in labels]
        if not candidates:
            continue
        choice = rng.choice(candidates)
        if choice not in seen:
            selected.append(choice)
            seen.add(choice)

    remaining = [idx for idx in frame.index.tolist() if idx not in seen]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, limit - len(selected))])
    selected = selected[:limit]
    rng.shuffle(selected)
    return frame.loc[selected].reset_index(drop=True)


def load_sdgi_parquet(
    data_dir: Path,
    language: str = "en",
    random_state: int = 42,
    test_random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load SDGi train/test parquet files and apply the shared language split."""
    train_path = data_dir / "train-00000-of-00001.parquet"
    test_path = data_dir / "test-00000-of-00001.parquet"
    train_df = pd.read_parquet(train_path, columns=["text", "labels", "metadata"])
    test_df = pd.read_parquet(test_path, columns=["text", "labels", "metadata"])

    if language not in {"xx", "all"}:
        # The HuggingFace-style metadata column stores the language; filtering
        # here keeps BERT and TF-IDF trained on the same subset.
        def get_lang(row) -> str:
            m = row.get("metadata") or {}
            return m.get("language", "")

        train_df = train_df[train_df.apply(get_lang, axis=1) == language]
        test_df = test_df[test_df.apply(get_lang, axis=1) == language]

    return train_df, test_df


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the same aggregate metrics reported by the BERT pipeline."""
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
    """Train and persist the sparse one-vs-rest SDG baseline."""
    set_seed(random_state)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_sdgi_parquet(data_dir, language, random_state, test_random_state)
    train_n = len(train_df)
    test_n = len(test_df)

    # Use the same coverage-aware sampling contract as BERT so baseline and
    # neural artifacts compare models rather than different training subsets.
    train_df = coverage_sample(train_df, train_limit, random_state)
    test_seed = random_state if test_random_state is None else test_random_state
    test_df = coverage_sample(test_df, test_limit, test_seed)

    # Preprocess before vectorization so saved vectorizer vocabulary matches the
    # exact normalization used at evaluation time.
    x_train_texts = np.asarray([preprocess_text(t) for t in train_df["text"]], dtype=object)
    x_test_texts = np.asarray([preprocess_text(t) for t in test_df["text"]], dtype=object)

    # MultiLabelBinarizer turns lists such as [3, 13] into the fixed 17-column
    # indicator matrix expected by sklearn metrics and one-vs-rest training.
    binarizer = MultiLabelBinarizer(classes=SDG_IDS)
    y_train = binarizer.fit_transform(train_df["labels"].tolist())
    y_test = binarizer.transform(test_df["labels"].tolist())

    # Binary TF-IDF intentionally behaves closer to a keyword baseline: repeated
    # terms matter less than whether a signal appears in the document at all.
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
            # Very small training subsets can contain only negatives or only
            # positives for a rare SDG. Store a placeholder so column positions
            # stay aligned with the 17-label contract.
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
            # Match the constant class seen during training for degenerate
            # labels instead of dropping the label column entirely.
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

    # Persist both a readable JSON report and the full sklearn payload. The JSON
    # feeds evaluation summaries; the joblib file is needed to reproduce
    # predictions from exactly the trained vectorizer and per-label models.
    output = {
        "run_config": {
            "method": "TF-IDF + linear SVM-style baseline",
            "language": language,
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "source_train_rows": train_n,
            "source_test_rows": test_n,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_limit": train_limit,
            "test_limit": test_limit,
            "sampling": "coverage_aware_by_sdg",
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


# Marker-facing baseline stage orchestration. The functions above implement the
# reusable model logic; the block below adapts it to the repository artifact grid.
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
    """Return True only when metadata and the files it points to all exist."""
    meta_path = artifact_metadata_path("tfidf", train_size, seed)
    if not meta_path.exists():
        return False
    meta = read_json(meta_path)
    model = project_path(meta["paths"]["model"])
    results = project_path(meta["paths"]["results"])
    return model.exists() and results.exists()


def train_one(args: argparse.Namespace, train_size: int, seed: int) -> dict[str, Any]:
    """Train or reuse one TF-IDF artifact for a size/seed condition."""
    out_dir = artifact_dir("tfidf", train_size, seed)
    meta_path = artifact_metadata_path("tfidf", train_size, seed)
    if artifact_complete(train_size, seed) and not args.force:
        # Reusing committed artifacts keeps the default sweep fast and prevents
        # accidental metric drift unless the caller explicitly asks for --force.
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

    # The artifact.json file is the handoff contract consumed by evaluate.py.
    # Keep paths relative so the artifact directory can move with the repo.
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
    """Build the direct-stage CLI used by main.py and standalone runs."""
    parser = argparse.ArgumentParser(description="Train TF-IDF SDG Lens artifacts.")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--train-sizes", nargs="+", type=int, default=TRAIN_SIZES)
    parser.add_argument("--language", choices=["en", "es", "fr", "all"], default="en")
    parser.add_argument("--test-samples", type=int, default=TEST_SAMPLES, help="Number of test samples (default: 1470, full test set).")
    parser.add_argument("--test-seed", type=int, default=TEST_SEED)
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    """Run the requested TF-IDF artifact grid and publish progress status."""
    args = build_parser().parse_args()
    if args.dry_run:
        # Dry runs still call train_one so skip/retrain decisions are visible.
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
            # Status is updated after every artifact so interrupted sweeps can
            # be inspected without reading the whole terminal log.
            write_status("baseline", "running", "tfidf_training", progress={"completed": completed, "total": total})
    write_status("baseline", "completed", "tfidf_training", progress={"completed": completed, "total": total})
    print("[baseline] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
