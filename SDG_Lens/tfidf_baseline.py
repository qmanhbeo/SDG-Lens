from __future__ import annotations

import argparse
import json
import random
import re
import string
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
DEFAULT_DATA_DIR = HERE / "data"
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
    x_train = vectorizer.fit_transform(x_train_texts)
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
        },
        output_dir / "tfidf_model.joblib",
    )

    print(f"TF-IDF baseline: micro_f1={test_metrics['micro_f1']:.4f}, macro_f1={test_metrics['macro_f1']:.4f}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TF-IDF baseline for SDGi")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--max_features", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_random_state", type=int, default=None)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--test_limit", type=int, default=None)
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    output_dir = args.output_dir or (HERE / "outputs" / "tfidf_baseline")
    run_tfidf_baseline(
        data_dir=args.data_dir,
        output_dir=output_dir,
        language=args.language,
        max_features=args.max_features,
        alpha=args.alpha,
        random_state=args.random_state,
        test_random_state=args.test_random_state,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
    )
    return 0


def main() -> int:
    return run_from_args(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit("Use `python main.py baseline`.")
