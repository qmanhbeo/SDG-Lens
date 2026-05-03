# Attribution:
# This coursework project is a simplified partial replication of the SDGi Corpus study.
# It is inspired by the published paper, public dataset, and authors' benchmark repository.
# This file is a re-implementation for coursework purposes and is not the original authors' code.

from __future__ import annotations

import argparse
import json
import random
import re
import string
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

# Ignore convergence warnings during the training of linear models to avoid excessively long terminal outputs.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 17 SDGs' integer label and display name.
SDG_IDS: list[int] = list(range(1, 18))
SDG_NAMES: list[str] = [f"SDG_{i}" for i in SDG_IDS]

# Table2 in articel: BOW-SVM and macro-F1
PAPER_REFERENCE_BOW_SVM_MACRO_F1: dict[tuple[str, str], float] = {
    ("s", "en"): 39.40,
    ("m", "en"): 57.34,
    ("l", "en"): 79.45,
    ("x", "en"): 60.23,
    ("x", "fr"): 47.81,
    ("x", "es"): 55.30,
    ("x", "xx"): 63.85,
}


@dataclass
class ConstantLabelModel:
    """
    When a particular label has only a single value in the training set (all 0s or all 1s),
    it is not possible to train a binary classifier properly, so a ‘constant model’ is used as a fallback.
    """
    constant: int

    def predict(self, n_rows: int) -> np.ndarray:
        return np.full(shape=(n_rows,), fill_value=int(self.constant), dtype=np.int64)


class RobustMultiLabelSGD:
    """
    Train a separate binary SGDClassifier for each SDG label. The loss function uses a hinge, so it can be regarded as a linear SVM-style approach
    If a particular label does not appear in both the positive and negative classes in the training set, it degenerates into a ConstantLabelModel
    """

    def __init__(
        self,
        alpha: float = 1e-4,
        random_state: int = 42,
        validation_fraction: float = 0.2,
    ) -> None:
        self.alpha = alpha
        self.random_state = random_state
        self.validation_fraction = validation_fraction
        self.models_: list[SGDClassifier | ConstantLabelModel] = []
        self.label_stats_: list[dict[str, int]] = []

    def fit(self, x_train: Any, y_train: np.ndarray) -> "RobustMultiLabelSGD":
        """Train a multi-class classifier using label-by-label training."""
        if y_train.ndim != 2:
            raise ValueError("y_train 必须是二维二值标签矩阵。")

        self.models_ = []
        self.label_stats_ = []

        # Train on the 17 SDG labels column by column.
        for col_idx in range(y_train.shape[1]):
            y_col = y_train[:, col_idx]
            unique_values = np.unique(y_col)
            positives = int(np.sum(y_col == 1))
            negatives = int(np.sum(y_col == 0))
            self.label_stats_.append({"positives": positives, "negatives": negatives})

            # If this label does not form two distinct classes (0 or 1) in the training set, use a constant prediction model.
            if unique_values.size < 2:
                self.models_.append(ConstantLabelModel(constant=int(unique_values[0])))
                continue

            clf = SGDClassifier(
                loss="hinge",
                penalty="l2",
                alpha=self.alpha,
                shuffle=True,
                learning_rate="optimal",
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=self.validation_fraction,
                class_weight="balanced",
            )
            clf.fit(x_train, y_col)
            self.models_.append(clf)

        return self

    def predict(self, x_test: Any) -> np.ndarray:
        """Concatenate the predicted results for each label to form a complete multi-label prediction matrix."""
        if not self.models_:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        n_rows = x_test.shape[0]
        predictions: list[np.ndarray] = []
        for model in self.models_:
            if isinstance(model, ConstantLabelModel):
                predictions.append(model.predict(n_rows))
            else:
                predictions.append(model.predict(x_test).astype(np.int64))

        return np.column_stack(predictions)


def set_seed(seed: int) -> None:
    """Fix the random seed to maximise the reproducibility of the results."""
    random.seed(seed)
    np.random.seed(seed)


def replace_numbers(text: str) -> str:
    """
    Change txt to NUM, to let datatype the same.
    """
    return re.sub(r"(\b\d+[\.,]?\d*\b)", "NUM", text)


def get_optional_stopwords() -> set[str]:
    try:
        from spacy.lang.en.stop_words import STOP_WORDS as STOP_WORDS_EN
        from spacy.lang.es.stop_words import STOP_WORDS as STOP_WORDS_ES
        from spacy.lang.fr.stop_words import STOP_WORDS as STOP_WORDS_FR
        return set(STOP_WORDS_EN) | set(STOP_WORDS_FR) | set(STOP_WORDS_ES)
    except Exception:
        return set()


def preprocess_text(
    text: str,
    remove_stopwords: bool = False,
    stopwords: set[str] | None = None,
) -> str:
    """
    text cleaning
    """
    if text is None:
        return ""

    cleaned = str(text).replace("\ufffe", "")
    cleaned = replace_numbers(cleaned)
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.lower().strip()

    if not remove_stopwords:
        return cleaned

    stopwords = stopwords or set()
    tokens: list[str] = []
    for token in cleaned.split():
        if token.isalnum() and 1 < len(token) < 35 and token not in stopwords:
            tokens.append(token)
    return " ".join(tokens)


# Data Retrieval and Filtering
def load_sdgi_dataset(dataset_source: str, from_disk: bool = False) -> DatasetDict:
    """
    Read data from Hugging Face.
    """
    dataset = load_from_disk(dataset_source) if from_disk else load_dataset(dataset_source)
    if not isinstance(dataset, DatasetDict):
        raise TypeError("Expected a DatasetDict with predefined train/test splits.")
    if "train" not in dataset or "test" not in dataset:
        raise ValueError("Dataset must contain 'train' and 'test' splits.")
    return dataset


def _matches_filter(example: dict[str, Any], size: str, language: str) -> bool:
    metadata = example.get("metadata", {}) or {}
    keep_size = size == "x" or metadata.get("size") == size
    keep_language = language == "xx" or metadata.get("language") == language
    return bool(keep_size and keep_language)


def filter_dataset(dataset: DatasetDict, size: str, language: str) -> DatasetDict:
    if size == "x" and language == "xx":
        return dataset

    filtered = DatasetDict()
    for split_name in ("train", "test"):
        filtered[split_name] = dataset[split_name].filter(
            lambda ex: _matches_filter(ex, size=size, language=language)
        )
    return filtered


def dataset_to_arrays(
    dataset: DatasetDict,
    remove_stopwords: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    HF Dataset -> numpy / pandas
    """
    stopwords = get_optional_stopwords() if remove_stopwords else set()

    def clean_split(split: Dataset) -> list[str]:
        return [
            preprocess_text(text=row["text"], remove_stopwords=remove_stopwords, stopwords=stopwords)
            for row in split
        ]

    x_train_texts = np.asarray(clean_split(dataset["train"]), dtype=object)
    x_test_texts = np.asarray(clean_split(dataset["test"]), dtype=object)

    # Convert the multi-label list into a 17-dimensional binary matrix.
    binariser = MultiLabelBinarizer(classes=SDG_IDS)
    y_train = binariser.fit_transform(dataset["train"]["labels"])
    y_test = binariser.transform(dataset["test"]["labels"])

    # Create a test set table to facilitate the subsequent saving of prediction results in CSV format.。
    test_rows: list[dict[str, Any]] = []
    for row in dataset["test"]:
        metadata = row.get("metadata", {}) or {}
        test_rows.append(
            {
                "text": row.get("text", ""),
                "gold_labels": row.get("labels", []),
                "language": metadata.get("language"),
                "size": metadata.get("size"),
                "country": metadata.get("country"),
                "type": metadata.get("type"),
                "year": metadata.get("year"),
            }
        )
    test_df = pd.DataFrame(test_rows)

    return x_train_texts, y_train, x_test_texts, y_test, test_df


# Result
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "subset_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
    }


def labels_from_binary_matrix(y: np.ndarray) -> list[list[int]]:
    return [[idx + 1 for idx, value in enumerate(row.tolist()) if int(value) == 1] for row in y]


# Graph
def save_test_metrics_plot(test_metrics: dict[str, float], output_dir: Path) -> None:
    names = ["f1_micro", "f1_macro", "f1_weighted", "subset_accuracy"]
    values = [test_metrics[name] * 100 for name in names]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values)
    plt.ylabel("Score (%)")
    plt.title("Test Metrics of the Partial Replication")
    plt.ylim(0, max(values) + 10)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.6,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "figure_test_metrics.png", dpi=200)
    plt.close()


def save_macro_comparison_plot(
    test_metrics: dict[str, float],
    size: str,
    language: str,
    output_dir: Path,
) -> None:
    paper_reference = PAPER_REFERENCE_BOW_SVM_MACRO_F1.get((size, language))
    if paper_reference is None:
        return

    your_macro = test_metrics["f1_macro"] * 100
    names = ["Paper BOW-SVM", "My replication"]
    values = [paper_reference, your_macro]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(names, values)
    plt.ylabel("Macro-F1 (%)")
    plt.title(f"Macro-F1 Comparison on sdgi-{size}-{language}")
    plt.ylim(0, max(values) + 10)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.6,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "figure_macro_f1_vs_paper.png", dpi=200)
    plt.close()


def save_label_distribution_plot(
    y_train: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
) -> None:
    train_share = y_train.sum(axis=0) / y_train.shape[0] * 100
    test_share = y_test.sum(axis=0) / y_test.shape[0] * 100

    x = np.arange(len(SDG_IDS))
    width = 0.38

    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, train_share, width=width, label="Train")
    plt.bar(x + width / 2, test_share, width=width, label="Test")
    plt.xticks(x, [str(i) for i in SDG_IDS])
    plt.xlabel("SDG label")
    plt.ylabel("Share of examples with label (%)")
    plt.title("Label Distribution in Train/Test Split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "figure_label_distribution.png", dpi=200)
    plt.close()


# Process
def run_replication(
    dataset_source: str,
    output_dir: Path,
    size: str,
    language: str,
    max_features: int,
    alpha: float,
    remove_stopwords: bool,
    random_state: int,
    from_disk: bool,
) -> dict[str, Any]:
    set_seed(random_state)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) read data
    dataset = load_sdgi_dataset(dataset_source=dataset_source, from_disk=from_disk)
    dataset = filter_dataset(dataset=dataset, size=size, language=language)

    train_n = len(dataset["train"])
    test_n = len(dataset["test"])
    if train_n == 0 or test_n == 0:
        raise ValueError(
            f"过滤后的数据为空（train={train_n}, test={test_n}），请更换 --size 或 --language。"
        )

    # 2) text and lable
    x_train_texts, y_train, x_test_texts, y_test, test_df = dataset_to_arrays(
        dataset=dataset,
        remove_stopwords=remove_stopwords,
    )

    # 3) TF-IDF
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

    # 4) Training a multi-class linear classifier
    model = RobustMultiLabelSGD(
        alpha=alpha,
        random_state=random_state,
        validation_fraction=0.2,
    )
    model.fit(x_train, y_train)

    # 5) Evaluation
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_metrics = evaluate_predictions(y_true=y_train, y_pred=y_train_pred)
    test_metrics = evaluate_predictions(y_true=y_test, y_pred=y_test_pred)

    train_report = classification_report(
        y_true=y_train,
        y_pred=y_train_pred,
        target_names=SDG_NAMES,
        zero_division=0,
    )
    test_report = classification_report(
        y_true=y_test,
        y_pred=y_test_pred,
        target_names=SDG_NAMES,
        zero_division=0,
    )

    # 6) compare difference
    comparison: dict[str, Any] = {}
    paper_reference = PAPER_REFERENCE_BOW_SVM_MACRO_F1.get((size, language))
    if paper_reference is not None:
        comparison = {
            "paper_reference_macro_f1_bow_svm": paper_reference,
            "your_macro_f1": round(test_metrics["f1_macro"] * 100, 2),
            "difference_points": round((test_metrics["f1_macro"] * 100) - paper_reference, 2),
        }

    summary = {
        "metadata": {
            "dataset_source": dataset_source,
            "from_disk": from_disk,
            "size": size,
            "language": language,
            "train_rows": train_n,
            "test_rows": test_n,
            "max_features": max_features,
            "alpha": alpha,
            "remove_stopwords": remove_stopwords,
            "random_state": random_state,
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "paper_comparison": comparison,
        "label_stats": model.label_stats_,
    }

    # 7) save result
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "train_report.txt").write_text(train_report, encoding="utf-8")
    (output_dir / "test_report.txt").write_text(test_report, encoding="utf-8")

    # 8) training set result
    predictions_df = test_df.copy()
    predictions_df["predicted_labels"] = labels_from_binary_matrix(y_test_pred)
    predictions_df["gold_label_count"] = predictions_df["gold_labels"].apply(len)
    predictions_df["predicted_label_count"] = predictions_df["predicted_labels"].apply(len)
    predictions_df.to_csv(output_dir / "test_predictions.csv", index=False)

    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")
    joblib.dump(model, output_dir / "robust_multilabel_sgd.joblib")

    # graph
    save_test_metrics_plot(test_metrics=test_metrics, output_dir=output_dir)
    save_macro_comparison_plot(
        test_metrics=test_metrics,
        size=size,
        language=language,
        output_dir=output_dir,
    )
    save_label_distribution_plot(y_train=y_train, y_test=y_test, output_dir=output_dir)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal one-file replication of the SDGi BOW-SVM supervised baseline."
    )
    parser.add_argument(
        "--dataset_source",
        type=str,
        default="UNDP/sdgi-corpus",
        help="ugging Face dataset id path.",
    )
    parser.add_argument(
        "--from_disk",
        action="store_true",
        help="Load the dataset from a local path created with datasets.save_to_disk().",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="x",
        choices=["s", "m", "l", "x"],
        help="Size subset. 'x' means all sizes.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "fr", "es", "xx"],
        help="Language subset. 'xx' means all supported languages.",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=10000,
        help="Maximum TF-IDF vocabulary size.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-4,
        help="Regularisation strength for the SGD hinge classifier.",
    )
    parser.add_argument(
        "--remove_stopwords",
        action="store_true",
        help="Attempt stop-word removal using spaCy stopword lists if available.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sdgi_outputs",
        help="Where output files should be written.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run_replication(
        dataset_source=args.dataset_source,
        output_dir=Path(args.output_dir),
        size=args.size,
        language=args.language,
        max_features=args.max_features,
        alpha=args.alpha,
        remove_stopwords=args.remove_stopwords,
        random_state=args.random_state,
        from_disk=args.from_disk,
    )


    print("=" * 72)
    print("SDGi minimal partial replication completed successfully")
    print("=" * 72)
    print(json.dumps(summary["metadata"], indent=2, ensure_ascii=False))

    print("\nTest metrics:")
    for key, value in summary["test_metrics"].items():
        print(f"  {key}: {value:.4f}")

    if summary["paper_comparison"]:
        print("\nComparison with paper (BOW SVM, macro-F1):")
        for key, value in summary["paper_comparison"].items():
            print(f"  {key}: {value}")

    print("\nOutput directory：", args.output_dir)
    print("Saved images include:")
    print("  - figure_test_metrics.png")
    print("  - figure_macro_f1_vs_paper.png")
    print("  - figure_label_distribution.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
