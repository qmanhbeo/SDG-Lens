# Minimal SDGi explainability prototype, upgraded for better signal:
# - train on a larger default subset for a few epochs
# - fine-tune only the last transformer layers instead of the whole encoder
# - use a lower multi-label prediction threshold
# Still simplified: no scheduler, no config files, no checkpointing, and
# attention remains a proxy explanation without SDGi rationale ground truth.

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shlex
import sys
import time as time_module
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer


SDG_IDS = list(range(1, 18))
SDG_NAMES = {idx: f"SDG_{idx}" for idx in SDG_IDS}
LABEL_NAMES = [SDG_NAMES[idx] for idx in SDG_IDS]
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

TRAIN_LIMIT = 2000
TEST_LIMIT = None
DEFAULT_EPOCHS = 3
DEFAULT_THRESHOLD = 0.3
TRAINABLE_ENCODER_LAYERS = 2


@dataclass
class ModelOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    attention_scores: torch.Tensor | None


class SDGiDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.texts = [clean_text(text) for text in frame["text"].tolist()]
        self.labels = np.stack([labels_to_vector(labels) for labels in frame["labels"]])

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "text": self.texts[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class BertMultiLabelAttentionClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int = 17,
        local_files_only: bool = True,
        trainable_encoder_layers: int = TRAINABLE_ENCODER_LAYERS,
        unfreeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            output_attentions=True,
        )
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"
        try:
            self.encoder = AutoModel.from_pretrained(
                model_name,
                config=config,
                local_files_only=local_files_only,
                attn_implementation="eager",
            )
        except TypeError:
            self.encoder = AutoModel.from_pretrained(
                model_name,
                config=config,
                local_files_only=local_files_only,
            )
        self.trainable_encoder_layers = int(trainable_encoder_layers)
        self.unfreeze_encoder = bool(unfreeze_encoder)
        self.encoder_trainable_mode = self._set_encoder_trainability()

        hidden = self.encoder.config.hidden_size
        dropout_prob = getattr(self.encoder.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden, num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def _set_encoder_trainability(self) -> str:
        if self.unfreeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = True
            return "all_layers"

        for param in self.encoder.parameters():
            param.requires_grad = False

        if self.trainable_encoder_layers <= 0:
            return "frozen"

        layers = getattr(getattr(self.encoder, "encoder", None), "layer", None)
        if layers is None:
            layers = getattr(getattr(self.encoder, "transformer", None), "layer", None)
        if layers is None:
            print("Could not find transformer layers; encoder stays frozen.")
            return "frozen"

        # Keep most of BERT frozen and update only the last N transformer blocks.
        # For BERT-base with 12 layers and N=2, this corresponds to layers 10 and 11.
        n_layers = min(self.trainable_encoder_layers, len(layers))
        for layer in layers[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        return f"last_{n_layers}_layers"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> ModelOutput:
        encoder_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_attentions": return_attention,
            "return_dict": True,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        if not any(param.requires_grad for param in self.encoder.parameters()):
            with torch.no_grad():
                enc = self.encoder(**encoder_kwargs)
        else:
            enc = self.encoder(**encoder_kwargs)

        pooled = enc.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(pooled))
        loss = self.loss_fn(logits, labels.float()) if labels is not None else None

        attention_scores = None
        if return_attention:
            if not enc.attentions:
                raise RuntimeError("Encoder did not return attentions.")
            last_layer_att = enc.attentions[-1]  # [batch, heads, seq, seq]
            cls_attention = last_layer_att[:, :, 0, :].mean(dim=1)
            cls_attention = cls_attention * attention_mask.float()
            cls_attention = cls_attention / (cls_attention.sum(dim=1, keepdim=True) + 1e-12)
            attention_scores = cls_attention

        return ModelOutput(loss=loss, logits=logits, attention_scores=attention_scores)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clean_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\ufffe", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def has_sdgi_split(data_dir: Path) -> bool:
    return (
        (data_dir / "train-00000-of-00001.parquet").exists()
        and (data_dir / "test-00000-of-00001.parquet").exists()
    )


def is_inside_project(path: Path) -> bool:
    try:
        path.resolve().relative_to(HERE)
    except ValueError:
        return False
    return True


def project_metadata_path(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(HERE))
    except ValueError:
        return str(path)


def project_io_path(path: Path | str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return HERE / path


def checkpoint_data_dir(saved_data_dir: Any) -> Path:
    """Keep copied checkpoints from reading the parent course repository."""
    if not saved_data_dir:
        return DEFAULT_DATA_DIR

    path = Path(saved_data_dir)
    if not path.is_absolute():
        project_relative = (HERE / path).resolve()
        if has_sdgi_split(project_relative):
            return project_relative
        return path

    if is_inside_project(path):
        return path

    if has_sdgi_split(DEFAULT_DATA_DIR):
        print(
            "Checkpoint data_dir points outside this standalone copy; "
            f"using local data at {DEFAULT_DATA_DIR}"
        )
        return DEFAULT_DATA_DIR

    return path


def compact_text(text: str, limit: int = 700) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def labels_to_vector(labels: Any) -> np.ndarray:
    vector = np.zeros(len(SDG_IDS), dtype=np.float32)
    for raw_label in labels:
        label = int(raw_label)
        if label < 1 or label > 17:
            raise ValueError(f"Unexpected SDG label {label}; expected 1..17.")
        vector[label - 1] = 1.0
    return vector


def labels_to_list(row: np.ndarray) -> list[int]:
    return [idx + 1 for idx, value in enumerate(row.tolist()) if int(value) == 1]


def coverage_sample(frame: pd.DataFrame, n_rows: int | None, seed: int) -> pd.DataFrame:
    if n_rows is None or n_rows <= 0 or n_rows >= len(frame):
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    rng = random.Random(seed)
    label_sets = {
        idx: {int(label) for label in labels}
        for idx, labels in zip(frame.index.tolist(), frame["labels"].tolist())
    }
    selected: list[int] = []
    seen: set[int] = set()

    for sdg_id in SDG_IDS:
        candidates = [idx for idx, labels in label_sets.items() if sdg_id in labels]
        if not candidates:
            continue
        choice = rng.choice(candidates)
        if choice not in seen:
            selected.append(choice)
            seen.add(choice)

    remaining = [idx for idx in frame.index.tolist() if idx not in seen]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, n_rows - len(selected))])
    selected = selected[:n_rows]
    rng.shuffle(selected)
    return frame.loc[selected].reset_index(drop=True)


def load_sdgi_split(data_dir: Path, split: str, language: str, max_samples: int, seed: int) -> pd.DataFrame:
    path = data_dir / f"{split}-00000-of-00001.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing SDGi parquet: {path}")

    frame = pd.read_parquet(path, columns=["text", "labels", "metadata"])
    if language != "all":
        frame = frame[
            frame["metadata"].map(lambda meta: (meta or {}).get("language") == language)
        ].copy()

    if frame.empty:
        raise ValueError(f"No {split} rows after language filter {language!r}.")

    return coverage_sample(frame, max_samples, seed)


def make_collate_fn(tokenizer: Any, max_length: int):
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [row["text"] for row in batch]
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded["labels"] = torch.stack([row["labels"] for row in batch], dim=0)
        encoded["texts"] = texts
        return encoded

    return collate_fn


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    keys = ["input_ids", "attention_mask", "token_type_ids", "labels"]
    return {key: batch[key].to(device) for key in keys if key in batch}


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def train_model(
    model: BertMultiLabelAttentionClassifier,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> list[dict[str, float]]:
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found.")
    optimizer = torch.optim.AdamW(params, lr=learning_rate)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for step, batch in enumerate(loader, start=1):
            batch_on_device = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(
                input_ids=batch_on_device["input_ids"],
                attention_mask=batch_on_device["attention_mask"],
                token_type_ids=batch_on_device.get("token_type_ids"),
                labels=batch_on_device["labels"],
                return_attention=False,
            )
            if out.loss is None:
                raise RuntimeError("Training loss was not computed.")
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            batch_size = batch_on_device["labels"].shape[0]
            total_loss += float(out.loss.detach().cpu()) * batch_size
            total_rows += batch_size
        epoch_loss = total_loss / max(total_rows, 1)
        print(f"Epoch {epoch + 1} - Loss: {epoch_loss:.4f}")
        history.append({"epoch": float(epoch + 1), "train_loss": epoch_loss})
    return history


def predictions_from_probs(probs: np.ndarray, threshold: float) -> np.ndarray:
    # Multi-label classifiers often need a lower threshold than 0.5 because
    # each label is predicted independently and positives are sparse.
    preds = (probs > threshold).astype(np.int64)
    empty_rows = np.where(preds.sum(axis=1) == 0)[0]
    if len(empty_rows):
        top_labels = probs[empty_rows].argmax(axis=1)
        preds[empty_rows, top_labels] = 1
    return preds


def evaluate_model(
    model: BertMultiLabelAttentionClassifier,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch_on_device = move_batch_to_device(batch, device)
            out = model(
                input_ids=batch_on_device["input_ids"],
                attention_mask=batch_on_device["attention_mask"],
                token_type_ids=batch_on_device.get("token_type_ids"),
                return_attention=False,
            )
            logits_all.append(out.logits.detach().cpu().numpy())
            labels_all.append(batch_on_device["labels"].detach().cpu().numpy())

    logits = np.concatenate(logits_all, axis=0)
    y_true = np.concatenate(labels_all, axis=0).astype(np.int64)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_pred = predictions_from_probs(probs, threshold)
    per_label = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics = {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "subset_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "per_label_f1": {
            str(label): float(score)
            for label, score in zip(SDG_IDS, per_label.tolist())
        },
        "average_true_labels": float(y_true.sum(axis=1).mean()),
        "average_predicted_labels": float(y_pred.sum(axis=1).mean()),
    }
    return metrics, y_true, y_pred, probs


def explain_examples(
    model: BertMultiLabelAttentionClassifier,
    tokenizer: Any,
    frame: pd.DataFrame,
    device: torch.device,
    max_length: int,
    threshold: float,
    n_examples: int,
    top_k: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    model.eval()

    for _, row in frame.head(n_examples).iterrows():
        text = clean_text(row["text"])
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        model_inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }

        with torch.no_grad():
            out = model(**model_inputs, return_attention=True)

        if out.attention_scores is None:
            raise RuntimeError("No attention scores returned for explanation.")

        logits = out.logits.detach().cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        pred = predictions_from_probs(probs, threshold)[0]
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].tolist())
        attention = out.attention_scores[0].detach().cpu().numpy()
        special_mask = encoded["special_tokens_mask"][0].numpy().tolist()
        input_mask = encoded["attention_mask"][0].numpy().tolist()

        scored_tokens = []
        for token, score, is_special, is_real_token in zip(tokens, attention, special_mask, input_mask):
            if is_special or not is_real_token:
                continue
            scored_tokens.append({"token": token, "score": float(score)})

        top_tokens = sorted(scored_tokens, key=lambda item: item["score"], reverse=True)[:top_k]
        top_tokens = [
            {"token": item["token"], "score": round(item["score"], 6)}
            for item in top_tokens
        ]

        top_probs = probs[0].argsort()[::-1][:5]
        examples.append(
            {
                "text": compact_text(text),
                "gold_labels": [int(label) for label in row["labels"]],
                "predicted_labels": labels_to_list(pred),
                "top_label_scores": [
                    {"label": int(idx + 1), "score": round(float(probs[0, idx]), 4)}
                    for idx in top_probs
                ],
                "top_tokens": top_tokens,
            }
        )

    return examples


def package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def write_json_local(payload: dict[str, Any] | list[Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def make_run_dir(output_dir: Path, timestamp: str) -> tuple[str, Path]:
    runs_root = output_dir / "runs"
    run_id = timestamp
    run_dir = runs_root / run_id
    suffix = 1
    while run_dir.exists():
        run_id = f"{timestamp}_{suffix:02d}"
        run_dir = runs_root / run_id
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def build_run_config(
    args: argparse.Namespace,
    *,
    mode: str,
    run_id: str,
    timestamp: str,
    run_dir: Path,
    device: torch.device,
    train_rows: int,
    test_rows: int,
    encoder_mode: str,
    trainable_params: int,
    checkpoint_loaded_from: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "mode": mode,
        "command": shlex.join([Path(sys.executable).name, *sys.argv]),
        "model_name": args.model_name,
        "num_labels": len(SDG_IDS),
        "label_names": LABEL_NAMES,
        "data_dir": project_metadata_path(args.data_dir),
        "dataset_source": project_metadata_path(args.data_dir),
        "language": args.language,
        "train_limit": int(args.train_samples),
        "test_limit": int(args.test_samples),
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "max_length": int(args.max_length),
        "learning_rate": float(args.learning_rate),
        "threshold": float(args.threshold),
        "unfrozen_layers": int(args.trainable_encoder_layers),
        "trainable_encoder_layers": int(args.trainable_encoder_layers),
        "unfreeze_encoder": bool(args.unfreeze_encoder),
        "encoder_trainable_mode": encoder_mode,
        "trainable_params": int(trainable_params),
        "seed": int(args.seed),
        "test_seed": int(args.test_seed),
        "device": str(device),
        "run_dir": project_metadata_path(run_dir),
        "checkpoint_loaded_from": checkpoint_loaded_from,
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "transformers_version": package_version("transformers"),
        "sklearn_version": package_version("scikit-learn"),
        "attention_note": "Last-layer CLS attention averaged over heads; proxy explanation only.",
    }


def checkpoint_payload(
    model: BertMultiLabelAttentionClassifier,
    args: argparse.Namespace,
    run_config: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "model_name": args.model_name,
        "num_labels": len(SDG_IDS),
        "label_names": LABEL_NAMES,
        "threshold": float(args.threshold),
        "train_limit": int(args.train_samples),
        "test_limit": int(args.test_samples),
        "epochs": int(args.epochs),
        "unfrozen_layers": int(args.trainable_encoder_layers),
        "trainable_encoder_layers": int(args.trainable_encoder_layers),
        "unfreeze_encoder": bool(args.unfreeze_encoder),
        "encoder_trainable_mode": model.encoder_trainable_mode,
        "seed": int(args.seed),
        "test_seed": int(args.test_seed),
        "language": args.language,
        "data_dir": run_config["data_dir"],
        "batch_size": int(args.batch_size),
        "max_length": int(args.max_length),
        "learning_rate": float(args.learning_rate),
        "run_config": run_config,
        "metrics": metrics,
    }


def write_run_readme(
    run_dir: Path,
    run_config: dict[str, Any],
    metrics: dict[str, Any],
    checkpoint_name: str = "model.pt",
) -> None:
    checkpoint_path = project_metadata_path(run_dir / checkpoint_name)
    text = f"""# SDGi Lens Run {run_config['run_id']}

This run {run_config['mode']} a minimal SDGi multi-label classifier and saved an
attention-based explanation proxy. Attention is not ground-truth rationale.

## Reload

```bash
python main.py eval --checkpoint {checkpoint_path}
```

## Metrics

- micro-F1: {metrics.get('micro_f1')}
- macro-F1: {metrics.get('macro_f1')}
- weighted-F1: {metrics.get('weighted_f1')}
- subset accuracy: {metrics.get('subset_accuracy')}
- average predicted labels: {metrics.get('average_predicted_labels')}

## Config

- model: {run_config['model_name']}
- data: {run_config['dataset_source']}
- train/test rows: {run_config['train_rows']} / {run_config['test_rows']}
- epochs: {run_config['epochs']}
- threshold: {run_config['threshold']}
- encoder mode: {run_config['encoder_trainable_mode']}
- command: `{run_config['command']}`

## Limitations

Prototype only: no scheduler, no checkpoint resume, no class weighting, and no
SDGi token-level rationale labels for explanation validation.
"""
    (run_dir / "README_run.md").write_text(text, encoding="utf-8")


def append_runs_index(output_dir: Path, run_config: dict[str, Any], metrics: dict[str, Any]) -> None:
    index_path = output_dir / "runs_index.csv"
    columns = [
        "run_id",
        "timestamp",
        "model_name",
        "train_limit",
        "test_limit",
        "epochs",
        "threshold",
        "unfrozen_layers",
        "seed",
        "micro_f1",
        "macro_f1",
        "average_predicted_labels",
        "run_dir",
    ]
    row = {
        "run_id": run_config["run_id"],
        "timestamp": run_config["timestamp"],
        "model_name": run_config["model_name"],
        "train_limit": run_config["train_limit"],
        "test_limit": run_config["test_limit"],
        "epochs": run_config["epochs"],
        "threshold": run_config["threshold"],
        "unfrozen_layers": run_config["unfrozen_layers"],
        "seed": run_config["seed"],
        "micro_f1": metrics.get("micro_f1"),
        "macro_f1": metrics.get("macro_f1"),
        "average_predicted_labels": metrics.get("average_predicted_labels"),
        "run_dir": run_config["run_dir"],
    }
    write_header = not index_path.exists()
    with index_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_experiment(
    model: BertMultiLabelAttentionClassifier,
    args: argparse.Namespace,
    output_dir: Path,
    run_config: dict[str, Any],
    metrics: dict[str, Any],
    examples: list[dict[str, Any]],
) -> Path:
    run_dir = project_io_path(run_config["run_dir"])
    combined = {
        "run_config": run_config,
        "metrics": metrics,
        "examples": examples,
        # Backward-compatible alias for earlier prototype outputs.
        "metadata": run_config,
    }

    torch.save(checkpoint_payload(model, args, run_config, metrics), run_dir / "model.pt")
    write_json_local(run_config, run_dir / "run_config.json")
    write_json_local(metrics, run_dir / "metrics.json")
    write_json_local(examples, run_dir / "examples.json")
    write_json_local(combined, run_dir / "results.json")
    write_json_local(combined, output_dir / "results.json")
    write_run_readme(run_dir, run_config, metrics)
    append_runs_index(output_dir, run_config, metrics)
    return run_dir


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    # This loader is for checkpoints produced by this local script. PyTorch
    # 2.6+ defaults to weights_only=True, which rejects some version metadata.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint {path} has no model_state_dict.")
    return checkpoint


def build_model_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal SDGi BERT attention prototype.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=HERE / "outputs")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--language", choices=["en", "es", "fr", "all"], default="en")
    parser.add_argument("--train_samples", type=int, default=TRAIN_LIMIT)
    parser.add_argument("--test_samples", type=int, default=TEST_LIMIT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--top_tokens", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--unfreeze_encoder", action="store_true")
    parser.add_argument("--trainable_encoder_layers", type=int, default=TRAINABLE_ENCODER_LAYERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_seed", type=int, default=None, help="Fixed test sampling seed; defaults to seed + 1.")
    parser.add_argument("--load", type=Path, default=None, help="Load a saved model.pt checkpoint.")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate a loaded checkpoint without training.")
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    if args.test_seed is None:
        args.test_seed = args.seed + 1
    device = pick_device(args.device)
    local_files_only = not args.allow_download
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.load is not None:
        if not args.eval_only:
            raise ValueError("--load currently requires --eval-only.")

        print(f"Loading checkpoint: {args.load}")
        checkpoint = load_checkpoint(args.load, device)
        saved_config = checkpoint.get("run_config", {})

        args.model_name = str(checkpoint.get("model_name", saved_config.get("model_name", args.model_name)))
        args.threshold = float(checkpoint.get("threshold", saved_config.get("threshold", args.threshold)))
        args.train_samples = int(checkpoint.get("train_limit", saved_config.get("train_limit", args.train_samples)))
        args.test_samples = int(checkpoint.get("test_limit", saved_config.get("test_limit", args.test_samples)))
        args.epochs = int(checkpoint.get("epochs", saved_config.get("epochs", args.epochs)))
        args.trainable_encoder_layers = int(
            checkpoint.get(
                "trainable_encoder_layers",
                checkpoint.get("unfrozen_layers", saved_config.get("unfrozen_layers", args.trainable_encoder_layers)),
            )
        )
        args.unfreeze_encoder = bool(checkpoint.get("unfreeze_encoder", saved_config.get("unfreeze_encoder", args.unfreeze_encoder)))
        args.language = str(checkpoint.get("language", saved_config.get("language", args.language)))
        args.data_dir = checkpoint_data_dir(
            checkpoint.get("data_dir", saved_config.get("data_dir", args.data_dir))
        )
        args.batch_size = int(checkpoint.get("batch_size", saved_config.get("batch_size", args.batch_size)))
        args.max_length = int(checkpoint.get("max_length", saved_config.get("max_length", args.max_length)))
        args.learning_rate = float(checkpoint.get("learning_rate", saved_config.get("learning_rate", args.learning_rate)))
        args.seed = int(checkpoint.get("seed", saved_config.get("seed", args.seed)))
        args.test_seed = int(checkpoint.get("test_seed", saved_config.get("test_seed", args.seed + 1)))
        set_seed(args.seed)

        print("Loading SDGi test parquet data...")
        test_frame = load_sdgi_split(args.data_dir, "test", args.language, args.test_samples, args.test_seed)
        print(f"test rows={len(test_frame)} language={args.language}")

        print(f"Rebuilding tokenizer/model: {args.model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name,
                use_fast=True,
                local_files_only=local_files_only,
            )
            model = BertMultiLabelAttentionClassifier(
                args.model_name,
                num_labels=int(checkpoint.get("num_labels", len(SDG_IDS))),
                local_files_only=local_files_only,
                trainable_encoder_layers=args.trainable_encoder_layers,
                unfreeze_encoder=args.unfreeze_encoder,
            ).to(device)
        except OSError as exc:
            raise RuntimeError(
                "Could not rebuild the model/tokenizer from the local HuggingFace cache. "
                "Use --allow_download if network access is available."
            ) from exc

        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(
            f"device={device} encoder_mode={model.encoder_trainable_mode} "
            f"trainable_params={trainable_params:,} max_length={args.max_length}"
        )

        collate_fn = make_collate_fn(tokenizer, args.max_length)
        test_loader = DataLoader(
            SDGiDataset(test_frame),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        print("Evaluating loaded checkpoint...")
        metrics, _, _, _ = evaluate_model(model, test_loader, device, args.threshold)
        metrics.update(
            {
                "last_train_loss": checkpoint.get("metrics", {}).get("last_train_loss"),
                "threshold": float(args.threshold),
                "empty_prediction_fallback": "top1",
            }
        )
        print(f"Final micro-F1={metrics['micro_f1']:.4f} macro-F1={metrics['macro_f1']:.4f}")

        print("Extracting attention explanations...")
        examples = explain_examples(
            model=model,
            tokenizer=tokenizer,
            frame=test_frame,
            device=device,
            max_length=args.max_length,
            threshold=args.threshold,
            n_examples=args.examples,
            top_k=args.top_tokens,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id, run_dir = make_run_dir(args.output_dir, timestamp)
        run_config = build_run_config(
            args,
            mode="eval_only",
            run_id=run_id,
            timestamp=timestamp,
            run_dir=run_dir,
            device=device,
            train_rows=int(saved_config.get("train_rows", args.train_samples)),
            test_rows=len(test_frame),
            encoder_mode=model.encoder_trainable_mode,
            trainable_params=trainable_params,
            checkpoint_loaded_from=str(args.load),
        )
        saved_run_dir = save_experiment(model, args, args.output_dir, run_config, metrics, examples)

        print("Done.")
        print(f"saved run {saved_run_dir}")
        print(f"latest summary {args.output_dir / 'results.json'}")
        return 0

    print("Loading SDGi parquet data...")
    train_frame = load_sdgi_split(args.data_dir, "train", args.language, args.train_samples, args.seed)
    test_frame = load_sdgi_split(args.data_dir, "test", args.language, args.test_samples, args.test_seed)
    print(f"train rows={len(train_frame)} test rows={len(test_frame)} language={args.language}")

    print(f"Loading tokenizer/model: {args.model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            use_fast=True,
            local_files_only=local_files_only,
        )
        model = BertMultiLabelAttentionClassifier(
            args.model_name,
            num_labels=17,
            local_files_only=local_files_only,
            trainable_encoder_layers=args.trainable_encoder_layers,
            unfreeze_encoder=args.unfreeze_encoder,
        ).to(device)
    except OSError as exc:
        raise RuntimeError(
            "Could not load the model/tokenizer from the local HuggingFace cache. "
            "Use --allow_download if network access is available, or pass a cached --model_name."
        ) from exc

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(
        f"device={device} encoder_mode={model.encoder_trainable_mode} "
        f"trainable_params={trainable_params:,} max_length={args.max_length}"
    )
    collate_fn = make_collate_fn(tokenizer, args.max_length)
    train_loader = DataLoader(
        SDGiDataset(train_frame),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        SDGiDataset(test_frame),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print("Training...")
    t0 = time_module.perf_counter()
    history = train_model(model, train_loader, device, args.epochs, args.learning_rate)
    training_time_seconds = time_module.perf_counter() - t0
    training_time_minutes = training_time_seconds / 60.0
    timing = {"training_time_seconds": training_time_seconds, "training_time_minutes": training_time_minutes}

    print("Evaluating...")
    metrics, _, _, _ = evaluate_model(model, test_loader, device, args.threshold)
    metrics.update(
        {
            "last_train_loss": float(history[-1]["train_loss"]) if history else None,
            "threshold": float(args.threshold),
            "empty_prediction_fallback": "top1",
            "training_time_seconds": training_time_seconds,
            "training_time_minutes": training_time_minutes,
        }
    )
    print(f"Final micro-F1={metrics['micro_f1']:.4f} macro-F1={metrics['macro_f1']:.4f}")

    print("Extracting attention explanations...")
    examples = explain_examples(
        model=model,
        tokenizer=tokenizer,
        frame=test_frame,
        device=device,
        max_length=args.max_length,
        threshold=args.threshold,
        n_examples=args.examples,
        top_k=args.top_tokens,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id, run_dir = make_run_dir(args.output_dir, timestamp)
    run_config = build_run_config(
        args,
        mode="train_eval",
        run_id=run_id,
        timestamp=timestamp,
        run_dir=run_dir,
        device=device,
        train_rows=len(train_frame),
        test_rows=len(test_frame),
        encoder_mode=model.encoder_trainable_mode,
        trainable_params=trainable_params,
    )
    run_config["timing"] = timing
    saved_run_dir = save_experiment(model, args, args.output_dir, run_config, metrics, examples)

    print("Done.")
    print(f"micro_f1={metrics['micro_f1']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    print(f"saved run {saved_run_dir}")
    print(f"latest summary {args.output_dir / 'results.json'}")
    for idx, example in enumerate(examples, start=1):
        tokens = ", ".join(item["token"] for item in example["top_tokens"][:8])
        print(f"example {idx}: pred={example['predicted_labels']} top_tokens=[{tokens}]")

    return 0


def model_cli_main() -> int:
    return run_from_args(build_model_parser().parse_args())


# Marker-facing training stage orchestration.
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
    bert_args_list = [
        "--data_dir", str(DATA_DIR),
        "--output_dir", str(out_dir),
        "--model_name", args.model_name,
        "--language", args.language,
        "--train_samples", str(train_size),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--max_length", str(args.max_length),
        "--learning_rate", str(args.learning_rate),
        "--threshold", str(args.threshold),
        "--examples", str(args.examples),
        "--top_tokens", str(args.top_tokens),
        "--device", args.device,
        "--trainable_encoder_layers", str(args.trainable_encoder_layers),
        "--seed", str(seed),
        "--test_seed", str(args.test_seed),
    ]
    if args.test_samples is not None:
        bert_args_list.extend(["--test_samples", str(args.test_samples)])
    if args.allow_download:
        bert_args_list.append("--allow_download")
    if args.unfreeze_encoder:
        bert_args_list.append("--unfreeze_encoder")
    bert_args = build_model_parser().parse_args(bert_args_list)

    run_from_args(bert_args)
    results_path = out_dir / "results.json"
    results = read_json(results_path)
    timing = (
        results.get("timing")
        or results.get("run_config", {}).get("timing")
        or {
            "training_time_seconds": results.get("metrics", {}).get("training_time_seconds"),
            "training_time_minutes": results.get("metrics", {}).get("training_time_minutes"),
        }
    )
    if timing.get("training_time_seconds") is None:
        print("[train] warning: no training_time_seconds found in results.json — artifact may be from an older run")
    run_config = results.get("run_config", {})
    run_dir = project_path(run_config["run_dir"])
    checkpoint = run_dir / "model.pt"

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
        "timing": timing,
    }
    write_json(meta_path, meta)
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train BERT SDG Lens artifacts.")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--train-sizes", nargs="+", type=int, default=TRAIN_SIZES)
    parser.add_argument("--language", choices=["en", "es", "fr", "all"], default="en")
    parser.add_argument("--test-samples", type=int, default=None, help="Number of test samples; omit or pass 0 for full test set (default: full test set).")
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
    parser.add_argument("--device", default="cuda")
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
