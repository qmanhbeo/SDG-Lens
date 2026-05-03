
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .utils import ensure_dir, sha256_file

DATASET_URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
SPLITS_URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/post_id_divisions.json"


@dataclass
class Example:
    post_id: str
    tokens: list[str]
    label: str
    label_id: int
    targets: list[str]
    rationale: list[int]


LABEL_TO_ID = {"hatespeech": 0, "offensive": 1, "normal": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def ensure_public_data(data_dir: str | Path) -> dict[str, Any]:
    raw_dir = ensure_dir(data_dir)
    dataset_path = raw_dir / "dataset.json"
    split_path = raw_dir / "post_id_divisions.json"
    _download(DATASET_URL, dataset_path)
    _download(SPLITS_URL, split_path)
    return {
        "dataset_path": str(dataset_path),
        "split_path": str(split_path),
        "dataset_sha256": sha256_file(dataset_path),
        "split_sha256": sha256_file(split_path),
    }


def _majority_label(annotators: list[dict[str, Any]]) -> str | None:
    labels = [a["label"].strip().lower() for a in annotators if "label" in a]
    counts = Counter(labels).most_common()
    if not counts:
        return None
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None
    return counts[0][0]


def _majority_targets(annotators: list[dict[str, Any]], majority_label: str) -> list[str]:
    selected: list[str] = []
    for ann in annotators:
        if ann.get("label", "").strip().lower() == majority_label:
            for tgt in ann.get("target", []):
                if tgt and tgt not in selected:
                    selected.append(str(tgt))
    return selected


def _majority_rationale(rationales: list[list[int]]) -> list[int]:
    if not rationales:
        return []
    length = len(rationales[0])
    out = []
    for i in range(length):
        votes = 0
        for row in rationales:
            if i < len(row) and int(row[i]) == 1:
                votes += 1
        out.append(1 if votes >= 2 else 0)
    return out


def load_examples(dataset_path: str | Path) -> list[Example]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples: list[Example] = []
    for post_id, row in data.items():
        annotators = row.get("annotators", [])
        majority_label = _majority_label(annotators)
        if majority_label is None:
            continue
        if majority_label not in LABEL_TO_ID:
            continue
        tokens = row.get("post_tokens", [])
        rationale = _majority_rationale(row.get("rationales", []))
        if rationale and len(rationale) != len(tokens):
            rationale = rationale[: len(tokens)]
            if len(rationale) < len(tokens):
                rationale = rationale + [0] * (len(tokens) - len(rationale))
        if not rationale:
            rationale = [0] * len(tokens)
        targets = _majority_targets(annotators, majority_label)
        examples.append(
            Example(
                post_id=post_id,
                tokens=[str(t) for t in tokens],
                label=majority_label,
                label_id=LABEL_TO_ID[majority_label],
                targets=targets,
                rationale=[int(x) for x in rationale],
            )
        )
    return examples


def load_split_ids(split_path: str | Path) -> dict[str, list[str]]:
    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    aliases = {
        "train": ["train", "training"],
        "valid": ["valid", "val", "validation", "dev"],
        "test": ["test"],
    }
    out: dict[str, list[str]] = {}
    for dest, keys in aliases.items():
        found = None
        for key in keys:
            if key in splits:
                found = splits[key]
                break
        if found is None:
            raise KeyError(f"Could not find split key for {dest} in {split_path}")
        out[dest] = [str(x) for x in found]
    return out


def split_examples(
    examples: list[Example],
    split_ids: dict[str, list[str]],
    subset_train: int | None = None,
    subset_valid: int | None = None,
    subset_test: int | None = None,
) -> dict[str, list[Example]]:
    by_id = {ex.post_id: ex for ex in examples}
    split_map = {}
    for split_name, ids in split_ids.items():
        rows = [by_id[i] for i in ids if i in by_id]
        split_map[split_name] = rows

    if subset_train is not None:
        split_map["train"] = split_map["train"][:subset_train]
    if subset_valid is not None:
        split_map["valid"] = split_map["valid"][:subset_valid]
    if subset_test is not None:
        split_map["test"] = split_map["test"][:subset_test]

    return split_map
