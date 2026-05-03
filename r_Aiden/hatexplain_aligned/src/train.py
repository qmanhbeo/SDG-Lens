
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data import Example
from .model import BertAttentionClassifier
from .utils import pick_device, set_seed


@dataclass
class EncodedExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    label: int
    rationale_targets: torch.Tensor
    word_ids: list[int | None]
    tokens: list[str]
    rationale_words: list[int]
    post_id: str
    targets: list[str]


class HatexplainDataset(Dataset):
    def __init__(self, encoded: list[EncodedExample]):
        self.encoded = encoded

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> EncodedExample:
        return self.encoded[idx]


def encode_examples(
    examples: list[Example],
    tokenizer: Any,
    max_length: int,
) -> list[EncodedExample]:
    encoded_rows: list[EncodedExample] = []
    for ex in examples:
        batch = tokenizer(
            ex.tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        word_ids = batch.word_ids(batch_index=0)
        rt = []
        for wid in word_ids:
            if wid is None or wid >= len(ex.rationale):
                rt.append(0)
            else:
                rt.append(int(ex.rationale[wid]))
        encoded_rows.append(
            EncodedExample(
                input_ids=batch["input_ids"][0],
                attention_mask=batch["attention_mask"][0],
                label=ex.label_id,
                rationale_targets=torch.tensor(rt, dtype=torch.float32),
                word_ids=word_ids,
                tokens=ex.tokens,
                rationale_words=ex.rationale,
                post_id=ex.post_id,
                targets=ex.targets,
            )
        )
    return encoded_rows


def collate_fn(batch: list[EncodedExample]) -> dict[str, Any]:
    return {
        "input_ids": torch.stack([b.input_ids for b in batch], dim=0),
        "attention_mask": torch.stack([b.attention_mask for b in batch], dim=0),
        "labels": torch.tensor([b.label for b in batch], dtype=torch.long),
        "rationale_targets": torch.stack([b.rationale_targets for b in batch], dim=0),
        "meta": batch,
    }


def build_loaders(
    split_map: dict[str, list[Example]],
    model_name: str,
    max_length: int,
    batch_size: int,
    eval_batch_size: int,
) -> tuple[Any, dict[str, DataLoader]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    encoded = {k: encode_examples(v, tokenizer, max_length) for k, v in split_map.items()}
    loaders = {
        "train": DataLoader(HatexplainDataset(encoded["train"]), batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
        "valid": DataLoader(HatexplainDataset(encoded["valid"]), batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn),
        "test": DataLoader(HatexplainDataset(encoded["test"]), batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn),
    }
    return tokenizer, loaders


def train_model(config: dict[str, Any], split_map: dict[str, list[Example]]):
    set_seed(int(config["seed"]))
    device = pick_device(config.get("device", "auto"))
    tokenizer, loaders = build_loaders(
        split_map=split_map,
        model_name=config["model_name"],
        max_length=int(config["max_length"]),
        batch_size=int(config["batch_size"]),
        eval_batch_size=int(config["eval_batch_size"]),
    )
    model = BertAttentionClassifier(
        model_name=config["model_name"],
        num_labels=3,
        att_lambda=float(config["attention_lambda"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    total_steps = max(1, len(loaders["train"]) * int(config["epochs"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * float(config["warmup_ratio"]))),
        num_training_steps=total_steps,
    )

    history = []
    model.train()
    for epoch in range(int(config["epochs"])):
        running_loss = 0.0
        running_cls = 0.0
        running_att = 0.0
        count = 0
        progress = tqdm(loaders["train"], desc=f"train epoch {epoch+1}", leave=False)
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
                rationale_targets=batch["rationale_targets"].to(device),
            )
            out.loss.backward()
            clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
            optimizer.step()
            scheduler.step()

            batch_size = batch["labels"].shape[0]
            running_loss += float(out.loss.detach().cpu()) * batch_size
            running_cls += float((out.cls_loss or torch.tensor(0.0)).detach().cpu()) * batch_size
            running_att += float((out.att_loss or torch.tensor(0.0)).detach().cpu()) * batch_size
            count += batch_size
            progress.set_postfix(loss=running_loss / max(count, 1))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": running_loss / max(count, 1),
                "train_cls_loss": running_cls / max(count, 1),
                "train_att_loss": running_att / max(count, 1),
            }
        )
    return {
        "model": model,
        "tokenizer": tokenizer,
        "loaders": loaders,
        "device": device,
        "history": history,
    }
