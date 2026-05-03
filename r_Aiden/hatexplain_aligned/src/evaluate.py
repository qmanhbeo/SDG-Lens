
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from .data import ID_TO_LABEL
from .train import EncodedExample


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _token_scores_to_word_scores(attn: np.ndarray, word_ids: list[int | None], n_words: int) -> list[float]:
    scores = [0.0] * n_words
    counts = [0] * n_words
    for idx, wid in enumerate(word_ids):
        if wid is None or wid >= n_words:
            continue
        scores[wid] += float(attn[idx])
        counts[wid] += 1
    return [scores[i] / counts[i] if counts[i] else 0.0 for i in range(n_words)]


def _topk_mask(scores: list[float], k: int) -> list[int]:
    k = max(1, min(k, len(scores)))
    order = np.argsort(np.asarray(scores))[::-1][:k]
    mask = [0] * len(scores)
    for idx in order:
        mask[int(idx)] = 1
    return mask


def _token_f1(gold: list[int], pred: list[int]) -> float:
    gold_set = {i for i, v in enumerate(gold) if v == 1}
    pred_set = {i for i, v in enumerate(pred) if v == 1}
    if not gold_set and not pred_set:
        return 1.0
    if not gold_set or not pred_set:
        return 0.0
    tp = len(gold_set & pred_set)
    prec = tp / len(pred_set) if pred_set else 0.0
    rec = tp / len(gold_set) if gold_set else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _iou(gold: list[int], pred: list[int]) -> float:
    gold_set = {i for i, v in enumerate(gold) if v == 1}
    pred_set = {i for i, v in enumerate(pred) if v == 1}
    if not gold_set and not pred_set:
        return 1.0
    union = gold_set | pred_set
    if not union:
        return 0.0
    return len(gold_set & pred_set) / len(union)


def _modified_tokens(tokens: list[str], mask: list[int], keep: bool) -> list[str]:
    if keep:
        kept = [tok for tok, m in zip(tokens, mask) if m == 1]
        return kept if kept else tokens[:1]
    removed = [tok for tok, m in zip(tokens, mask) if m == 0]
    return removed if removed else tokens[:1]


def _predict_single(model, tokenizer, device, tokens: list[str], max_length: int) -> tuple[np.ndarray, np.ndarray, list[int | None]]:
    batch = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    with torch.no_grad():
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
    logits = out.logits.detach().cpu().numpy()[0]
    attn = out.attention_scores.detach().cpu().numpy()[0]
    return logits, attn, batch.word_ids(batch_index=0)


def evaluate_all(
    model,
    tokenizer,
    device,
    test_loader,
    max_length: int,
    faithfulness_max_examples: int,
    min_group_size: int,
) -> dict[str, Any]:
    model.eval()
    logits_all = []
    y_true = []
    metadata: list[EncodedExample] = []
    attn_all = []

    for batch in tqdm(test_loader, desc="evaluate", leave=False):
        with torch.no_grad():
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
        logits_all.append(out.logits.detach().cpu().numpy())
        attn_all.append(out.attention_scores.detach().cpu().numpy())
        y_true.extend(batch["labels"].numpy().tolist())
        metadata.extend(batch["meta"])

    logits_np = np.concatenate(logits_all, axis=0)
    attn_np = np.concatenate(attn_all, axis=0)
    probs = _softmax(logits_np, axis=1)
    y_pred = probs.argmax(axis=1)

    cls_metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "auroc_ovr": float(roc_auc_score(np.eye(3)[y_true], probs, multi_class="ovr")),
    }

    token_f1s = []
    ious = []
    comprehensiveness = []
    sufficiency = []
    per_example = []

    for idx, meta in enumerate(metadata):
        if sum(meta.rationale_words) == 0:
            continue
        word_scores = _token_scores_to_word_scores(attn_np[idx], meta.word_ids, len(meta.tokens))
        pred_mask = _topk_mask(word_scores, k=sum(meta.rationale_words))
        tok_f1 = _token_f1(meta.rationale_words, pred_mask)
        iou = _iou(meta.rationale_words, pred_mask)
        token_f1s.append(tok_f1)
        ious.append(iou)

        if len(comprehensiveness) < faithfulness_max_examples:
            orig_prob = float(probs[idx, y_true[idx]])
            removed_tokens = _modified_tokens(meta.tokens, pred_mask, keep=False)
            kept_tokens = _modified_tokens(meta.tokens, pred_mask, keep=True)
            rem_logits, _, _ = _predict_single(model, tokenizer, device, removed_tokens, max_length)
            keep_logits, _, _ = _predict_single(model, tokenizer, device, kept_tokens, max_length)
            rem_prob = float(_softmax(rem_logits[None, :], axis=1)[0, y_true[idx]])
            keep_prob = float(_softmax(keep_logits[None, :], axis=1)[0, y_true[idx]])
            comprehensiveness.append(orig_prob - rem_prob)
            sufficiency.append(orig_prob - keep_prob)

        per_example.append(
            {
                "post_id": meta.post_id,
                "gold_label": ID_TO_LABEL[y_true[idx]],
                "pred_label": ID_TO_LABEL[int(y_pred[idx])],
                "targets": meta.targets,
                "token_f1": tok_f1,
                "iou": iou,
            }
        )

    plausibility = {
        "token_f1": float(np.mean(token_f1s)) if token_f1s else None,
        "iou": float(np.mean(ious)) if ious else None,
    }
    faithfulness = {
        "comprehensiveness": float(np.mean(comprehensiveness)) if comprehensiveness else None,
        "sufficiency": float(np.mean(sufficiency)) if sufficiency else None,
    }

    community_rows = defaultdict(lambda: {"y_true": [], "y_pred": [], "toxic_fp_base": []})
    for yt, yp, meta in zip(y_true, y_pred, metadata):
        toxic_true = 0 if yt == 2 else 1
        toxic_pred = 0 if yp == 2 else 1
        for tgt in meta.targets:
            community_rows[tgt]["y_true"].append(yt)
            community_rows[tgt]["y_pred"].append(yp)
            community_rows[tgt]["toxic_fp_base"].append((toxic_true, toxic_pred))

    subgroup_bias = []
    for target, row in community_rows.items():
        support = len(row["y_true"])
        if support < min_group_size:
            continue
        group_f1 = float(f1_score(row["y_true"], row["y_pred"], average="macro"))
        normal_examples = [pair for pair in row["toxic_fp_base"] if pair[0] == 0]
        if normal_examples:
            fp_rate = sum(int(pred == 1) for _, pred in normal_examples) / len(normal_examples)
        else:
            fp_rate = None
        subgroup_bias.append(
            {
                "target": target,
                "support": support,
                "macro_f1": group_f1,
                "toxic_false_positive_rate_on_normal": fp_rate,
            }
        )
    subgroup_bias = sorted(subgroup_bias, key=lambda x: (-x["support"], x["target"]))

    return {
        "classification": cls_metrics,
        "plausibility": plausibility,
        "faithfulness": faithfulness,
        "subgroup_bias": subgroup_bias,
        "examples_audited": per_example[:50],
    }
