"""Export full BERT test-set prediction + attention-token examples.

This script reloads an existing trained BERT artifact, runs inference over the full
English test split, extracts prediction + attention-token examples, and exports them
in marker-friendly formats.

No retraining - only inference on existing checkpoint weights.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoTokenizer

# Relative imports from train.py - reusing existing logic without triggering training
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train import (
    BertMultiLabelAttentionClassifier,
    clean_text,
    compact_text,
    labels_to_list,
    load_sdgi_split,
    predictions_from_probs,
    SDG_IDS,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "examples_results"
DEFAULT_TEST_SEED = 43


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a local checkpoint and validate that model weights are present."""
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint {path} has no model_state_dict.")
    return checkpoint


def find_latest_run(artifact_dir: Path) -> Path:
    """Find the most recent run directory under the artifact directory."""
    runs_dir = artifact_dir / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"No runs directory found in {artifact_dir}")

    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {runs_dir}")

    # Sort by name (timestamp) descending - most recent first
    latest = sorted(run_dirs, key=lambda d: d.name, reverse=True)[0]
    return latest


def export_full_examples(
    model: BertMultiLabelAttentionClassifier,
    tokenizer: Any,
    frame: pd.DataFrame,
    device: torch.device,
    max_length: int,
    threshold: float,
    top_k_tokens: int,
    top_k_labels: int,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Extract full test-set examples with label scores and attention tokens."""
    examples: list[dict[str, Any]] = []
    model.eval()

    # Apply limit if specified, otherwise process all rows
    if limit is not None and limit > 0:
        frame_to_process = frame.head(limit)
    else:
        frame_to_process = frame

    total = len(frame_to_process)
    print(f"Processing {total} examples...")

    for idx, (_, row) in enumerate(frame_to_process.iterrows()):
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  [{idx + 1}/{total}]")

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
        for token, score, is_special, is_real_token in zip(
            tokens, attention, special_mask, input_mask
        ):
            if is_special or not is_real_token:
                continue
            scored_tokens.append({"token": token, "score": float(score)})

        top_tokens = sorted(scored_tokens, key=lambda item: item["score"], reverse=True)[
            :top_k_tokens
        ]
        top_tokens = [
            {"token": item["token"], "score": round(item["score"], 6)}
            for item in top_tokens
        ]

        top_probs = probs[0].argsort()[::-1][:top_k_labels]
        examples.append(
            {
                "example_index": idx,
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


def examples_to_jsonl(examples: list[dict[str, Any]], output_path: Path) -> None:
    """Write examples as JSONL (one JSON object per line)."""
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def examples_to_csv(examples: list[dict[str, Any]], output_path: Path) -> None:
    """Write examples as CSV with separate columns for each prediction/token/label."""
    if not examples:
        pd.DataFrame().to_csv(output_path, index=False)
        return

    # Determine max counts from the first example (all should be consistent)
    max_pred_labels = 17  # max possible SDGs
    max_top_tokens = max(len(ex.get("top_tokens", [])) for ex in examples)
    max_top_labels = max(len(ex.get("top_label_scores", [])) for ex in examples)

    rows = []
    for ex in examples:
        row = {
            "example_index": ex["example_index"],
            "text": ex["text"][:500],
            "gold_labels": ",".join(map(str, ex["gold_labels"])),
        }

        # Predicted labels: up to 17 separate columns
        pred_labels = ex.get("predicted_labels", [])
        for i in range(1, max_pred_labels + 1):
            row[f"pred_label_{i}"] = pred_labels[i - 1] if i <= len(pred_labels) else ""

        # Top label scores: separate label and score columns
        top_labels = ex.get("top_label_scores", [])
        for i in range(1, max_top_labels + 1):
            if i <= len(top_labels):
                row[f"top_label_{i}"] = top_labels[i - 1]["label"]
                row[f"top_label_{i}_score"] = top_labels[i - 1]["score"]
            else:
                row[f"top_label_{i}"] = ""
                row[f"top_label_{i}_score"] = ""

        # Top tokens: separate token and score columns
        top_tokens = ex.get("top_tokens", [])
        for i in range(1, max_top_tokens + 1):
            if i <= len(top_tokens):
                row[f"top_token_{i}"] = top_tokens[i - 1]["token"]
                row[f"top_token_{i}_score"] = top_tokens[i - 1]["score"]
            else:
                row[f"top_token_{i}"] = ""
                row[f"top_token_{i}_score"] = ""

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL, escapechar="\\")


def examples_to_markdown(examples: list[dict[str, Any]], output_path: Path) -> None:
    """Write examples as readable Markdown."""
    lines = []

    lines.append("# Full Test-Set Attention Examples\n")
    lines.append(f"**Total examples:** {len(examples)}\n")
    lines.append(
        "*Note: Attention tokens are proxy evidence, not validated causal explanations.*\n"
    )
    lines.append("---")

    for ex in examples:
        lines.append(f"\n## Example {ex['example_index'] + 1}\n")

        lines.append("### Text\n")
        lines.append(f"```\n{ex['text']}\n```\n")

        lines.append("### Labels\n")
        lines.append(f"- **Gold labels:** {ex['gold_labels']}")
        lines.append(f"- **Predicted labels:** {ex['predicted_labels']}\n")

        lines.append("### Top Label Scores\n")
        lines.append("| SDG | Score |")
        lines.append("|-----|-------|")
        for item in ex["top_label_scores"]:
            lines.append(f"| {item['label']} | {item['score']} |")
        lines.append("")

        lines.append("### Top Attended Tokens\n")
        lines.append("| Token | Attention Score |")
        lines.append("|-------|-----------------|")
        for item in ex["top_tokens"]:
            lines.append(f"| {item['token']} | {item['score']} |")
        lines.append("")

        lines.append("---")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export full BERT test-set prediction + attention-token examples"
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "bert_train4000_seed42",
        help="BERT artifact directory (default: artifacts/bert_train4000_seed42)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Explicit run directory (default: auto-detect latest)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for exported examples",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples (default: all English test examples)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--top-tokens",
        type=int,
        default=12,
        help="Number of top attended tokens to export (default: 12)",
    )
    parser.add_argument(
        "--top-labels",
        type=int,
        default=5,
        help="Number of top label scores to export (default: 5)",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow downloading HuggingFace model files if not cached",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        default=DEFAULT_TEST_SEED,
        help=f"Test set sampling seed (default: {DEFAULT_TEST_SEED})",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Max token length for model input (default: 128)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Prediction threshold (default: 0.3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without running export.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.dry_run:
        print(f"[export] Would export attention examples from: {args.artifact_dir}")
        print(f"[export] Run dir: auto-detect latest")
        print(f"[export] Output dir: {args.output_dir}")
        print(f"[export] Limit: {args.limit if args.limit else 'all (1057) English test examples'}")
        print(f"[export] Device: {args.device}")
        print(f"[export] Top tokens: {args.top_tokens}")
        print(f"[export] Top labels: {args.top_labels}")
        print("[export] dry run complete")
        return 0

    # Resolve run directory
    if args.run_dir is not None:
        run_dir = args.run_dir
    else:
        run_dir = find_latest_run(args.artifact_dir)

    checkpoint_path = run_dir / "model.pt"
    if not checkpoint_path.exists():
        # Check if checkpoint is at top level of artifact dir (older structure)
        checkpoint_path = args.artifact_dir / "model.pt"
        if not checkpoint_path.exists():
            print(f"ERROR: No model.pt found in {run_dir} or {args.artifact_dir}")
            return 1

    # Load run config for metadata
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.exists():
        run_config_path = args.artifact_dir / "run_config.json"

    run_config = {}
    if run_config_path.exists():
        run_config = json.loads(run_config_path.read_text())

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, device)
    saved_config = checkpoint.get("run_config", {})

    # Extract model config
    model_name = saved_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    max_length = saved_config.get("max_length", args.max_length)
    threshold = saved_config.get("threshold", args.threshold)
    num_labels = saved_config.get("num_labels", 17)

    print(f"Model: {model_name}")
    print(f"Max length: {max_length}")
    print(f"Threshold: {threshold}")

    # Load tokenizer
    print("Loading tokenizer...")
    local_files_only = not args.allow_download
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only
    )

    # Build model and load weights
    print("Building model...")
    model = BertMultiLabelAttentionClassifier(
        model_name=model_name,
        num_labels=num_labels,
        local_files_only=local_files_only,
        trainable_encoder_layers=saved_config.get("trainable_encoder_layers", 2),
        unfreeze_encoder=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    print("Model loaded and ready for inference.")

    # Load test split (English only, same as training/evaluation)
    print("Loading test split...")
    data_dir = saved_config.get("data_dir", DEFAULT_DATA_DIR)
    language = saved_config.get("language", "en")
    test_samples = saved_config.get("test_limit", 1470)

    # Load full test split - pass a large number to get all English test rows
    # Then limit to 1057 (the English-filtered test set)
    test_frame = load_sdgi_split(
        Path(data_dir), "test", language=language, max_samples=2000, seed=args.test_seed
    )
    print(f"Loaded {len(test_frame)} English test examples")

    # Export full examples
    print(f"Exporting attention examples (limit={args.limit}, top_tokens={args.top_tokens}, top_labels={args.top_labels})...")
    examples = export_full_examples(
        model=model,
        tokenizer=tokenizer,
        frame=test_frame,
        device=device,
        max_length=max_length,
        threshold=threshold,
        top_k_tokens=args.top_tokens,
        top_k_labels=args.top_labels,
        limit=args.limit,
    )

    # Prepare output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing outputs
    if not args.overwrite:
        for fname in ["full_examples.jsonl", "full_examples.csv", "full_examples.md"]:
            if (args.output_dir / fname).exists():
                print(f"ERROR: Output {args.output_dir / fname} already exists. Use --overwrite to replace.")
                return 1

    # Write outputs
    print(f"Writing outputs to {args.output_dir}...")
    examples_to_jsonl(examples, args.output_dir / "full_examples.jsonl")
    examples_to_csv(examples, args.output_dir / "full_examples.csv")
    examples_to_markdown(examples, args.output_dir / "full_examples.md")

    # Summary
    print("\n" + "=" * 50)
    print("EXPORT SUMMARY")
    print("=" * 50)
    print(f"Artifact: {args.artifact_dir.name}")
    print(f"Run: {run_dir.name}")
    print(f"Model: {model_name}")
    print(f"Test seed: {args.test_seed}")
    print(f"Examples exported: {len(examples)}")
    print(f"Output files:")
    print(f"  - {args.output_dir / 'full_examples.jsonl'}")
    print(f"  - {args.output_dir / 'full_examples.csv'}")
    print(f"  - {args.output_dir / 'full_examples.md'}")
    print("=" * 50)
    print("NOTE: Attention tokens are proxy evidence, not validated causal explanations.")
    print("SDGi provides SDG labels but no token-level rationale ground truth.")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    import numpy as np
    sys.exit(main())