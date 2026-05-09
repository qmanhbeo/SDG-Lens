"""Public command-line entrypoint for the SDG Lens pipeline.

The individual files under ``scripts/`` remain runnable on their own, but this
module is the stable interface a marker or future maintainer is expected to use.
It launches stages as subprocesses so heavy dependencies such as PyTorch,
transformers, matplotlib, and LaTeX tooling are only imported by the stage that
actually needs them.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Resolve paths from this file rather than from the caller's current directory.
# That keeps commands reproducible whether they are launched from the repo root,
# an IDE, or another shell location.
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# These defaults define the marker-facing sweep. Stage modules have their own
# defaults too, but the orchestrator repeats them so ``python main.py sweep`` is
# explicit about which experimental grid it will run.
TRAIN_SIZES_DEFAULT = [1000, 2000, 4000]
SEEDS_DEFAULT = [42, 43, 44, 45, 46]


def stage_command(script_name: str, args: argparse.Namespace) -> list[str]:
    """Translate top-level CLI options into the subprocess command for a stage."""
    command = [sys.executable, str(SCRIPTS_DIR / script_name)]

    # Dry runs should traverse the same orchestration path as real runs, but
    # stop inside each stage before filesystem-heavy or compute-heavy work.
    if getattr(args, "dry_run", False):
        command.append("--dry-run")

    if script_name == "train.py":
        # Keep every forwarded option spelled out here. It is a little verbose,
        # but it makes the public CLI contract obvious and avoids leaking
        # unrelated options into a stage parser that does not understand them.
        seeds = getattr(args, "seeds", SEEDS_DEFAULT) or []
        if seeds:
            command.extend(["--seeds", *[str(v) for v in seeds]])
        sizes = getattr(args, "train_sizes", TRAIN_SIZES_DEFAULT) or []
        if sizes:
            command.extend(["--train-sizes", *[str(v) for v in sizes]])
        if (v := getattr(args, "language", None)) is not None:
            command.extend(["--language", v])
        if (v := getattr(args, "test_samples", None)) is not None:
            command.extend(["--test-samples", str(v)])
        if (v := getattr(args, "test_seed", None)) is not None:
            command.extend(["--test-seed", str(v)])
        if (v := getattr(args, "epochs", None)) is not None:
            command.extend(["--epochs", str(v)])
        if (v := getattr(args, "batch_size", None)) is not None:
            command.extend(["--batch-size", str(v)])
        if (v := getattr(args, "max_length", None)) is not None:
            command.extend(["--max-length", str(v)])
        if (v := getattr(args, "learning_rate", None)) is not None:
            command.extend(["--learning-rate", str(v)])
        if (v := getattr(args, "threshold", None)) is not None:
            command.extend(["--threshold", str(v)])
        if (v := getattr(args, "examples", None)) is not None:
            command.extend(["--examples", str(v)])
        if (v := getattr(args, "top_tokens", None)) is not None:
            command.extend(["--top-tokens", str(v)])
        if (v := getattr(args, "device", None)) is not None:
            command.extend(["--device", v])
        if (v := getattr(args, "trainable_encoder_layers", None)) is not None:
            command.extend(["--trainable-encoder-layers", str(v)])
        if getattr(args, "unfreeze_encoder", False):
            command.append("--unfreeze-encoder")
        if getattr(args, "allow_download", False):
            command.append("--allow-download")
        if getattr(args, "force", False):
            command.append("--force")
        if (v := getattr(args, "model_name", None)) is not None:
            command.extend(["--model-name", v])

    elif script_name == "baseline.py":
        # The baseline participates in the same seed/size grid as BERT so the
        # evaluation stage can compare conditions one-for-one.
        seeds = getattr(args, "seeds", SEEDS_DEFAULT) or []
        if seeds:
            command.extend(["--seeds", *[str(v) for v in seeds]])
        sizes = getattr(args, "train_sizes", TRAIN_SIZES_DEFAULT) or []
        if sizes:
            command.extend(["--train-sizes", *[str(v) for v in sizes]])
        if (v := getattr(args, "language", None)) is not None:
            command.extend(["--language", v])
        if (v := getattr(args, "test_samples", None)) is not None:
            command.extend(["--test-samples", str(v)])
        if (v := getattr(args, "test_seed", None)) is not None:
            command.extend(["--test-seed", str(v)])
        if (v := getattr(args, "max_features", None)) is not None:
            command.extend(["--max-features", str(v)])
        if (v := getattr(args, "alpha", None)) is not None:
            command.extend(["--alpha", str(v)])
        if getattr(args, "force", False):
            command.append("--force")

    elif script_name == "evaluate.py":
        # Evaluation only needs model-loading controls; it discovers the trained
        # artifacts from artifact metadata written by train.py and baseline.py.
        if (v := getattr(args, "device", None)) is not None:
            command.extend(["--device", v])
        if getattr(args, "allow_download", False):
            command.append("--allow-download")
        if getattr(args, "allow_missing", False):
            command.append("--allow-missing")

    elif script_name == "compile_manuscript.py":
        # The compile stage only needs to know which TeX source to compile; all
        # visualization dependencies are checked by the stage itself.
        if (v := getattr(args, "tex", None)) is not None:
            command.extend(["--tex", v])

    elif script_name == "export_attention_examples.py":
        # Export attention examples: forward device, allow-download, overwrite, limit
        if (v := getattr(args, "device", None)) is not None:
            command.extend(["--device", v])
        if getattr(args, "allow_download", False):
            command.append("--allow-download")
        if getattr(args, "overwrite", False):
            command.append("--overwrite")
        if (v := getattr(args, "limit", None)) is not None:
            command.extend(["--limit", str(v)])

    return command


def run_stage(label: str, script_name: str, args: argparse.Namespace) -> None:
    """Run one pipeline stage and fail fast with a readable stage label."""
    command = stage_command(script_name, args)
    print(f"[main] {label}: {' '.join(command)}", flush=True)
    env = os.environ.copy()

    # Avoid littering the submitted repository with __pycache__ files when the
    # marker-facing pipeline is run from a clean checkout.
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    if result.returncode != 0:
        raise SystemExit(f"[main] {label} failed with exit code {result.returncode}")


def cmd_sweep(args: argparse.Namespace) -> int:
    """Run the reproducible ML pipeline in dependency order."""
    # The ordering reflects the artifact contract: BERT and TF-IDF write
    # metadata, evaluation aggregates metadata, visualization reads evaluation
    # outputs, and export generates full test-set attention examples.
    # Manuscript compilation is optional and can be run separately.
    stages = [
        ("1/5 train neural model", "train.py"),
        ("2/5 train baseline", "baseline.py"),
        ("3/5 evaluate artifacts", "evaluate.py"),
        ("4/5 visualize results", "visualize.py"),
        ("5/5 export attention examples", "export_attention_examples.py"),
    ]
    for label, script_name in stages:
        # Auto-overwrite exports during sweep for idempotent behavior
        if script_name == "export_attention_examples.py":
            args.overwrite = True
        run_stage(label, script_name, args)
    print("[main] sweep complete")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the public parser while keeping stage-specific options isolated."""
    parser = argparse.ArgumentParser(
        description="Marker-facing SDG Lens pipeline orchestrator.",
        epilog="Run `python main.py <command> --help` for command-specific help.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sweep = subparsers.add_parser(
        "sweep",
        help="Run train, baseline, evaluate, visualize, and export attention examples.",
        description="Run train, baseline, evaluate, visualize, and export attention examples (compile is optional, run separately).",
    )
    _add_sweep_args(sweep)

    train_parser = subparsers.add_parser(
        "train",
        help="Train BERT artifacts.",
        description="Run scripts/train.py to produce BERT model artifacts.",
    )
    _add_train_args(train_parser)

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Train TF-IDF artifacts.",
        description="Run scripts/baseline.py to produce TF-IDF model artifacts.",
    )
    _add_baseline_args(baseline_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate trained artifacts.",
        description="Run scripts/evaluate.py to compute per-seed and aggregate metrics.",
    )
    _add_evaluate_args(evaluate_parser)

    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Generate tables and charts.",
        description="Run scripts/visualize.py to produce manuscript-ready outputs.",
    )
    _add_visualize_args(visualize_parser)

    compile_parser = subparsers.add_parser(
        "compile",
        help="Compile manuscript to PDF.",
        description="Run scripts/compile_manuscript.py to build the PDF.",
    )
    _add_compile_args(compile_parser)

    export_parser = subparsers.add_parser(
        "export-examples",
        help="Export full test-set attention examples.",
        description="Run scripts/export_attention_examples.py to export full attention examples.",
    )
    _add_export_args(export_parser)

    sweep.set_defaults(func=cmd_sweep)
    train_parser.set_defaults(
        func=lambda a: (run_stage("train", "train.py", a), 0)[1]
    )
    baseline_parser.set_defaults(
        func=lambda a: (run_stage("baseline", "baseline.py", a), 0)[1]
    )
    evaluate_parser.set_defaults(
        func=lambda a: (run_stage("evaluate", "evaluate.py", a), 0)[1]
    )
    visualize_parser.set_defaults(
        func=lambda a: (run_stage("visualize", "visualize.py", a), 0)[1]
    )
    compile_parser.set_defaults(
        func=lambda a: (run_stage("compile", "compile_manuscript.py", a), 0)[1]
    )
    export_parser.set_defaults(
        func=lambda a: (run_stage("export-examples", "export_attention_examples.py", a), 0)[1]
    )
    return parser


# The argument helpers below intentionally duplicate a few options across
# subcommands. That makes ``python main.py <stage> --help`` useful without
# requiring readers to inspect the underlying stage script.
def _add_sweep_args(p: argparse.ArgumentParser) -> None:
    """Arguments shared by the end-to-end sweep."""
    p.add_argument(
        "--train-sizes", nargs="+", type=int, default=TRAIN_SIZES_DEFAULT,
        help="Training set sizes (default: %(default)s).",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS_DEFAULT,
        help="Random seeds (default: %(default)s).",
    )
    p.add_argument(
        "--language", choices=["en", "es", "fr", "all"], default="en",
        help="Text language filter (default: en).",
    )
    p.add_argument(
        "--test-samples", type=int, default=None,
        help="Number of test samples (default: 1470, full test set).",
    )
    p.add_argument(
        "--test-seed", type=int, default=43,
        help="Fixed test sampling seed (default: 43).",
    )
    p.add_argument(
        "--device", default="cuda",
        help="Compute device for train and evaluate (default: cuda).",
    )
    p.add_argument(
        "--allow-download", action="store_true",
        help="Allow downloading model weights from HuggingFace.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Retrain existing artifacts instead of reusing them.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print stages without executing them.",
    )


def _add_train_args(p: argparse.ArgumentParser) -> None:
    """Arguments forwarded to the BERT training stage."""
    p.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS_DEFAULT,
        help="Random seeds (default: %(default)s).",
    )
    p.add_argument(
        "--train-sizes", nargs="+", type=int, default=TRAIN_SIZES_DEFAULT,
        help="Training set sizes (default: %(default)s).",
    )
    p.add_argument(
        "--language", choices=["en", "es", "fr", "all"], default="en",
        help="Text language filter (default: en).",
    )
    p.add_argument(
        "--test-samples", type=int, default=None,
        help="Number of test samples (default: 1470, full test set).",
    )
    p.add_argument(
        "--test-seed", type=int, default=43,
        help="Fixed test sampling seed (default: 43).",
    )
    p.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs (default: 3).",
    )
    p.add_argument(
        "--batch-size", type=int, default=8,
        help="Training batch size (default: 8).",
    )
    p.add_argument(
        "--max-length", type=int, default=128,
        help="Tokenization max length (default: 128).",
    )
    p.add_argument(
        "--learning-rate", type=float, default=5e-5,
        help="AdamW learning rate (default: 5e-5).",
    )
    p.add_argument(
        "--threshold", type=float, default=0.3,
        help="Multi-label prediction threshold (default: 0.3).",
    )
    p.add_argument(
        "--examples", type=int, default=5,
        help="Number of examples for attention explanation (default: 5).",
    )
    p.add_argument(
        "--top-tokens", type=int, default=12,
        help="Top tokens to show per explanation (default: 12).",
    )
    p.add_argument(
        "--device", default="cuda",
        help="Compute device (default: cuda).",
    )
    p.add_argument(
        "--trainable-encoder-layers", type=int, default=2,
        help="Number of trainable encoder layers (default: 2).",
    )
    p.add_argument(
        "--unfreeze-encoder", action="store_true",
        help="Unfreeze the full encoder (default: last 2 layers only).",
    )
    p.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model name (default: all-MiniLM-L6-v2).",
    )
    p.add_argument(
        "--allow-download", action="store_true",
        help="Allow downloading model weights from HuggingFace.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Retrain existing artifacts instead of reusing them.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print stages without executing them.",
    )


def _add_baseline_args(p: argparse.ArgumentParser) -> None:
    """Arguments forwarded to the TF-IDF baseline stage."""
    p.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS_DEFAULT,
        help="Random seeds (default: %(default)s).",
    )
    p.add_argument(
        "--train-sizes", nargs="+", type=int, default=TRAIN_SIZES_DEFAULT,
        help="Training set sizes (default: %(default)s).",
    )
    p.add_argument(
        "--language", choices=["en", "es", "fr", "all"], default="en",
        help="Text language filter (default: en).",
    )
    p.add_argument(
        "--test-samples", type=int, default=None,
        help="Number of test samples (default: 1470, full test set).",
    )
    p.add_argument(
        "--test-seed", type=int, default=43,
        help="Fixed test sampling seed (default: 43).",
    )
    p.add_argument(
        "--max-features", type=int, default=10000,
        help="TF-IDF max features (default: 10000).",
    )
    p.add_argument(
        "--alpha", type=float, default=1e-4,
        help="SGDClassifier regularization alpha (default: 1e-4).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Retrain existing artifacts instead of reusing them.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print stages without executing them.",
    )


def _add_evaluate_args(p: argparse.ArgumentParser) -> None:
    """Arguments forwarded to artifact evaluation."""
    p.add_argument(
        "--device", default="cuda",
        help="Compute device (default: cuda).",
    )
    p.add_argument(
        "--allow-download", action="store_true",
        help="Allow downloading model weights from HuggingFace.",
    )
    p.add_argument(
        "--allow-missing", action="store_true",
        help="Allow missing artifacts instead of failing.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print stages without executing them.",
    )


def _add_visualize_args(p: argparse.ArgumentParser) -> None:
    """Arguments forwarded to visualization generation."""
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print stages without executing them.",
    )


def _add_export_args(p: argparse.ArgumentParser) -> None:
    """Arguments forwarded to attention example export."""
    p.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "bert_train4000_seed42",
        help="BERT artifact directory (default: artifacts/bert_train4000_seed42).",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Explicit run directory (default: auto-detect latest).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "examples_results",
        help="Output directory for exported examples.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples (default: all 1057 English test examples).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda).",
    )
    p.add_argument(
        "--top-tokens",
        type=int,
        default=12,
        help="Number of top attended tokens to export (default: 12).",
    )
    p.add_argument(
        "--top-labels",
        type=int,
        default=5,
        help="Number of top label scores to export (default: 5).",
    )
    p.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow downloading HuggingFace model files if not cached.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    p.add_argument(
        "--test-seed",
        type=int,
        default=43,
        help="Test set sampling seed (default: 43).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print stages without executing them.",
    )


def _add_compile_args(p: argparse.ArgumentParser) -> None:
    """Arguments forwarded to LaTeX compilation."""
    p.add_argument(
        "--tex",
        default="sdg_lens_manuscript.tex",
        help="LaTeX source filename (default: sdg_lens_manuscript.tex).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print stages without executing them.",
    )


def main() -> int:
    """Parse the requested command and dispatch to its stage runner."""
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
