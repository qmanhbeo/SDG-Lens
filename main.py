from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def stage_command(script_name: str, args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(SCRIPTS_DIR / script_name)]
    if args.dry_run:
        command.append("--dry-run")
    if args.force and script_name in {"train.py", "baseline.py"}:
        command.append("--force")
    if args.device and script_name in {"train.py", "evaluate.py"}:
        command.extend(["--device", args.device])
    if args.allow_download and script_name in {"train.py", "evaluate.py"}:
        command.append("--allow-download")
    return command


def run_stage(label: str, script_name: str, args: argparse.Namespace) -> None:
    command = stage_command(script_name, args)
    print(f"[main] {label}: {' '.join(command)}", flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    if result.returncode != 0:
        raise SystemExit(f"[main] {label} failed with exit code {result.returncode}")


def cmd_sweep(args: argparse.Namespace) -> int:
    stages = [
        ("1/5 train neural model", "train.py"),
        ("2/5 train baseline", "baseline.py"),
        ("3/5 evaluate artifacts", "evaluate.py"),
        ("4/5 visualize results", "visualize.py"),
        ("5/5 compile manuscript", "compile_manuscript.py"),
    ]
    for label, script_name in stages:
        run_stage(label, script_name, args)
    print("[main] sweep complete")
    return 0


def add_shared_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--device", default="cuda")
    subparser.add_argument("--allow-download", action="store_true")
    subparser.add_argument("--force", action="store_true")
    subparser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Marker-facing SDG Lens pipeline orchestrator.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    sweep = subparsers.add_parser("sweep", help="Run all stages in order.")
    add_shared_args(sweep)
    sweep.set_defaults(func=cmd_sweep)

    for label, script_name, help_text in [
        ("train", "train.py", "Train BERT artifacts."),
        ("baseline", "baseline.py", "Train TF-IDF artifacts."),
        ("evaluate", "evaluate.py", "Evaluate artifacts."),
        ("visualize", "visualize.py", "Generate tables and charts."),
        ("compile", "compile_manuscript.py", "Compile manuscript to PDF."),
    ]:
        sub = subparsers.add_parser(label, help=help_text)
        add_shared_args(sub)
        sub.set_defaults(func=lambda a, l=label, s=script_name: (run_stage(l, s, a), 0)[1])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
