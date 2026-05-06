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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Marker-facing SDG Lens pipeline orchestrator.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    sweep = subparsers.add_parser("sweep", help="Run train, baseline, evaluate, visualize, and manuscript compile stages.")
    sweep.add_argument("--device", default="cpu")
    sweep.add_argument("--allow-download", action="store_true")
    sweep.add_argument("--force", action="store_true", help="Retrain model artifacts instead of reusing complete artifacts.")
    sweep.add_argument("--dry-run", action="store_true", help="Print and validate stages without training or writing final outputs.")
    sweep.set_defaults(func=cmd_sweep)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
