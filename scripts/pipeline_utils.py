"""Shared filesystem and metadata helpers for the SDG Lens stages.

The training, baseline, evaluation, visualization, and manuscript scripts all
need to agree on artifact names, output directories, and progress-status files.
Keeping those conventions here prevents subtle drift between independently
runnable stage scripts.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


# All project paths are anchored at the repository root, not at the directory
# from which a script happens to be launched.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directory ownership is intentionally centralized so every stage writes into
# the same submitted artifact layout documented in README.md.
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"
JOB_STATUS_DIR = ARTIFACTS_DIR / "job_status"

# This smaller default grid is used by stage scripts when run directly. The
# public sweep in main.py can pass a wider grid without changing these helpers.
SEEDS = [42, 43, 44]
TRAIN_SIZES = [2000, 4000]
TEST_SEED = 43
TEST_SAMPLES = 1470


def now_iso() -> str:
    """Return a compact wall-clock timestamp for human-readable metadata."""
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically so interrupted stages do not leave half-written files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    # Replace only after the temporary file is complete. This keeps status and
    # artifact metadata readable even if a long-running job is interrupted.
    tmp_path.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object from disk."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rel_path(path: Path | str) -> str:
    """Store paths relative to the repo when possible for portable metadata."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def project_path(path: Path | str) -> Path:
    """Resolve artifact metadata paths back to absolute local paths."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def artifact_id(model_type: str, train_size: int, seed: int) -> str:
    """Return the stable identifier used by metadata, folders, and logs."""
    return f"{model_type}_train{train_size}_seed{seed}"


def artifact_dir(model_type: str, train_size: int, seed: int) -> Path:
    """Return the directory that owns one model/size/seed artifact."""
    return ARTIFACTS_DIR / artifact_id(model_type, train_size, seed)


def artifact_metadata_path(model_type: str, train_size: int, seed: int) -> Path:
    """Return the canonical metadata file for one trained artifact."""
    return artifact_dir(model_type, train_size, seed) / "artifact.json"


def write_status(job: str, status: str, phase: str, **extra: Any) -> None:
    """Publish a lightweight heartbeat for long or multi-stage commands."""
    payload = {
        "job": job,
        "status": status,
        "phase": phase,
        "updated_at": now_iso(),
        "last_heartbeat": now_iso(),
        **extra,
    }
    # Each stage owns exactly one status file, which makes it easy to inspect
    # progress without scraping stdout from a long-running command.
    write_json(JOB_STATUS_DIR / f"{job}.json", payload)


def load_artifact_metadata() -> list[dict[str, Any]]:
    """Load every artifact metadata file discovered under artifacts/."""
    artifacts: list[dict[str, Any]] = []
    for path in sorted(ARTIFACTS_DIR.glob("*/artifact.json")):
        artifacts.append(read_json(path))
    return artifacts


def required_conditions() -> list[tuple[str, int, int]]:
    """Return the model/size/seed grid expected by direct stage defaults."""
    return [
        (model_type, train_size, seed)
        for model_type in ("bert", "tfidf")
        for train_size in TRAIN_SIZES
        for seed in SEEDS
    ]


def ensure_base_dirs() -> None:
    """Create the top-level output directories expected by every stage."""
    for path in (ARTIFACTS_DIR, RESULTS_DIR, OUTPUTS_DIR, MANUSCRIPT_DIR):
        path.mkdir(parents=True, exist_ok=True)
