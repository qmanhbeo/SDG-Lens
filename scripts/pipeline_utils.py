from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"
JOB_STATUS_DIR = ARTIFACTS_DIR / "job_status"

SEEDS = [42, 43, 44]
TRAIN_SIZES = [2000, 4000]
TEST_SEED = 43
TEST_SAMPLES = 300


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rel_path(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def project_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def artifact_id(model_type: str, train_size: int, seed: int) -> str:
    return f"{model_type}_train{train_size}_seed{seed}"


def artifact_dir(model_type: str, train_size: int, seed: int) -> Path:
    return ARTIFACTS_DIR / artifact_id(model_type, train_size, seed)


def artifact_metadata_path(model_type: str, train_size: int, seed: int) -> Path:
    return artifact_dir(model_type, train_size, seed) / "artifact.json"


def write_status(job: str, status: str, phase: str, **extra: Any) -> None:
    payload = {
        "job": job,
        "status": status,
        "phase": phase,
        "updated_at": now_iso(),
        "last_heartbeat": now_iso(),
        **extra,
    }
    write_json(JOB_STATUS_DIR / f"{job}.json", payload)


def load_artifact_metadata() -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(ARTIFACTS_DIR.glob("*/artifact.json")):
        artifacts.append(read_json(path))
    return artifacts


def required_conditions() -> list[tuple[str, int, int]]:
    return [
        (model_type, train_size, seed)
        for model_type in ("bert", "tfidf")
        for train_size in TRAIN_SIZES
        for seed in SEEDS
    ]


def ensure_base_dirs() -> None:
    for path in (ARTIFACTS_DIR, RESULTS_DIR, OUTPUTS_DIR, MANUSCRIPT_DIR):
        path.mkdir(parents=True, exist_ok=True)
