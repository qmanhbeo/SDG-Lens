
from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import ensure_dir, save_json


def write_outputs(config: dict[str, Any], data_meta: dict[str, Any], train_state: dict[str, Any], metrics: dict[str, Any]) -> str:
    out_dir = ensure_dir(config["output_dir"])
    results_path = Path(out_dir) / "results.json"
    payload = {
        "project_name": config["project_name"],
        "mode": config["mode"],
        "seed": config["seed"],
        "model_name": config["model_name"],
        "data": data_meta,
        "training_history": train_state["history"],
        "metrics": metrics,
    }
    save_json(payload, results_path)

    summary_path = Path(out_dir) / "metrics_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Classification\n")
        for k, v in metrics["classification"].items():
            f.write(f"- {k}: {v}\n")
        f.write("\nPlausibility\n")
        for k, v in metrics["plausibility"].items():
            f.write(f"- {k}: {v}\n")
        f.write("\nFaithfulness\n")
        for k, v in metrics["faithfulness"].items():
            f.write(f"- {k}: {v}\n")
    return str(results_path)
