
from __future__ import annotations

import argparse
from pathlib import Path

from .data import ensure_public_data, load_examples, load_split_ids, split_examples
from .evaluate import evaluate_all
from .export_results import write_outputs
from .train import train_model
from .utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command HateXplain reproduction.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_meta = ensure_public_data(config["data_dir"])
    examples = load_examples(data_meta["dataset_path"])
    split_ids = load_split_ids(data_meta["split_path"])
    split_map = split_examples(
        examples,
        split_ids,
        subset_train=config.get("subset_train"),
        subset_valid=config.get("subset_valid"),
        subset_test=config.get("subset_test"),
    )

    train_state = train_model(config, split_map)
    metrics = evaluate_all(
        model=train_state["model"],
        tokenizer=train_state["tokenizer"],
        device=train_state["device"],
        test_loader=train_state["loaders"]["test"],
        max_length=int(config["max_length"]),
        faithfulness_max_examples=int(config["faithfulness_max_examples"]),
        min_group_size=int(config["min_group_size"]),
    )
    result_path = write_outputs(config, data_meta, train_state, metrics)
    print(f"Finished successfully. Results written to {result_path}")


if __name__ == "__main__":
    main()
