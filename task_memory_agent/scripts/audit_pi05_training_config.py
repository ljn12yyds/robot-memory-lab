#!/usr/bin/env python3
"""Audit PI0.5 chunk padding and training/evaluation configuration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-config", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--pi05-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    info = json.loads((args.dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    policy = json.loads(args.checkpoint_config.read_text(encoding="utf-8"))
    train = json.loads(args.train_config.read_text(encoding="utf-8"))
    source = args.pi05_source.read_text(encoding="utf-8")
    tables = [pq.read_table(path, columns=["episode_index"]) for path in sorted((args.dataset_root / "data").glob("**/*.parquet"))]
    frame = pa.concat_tables(tables).to_pandas()
    lengths = frame.groupby("episode_index").size().astype(int).tolist()
    chunk_size = int(policy["chunk_size"])
    padded_steps = sum(
        max(0, chunk_size - (length - frame_index))
        for length in lengths
        for frame_index in range(length)
    )
    total_chunk_steps = sum(lengths) * chunk_size
    samples_with_padding = sum(min(chunk_size - 1, length) for length in lengths)

    eval_freq = train.get("eval_freq")
    dataset_split = info.get("splits", {})
    checks = [
        {
            "name": "pi05_masks_padded_action_loss",
            "status": "PASS" if "action_is_pad" in source else "FAIL",
            "details": {
                "source_mentions_action_is_pad": "action_is_pad" in source,
                "samples_with_padding": samples_with_padding,
                "sample_fraction_with_padding": samples_with_padding / sum(lengths),
                "padded_action_step_fraction": padded_steps / total_chunk_steps,
            },
        },
        {
            "name": "held_out_validation_split",
            "status": "PASS" if any(key != "train" for key in dataset_split) else "FAIL",
            "details": {"splits": dataset_split},
        },
        {
            "name": "periodic_evaluation_enabled",
            "status": "PASS" if eval_freq not in (None, 0) else "FAIL",
            "details": {"eval_freq": eval_freq},
        },
    ]
    payload = {
        "status": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL",
        "chunk_size": chunk_size,
        "dataset_frames": sum(lengths),
        "checks": checks,
        "optimizer": train.get("optimizer"),
        "scheduler": train.get("scheduler"),
        "steps_in_this_run": train.get("steps"),
        "batch_size": train.get("batch_size"),
        "gradient_accumulation_steps": train.get("gradient_accumulation_steps"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
