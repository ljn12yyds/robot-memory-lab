#!/usr/bin/env python3
"""Compare raw RoboTwin actions with converted LeRobot actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq


def joint_vector(group: h5py.Group) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(group["left_arm_joint_states"], dtype=np.float32),
            np.asarray(group["left_ee_joint_states"], dtype=np.float32),
            np.asarray(group["right_arm_joint_states"], dtype=np.float32),
            np.asarray(group["right_ee_joint_states"], dtype=np.float32),
        ],
        axis=1,
    )


def summarize(values: np.ndarray) -> dict:
    unique, counts = np.unique(np.round(values, 6), return_counts=True)
    largest = np.argsort(counts)[-8:][::-1]
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "below_0_5": int(np.sum(values < 0.5)),
        "top_values": [
            {"value": float(unique[index]), "count": int(counts[index])} for index in largest
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--stats", type=Path)
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as source:
        raw = joint_vector(source["action"])
    table = pq.read_table(args.parquet, columns=["episode_index", "frame_index", "action"])
    episode_indices = np.asarray(table["episode_index"].to_numpy())
    all_converted = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    converted = all_converted[episode_indices == 0]

    expected = raw[: len(converted)]
    delta = np.abs(converted - expected)
    payload = {
        "raw_shape": list(raw.shape),
        "converted_episode0_shape": list(converted.shape),
        "comparison_definition": "converted action[t] versus raw action[t]",
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
        "raw_right_gripper": summarize(expected[:, 13]),
        "converted_right_gripper": summarize(converted[:, 13]),
        "global_empirical_grippers": {
            side: {
                **summarize(all_converted[:, column]),
                "q01_q10_q50_q90_q99": np.quantile(
                    all_converted[:, column], [0.01, 0.10, 0.50, 0.90, 0.99]
                ).astype(float).tolist(),
            }
            for side, column in (("left", 6), ("right", 13))
        },
        "first_mismatch": None,
    }
    empirical = np.quantile(all_converted, [0.01, 0.99], axis=0)
    payload["global_empirical_action_q01"] = empirical[0].astype(float).tolist()
    payload["global_empirical_action_q99"] = empirical[1].astype(float).tolist()
    if args.stats:
        stats = json.loads(args.stats.read_text(encoding="utf-8"))["action"]
        reported = np.asarray([stats["q01"], stats["q99"]], dtype=np.float64)
        payload["reported_minus_empirical_quantiles"] = {
            "q01": (reported[0] - empirical[0]).astype(float).tolist(),
            "q99": (reported[1] - empirical[1]).astype(float).tolist(),
            "max_abs_error": float(np.abs(reported - empirical).max()),
        }
    mismatches = np.argwhere(delta > 1e-6)
    if len(mismatches):
        row, column = (int(value) for value in mismatches[0])
        payload["first_mismatch"] = {
            "row": row,
            "column": column,
            "raw": float(expected[row, column]),
            "converted": float(converted[row, column]),
        }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
