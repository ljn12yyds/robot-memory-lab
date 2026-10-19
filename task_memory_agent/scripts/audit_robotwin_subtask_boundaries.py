#!/usr/bin/env python3
"""Replay representative demonstrations and verify language-labeled stage boundaries."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROBOTWIN_ROOT = Path(os.environ.get("ROBOTWIN_ROOT", "/home/user/ljn/RoboTwin")).resolve()
AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOTWIN_ROOT))
sys.path.insert(0, str(AGENT_ROOT))
sys.path.insert(0, str(AGENT_ROOT / "scripts"))

from envs.stack_blocks_three import stack_blocks_three
from run_robotwin_grounded_stack import _load_robotwin_args, live_stack_predicates


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


def expected_predicate(instruction: str) -> tuple[str, str]:
    base = re.fullmatch(
        r"pick up the (red|green|blue) block and place it at the center as the tower base",
        instruction,
    )
    if base:
        return f"{base.group(1)}_block", "centered_base"
    stacked = re.fullmatch(
        r"pick up the (red|green|blue) block and place it stably on top of the (red|green|blue) block",
        instruction,
    )
    if stacked:
        return f"{stacked.group(1)}_block", f"on_{stacked.group(2)}"
    raise ValueError(instruction)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--sim-output", type=Path, required=True)
    parser.add_argument("--config", default="memory_pi05_permutations30")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tables = [pq.read_table(path) for path in sorted((args.dataset_root / "data").glob("**/*.parquet"))]
    frame = pa.concat_tables(tables).to_pandas()
    tasks = pq.read_table(args.dataset_root / "meta/tasks.parquet").to_pandas()
    task_names = list(tasks.index)
    results: list[dict] = []

    for raw_episode in range(args.episodes):
        with h5py.File(args.raw_data / f"episode_{raw_episode:07d}.hdf5", "r") as source:
            actions = joint_vector(source["action"])
        env = stack_blocks_three()
        cursor = 0
        try:
            episode_output = args.sim_output / f"episode_{raw_episode:02d}"
            episode_output.mkdir(parents=True, exist_ok=True)
            env.setup_demo(
                now_ep_num=raw_episode,
                seed=raw_episode + 1,
                **_load_robotwin_args(args.config, episode_output),
            )
            env.last_gripper = None
            env.last_actor = None
            for stage in range(3):
                converted_episode = raw_episode * 3 + stage
                rows = frame[frame["episode_index"] == converted_episode]
                length = len(rows)
                task_id = int(rows["task_index"].iloc[0])
                instruction = str(task_names[task_id])
                for action in actions[cursor : cursor + length]:
                    env.take_action(action, action_type="qpos")
                cursor += length
                predicates = live_stack_predicates(env)
                key, value = expected_predicate(instruction)
                results.append(
                    {
                        "raw_episode": raw_episode,
                        "seed": raw_episode + 1,
                        "stage": stage,
                        "instruction": instruction,
                        "segment_length": length,
                        "expected": {key: value},
                        "observed": predicates,
                        "passed": predicates.get(key) == value,
                    }
                )
            if cursor != len(actions):
                raise ValueError(f"episode {raw_episode}: replayed {cursor}, actions {len(actions)}")
        finally:
            env.close_env()

    payload = {
        "status": "PASS" if all(item["passed"] for item in results) else "FAIL",
        "boundaries_checked": len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
