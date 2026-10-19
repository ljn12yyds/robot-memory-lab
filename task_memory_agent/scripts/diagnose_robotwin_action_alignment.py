#!/usr/bin/env python3
"""Replay recorded RoboTwin qpos actions and verify state/action alignment."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np


ROBOTWIN_ROOT = Path(os.environ.get("ROBOTWIN_ROOT", "/home/user/ljn/RoboTwin")).resolve()
AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOTWIN_ROOT))
sys.path.insert(0, str(AGENT_ROOT))
sys.path.insert(0, str(AGENT_ROOT / "scripts"))

from envs.stack_blocks_three import stack_blocks_three
from run_robotwin_grounded_stack import _load_robotwin_args, live_stack_predicates


def _joint_vector(group: h5py.Group) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(group["left_arm_joint_states"], dtype=np.float32),
            np.asarray(group["left_ee_joint_states"], dtype=np.float32),
            np.asarray(group["right_arm_joint_states"], dtype=np.float32),
            np.asarray(group["right_ee_joint_states"], dtype=np.float32),
        ],
        axis=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--config", default="memory_pi05_permutations30")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=160)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as dataset:
        states = _joint_vector(dataset["state"])
        actions = _joint_vector(dataset["action"])

    if not (0 <= args.start < args.end <= len(actions)):
        raise ValueError((args.start, args.end, len(actions)))

    temporal_error = np.abs(actions[:-1] - states[1:])
    selected = actions[args.start : args.end]
    grippers = selected[:, [6, 13]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sim_output = args.output.parent / "sim"
    sim_output.mkdir(parents=True, exist_ok=True)
    env = stack_blocks_three()
    tracking_errors: list[float] = []
    red_positions: list[list[float]] = []
    try:
        env.setup_demo(
            now_ep_num=args.episode_index,
            seed=args.seed,
            **_load_robotwin_args(args.config, sim_output),
        )
        env.last_gripper = None
        env.last_actor = None
        initial_observation = env.get_obs()
        initial_qpos = np.asarray(initial_observation["joint_action"]["vector"], dtype=np.float32)
        initial_state_error = np.abs(initial_qpos - states[args.start])
        initial_predicates = live_stack_predicates(env)

        for action in selected:
            env.take_action(action, action_type="qpos")
            current = np.asarray(env.get_obs()["joint_action"]["vector"], dtype=np.float32)
            tracking_errors.append(float(np.max(np.abs(current - action))))
            red_positions.append(np.asarray(env.block1.get_pose().p).astype(float).tolist())

        final_predicates = live_stack_predicates(env)
        red_start = np.asarray(red_positions[0])
        red_end = np.asarray(red_positions[-1])
        close_indices = {
            "left": np.flatnonzero(grippers[:, 0] < 0.5).astype(int).tolist(),
            "right": np.flatnonzero(grippers[:, 1] < 0.5).astype(int).tolist(),
        }
        payload = {
            "hdf5": str(args.hdf5),
            "episode_index": args.episode_index,
            "seed": args.seed,
            "segment": [args.start, args.end],
            "actions_replayed": len(selected),
            "state_action_temporal_alignment": {
                "definition": "action[t] should equal state[t+1]",
                "max_abs_error": float(temporal_error.max()),
                "mean_abs_error": float(temporal_error.mean()),
            },
            "reset_alignment": {
                "max_abs_error": float(initial_state_error.max()),
                "mean_abs_error": float(initial_state_error.mean()),
            },
            "simulator_qpos_tracking": {
                "max_abs_error": float(max(tracking_errors)),
                "mean_max_abs_error": float(np.mean(tracking_errors)),
            },
            "recorded_grippers": {
                "left_min_max": [float(grippers[:, 0].min()), float(grippers[:, 0].max())],
                "right_min_max": [float(grippers[:, 1].min()), float(grippers[:, 1].max())],
                "left_first_closed_offset": close_indices["left"][0] if close_indices["left"] else None,
                "right_first_closed_offset": close_indices["right"][0] if close_indices["right"] else None,
            },
            "red_block": {
                "start_xyz": red_start.tolist(),
                "end_xyz": red_end.tolist(),
                "displacement": float(np.linalg.norm(red_end - red_start)),
            },
            "initial_predicates": initial_predicates,
            "final_predicates": final_predicates,
            "first_subtask_succeeded": final_predicates.get("red_block") == "centered_base",
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
