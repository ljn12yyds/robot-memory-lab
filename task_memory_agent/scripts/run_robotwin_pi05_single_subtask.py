#!/usr/bin/env python3
"""Run one PI0.5 subtask from a grounded RoboTwin prerequisite state."""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

ROBOTWIN_ROOT = Path(os.environ.get("ROBOTWIN_ROOT", "/home/user/ljn/RoboTwin")).resolve()
AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOTWIN_ROOT))
sys.path.insert(0, str(AGENT_ROOT))
sys.path.insert(0, str(AGENT_ROOT / "scripts"))

from envs.stack_blocks_three import stack_blocks_three
from run_robotwin_grounded_stack import _load_robotwin_args, live_stack_predicates


INSTRUCTIONS = {
    0: "pick up the red block and place it at the center as the tower base",
    1: "pick up the green block and place it stably on top of the red block",
    2: "pick up the blue block and place it stably on top of the green block",
}


class RemotePI05:
    def __init__(self, host: str, port: int) -> None:
        self.connection = Client((host, port), authkey=b"robotwin-pi05")

    def reset(self) -> None:
        self.connection.send({"op": "reset"})
        if not self.connection.recv().get("ok"):
            raise RuntimeError("policy reset failed")

    def predict(self, observation: dict[str, np.ndarray], instruction: str) -> np.ndarray:
        self.connection.send({"op": "predict", "observation": observation, "instruction": instruction})
        response = self.connection.recv()
        if not response.get("ok"):
            raise RuntimeError(response)
        return np.asarray(response["actions"], dtype=np.float32)

    def close(self) -> None:
        self.connection.close()


def encode_observation(raw: dict) -> dict[str, np.ndarray]:
    cameras = raw["observation"]
    encoded = {
        "observation.images.cam_high": cameras["head_camera"]["rgb"],
        "observation.images.cam_left_wrist": cameras["left_camera"]["rgb"],
        "observation.images.cam_right_wrist": cameras["right_camera"]["rgb"],
        "observation.state": raw["joint_action"]["vector"],
    }
    for key, value in list(encoded.items()):
        value = np.asarray(value)
        if key.startswith("observation.images"):
            if value.shape[-1] == 3:
                value = np.transpose(value, (2, 0, 1))
            value = value.astype(np.float32)
            if value.max() > 1.0:
                value /= 255.0
        else:
            value = value.astype(np.float32)
        encoded[key] = np.ascontiguousarray(value)
    return encoded


def pose_dict(env: stack_blocks_three) -> dict[str, list[float]]:
    return {
        name: np.asarray(actor.get_pose().p, dtype=float).tolist()
        for name, actor in (("red", env.block1), ("green", env.block2), ("blue", env.block3))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--config", default="memory_smoke")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6025)
    parser.add_argument("--actions-per-chunk", type=int, default=20)
    parser.add_argument("--max-chunks", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    env = stack_blocks_three()
    backend = RemotePI05(args.host, args.port)
    try:
        env.setup_demo(
            now_ep_num=args.episode_index,
            seed=args.seed,
            **_load_robotwin_args(args.config, args.output.parent),
        )
        env.last_gripper = None
        env.last_actor = None
        # Establish valid prerequisites with RoboTwin's own expert primitive.
        for prerequisite in range(args.subtask):
            env.last_actor = None if prerequisite == 0 else [env.block1, env.block2, env.block3][prerequisite - 1]
            env.pick_and_place_block([env.block1, env.block2, env.block3][prerequisite])
        before = pose_dict(env)
        instruction = INSTRUCTIONS[args.subtask]
        chunks = []
        backend.reset()
        for chunk_index in range(args.max_chunks):
            raw = env.get_obs()
            actions = backend.predict(encode_observation(raw), instruction)
            if actions.shape != (50, 14) or not np.isfinite(actions).all():
                raise ValueError(f"invalid action chunk: {actions.shape}")
            for action in actions[: args.actions_per_chunk]:
                env.take_action(action, action_type="qpos")
            after = pose_dict(env)
            chunks.append({
                "chunk": chunk_index,
                "actions_sent": args.actions_per_chunk,
                "poses": after,
                "predicted_grippers": actions[:, [6, 13]].tolist(),
                "predicates": live_stack_predicates(env),
            })
        final = pose_dict(env)
        target = ("red", "green", "blue")[args.subtask]
        result = {
            "subtask": args.subtask,
            "instruction": instruction,
            "checkpoint": str(args.checkpoint),
            "prerequisites_executed_by_robotwin_expert": args.subtask,
            "initial_poses_after_prerequisites": before,
            "final_poses": final,
            "target_displacement": float(np.linalg.norm(np.asarray(final[target]) - np.asarray(before[target]))),
            "robotwin_success": bool(env.check_success()),
            "final_predicates": live_stack_predicates(env),
            "chunks": chunks,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        backend.close()
        env.close_env()


if __name__ == "__main__":
    main()
