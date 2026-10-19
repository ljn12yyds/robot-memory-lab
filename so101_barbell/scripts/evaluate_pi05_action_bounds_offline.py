#!/usr/bin/env python3
"""Compare raw and [-1,1]-bounded PI0.5 actions on recorded observations.

This script opens only a LeRobot dataset and a checkpoint. It never constructs a
robot, camera, motor bus, or teleoperator object, so it cannot command hardware.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
SOFT_LOW = torch.tensor([-72.64, -104.86, -96.64, -100.73, -167.63, 1.0])
SOFT_HIGH = torch.tensor([72.64, 104.86, 96.64, 100.73, 167.63, 99.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="barbell/so101_barbell_100_merged_v1")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 39754, 79507])
    parser.add_argument("--samples-per-frame", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def summarize(actions: torch.Tensor, state: torch.Tensor) -> dict:
    values = actions.detach().float().cpu()[0, :, :6]
    outside = (values < SOFT_LOW) | (values > SOFT_HIGH)
    first_delta = (values[0] - state.cpu()).abs()
    return {
        "min_deg": dict(zip(JOINT_NAMES, values.min(dim=0).values.tolist(), strict=True)),
        "max_deg": dict(zip(JOINT_NAMES, values.max(dim=0).values.tolist(), strict=True)),
        "outside_soft_values": int(outside.sum()),
        "chunks_with_any_soft_violation": int(outside.any()),
        "first_step_max_delta_deg": float(first_delta.max()),
        "first_step_over_20deg": int(first_delta.max() > 20.0),
    }


def main() -> None:
    args = parse_args()
    if args.samples_per_frame < 1:
        raise ValueError("--samples-per-frame must be positive")

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    config = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
    config.device = args.device
    config.dtype = "bfloat16"
    policy = PI05Policy.from_pretrained(
        args.checkpoint, config=config, strict=True, local_files_only=True
    ).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(args.checkpoint)
    )

    records = []
    totals = {
        "raw_chunks": 0,
        "raw_chunks_with_soft_violation": 0,
        "bounded_chunks_with_soft_violation": 0,
        "raw_first_steps_over_20deg": 0,
        "bounded_first_steps_over_20deg": 0,
        "raw_normalized_values_clipped": 0,
    }
    for index in args.indices:
        sample = dataset[index]
        observation = {
            "observation.images.front": sample["observation.images.front"],
            "observation.images.wrist": sample["observation.images.wrist"],
            "observation.state": sample["observation.state"],
            "task": sample["task"],
            "robot_type": "so101_follower",
        }
        processed = preprocessor(observation)
        state = sample["observation.state"].detach().float().cpu()
        for repeat in range(args.samples_per_frame):
            seed = 1000 + index * args.samples_per_frame + repeat
            torch.manual_seed(seed)
            with torch.inference_mode():
                normalized = policy.predict_action_chunk(processed)
                raw = postprocessor(normalized)
                bounded = postprocessor(normalized.clamp(-1.0, 1.0))
            raw_summary = summarize(raw, state)
            bounded_summary = summarize(bounded, state)
            clipped_values = int(((normalized < -1.0) | (normalized > 1.0)).sum().item())
            totals["raw_chunks"] += 1
            totals["raw_chunks_with_soft_violation"] += raw_summary[
                "chunks_with_any_soft_violation"
            ]
            totals["bounded_chunks_with_soft_violation"] += bounded_summary[
                "chunks_with_any_soft_violation"
            ]
            totals["raw_first_steps_over_20deg"] += raw_summary["first_step_over_20deg"]
            totals["bounded_first_steps_over_20deg"] += bounded_summary["first_step_over_20deg"]
            totals["raw_normalized_values_clipped"] += clipped_values
            records.append(
                {
                    "dataset_index": index,
                    "repeat": repeat,
                    "seed": seed,
                    "state_deg": dict(zip(JOINT_NAMES, state.tolist(), strict=True)),
                    "normalized_values_outside_minus1_plus1": clipped_values,
                    "raw": raw_summary,
                    "bounded": bounded_summary,
                }
            )

    totals["raw_soft_violation_rate"] = (
        totals["raw_chunks_with_soft_violation"] / totals["raw_chunks"]
    )
    totals["bounded_soft_violation_rate"] = (
        totals["bounded_chunks_with_soft_violation"] / totals["raw_chunks"]
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "indices": args.indices,
        "samples_per_frame": args.samples_per_frame,
        "totals": totals,
        "records": records,
        "hardware_accessed": False,
        "actions_sent": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(totals, indent=2, ensure_ascii=False))
    print(f"OFFLINE ONLY; report: {args.output}")


if __name__ == "__main__":
    main()
