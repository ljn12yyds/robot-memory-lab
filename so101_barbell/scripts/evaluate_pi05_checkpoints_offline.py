#!/usr/bin/env python3
"""Compare PI0.5 checkpoints on identical recorded SO101 observations.

The dataset was used for training, so this measures imitation fit/progression, not
generalization. The script never creates robot, camera, bus, or teleop objects.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy


JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
STAGES = {"initial": 0.05, "approach": 0.25, "grasp": 0.50, "lift": 0.75, "place": 0.95}
SOFT_LOW = torch.tensor([-72.64, -104.86, -96.64, -100.73, -167.63, 1.0])
SOFT_HIGH = torch.tensor([72.64, 104.86, 96.64, 100.73, 167.63, 99.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, help="LABEL=/absolute/path")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="barbell/so101_barbell_100_merged_v1")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 24, 49, 74, 99])
    parser.add_argument("--samples-per-frame", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def parse_checkpoints(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Checkpoint must be LABEL=PATH: {value}")
        label, path = value.split("=", 1)
        result.append((label, Path(path)))
    return result


def episode_rows(root: Path) -> dict[int, dict]:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    table = pq.read_table([str(path) for path in files])
    return {int(row["episode_index"]): row for row in table.to_pylist()}


def main() -> None:
    args = parse_args()
    checkpoints = parse_checkpoints(args.checkpoint)
    rows = episode_rows(args.dataset_root)
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)

    selected = []
    for episode in args.episodes:
        row = rows[episode]
        start, length = int(row["dataset_from_index"]), int(row["length"])
        for stage, fraction in STAGES.items():
            offset = min(length - 1, max(0, round((length - 1) * fraction)))
            selected.append((episode, stage, start + offset))

    cached = []
    for episode, stage, index in selected:
        sample = dataset[index]
        cached.append(
            {
                "episode": episode,
                "stage": stage,
                "index": index,
                "observation": {
                    "observation.images.front": sample["observation.images.front"],
                    "observation.images.wrist": sample["observation.images.wrist"],
                    "observation.state": sample["observation.state"],
                    "task": sample["task"],
                    "robot_type": "so101_follower",
                },
                "target": sample["action"].detach().float().cpu()[:6],
            }
        )

    reports = []
    for label, checkpoint in checkpoints:
        config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
        config.device, config.dtype = args.device, "bfloat16"
        policy = PI05Policy.from_pretrained(
            checkpoint, config=config, strict=True, local_files_only=True
        ).eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config, pretrained_path=str(checkpoint)
        )

        records = []
        for item in cached:
            processed = preprocessor(item["observation"])
            for repeat in range(args.samples_per_frame):
                seed = 100_000 + item["index"] * args.samples_per_frame + repeat
                torch.manual_seed(seed)
                with torch.inference_mode():
                    normalized = policy.predict_action_chunk(processed)
                    raw = postprocessor(normalized).detach().float().cpu()[0, 0, :6]
                    bounded = postprocessor(normalized.clamp(-1.0, 1.0)).detach().float().cpu()[0, 0, :6]
                records.append(
                    {
                        "episode": item["episode"],
                        "stage": item["stage"],
                        "index": item["index"],
                        "repeat": repeat,
                        "target": item["target"].tolist(),
                        "raw": raw.tolist(),
                        "bounded": bounded.tolist(),
                        "normalized_outside_count": int(((normalized < -1) | (normalized > 1)).sum()),
                        "raw_soft_violation": bool(((raw < SOFT_LOW) | (raw > SOFT_HIGH)).any()),
                        "bounded_soft_violation": bool(((bounded < SOFT_LOW) | (bounded > SOFT_HIGH)).any()),
                    }
                )

        target = torch.tensor([row["target"] for row in records])
        raw = torch.tensor([row["raw"] for row in records])
        bounded = torch.tensor([row["bounded"] for row in records])
        abs_error = (bounded - target).abs()
        stage_metrics = {}
        for stage in STAGES:
            mask = torch.tensor([row["stage"] == stage for row in records])
            stage_metrics[stage] = {
                "mae_all_deg": float(abs_error[mask].mean()),
                "mae_gripper": float(abs_error[mask, 5].mean()),
            }
        reports.append(
            {
                "label": label,
                "checkpoint": str(checkpoint),
                "sample_predictions": len(records),
                "bounded_mae_all_deg": float(abs_error.mean()),
                "bounded_rmse_all_deg": float(torch.sqrt(((bounded - target) ** 2).mean())),
                "bounded_mae_by_joint": dict(zip(JOINT_NAMES, abs_error.mean(dim=0).tolist(), strict=True)),
                "raw_mae_all_deg": float((raw - target).abs().mean()),
                "raw_chunks_with_first_action_soft_violation": int(sum(row["raw_soft_violation"] for row in records)),
                "bounded_chunks_with_first_action_soft_violation": int(sum(row["bounded_soft_violation"] for row in records)),
                "normalized_chunk_values_outside_total": int(sum(row["normalized_outside_count"] for row in records)),
                "stage_metrics": stage_metrics,
                "records": records,
            }
        )
        del policy, preprocessor, postprocessor
        gc.collect()
        torch.cuda.empty_cache()

    payload = {
        "evaluation_kind": "training-set representative-frame imitation fit; not held-out generalization",
        "episodes": args.episodes,
        "stages": STAGES,
        "frames_per_checkpoint": len(selected),
        "samples_per_frame": args.samples_per_frame,
        "checkpoints": reports,
        "hardware_accessed": False,
        "actions_sent": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = [{key: value for key, value in report.items() if key not in {"records", "stage_metrics", "checkpoint"}} for report in reports]
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"OFFLINE ONLY; report: {args.output}")


if __name__ == "__main__":
    main()
