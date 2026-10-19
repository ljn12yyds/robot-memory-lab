#!/usr/bin/env python3
"""Probe a PI0.5 checkpoint on recorded RoboTwin observations."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import torch


DEFAULT_INDICES = [0, 168, 309]
RENAME_MAP = {
    "observation.images.cam_high": "observation.images.base_0_rgb",
    "observation.images.cam_left_wrist": "observation.images.left_wrist_0_rgb",
    "observation.images.cam_right_wrist": "observation.images.right_wrist_0_rgb",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline zero-shot PI0.5 probe; never sends actions to a robot or simulator"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--index", type=int, action="append", dest="indices")
    parser.add_argument(
        "--episode-range",
        type=int,
        nargs=2,
        metavar=("START", "STOP"),
        help="Select the first frame of each episode in the half-open range [START, STOP).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy  # noqa: F401 - registers config type

    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        video_backend="pyav",
    )
    if args.indices and args.episode_range:
        parser.error("--index and --episode-range are mutually exclusive")
    if args.episode_range:
        start, stop = args.episode_range
        requested_episodes = set(range(start, stop))
        first_frame_by_episode = {}
        for index, episode_index in enumerate(dataset.hf_dataset["episode_index"]):
            episode_index = int(episode_index)
            if episode_index in requested_episodes and episode_index not in first_frame_by_episode:
                first_frame_by_episode[episode_index] = index
        missing = sorted(requested_episodes - first_frame_by_episode.keys())
        if missing:
            raise ValueError(f"dataset is missing requested episodes: {missing}")
        indices = [first_frame_by_episode[episode] for episode in range(start, stop)]
    else:
        indices = args.indices or DEFAULT_INDICES
    subtasks = list(dict.fromkeys(str(dataset[index]["task"]) for index in indices))
    config = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
    config.pretrained_path = str(args.checkpoint)
    config.device = args.device
    config.dtype = "bfloat16"
    policy = make_policy(config, ds_meta=dataset.meta, rename_map=RENAME_MAP).eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.checkpoint),
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
            "device_processor": {"device": "cpu"},
        },
    )

    results = []
    for index in indices:
        sample = dataset[index]
        observation = {
            key: value
            for key, value in sample.items()
            if key.startswith("observation.")
        }
        predictions = {}
        for instruction in subtasks:
            policy.reset()
            model_input = {**observation, "task": instruction}
            processed = preprocessor(model_input)
            torch.manual_seed(args.seed)
            with torch.inference_mode():
                predicted = postprocessor(policy.predict_action_chunk(processed))
            predictions[instruction] = predicted.detach().float().cpu()[0, :, :14]

        expert = sample["action"].detach().float().cpu()
        correct_instruction = str(sample["task"])
        correct = predictions[correct_instruction]
        pairwise = []
        for left, right in combinations(subtasks, 2):
            delta = predictions[left] - predictions[right]
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "mean_absolute_difference": float(delta.abs().mean()),
                    "first_action_l2_difference": float(torch.linalg.vector_norm(delta[0])),
                }
            )
        results.append(
            {
                "dataset_index": index,
                "episode_index": int(sample["episode_index"]),
                "recorded_subtask": correct_instruction,
                "expert_first_action": expert.tolist(),
                "predicted_first_action": correct[0].tolist(),
                "first_action_mae": float((correct[0] - expert).abs().mean()),
                "first_action_arm_mae": float(
                    (correct[0, [*range(6), *range(7, 13)]] - expert[[*range(6), *range(7, 13)]])
                    .abs()
                    .mean()
                ),
                "first_action_gripper_mae": float(
                    (correct[0, [6, 13]] - expert[[6, 13]]).abs().mean()
                ),
                "predicted_chunk_min": float(correct.min()),
                "predicted_chunk_max": float(correct.max()),
                "predicted_chunk_finite": bool(torch.isfinite(correct).all()),
                "pairwise_language_distances": pairwise,
            }
        )

    all_pairwise = [
        distance["mean_absolute_difference"]
        for result in results
        for distance in result["pairwise_language_distances"]
    ]
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_kind": "checkpoint supplied by --checkpoint",
        "dataset": args.repo_id,
        "normalization": "target dataset statistics; this probe performs no optimizer step",
        "hardware_accessed": False,
        "simulator_actions_sent": 0,
        "chunk_shape": [50, 14],
        "summary": {
            "num_observations": len(results),
            "mean_first_action_mae": sum(item["first_action_mae"] for item in results)
            / len(results),
            "mean_first_action_arm_mae": sum(
                item["first_action_arm_mae"] for item in results
            )
            / len(results),
            "mean_first_action_gripper_mae": sum(
                item["first_action_gripper_mae"] for item in results
            )
            / len(results),
            "mean_pairwise_language_action_difference": sum(all_pairwise) / len(all_pairwise),
            "all_predictions_finite": all(item["predicted_chunk_finite"] for item in results),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
