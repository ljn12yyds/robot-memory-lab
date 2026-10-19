#!/usr/bin/env python3
"""Serve a LeRobot PI0.5 checkpoint to a RoboTwin process over localhost."""
from __future__ import annotations

import argparse
import json
from multiprocessing.connection import Listener
from pathlib import Path

import torch


RENAME_MAP = {
    "observation.images.cam_high": "observation.images.base_0_rgb",
    "observation.images.cam_left_wrist": "observation.images.left_wrist_0_rgb",
    "observation.images.cam_right_wrist": "observation.images.right_wrist_0_rgb",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6025)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed the diffusion sampler for reproducible online action chunks.",
    )
    parser.add_argument(
        "--calibrate-binary-grippers",
        action="store_true",
        help="Remap gripper outputs from erroneous nonzero quantile bounds to [0, 1].",
    )
    parser.add_argument(
        "--action-quantiles",
        type=Path,
        help="Empirical q01/q99 JSON used to decode actions from corrected physical bounds.",
    )
    args = parser.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy  # noqa: F401 - registers config type

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, video_backend="pyav")
    corrected_quantiles = None
    if args.action_quantiles:
        corrected_quantiles = json.loads(args.action_quantiles.read_text(encoding="utf-8"))
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

    listener = Listener((args.host, args.port), authkey=b"robotwin-pi05")
    print(f"READY {args.host}:{args.port}", flush=True)
    running = True
    while running:
        connection = listener.accept()
        try:
            while True:
                try:
                    request = connection.recv()
                except EOFError:
                    break
                operation = request.get("op")
                if operation == "shutdown":
                    connection.send({"ok": True})
                    running = False
                    break
                if operation == "reset":
                    policy.reset()
                    connection.send({"ok": True})
                    continue
                if operation != "predict":
                    connection.send({"ok": False, "error": f"unknown operation: {operation}"})
                    continue
                model_input = {
                    key: torch.from_numpy(value) for key, value in request["observation"].items()
                }
                model_input["task"] = request["instruction"]
                processed = preprocessor(model_input)
                if args.seed is not None:
                    torch.manual_seed(args.seed)
                with torch.inference_mode():
                    actions = postprocessor(policy.predict_action_chunk(processed))
                actions = actions.detach().float().cpu()[0, :, :14].numpy()
                if corrected_quantiles is not None:
                    old_lower = torch.tensor(dataset.meta.stats["action"]["q01"]).numpy()
                    old_upper = torch.tensor(dataset.meta.stats["action"]["q99"]).numpy()
                    new_lower = torch.tensor(corrected_quantiles["q01"]).numpy()
                    new_upper = torch.tensor(corrected_quantiles["q99"]).numpy()
                    unit_actions = (actions - old_lower) / (old_upper - old_lower)
                    actions = (new_lower + unit_actions * (new_upper - new_lower)).clip(
                        new_lower, new_upper
                    )
                elif args.calibrate_binary_grippers:
                    for index in (6, 13):
                        lower = float(dataset.meta.stats["action"]["q01"][index])
                        upper = float(dataset.meta.stats["action"]["q99"][index])
                        actions[:, index] = ((actions[:, index] - lower) / (upper - lower)).clip(
                            0.0, 1.0
                        )
                connection.send({"ok": True, "actions": actions})
        finally:
            connection.close()
    listener.close()


if __name__ == "__main__":
    main()
