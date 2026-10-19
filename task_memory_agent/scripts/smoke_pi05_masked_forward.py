#!/usr/bin/env python3
"""Run one padded PI0.5 training batch through the masked loss path."""
from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import torch
from torch.utils.data._utils.collate import default_collate


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy  # noqa: F401

    config = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
    config.pretrained_path = str(args.checkpoint)
    config.device = args.device
    config.dtype = "bfloat16"
    delta_timestamps = {"action": [index / 15 for index in range(config.chunk_size)]}
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    first_episode_length = int(dataset.meta.episodes[0]["length"])
    sample = dataset[first_episode_length - 1]
    batch = default_collate([sample])
    policy = make_policy(config, ds_meta=dataset.meta, rename_map=RENAME_MAP).train()
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
    )
    processed = preprocessor(batch)
    torch.manual_seed(123)
    with torch.no_grad():
        loss, details = policy(processed)
    reconstructed_actions = postprocessor(processed["action"].detach().clone()).cpu()
    raw_actions = batch["action"].float().cpu()

    original_model_forward = policy.model.forward

    def fake_loss_forward(_self, images, img_masks, tokens, masks, actions):
        fake_losses = torch.full_like(actions, 100.0)
        fake_losses[:, 0, :] = 1.0
        return fake_losses

    policy.model.forward = types.MethodType(fake_loss_forward, policy.model)
    try:
        with torch.no_grad():
            mask_unit_loss, _ = policy(processed)
    finally:
        policy.model.forward = original_model_forward
    payload = {
        "status": (
            "PASS"
            if torch.isfinite(loss)
            and abs(float(mask_unit_loss.item()) - 1.0) < 1e-6
            and float((reconstructed_actions - raw_actions).abs().max()) < 1e-5
            else "FAIL"
        ),
        "sample_index": first_episode_length - 1,
        "action_shape": list(processed["action"].shape),
        "action_is_pad_shape": list(processed["action_is_pad"].shape),
        "padded_steps": int(processed["action_is_pad"].sum().item()),
        "valid_steps": int((~processed["action_is_pad"]).sum().item()),
        "loss": float(loss.item()),
        "deterministic_mask_test_loss": float(mask_unit_loss.item()),
        "normalization_roundtrip_max_abs_error": float(
            (reconstructed_actions - raw_actions).abs().max()
        ),
        "loss_per_dim_count": len(details["loss_per_dim"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
