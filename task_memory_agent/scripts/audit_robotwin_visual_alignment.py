#!/usr/bin/env python3
"""Compare online RoboTwin images with raw HDF5 and converted training video."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np


ROBOTWIN_ROOT = Path(os.environ.get("ROBOTWIN_ROOT", "/home/user/ljn/RoboTwin")).resolve()
AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOTWIN_ROOT))
sys.path.insert(0, str(AGENT_ROOT))
sys.path.insert(0, str(AGENT_ROOT / "scripts"))

from envs.stack_blocks_three import stack_blocks_three
from run_robotwin_grounded_stack import _load_robotwin_args


CAMERAS = {
    "cam_high": ("head_camera", "cam_head"),
    "cam_left_wrist": ("left_camera", "cam_left_wrist"),
    "cam_right_wrist": ("right_camera", "cam_right_wrist"),
}


def resize_like(image: np.ndarray, target: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_AREA)


def mae(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", default="memory_pi05_permutations30")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--sim-output", type=Path, required=True)
    parser.add_argument("--decoded-video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    env = stack_blocks_three()
    results: dict[str, dict] = {}
    try:
        env.setup_demo(
            now_ep_num=args.episode_index,
            seed=args.seed,
            **_load_robotwin_args(args.config, args.sim_output),
        )
        online = env.get_obs()["observation"]
        with h5py.File(args.hdf5, "r") as source:
            for dataset_key, (online_key, raw_key) in CAMERAS.items():
                encoded = np.frombuffer(source[f"vision/{raw_key}/colors"][0], dtype=np.uint8)
                raw_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                online_image = np.asarray(online[online_key]["rgb"], dtype=np.uint8)
                raw_resized = resize_like(raw_bgr, online_image)
                raw_online_direct = mae(raw_resized, online_image)
                raw_online_swapped = mae(raw_resized[..., ::-1], online_image)

                training_rgb = np.load(args.decoded_video_dir / f"{dataset_key}.npy")
                video_resized = resize_like(training_rgb, raw_bgr)
                video_raw_direct = mae(video_resized, raw_bgr)
                video_raw_swapped = mae(video_resized[..., ::-1], raw_bgr)
                training_online_direct = mae(resize_like(training_rgb, online_image), online_image)
                training_online_swapped = mae(
                    resize_like(training_rgb[..., ::-1], online_image), online_image
                )

                results[dataset_key] = {
                    "online_shape": list(online_image.shape),
                    "raw_shape": list(raw_bgr.shape),
                    "training_rgb_shape": list(training_rgb.shape),
                    "raw_to_online_mae_direct": raw_online_direct,
                    "raw_to_online_mae_channel_swapped": raw_online_swapped,
                    "raw_online_best_transform": (
                        "channel_swap" if raw_online_swapped < raw_online_direct else "direct"
                    ),
                    "video_to_raw_mae_direct": video_raw_direct,
                    "video_to_raw_mae_channel_swapped": video_raw_swapped,
                    "video_raw_best_transform": (
                        "channel_swap" if video_raw_swapped < video_raw_direct else "direct"
                    ),
                    "training_rgb_to_online_mae_direct": training_online_direct,
                    "training_rgb_to_online_mae_channel_swapped": training_online_swapped,
                    "training_online_best_transform": (
                        "channel_swap"
                        if training_online_swapped < training_online_direct
                        else "direct"
                    ),
                }
    finally:
        env.close_env()

    passed = all(
        item["raw_online_best_transform"] == "direct"
        and item["video_raw_best_transform"] == "direct"
        and item["training_online_best_transform"] == "direct"
        and item["raw_to_online_mae_direct"] < 30.0
        and item["video_to_raw_mae_direct"] < 10.0
        for item in results.values()
    )
    payload = {"status": "PASS" if passed else "FAIL", "cameras": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
