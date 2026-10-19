#!/usr/bin/env python
"""使用已录制数据离线检查 SmolVLA checkpoint。

读取数据集中的观测并运行模型，只保存预测结果；完全不连接机械臂。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def tensor_summary(value: torch.Tensor) -> dict:
    """把动作张量压缩为形状和统计量，避免报告保存整个 chunk。"""
    value = value.detach().float().cpu()
    return {
        "shape": list(value.shape),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "contains_nan": bool(torch.isnan(value).any()),
        "contains_inf": bool(torch.isinf(value).any()),
    }


def parse_args() -> argparse.Namespace:
    """解析 checkpoint、历史数据集、抽样帧和报告路径。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="barbell/so101_barbell_100_merged_v1")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 39754, 79507])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """运行若干历史观测，确认 checkpoint 输出有限且维度正确。"""
    # ---------- 1. 固定设备和随机种子，保证重复测试可比较 ----------
    args = parse_args()
    device = torch.device("cuda")
    torch.manual_seed(1000)

    # ---------- 2. 加载数据、模型及其归一化处理器 ----------
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # ---------- 3. 对指定帧执行动作 chunk 预测 ----------
    results = []
    for index in args.indices:
        sample = dataset[index]
        observation = {
            "observation.images.front": sample["observation.images.front"],
            "observation.images.wrist": sample["observation.images.wrist"],
            "observation.state": sample["observation.state"],
            "task": sample["task"],
        }

        # preprocessor 使用训练 checkpoint 内保存的均值/方差归一化状态与动作。
        processed = preprocessor(observation)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            normalized_chunk = policy.predict_action_chunk(processed)
            action_chunk = postprocessor(normalized_chunk)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000

        state = sample["observation.state"].detach().float().cpu()
        chunk = action_chunk.detach().float().cpu()
        first_action = chunk[0, 0]
        first_delta = first_action - state

        results.append(
            {
                "dataset_index": index,
                "episode_index": int(sample["episode_index"]),
                "frame_index": int(sample["frame_index"]),
                "task": sample["task"],
                "state_deg": dict(zip(JOINT_NAMES, map(float, state), strict=True)),
                "action_chunk": tensor_summary(chunk),
                "first_action_deg": dict(zip(JOINT_NAMES, map(float, first_action), strict=True)),
                "first_action_minus_state_deg": dict(
                    zip(JOINT_NAMES, map(float, first_delta), strict=True)
                ),
                "inference_ms": elapsed_ms,
            }
        )

    # ---------- 4. 检查 NaN/Inf，并保存诊断报告 ----------
    all_finite = all(
        not item["action_chunk"]["contains_nan"] and not item["action_chunk"]["contains_inf"]
        for item in results
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "device": str(device),
        "samples": results,
        "all_outputs_finite": all_finite,
        "motion_command_sent": False,
        "note": "OFFLINE ONLY: no camera, serial port, or robot object was opened.",
    }

    if not all_finite or not all(math.isfinite(item["inference_ms"]) for item in results):
        raise RuntimeError("Offline inference produced a non-finite result.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("NO ROBOT MOTION COMMAND WAS SENT")


if __name__ == "__main__":
    main()
