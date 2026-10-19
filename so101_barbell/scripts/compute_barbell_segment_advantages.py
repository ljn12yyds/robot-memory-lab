#!/usr/bin/env python3
"""把人工标注的 Episode 结果转换为逐帧训练 advantage。

普通 Episode 先根据成败得到一个基础 advantage。成功的人工介入 Episode
进一步切成三段：

1. 早期自主段：保留基础 advantage，因为这些动作可能是正确的；
2. 接管前风险窗口：降低权重，因为它最可能直接触发人工接管；
3. 接管后人工段：提高权重，让模型重点学习纠正动作。

这里的分数是人工设计的训练权重信号，并非 critic 学出的严格 AWR advantage。
本脚本只处理数据，不连接机械臂。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """解析数据路径、AWR 温度 beta 和 advantage 裁剪范围。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--clip-min", type=float, default=-2.0)
    parser.add_argument("--clip-max", type=float, default=2.0)
    parser.add_argument(
        "--intervention-policy-advantage",
        type=float,
        default=-0.5,
        help="成功人工接管 Episode 中，接管前风险窗口的 advantage",
    )
    parser.add_argument(
        "--intervention-human-advantage",
        type=float,
        default=1.0,
        help="成功人工接管 Episode 中，接管后人工纠正片段的 advantage",
    )
    parser.add_argument(
        "--intervention-risk-window-frames",
        type=int,
        default=90,
        help="接管前多少帧视为风险窗口；30 fps 下默认 90 帧约等于 3 秒",
    )
    return parser.parse_args()


def load_labels(path: Path) -> dict[int, dict[str, Any]]:
    """统一数值标签，同时保留人工接管标志和接管起始帧。"""
    raw = json.loads(path.read_text())
    rows = raw["episodes"] if isinstance(raw, dict) else raw
    labels: dict[int, dict[str, Any]] = {}
    # BC Episode 可以直接给固定 advantage；rollout 通常只给 success。
    for row in rows:
        episode = int(row["episode_index"])
        if "advantage" in row:
            label: dict[str, Any] = {"advantage": float(row["advantage"])}
        elif "return" in row:
            label = {"return": float(row["return"])}
        elif "success" in row:
            label = {
                "return": float(bool(row["success"])),
                "success": bool(row["success"]),
            }
        else:
            raise ValueError(
                f"Episode {episode} has neither 'advantage', 'return', nor 'success'"
            )
        label["human_intervention"] = bool(row.get("human_intervention", False))
        label["intervention_start_frame"] = row.get("intervention_start_frame")
        labels[episode] = label
    return labels


def main() -> None:
    """校验标签、计算 Episode advantage，再展开到数据集每一帧。"""
    # ---------- 1. 参数合法性 ----------
    args = parse_args()
    if args.beta <= 0:
        raise ValueError("--beta must be positive")
    if args.intervention_risk_window_frames < 0:
        raise ValueError("--intervention-risk-window-frames must be non-negative")

    # ---------- 2. 只读取全局帧索引和 Episode 索引 ----------
    parquet_paths = sorted((args.dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No dataset parquet files under {args.dataset_root / 'data'}")

    frames = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=["index", "episode_index", "frame_index"],
            )
            for path in parquet_paths
        ],
        ignore_index=True,
    )
    # ---------- 3. 确认标签与数据集 Episode 一一对应 ----------
    labels = load_labels(args.labels)
    episodes = sorted(int(v) for v in frames["episode_index"].unique())
    missing = sorted(set(episodes) - set(labels))
    extra = sorted(set(labels) - set(episodes))
    if missing:
        raise ValueError(f"Missing labels for episodes: {missing[:20]}")
    if extra:
        raise ValueError(f"Labels reference absent episodes: {extra[:20]}")

    # ---------- 4. 用 rollout 回报估计 baseline/std ----------
    # 直接指定 advantage 的 BC Episode 不参与这个统计，避免改变 rollout 基线。
    return_episodes = [episode for episode in episodes if "return" in labels[episode]]
    if return_episodes:
        returns = np.array(
            [labels[episode]["return"] for episode in return_episodes], dtype=np.float64
        )
        baseline = float(returns.mean())
        std = float(returns.std())
        denom = max(std * args.beta, 1e-6)
    else:
        baseline = 0.0
        std = 0.0
        denom = 1.0

    # ---------- 5. 计算并裁剪每条 Episode 的 advantage ----------
    episode_advantage: dict[int, float] = {}
    for episode in episodes:
        label = labels[episode]
        if "advantage" in label:
            advantage = label["advantage"]
        else:
            advantage = (label["return"] - baseline) / denom
        episode_advantage[episode] = float(
            np.clip(advantage, args.clip_min, args.clip_max)
        )

    # ---------- 6. 将 Episode advantage 展开到其全部帧 ----------
    output = args.output or args.dataset_root / "segment_advantages.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = frames.copy()
    result["advantage"] = result["episode_index"].map(episode_advantage).astype("float32")

    # ---------- 7. 对成功的人工纠正 Episode 做三段式赋权 ----------
    # 早期自主段保留该 Episode 的基础 advantage。只降低接管前一小段风险
    # 窗口，避免把正确的接近、对准等早期动作一起惩罚。
    #
    # 只给最终成功的人工段正权重。若接管后仍失败，则整条轨迹沿用失败
    # Episode 的低 advantage，避免把未完成的人工动作直接当成专家标签。
    intervention_episodes = 0
    intervention_frames = 0
    risk_frames = 0
    for episode in episodes:
        label = labels[episode]
        if not label.get("human_intervention", False):
            continue
        start = label.get("intervention_start_frame")
        if start is None:
            raise ValueError(
                f"Episode {episode} is marked as intervention but has no start frame"
            )
        start = int(start)
        episode_mask = result["episode_index"] == episode
        episode_frame_count = int(episode_mask.sum())
        if start < 0 or start >= episode_frame_count:
            raise ValueError(
                f"Episode {episode} intervention_start_frame={start} is outside "
                f"its {episode_frame_count} frames"
            )
        intervention_episodes += 1
        if bool(label.get("success", label.get("return", 0.0) > 0.0)):
            risk_start = max(0, start - args.intervention_risk_window_frames)
            risk = (
                episode_mask
                & (result["frame_index"] >= risk_start)
                & (result["frame_index"] < start)
            )
            human = episode_mask & (result["frame_index"] >= start)
            result.loc[risk, "advantage"] = args.intervention_policy_advantage
            result.loc[human, "advantage"] = args.intervention_human_advantage
            risk_frames += int(risk.sum())
            intervention_frames += int(human.sum())

    result["advantage"] = result["advantage"].clip(
        lower=args.clip_min,
        upper=args.clip_max,
    ).astype("float32")
    result.to_parquet(output, index=False)

    # ---------- 8. 打印训练前可人工核对的摘要 ----------
    summary = {
        "dataset_root": str(args.dataset_root.resolve()),
        "labels": str(args.labels.resolve()),
        "output": str(output.resolve()),
        "episodes": len(episodes),
        "frames": len(result),
        "return_mean": baseline,
        "return_std": std,
        "advantage_min": float(result["advantage"].min()),
        "advantage_max": float(result["advantage"].max()),
        "effective_weight_min": float(np.exp(result["advantage"].min())),
        "effective_weight_max": float(np.exp(result["advantage"].max())),
        "intervention_episodes": intervention_episodes,
        "intervention_risk_frames": risk_frames,
        "successful_human_correction_frames": intervention_frames,
        "intervention_risk_window_frames": args.intervention_risk_window_frames,
        "intervention_policy_advantage": args.intervention_policy_advantage,
        "intervention_human_advantage": args.intervention_human_advantage,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
