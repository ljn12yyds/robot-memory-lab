#!/usr/bin/env python3
"""分析遥操作侧车日志中的 Leader、实际指令和 Follower 延迟。

输入是采集脚本生成的 CSV；输出关节误差与延迟统计 JSON。
本脚本只读文件，不连接或控制机械臂。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


JOINTS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


def main() -> None:
    """读取 CSV，并按关节和 Episode 汇总跟踪/裁剪误差。"""
    # ---------- 1. 解析输入输出路径 ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    # ---------- 2. 读取日志并只选择真正录制阶段 ----------
    # RESET 阶段的动作只用于人工复位，不应混入训练动作延迟统计。
    data = pd.read_csv(args.sidecar)
    recording = data[data["phase"] == "RECORDING"].copy()
    report: dict = {
        "sidecar": str(args.sidecar),
        "total_samples": int(len(data)),
        "recording_samples": int(len(recording)),
        "reset_samples": int((data["phase"] == "RESET").sum()),
        "episodes": {},
        "aggregate": {},
    }

    # ---------- 3. 全数据集逐关节统计 ----------
    # raw-state：主臂目标与从臂状态差；raw-sent：安全层裁剪掉的幅度。
    for joint in JOINTS:
        raw = recording[f"raw.{joint}"].to_numpy(dtype=float)
        sent = recording[f"sent.{joint}"].to_numpy(dtype=float)
        state = recording[f"state.{joint}"].to_numpy(dtype=float)
        raw_state = np.abs(raw - state)
        raw_sent = np.abs(raw - sent)
        report["aggregate"][joint] = {
            "mean_abs_raw_minus_state": float(raw_state.mean()),
            "p95_abs_raw_minus_state": float(np.percentile(raw_state, 95)),
            "max_abs_raw_minus_state": float(raw_state.max()),
            "mean_abs_raw_minus_sent": float(raw_sent.mean()),
            "fraction_raw_minus_sent_gt_5": float(np.mean(raw_sent > 5.0)),
        }

    # ---------- 4. 逐 Episode 重点分析腕部关节 ----------
    # 连续裁剪时间可以帮助定位腕部电机延迟或机械阻力。
    for episode, group in recording.groupby("episode_index", sort=True):
        episode_report = {}
        for joint in ("wrist_flex.pos", "wrist_roll.pos"):
            raw = group[f"raw.{joint}"].to_numpy(dtype=float)
            sent = group[f"sent.{joint}"].to_numpy(dtype=float)
            state = group[f"state.{joint}"].to_numpy(dtype=float)
            error = np.abs(raw - state)
            clipped = np.abs(raw - sent) > 0.25
            best = current = 0
            for value in clipped:
                current = current + 1 if value else 0
                best = max(best, current)
            episode_report[joint] = {
                "mean_abs_raw_minus_state": float(error.mean()),
                "p95_abs_raw_minus_state": float(np.percentile(error, 95)),
                "max_abs_raw_minus_state": float(error.max()),
                "longest_continuous_clipping_frames": int(best),
                "longest_continuous_clipping_seconds_at_30fps": float(best / 30),
            }
        report["episodes"][str(int(episode))] = episode_report

    # ---------- 5. 保存 JSON 报告 ----------
    output = args.output or args.sidecar.with_name("raw_leader_analysis.json")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Analysis written to: {output}")


if __name__ == "__main__":
    main()
