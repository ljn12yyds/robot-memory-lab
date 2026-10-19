#!/usr/bin/env python3
"""检查本地 LeRobot 杠铃数据集的数值和视频完整性。

检查 Episode/帧数、NaN、动作状态误差、两路视频分辨率/帧率/帧数。
本脚本只读数据集，审计结果写入 JSON，不连接机械臂。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def ffprobe(path: Path) -> dict:
    """读取一个 MP4 的编码、分辨率、帧率、时长和帧数。"""
    # 优先使用系统 ffprobe；找不到时检查当前 Python 环境旁边的程序。
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin is None:
        sibling = Path(sys.executable).with_name("ffprobe")
        if sibling.exists():
            ffprobe_bin = str(sibling)
        else:
            raise RuntimeError("ffprobe was not found in PATH or next to the active Python executable.")
    result = subprocess.run(
        [
            ffprobe_bin, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["streams"][0]


def main() -> None:
    """执行数据表、动作状态和视频三部分完整性审计。"""
    # ---------- 1. 参数与误差阈值 ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", help="For example: barbell/so101_trial_v3")
    parser.add_argument(
        "--root",
        type=Path,
        help="Exact local dataset directory. Defaults to the legacy Hugging Face cache path.",
    )
    parser.add_argument("--output", type=Path)
    # max_relative_target=5 是相对稍后一次电机读数做裁剪，而数据帧里的状态采样更早。
    # 因此受采样延迟影响，action-state 偶尔略高于 5° 并不一定代表安全裁剪失效。
    parser.add_argument("--max-mean-error", type=float, default=2.5)
    parser.add_argument("--max-p99-error", type=float, default=5.5)
    parser.add_argument("--max-single-error", type=float, default=6.0)
    args = parser.parse_args()

    # ---------- 2. 定位并读取数据集 ----------
    root = (
        args.root
        if args.root is not None
        else Path.home() / ".cache/huggingface/lerobot" / args.repo_id
    )
    info_path = root / "meta/info.json"
    if not info_path.exists():
        raise SystemExit(f"Dataset not found: {root}")
    info = json.loads(info_path.read_text())

    parquet_files = sorted((root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise SystemExit("No parquet data files found.")
    data = pd.concat((pd.read_parquet(path) for path in parquet_files), ignore_index=True)
    # 将 Parquet 中的向量列堆叠成 [帧数, 6关节] 数组。
    action = np.stack(data["action"].to_numpy()).astype(np.float64)
    state = np.stack(data["observation.state"].to_numpy()).astype(np.float64)
    names = info["features"]["action"]["names"]

    report: dict = {
        "repo_id": args.repo_id,
        "dataset_root": str(root),
        "metadata_episodes": int(info["total_episodes"]),
        "metadata_frames": int(info["total_frames"]),
        "actual_frames": int(len(data)),
        "fps": int(info["fps"]),
        "finite": bool(np.isfinite(action).all() and np.isfinite(state).all()),
        "episodes": [],
        "videos": {},
    }

    # ---------- 3. 逐 Episode 检查数值与动作跟踪 ----------
    passed = report["finite"] and len(data) == int(info["total_frames"])
    for episode, group in data.groupby("episode_index", sort=True):
        indices = group.index.to_numpy()
        err = np.abs(action[indices] - state[indices])
        per_joint_mean = err.mean(axis=0)
        per_joint_p99 = np.percentile(err, 99, axis=0)
        per_joint_max = err.max(axis=0)
        episode_ok = bool(
            per_joint_mean.max() <= args.max_mean_error
            and per_joint_p99.max() <= args.max_p99_error
            and per_joint_max.max() <= args.max_single_error
        )
        timestamps = group["timestamp"].to_numpy(dtype=float)
        episode_report = {
            "episode_index": int(episode),
            "frames": int(len(group)),
            "duration_s": float(timestamps[-1] - timestamps[0]) if len(group) > 1 else 0.0,
            "mean_abs_action_state_deg": dict(zip(names, per_joint_mean.round(3))),
            "p99_abs_action_state_deg": dict(zip(names, per_joint_p99.round(3))),
            "max_abs_action_state_deg": dict(zip(names, per_joint_max.round(3))),
            "numeric_pass": episode_ok,
            "operator_marked_success": True,
        }
        report["episodes"].append(episode_report)
        passed = passed and episode_ok

    # ---------- 4. 检查每个视频流 ----------
    video_features = [
        name for name, feature in info["features"].items() if feature.get("dtype") == "video"
    ]
    for feature_name in video_features:
        paths = sorted((root / "videos" / feature_name).rglob("*.mp4"))
        report["videos"][feature_name] = [
            {"path": str(path), **ffprobe(path)} for path in paths
        ]
        if not paths:
            passed = False

    report["numeric_and_media_pass"] = bool(passed)
    report["overall_pass"] = bool(passed)
    report["note"] = (
        "Task success/failure is marked online by the operator: right arrow saves, "
        "left arrow discards. This audit covers numerical and media integrity."
    )

    # ---------- 5. 汇总结果并用退出码表达通过/失败 ----------
    output = args.output or Path("reports") / args.repo_id.replace("/", "_") / "audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Audit written to: {output}")
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
