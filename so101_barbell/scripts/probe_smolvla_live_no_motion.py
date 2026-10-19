#!/usr/bin/env python
"""读取真实相机和关节状态进行 SmolVLA 在线探针，但绝不执行动作。

脚本只读取从臂位置与两路图像，模型预测仅写入 JSON，适合真实 rollout 前核对。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
# 任务文本必须与训练数据完全一致，否则语言条件会发生分布偏移。
TASK = "Lift the blue barbell with the left arm and place it in the orange tray"


def parse_args() -> argparse.Namespace:
    """解析模型、真实设备路径、相机预热帧数和重复推理次数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--follower-port",
        default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_FOLLOWER_SERIAL-if00",
    )
    parser.add_argument(
        "--front-camera",
        default="/dev/v4l/by-path/pci-0000:05:00.0-usb-0:2:1.0-video-index0",
    )
    parser.add_argument(
        "--wrist-camera",
        default="/dev/v4l/by-path/pci-0000:00:14.0-usb-0:4.4:1.0-video-index0",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--inferences", type=int, default=3)
    return parser.parse_args()


def open_camera(path: str) -> cv2.VideoCapture:
    """用与采集相同的 V4L2/MJPG/640×480@30 配置打开相机。"""
    capture = cv2.VideoCapture(path, cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_FPS, 30)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera: {path}")
    return capture


def read_warmed_frame(capture: cv2.VideoCapture, count: int) -> tuple[np.ndarray, dict]:
    """丢弃相机启动阶段的不稳定帧，返回最后一帧和采集统计。"""
    frame = None
    successes = 0
    started = time.perf_counter()
    for _ in range(count):
        ok, candidate = capture.read()
        if ok and candidate is not None:
            frame = candidate
            successes += 1
    elapsed = time.perf_counter() - started
    if frame is None:
        raise RuntimeError("Camera returned no valid frames.")
    return frame, {
        "requested_frames": count,
        "successful_frames": successes,
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "channels": int(frame.shape[2]),
        "dtype": str(frame.dtype),
        "capture_ms": elapsed * 1000,
    }


def image_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """把 HWC uint8 RGB 图像转换为模型需要的 CHW float32 [0,1] 张量。"""
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255.0)


def main() -> None:
    """采集一次真实观测，在 GPU 上重复预测并保存动作 chunk 统计。"""
    # ---------- 1. 参数与设备存在性检查 ----------
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in (args.follower_port, args.front_camera, args.wrist_camera):
        if not Path(path).exists():
            raise FileNotFoundError(path)

    # ---------- 2. 打开两路相机并构造只读从臂对象 ----------
    front_capture = open_camera(args.front_camera)
    wrist_capture = open_camera(args.wrist_camera)
    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.follower_port,
            id="SO101_FOLLOWER_ID",
            use_degrees=True,
            cameras={},
        )
    )

    # ---------- 3. 读取图像和关节状态 ----------
    try:
        front_bgr, front_stats = read_warmed_frame(front_capture, args.warmup_frames)
        wrist_bgr, wrist_stats = read_warmed_frame(wrist_capture, args.warmup_frames)

        # 与训练数据保持一致：顶部相机在软件中旋转 180°。
        front_bgr = cv2.rotate(front_bgr, cv2.ROTATE_180)
        front_rgb = cv2.cvtColor(front_bgr, cv2.COLOR_BGR2RGB)
        wrist_rgb = cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB)

        # 只连接总线读取状态；不调用 robot.connect/configure 或任何写方法。
        robot.bus.connect()
        try:
            torque = robot.bus.sync_read("Torque_Enable", normalize=False)
            position = robot.bus.sync_read("Present_Position")
        finally:
            robot.bus.disconnect(disable_torque=False)
    finally:
        front_capture.release()
        wrist_capture.release()

    # 保存模型实际看到的画面，便于人工核对相机角色和旋转方向。
    cv2.imwrite(str(args.output_dir / "front_live.jpg"), front_bgr)
    cv2.imwrite(str(args.output_dir / "wrist_live.jpg"), wrist_bgr)

    # ---------- 4. 按训练特征名称构造 observation ----------
    state = torch.tensor([float(position[name]) for name in JOINT_NAMES], dtype=torch.float32)
    observation = {
        "observation.images.front": image_to_tensor(front_rgb),
        "observation.images.wrist": image_to_tensor(wrist_rgb),
        "observation.state": state,
        "task": TASK,
    }

    # ---------- 5. 加载 checkpoint 及其归一化处理器 ----------
    device = torch.device("cuda")
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    processed = preprocessor(observation)

    # ---------- 6. 重复预测动作 chunk ----------
    # 每次改变随机种子，用于观察扩散/流模型输出是否稳定，而不是执行这些动作。
    probes = []
    for probe_index in range(args.inferences):
        torch.manual_seed(1000 + probe_index)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            action_chunk = postprocessor(policy.predict_action_chunk(processed))
        torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - started) * 1000

        chunk = action_chunk.detach().float().cpu()
        trajectory = chunk[0].numpy()
        first_action = chunk[0, 0]
        delta = first_action - state
        # 统计每个关节在 50 步 chunk 内的范围、总变化和方向反转次数。
        per_joint = {}
        for joint_index, joint_name in enumerate(JOINT_NAMES):
            values = trajectory[:, joint_index]
            steps = np.diff(values)
            nonzero_steps = steps[np.abs(steps) >= 0.05]
            reversals = (
                int(np.sum(np.sign(nonzero_steps[1:]) != np.sign(nonzero_steps[:-1])))
                if len(nonzero_steps) >= 2
                else 0
            )
            per_joint[joint_name] = {
                "start_deg": float(values[0]),
                "end_deg": float(values[-1]),
                "min_deg": float(values.min()),
                "max_deg": float(values.max()),
                "range_deg": float(values.max() - values.min()),
                "net_change_deg": float(values[-1] - values[0]),
                "max_abs_delta_from_state_deg": float(
                    np.max(np.abs(values - float(state[joint_index])))
                ),
                "mean_abs_step_deg": float(np.mean(np.abs(steps))),
                "max_abs_step_deg": float(np.max(np.abs(steps))),
                "total_variation_deg": float(np.sum(np.abs(steps))),
                "direction_reversals_over_0_05deg": reversals,
            }
        probes.append(
            {
                "probe_index": probe_index,
                "inference_ms": inference_ms,
                "action_shape": list(chunk.shape),
                "action_min_deg": float(chunk.min()),
                "action_max_deg": float(chunk.max()),
                "contains_nan": bool(torch.isnan(chunk).any()),
                "contains_inf": bool(torch.isinf(chunk).any()),
                "first_action_deg": dict(zip(JOINT_NAMES, map(float, first_action), strict=True)),
                "first_action_minus_state_deg": dict(
                    zip(JOINT_NAMES, map(float, delta), strict=True)
                ),
                "max_abs_first_delta_deg": float(delta.abs().max()),
                "per_joint_trajectory_stats": per_joint,
                "full_action_chunk_deg": trajectory.tolist(),
            }
        )

    # ---------- 7. 保存报告 ----------
    report = {
        "checkpoint": str(args.checkpoint),
        "task": TASK,
        "follower_port": args.follower_port,
        "front_camera": args.front_camera,
        "wrist_camera": args.wrist_camera,
        "camera_stats": {"front": front_stats, "wrist": wrist_stats},
        "state_deg": dict(zip(JOINT_NAMES, map(float, state), strict=True)),
        "torque_flags": {name: int(value) for name, value in torque.items()},
        "probes": probes,
        "all_outputs_finite": all(
            not probe["contains_nan"] and not probe["contains_inf"] for probe in probes
        ),
        "motion_command_sent": False,
        "safety": (
            "Bus reads only. robot.connect(), send_action(), goal-position writes, "
            "torque writes, homing, and teleoperation were not called."
        ),
    }
    output_path = args.output_dir / "live_probe.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("NO ROBOT MOTION COMMAND WAS SENT")


if __name__ == "__main__":
    main()
