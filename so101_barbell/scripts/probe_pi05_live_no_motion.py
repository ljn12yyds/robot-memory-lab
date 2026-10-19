#!/usr/bin/env python3
"""使用真实 SO101 观测测试 PI0.5，但绝不向机械臂发送动作。

本脚本用于真实执行前的最后一道只读检查：
1. 读取顶部、左腕、右腕三路相机；
2. 只读获取左从臂六个关节的位置；
3. 加载本地 PI0.5 checkpoint 并预测一个动作 chunk；
4. 保存三张模型输入图和动作统计，方便人工核对。

安全保证：脚本不调用 ``robot.connect()`` 或 ``robot.send_action()``，
只直接连接电机总线读取 Present_Position，退出时不会改变扭矩状态。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy
# Evo-RL 使用新版 LeRobot：SO100/SO101 共用 ``so_follower`` 实现，
# ``SO101Follower*`` 在这里是保持硬件语义清晰的类型别名。
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SO101Follower


# SO101 数据集和动作使用的固定关节顺序；顺序错误会导致动作对应到错误电机。
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
TASK = "Lift the blue barbell with the left arm and place it in the orange tray"
# 这份校准由当前左从臂 SO101_FOLLOWER_ID 采集。Evo-RL 新版类名是 so_follower，
# 但应继续读取旧项目已经验证过的 so101_follower 校准目录。
CALIBRATION_DIR = Path(
    "/home/user/.cache/huggingface/lerobot/calibration/robots/so101_follower"
)


def parse_args() -> argparse.Namespace:
    """集中定义设备路径，便于换 USB 口后从命令行覆盖。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="cuda")
    parser.add_argument(
        "--follower-port",
        default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_FOLLOWER_SERIAL-if00",
    )
    parser.add_argument(
        "--base-camera",
        default="/dev/v4l/by-path/pci-0000:05:00.0-usb-0:2:1.0-video-index0",
    )
    parser.add_argument(
        "--left-wrist-camera",
        default="/dev/v4l/by-path/pci-0000:00:14.0-usb-0:4.4:1.0-video-index0",
    )
    parser.add_argument(
        "--right-wrist-camera",
        default="/dev/v4l/by-path/pci-0000:00:14.0-usb-0:3.4:1.0-video-index0",
    )
    parser.add_argument("--warmup-frames", type=int, default=30)
    return parser.parse_args()


def open_camera(path: str) -> cv2.VideoCapture:
    """按采集时的 MJPG、640×480、30 FPS 参数打开相机。"""
    capture = cv2.VideoCapture(path, cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_FPS, 30)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开相机：{path}")
    return capture


def read_warmed_frame(capture: cv2.VideoCapture, count: int) -> np.ndarray:
    """丢弃启动初期曝光不稳定的画面，只保留最后一张有效帧。"""
    frame = None
    for _ in range(count):
        ok, candidate = capture.read()
        if ok and candidate is not None:
            frame = candidate
    if frame is None:
        raise RuntimeError("相机预热后仍未返回有效画面")
    return frame


def image_tensor(bgr: np.ndarray) -> torch.Tensor:
    """把 OpenCV 的 BGR uint8 转换为 LeRobot 使用的 CHW RGB [0,1]。"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255.0)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 检查所有输入，避免模型加载后才发现设备路径错误 ----------
    required = [
        args.checkpoint / "model.safetensors",
        Path(args.follower_port),
        Path(args.base_camera),
        Path(args.left_wrist_camera),
        Path(args.right_wrist_camera),
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    # ---------- 2. 读取三路真实图像 ----------
    captures = {
        "base": open_camera(args.base_camera),
        "left_wrist": open_camera(args.left_wrist_camera),
        "right_wrist": open_camera(args.right_wrist_camera),
    }
    try:
        frames = {name: read_warmed_frame(cap, args.warmup_frames) for name, cap in captures.items()}
    finally:
        for capture in captures.values():
            capture.release()

    # 顶部相机在既有哑铃数据中旋转了 180°，现场推理必须保持一致。
    frames["base"] = cv2.rotate(frames["base"], cv2.ROTATE_180)
    for name, frame in frames.items():
        cv2.imwrite(str(args.output_dir / f"{name}.jpg"), frame)

    # ---------- 3. 只读获取左从臂关节状态 ----------
    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.follower_port,
            id="SO101_FOLLOWER_ID",
            use_degrees=True,
            cameras={},
            calibration_dir=CALIBRATION_DIR,
        )
    )
    robot.bus.connect()
    try:
        position = robot.bus.sync_read("Present_Position")
        torque = robot.bus.sync_read("Torque_Enable", normalize=False)
    finally:
        # disable_torque=False：只读探针不得擅自改变用户原来的扭矩状态。
        robot.bus.disconnect(disable_torque=False)

    state = torch.tensor([float(position[name]) for name in JOINT_NAMES], dtype=torch.float32)

    # ---------- 4. 以 BF16 严格加载本地 PI0.5 ----------
    config = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
    config.device = args.gpu
    config.dtype = "bfloat16"
    policy = PI05Policy.from_pretrained(
        args.checkpoint,
        config=config,
        strict=True,
        local_files_only=True,
    ).eval()
    # 必须从“当前 checkpoint”加载处理器，而不能临时创建空统计处理器。
    # 基础权重输出的是归一化动作；训练 checkpoint 则在这两个 JSON/状态文件中
    # 保存了本任务数据的 quantile 统计。postprocessor 会据此把网络输出恢复为
    # SO101 的真实 6 维关节角度，否则归一化数值绝不能直接发送给机械臂。
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.checkpoint),
    )

    # PI0.5 基础权重期望这三个标准相机名称；SO101 的6维状态由处理器补齐到32维。
    observation = {
        "observation.images.base_0_rgb": image_tensor(frames["base"]),
        "observation.images.left_wrist_0_rgb": image_tensor(frames["left_wrist"]),
        "observation.images.right_wrist_0_rgb": image_tensor(frames["right_wrist"]),
        "observation.state": state,
        "task": TASK,
        "robot_type": "so101_follower",
    }
    processed = preprocessor(observation)

    # ---------- 5. 仅预测并记录，不执行 ----------
    torch.manual_seed(42)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        chunk = postprocessor(policy.predict_action_chunk(processed)).detach().float().cpu()
    torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - started) * 1000

    # SO101 桥接会使用前6维；这里单独保存它们，方便检查数值是否合理。
    so101_chunk = chunk[0, :, : len(JOINT_NAMES)]
    report = {
        "checkpoint": str(args.checkpoint),
        "task": TASK,
        "inference_ms": inference_ms,
        "chunk_shape": list(chunk.shape),
        "so101_chunk_shape": list(so101_chunk.shape),
        "all_finite": bool(torch.isfinite(chunk).all()),
        "current_state": dict(zip(JOINT_NAMES, state.tolist(), strict=True)),
        "torque_enabled": {name: bool(value) for name, value in torque.items()},
        "first_predicted_action": dict(zip(JOINT_NAMES, so101_chunk[0].tolist(), strict=True)),
        "predicted_min": dict(zip(JOINT_NAMES, so101_chunk.min(dim=0).values.tolist(), strict=True)),
        "predicted_max": dict(zip(JOINT_NAMES, so101_chunk.max(dim=0).values.tolist(), strict=True)),
        "actions_sent": 0,
    }
    report_path = args.output_dir / "probe.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"只读推理完成，未发送动作。报告：{report_path}")


if __name__ == "__main__":
    main()
