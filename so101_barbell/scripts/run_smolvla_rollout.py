#!/usr/bin/env python
"""SmolVLA/PI0.5 在左侧 SO-101 从臂上的连续闭环推理。

用途：
    读取顶部/腕部相机和真实关节状态，让模型连续预测并控制从臂。

安全机制：
    默认只有持续按住空格才发送动作；多轮自动评测需显式授权，RESET 时关闭扭矩。
    Esc 始终退出并关闭扭矩。
    动作还会经过单步变化限制、校准软限位和电机跟踪失败检测。

输出：
    ``--output`` 指定的 JSON 诊断日志。该 JSON 只用于分析，不是训练数据集。
    需要采集可训练的策略轨迹时，请运行 record_barbell_policy_rollouts.sh。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from threading import Event

import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import Cv2Rotation, OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
# 新版 Evo-RL 把 SO100/SO101 合并为 so_follower；旧版 LeRobot 仍使用
# so101_follower。兼容两种导入路径，避免维护两套几乎相同的安全执行器。
try:
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.robots.so_follower.so_follower import SO101Follower
except ImportError:
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
    from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.utils.robot_utils import precise_sleep


# ---------- 已确认的硬件、任务和动作顺序 ----------
# 动作顺序必须与 checkpoint 的 6 维 action 特征顺序一致。
FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_FOLLOWER_SERIAL-if00"
FRONT_CAMERA = "/dev/v4l/by-path/pci-0000:05:00.0-usb-0:2:1.0-video-index0"
WRIST_CAMERA = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:4.4:1.0-video-index0"
RIGHT_WRIST_CAMERA = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:3.4:1.0-video-index0"
CALIBRATION_DIR = Path(
    "/home/user/.cache/huggingface/lerobot/calibration/robots/so101_follower"
)
TASK = "Lift the blue barbell with the left arm and place it in the orange tray"
ACTION_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

_original_enable_torque = FeetechMotorsBus.enable_torque
_original_disable_torque = FeetechMotorsBus.disable_torque


def _enable_torque_with_retries(self, motors=None, num_retry=0):
    """把底层扭矩启用写入的最少重试次数提高到 5。"""
    return _original_enable_torque(self, motors, num_retry=max(num_retry, 5))


def _disable_torque_with_retries(self, motors=None, num_retry=0):
    """退出时尽力关闭所有电机扭矩，降低偶发串口丢包影响。"""
    return _original_disable_torque(self, motors, num_retry=max(num_retry, 5))


# 只在本进程中替换方法，不会修改安装目录中的 LeRobot 源码。
FeetechMotorsBus.enable_torque = _enable_torque_with_retries
FeetechMotorsBus.disable_torque = _disable_torque_with_retries


def parse_args() -> argparse.Namespace:
    """解析模型、输出、安全限速、控制频率和跟踪超时。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Optional active-execution time limit in seconds; omitted means unlimited.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help=(
            "Number of labeled evaluation episodes. Press s for success; an episode "
            "that reaches --duration-s is labeled failure (default: 1)."
        ),
    )
    parser.add_argument(
        "--auto-execute-episodes",
        action="store_true",
        help=(
            "Run each evaluation episode automatically, like the RL rollout recorder. "
            "Space is then used only to confirm RESET between episodes."
        ),
    )
    parser.add_argument(
        "--allow-long-duration",
        action="store_true",
        help="Explicitly authorize a duration above 10 seconds after short tests pass.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-step-deg", type=float, default=5.0)
    parser.add_argument(
        "--replan-every-steps",
        type=int,
        default=10,
        help=(
            "Discard the unexecuted action-chunk tail and infer again from the latest "
            "camera/state observation after this many executed actions (default: 10)."
        ),
    )
    parser.add_argument(
        "--max-target-outside-soft-limit-deg",
        type=float,
        default=2.0,
        help="Reject a model target this far outside a calibration-derived soft limit.",
    )
    parser.add_argument(
        "--tracking-failure-cycles",
        type=int,
        default=3,
        help="Deprecated compatibility option; time-based tracking is used instead.",
    )
    parser.add_argument("--tracking-failure-timeout-s", type=float, default=2.0)
    parser.add_argument(
        "--use-right-wrist-camera",
        action="store_true",
        help=(
            "Provide the real right-wrist image to PI0.5. Keep disabled for checkpoints "
            "trained from the historical two-camera barbell dataset."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """连接真实设备，运行带死手开关的 SmolVLA/PI0.5 闭环控制。"""
    # ---------- 1. 执行授权与参数约束 ----------
    args = parse_args()
    if not args.execute:
        raise SystemExit("Execution gate closed: pass --execute after the physical safety confirmation.")
    if args.duration_s is not None and args.duration_s <= 0:
        raise SystemExit("--duration-s must be positive when provided.")
    if args.num_episodes <= 0:
        raise SystemExit("--num-episodes must be positive.")
    if args.num_episodes > 1 and args.duration_s is None:
        raise SystemExit("--duration-s is required when --num-episodes is greater than 1.")
    # 首次真实测试曾两次把建议的 3 秒手动改成 180 秒。长时间执行必须增加
    # 独立显式授权，避免修改一个数字就绕过短测流程。
    if args.duration_s is not None and args.duration_s > 10 and not args.allow_long_duration:
        raise SystemExit(
            "Duration safety gate closed: durations above 10 seconds require "
            "--allow-long-duration after short tests pass."
        )
    if not 0 < args.max_step_deg <= 5.0:
        raise SystemExit("--max-step-deg must be in (0, 5.0].")
    if not 0 <= args.max_target_outside_soft_limit_deg <= 5.0:
        raise SystemExit("--max-target-outside-soft-limit-deg must be in [0, 5.0].")
    if args.fps != 30.0:
        raise SystemExit("--fps must be 30 to match the recorded dataset.")
    if args.tracking_failure_cycles < 2:
        raise SystemExit("--tracking-failure-cycles must be at least 2.")
    if args.tracking_failure_timeout_s < 1.0:
        raise SystemExit("--tracking-failure-timeout-s must be at least 1.0.")

    from pynput import keyboard

    # ---------- 2. 先加载模型，再启用机械臂 ----------
    # 如果 checkpoint 加载失败，机械臂此时尚未连接或锁定。
    device = torch.device("cuda")
    # 先读 checkpoint 类型，再由 LeRobot 工厂选择 SmolVLA 或 PI0.5 类。
    # local_files_only 防止真实执行前因网络访问而长时间卡住。
    policy_config = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
    policy_config.device = str(device)
    policy_class = get_policy_class(policy_config.type)
    print(f"Loading {policy_config.type} checkpoint before enabling follower torque...")
    policy = policy_class.from_pretrained(
        args.checkpoint,
        config=policy_config,
        strict=True,
        local_files_only=True,
    ).to(device).eval()
    if not 1 <= args.replan_every_steps <= policy.config.n_action_steps:
        raise SystemExit(
            "--replan-every-steps must be between 1 and "
            f"the policy chunk length ({policy.config.n_action_steps})."
        )
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # ---------- 3. 构造与各模型训练接口一致的相机配置 ----------
    # PI0.5 使用标准三相机名称；SmolVLA 继续使用旧数据的 front/wrist 名称。
    base_camera = OpenCVCameraConfig(
            index_or_path=Path(FRONT_CAMERA),
            width=640,
            height=480,
            fps=30,
            rotation=Cv2Rotation.ROTATE_180,
            fourcc="MJPG",
        )
    left_wrist_camera = OpenCVCameraConfig(
            index_or_path=Path(WRIST_CAMERA),
            width=640,
            height=480,
            fps=30,
            fourcc="MJPG",
        )
    if policy_config.type == "pi05":
        cameras = {
            "base_0_rgb": base_camera,
            "left_wrist_0_rgb": left_wrist_camera,
        }
        # 当前 22K checkpoint 使用旧双相机数据训练，训练时缺少的右腕图像由
        # PI0.5 自动补成 masked 空输入。默认继续保持这个分布；只有未来使用
        # 三相机数据重新训练后，才应显式传入 --use-right-wrist-camera。
        if args.use_right_wrist_camera:
            cameras["right_wrist_0_rgb"] = OpenCVCameraConfig(
                index_or_path=Path(RIGHT_WRIST_CAMERA),
                width=640,
                height=480,
                fps=30,
                fourcc="MJPG",
            )
    else:
        cameras = {"front": base_camera, "wrist": left_wrist_camera}
    # max_relative_target 是 LeRobot 内部最后一道单步相对变化保护。
    robot = SO101Follower(
        SO101FollowerConfig(
            port=FOLLOWER_PORT,
            id="SO101_FOLLOWER_ID",
            use_degrees=True,
            cameras=cameras,
            calibration_dir=CALIBRATION_DIR,
            max_relative_target=args.max_step_deg,
            disable_torque_on_disconnect=True,
        )
    )
    # build_inference_frame 需要 LeRobot 数据集格式的特征描述。
    action_features = hw_to_dataset_features(robot.action_features, "action")
    observation_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **observation_features}

    # 把本机械臂的校准范围转换为策略使用的角度单位，并在物理端点内保留小余量，
    # 避免浮点取整把目标恰好压在硬限位上。
    soft_limits = {}
    for name, motor in robot.bus.motors.items():
        calibration = robot.calibration[name]
        if name == "gripper":
            low, high, margin = 0.0, 100.0, 1.0
        else:
            resolution = robot.bus.model_resolution_table[motor.model] - 1
            midpoint = (calibration.range_min + calibration.range_max) / 2
            low = (calibration.range_min - midpoint) * 360 / resolution
            high = (calibration.range_max - midpoint) * 360 / resolution
            margin = 0.5
        soft_limits[f"{name}.pos"] = (low + margin, high - margin)

    # ---------- 4. 键盘死手开关 ----------
    # Event 可在 pynput 线程和控制主线程之间安全共享状态。
    space_held = Event()
    physical_space_down = Event()
    stop_requested = Event()
    success_requested = Event()
    reset_ack_requested = Event()
    reset_space_armed = Event()
    control_phase = {"value": "episode"}

    def on_press(key):
        """执行阶段空格是死手开关；RESET 阶段空格只确认复位完成。"""
        if key == keyboard.Key.space:
            physical_space_down.set()
            if control_phase["value"] == "reset":
                if reset_space_armed.is_set():
                    reset_ack_requested.set()
            elif control_phase["value"] == "episode" and not args.auto_execute_episodes:
                space_held.set()
        elif key == keyboard.Key.esc:
            stop_requested.set()
            space_held.clear()
        elif (
            control_phase["value"] == "episode"
            and (getattr(key, "char", "") or "").lower() == "s"
        ):
            success_requested.set()
            space_held.clear()

    def on_release(key):
        """一旦松开空格，下一周期不再发送新动作。"""
        if key == keyboard.Key.space:
            physical_space_down.clear()
            if not args.auto_execute_episodes:
                space_held.clear()
            if control_phase["value"] == "reset":
                reset_space_armed.set()

    if args.auto_execute_episodes:
        input(
            "AUTO EVALUATION: clear the workspace and place the objects at the "
            "episode start pose, then press Enter..."
        )

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    # ---------- 5. 运行状态与诊断缓存 ----------
    connected = False
    active_elapsed = 0.0
    first_execution_wall_time = None
    session_first_execution_wall_time = None
    records = []
    all_records = []
    episodes = []
    episode_index = 0
    episode_finalized = False
    stop_reason = "not_started"
    previous_command = None
    previous_state_at_send = None
    last_observed_state = None
    last_joint_progress_time = None
    last_tracking_error_sign = np.zeros(5)
    action_index_in_chunk = 0
    steps_since_replan = 0
    replan_count = 0

    def finalize_episode(outcome: str, outcome_source: str, episode_stop_reason: str) -> None:
        """冻结当前轮的标签和统计，并把逐步记录合并到会话报告。"""
        nonlocal episode_finalized
        if episode_finalized:
            return
        record_start = len(all_records)
        for record in records:
            record["episode_index"] = episode_index
            all_records.append(record)
        episodes.append(
            {
                "episode_index": episode_index,
                "outcome": outcome,
                "outcome_source": outcome_source,
                "stop_reason": episode_stop_reason,
                "active_execution_s": active_elapsed,
                "wall_time_since_first_execution_s": (
                    time.perf_counter() - first_execution_wall_time
                    if first_execution_wall_time is not None
                    else 0.0
                ),
                "commands_sent": len(records),
                "replan_count": replan_count,
                "record_start_index": record_start,
                "record_end_index_exclusive": len(all_records),
            }
        )
        episode_finalized = True

    def wait_for_manual_reset() -> bool:
        """关闭扭矩等待人工复位；返回 False 表示用户按 Esc 终止会话。"""
        control_phase["value"] = "reset"
        space_held.clear()
        reset_ack_requested.clear()
        reset_space_armed.clear()
        if not physical_space_down.is_set():
            reset_space_armed.set()
        robot.bus.disable_torque(num_retry=5)
        print("RESET: follower torque is OFF. Reset the arm and scene safely.")
        print("Press SPACE once when reset is complete; ESC aborts the session.")
        while not reset_ack_requested.wait(timeout=0.05):
            if stop_requested.is_set():
                return False
        # RESET 中的这次空格只作确认，绝不能顺带驱动下一轮。
        space_held.clear()
        control_phase["value"] = "transition"
        robot.bus.enable_torque(num_retry=5)
        torque = robot.bus.sync_read("Torque_Enable", normalize=False, num_retry=5)
        if not all(int(value) == 1 for value in torque.values()):
            raise RuntimeError(f"Not all motors enabled torque after reset: {torque}")
        robot.get_observation()
        return True

    def prepare_next_episode() -> None:
        """清空跨轮动作队列和跟踪状态，确保新一轮只使用新观测。"""
        nonlocal episode_index, episode_finalized, active_elapsed
        nonlocal first_execution_wall_time, records, previous_command
        nonlocal previous_state_at_send, last_observed_state, last_joint_progress_time
        nonlocal last_tracking_error_sign, action_index_in_chunk, steps_since_replan
        nonlocal replan_count
        episode_index += 1
        episode_finalized = False
        active_elapsed = 0.0
        first_execution_wall_time = None
        records = []
        previous_command = None
        previous_state_at_send = None
        last_observed_state = None
        last_joint_progress_time = None
        last_tracking_error_sign = np.zeros(5)
        action_index_in_chunk = 0
        steps_since_replan = 0
        replan_count = 0
        success_requested.clear()
        reset_ack_requested.clear()
        policy.reset()
        torch.manual_seed(1000 + episode_index)
        control_phase["value"] = "episode"
        if args.auto_execute_episodes:
            space_held.set()
        print(f"EPISODE {episode_index + 1}/{args.num_episodes} READY.")
        if args.auto_execute_episodes:
            print("AUTO EXECUTION STARTED; s=success; ESC=abort.")
        else:
            print("RELEASE SPACE, then HOLD SPACE to execute; s=success; ESC=abort.")

    # ---------- 6. 连接设备并确认扭矩确实启用 ----------
    try:
        robot.connect()
        connected = True
        # USB/Feetech 总线刚连接后的第一包偶尔会丢失。这里只重试只读的
        # Torque_Enable 握手；仍未成功就禁止进入 READY，更不会发送动作。
        torque = robot.bus.sync_read("Torque_Enable", normalize=False, num_retry=5)
        if not all(int(value) == 1 for value in torque.values()):
            raise RuntimeError(f"Not all motors enabled torque: {torque}")

        robot.get_observation()
        policy.reset()
        torch.manual_seed(1000)
        if args.auto_execute_episodes:
            space_held.set()
        print("Calibration-derived soft limits:")
        for name in ACTION_NAMES:
            low, high = soft_limits[name]
            print(f"  {name:22s} [{low:+8.2f}, {high:+8.2f}]")
        print(f"EPISODE 1/{args.num_episodes} READY.")
        if args.auto_execute_episodes:
            print("AUTO EXECUTION STARTED; s=success; ESC aborts.")
        else:
            print("HOLD SPACE to execute; RELEASE pauses; s=success; ESC aborts.")
        if args.duration_s is None:
            print("No time limit. RELEASE SPACE to pause; ESC aborts and disables follower torque.")
        else:
            print(
                f"No s within {args.duration_s:.0f} active seconds = FAILURE. "
                "Torque then turns off for manual RESET."
            )

        # ---------- 7. 30Hz 闭环控制主循环 ----------
        while True:
            if stop_requested.is_set():
                stop_reason = "escape"
                if first_execution_wall_time is not None or records:
                    finalize_episode("aborted", "escape", "escape")
                break
            if success_requested.is_set() and (
                args.duration_s is None or active_elapsed < args.duration_s
            ):
                finalize_episode("success", "key_s", "success_key")
                print(
                    f"EPISODE {episode_index + 1}/{args.num_episodes}: SUCCESS "
                    f"({len(records)} commands, {active_elapsed:.2f}s active)."
                )
                if episode_index + 1 >= args.num_episodes:
                    stop_reason = "evaluation_complete"
                    break
                if not wait_for_manual_reset():
                    stop_reason = "escape_during_reset"
                    break
                prepare_next_episode()
                continue
            if args.duration_s is not None and active_elapsed >= args.duration_s:
                finalize_episode("failure", "timeout", "episode_time_limit")
                print(
                    f"EPISODE {episode_index + 1}/{args.num_episodes}: FAILURE (timeout; "
                    f"{len(records)} commands, {active_elapsed:.2f}s active)."
                )
                if episode_index + 1 >= args.num_episodes:
                    stop_reason = "evaluation_complete"
                    break
                if not wait_for_manual_reset():
                    stop_reason = "escape_during_reset"
                    break
                prepare_next_episode()
                continue
            if not space_held.wait(timeout=0.05):
                continue
            if first_execution_wall_time is None:
                first_execution_wall_time = time.perf_counter()
                if session_first_execution_wall_time is None:
                    session_first_execution_wall_time = first_execution_wall_time
            elapsed = active_elapsed

            loop_started = time.perf_counter()
            observation = robot.get_observation()
            current = np.asarray([float(observation[name]) for name in ACTION_NAMES])

            # 舵机不一定在每个 33ms 周期都有可测位移。只有某关节持续存在明显误差，
            # 且超过真实时间窗口都没有进展时才判定卡滞；夹爪允许接触物体，不参与判定。
            tracking = None
            if previous_command is not None and previous_state_at_send is not None:
                tracking_now = time.perf_counter()
                if last_observed_state is None:
                    last_observed_state = current.copy()
                    last_joint_progress_time = np.full(5, tracking_now)
                observed_step = np.abs(current[:5] - last_observed_state[:5])
                requested = np.abs(previous_command[:5] - previous_state_at_send[:5])
                signed_error = previous_command[:5] - current[:5]
                error_now = np.abs(signed_error)
                error_sign = np.sign(signed_error)
                for i in range(5):
                    no_meaningful_error = error_now[i] <= 0.75
                    direction_changed = (
                        last_tracking_error_sign[i] != 0
                        and error_sign[i] != last_tracking_error_sign[i]
                    )
                    if (
                        observed_step[i] >= 0.05
                        or no_meaningful_error
                        or direction_changed
                    ):
                        last_joint_progress_time[i] = tracking_now
                seconds_without_progress = tracking_now - last_joint_progress_time
                stalled = (error_now > 0.75) & (
                    seconds_without_progress >= args.tracking_failure_timeout_s
                )
                stalled_joints = [ACTION_NAMES[i] for i in np.flatnonzero(stalled)]
                tracking = {
                    "requested_deg": requested.tolist(),
                    "remaining_error_deg": error_now.tolist(),
                    "observed_step_deg": observed_step.tolist(),
                    "seconds_without_progress": seconds_without_progress.tolist(),
                    "stalled_joints": stalled_joints,
                }
                last_observed_state = current.copy()
                last_tracking_error_sign = error_sign
                if bool(stalled.any()):
                    stop_reason = "target_tracking_failure"
                    break
            elif last_observed_state is None:
                last_observed_state = current.copy()
                last_joint_progress_time = np.full(5, time.perf_counter())

            # ---------- 7.1 把最新真实观测转换成模型输入 ----------
            frame = build_inference_frame(
                observation=observation,
                ds_features=dataset_features,
                device=device,
                task=TASK,
                robot_type="so101_follower",
            )
            processed = preprocessor(frame)
            # PI0.5/SmolVLA 会把一次预测得到的整个动作 chunk 缓存在策略内部。
            # 每执行指定步数就丢弃尚未执行的尾部，让下一次 select_action 使用
            # 本周期刚采集的相机图像和关节状态重新生成动作，实现滚动闭环规划。
            replanned = steps_since_replan >= args.replan_every_steps
            if replanned:
                policy.reset()
                steps_since_replan = 0
                action_index_in_chunk = 0
                replan_count += 1
            torch.cuda.synchronize()
            inference_started = time.perf_counter()
            with torch.inference_mode():
                # 与 LeRobot 官方 SmolVLA 推理一致：动作队列为空时预测一个新 chunk，
                # 每个控制周期只取队列中的下一个动作。
                action_tensor = postprocessor(policy.select_action(processed))
            torch.cuda.synchronize()
            inference_ms = (time.perf_counter() - inference_started) * 1000

            # ---------- 7.2 反归一化、异常跳变检查和安全裁剪 ----------
            model_action = make_robot_action(action_tensor, dataset_features)
            predicted = np.asarray([float(model_action[name]) for name in ACTION_NAMES])
            if not np.isfinite(predicted).all():
                raise RuntimeError("Model produced NaN or Inf.")
            raw_delta = predicted - current
            if float(np.abs(raw_delta).max()) > 20.0:
                raise RuntimeError(
                    f"Model target rejected: raw first-step delta {np.abs(raw_delta).max():.2f}°."
                )
            # 不能仅把越界目标裁到软限位后继续执行。若模型持续要求一个不可达位置，
            # 逐周期裁剪会把关节一直压在边界上，raw_delta 还会不断累积。允许最多
            # 2° 数值余量；明显越界时立即停机并指出具体关节。
            outside_soft_limits = []
            for i, name in enumerate(ACTION_NAMES):
                low, high = soft_limits[name]
                if predicted[i] < low - args.max_target_outside_soft_limit_deg:
                    outside_soft_limits.append(
                        f"{name}={predicted[i]:.2f}° below {low:.2f}°"
                    )
                elif predicted[i] > high + args.max_target_outside_soft_limit_deg:
                    outside_soft_limits.append(
                        f"{name}={predicted[i]:.2f}° above {high:.2f}°"
                    )
            if outside_soft_limits:
                raise RuntimeError(
                    "Model target rejected outside soft limits: " + "; ".join(outside_soft_limits)
                )
            command = current + np.clip(raw_delta, -args.max_step_deg, args.max_step_deg)
            for i, name in enumerate(ACTION_NAMES):
                low, high = soft_limits[name]
                command[i] = np.clip(command[i], low, high)
            action = dict(zip(ACTION_NAMES, map(float, command), strict=True))
            # ---------- 7.3 下发动作并回读 Goal_Position ----------
            # 回读用于确认串口写入确实到达电机寄存器。
            # 推理可能耗时数百毫秒；期间若松开空格、按 s 或 Esc，本步也必须取消。
            if (
                not space_held.is_set()
                or success_requested.is_set()
                or stop_requested.is_set()
            ):
                continue
            sent = robot.send_action(action)
            sent_vector = np.asarray([float(sent[name]) for name in ACTION_NAMES])
            goal_register = robot.bus.sync_read("Goal_Position")
            goal_register_vector = np.asarray(
                [float(goal_register[name.removesuffix(".pos")]) for name in ACTION_NAMES]
            )
            goal_register_error = np.abs(goal_register_vector - sent_vector)
            if float(goal_register_error.max()) > 0.5:
                raise RuntimeError(
                    "Goal_Position readback mismatch after send_action: "
                    f"maximum error {goal_register_error.max():.2f}°."
                )
            queue_position = action_index_in_chunk
            action_index_in_chunk += 1
            steps_since_replan += 1
            # ---------- 7.4 保存本周期诊断信息 ----------
            records.append(
                {
                    "elapsed_s": elapsed,
                    "inference_ms": inference_ms,
                    "action_index_in_chunk": queue_position,
                    "replanned_from_latest_observation": replanned,
                    "current_deg": dict(zip(ACTION_NAMES, map(float, current), strict=True)),
                    "predicted_deg": dict(zip(ACTION_NAMES, map(float, predicted), strict=True)),
                    "sent_deg": {name: float(value) for name, value in sent.items()},
                    "goal_position_readback_deg": dict(
                        zip(ACTION_NAMES, map(float, goal_register_vector), strict=True)
                    ),
                    "max_goal_position_readback_error_deg": float(goal_register_error.max()),
                    "tracking_previous_command": tracking,
                    "max_abs_raw_delta_deg": float(np.abs(raw_delta).max()),
                    "max_abs_sent_delta_deg": float(
                        max(abs(float(sent[name]) - current[i]) for i, name in enumerate(ACTION_NAMES))
                    ),
                }
            )
            previous_state_at_send = current.copy()
            previous_command = sent_vector
            print(
                f"episode={episode_index + 1:02d} command={len(records):02d} "
                f"elapsed={elapsed:.2f}s "
                f"inference={inference_ms:.0f}ms "
                f"chunk_step={queue_position:02d}/{policy.config.n_action_steps - 1:02d} "
                f"replan={'yes' if replanned else 'no'} "
                f"raw_delta={np.abs(raw_delta).max():.2f}° "
                f"sent_delta<={args.max_step_deg:.2f}°"
            )
            precise_sleep(max(1.0 / args.fps - (time.perf_counter() - loop_started), 0.0))
            active_elapsed += time.perf_counter() - loop_started
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        if first_execution_wall_time is not None or records:
            finalize_episode("aborted", "keyboard_interrupt", stop_reason)
    except Exception as error:
        # 保留原异常继续抛出，但先写入准确的停止原因；这样报告不会再误写成
        # not_started，事后能够区分模型越界、跟踪失败和硬件通信异常。
        stop_reason = f"safety_exception:{type(error).__name__}:{error}"
        if first_execution_wall_time is not None or records:
            finalize_episode("failure", "safety_exception", stop_reason)
        raise
    finally:
        # ---------- 8. 无论正常/异常退出都停止监听、断开设备并写报告 ----------
        control_phase["value"] = "done"
        space_held.clear()
        if not episode_finalized and (first_execution_wall_time is not None or records):
            if stop_reason == "target_tracking_failure":
                finalize_episode("failure", "safety_gate", stop_reason)
            else:
                finalize_episode("aborted", "session_stopped", stop_reason)
        listener.stop()
        if connected:
            try:
                robot.disconnect()
            except Exception as error:
                print(f"Warning during disconnect: {error}")
                if robot.bus.is_connected:
                    try:
                        robot.bus.disable_torque(num_retry=5)
                    finally:
                        robot.bus.disconnect(disable_torque=False)
        report = {
            "checkpoint": str(args.checkpoint),
            "duration_limit_s": args.duration_s,
            "requested_num_episodes": args.num_episodes,
            "completed_num_episodes": sum(
                episode["outcome"] in {"success", "failure"} for episode in episodes
            ),
            "success_count": sum(episode["outcome"] == "success" for episode in episodes),
            "failure_count": sum(episode["outcome"] == "failure" for episode in episodes),
            "success_rate": (
                sum(episode["outcome"] == "success" for episode in episodes)
                / sum(episode["outcome"] in {"success", "failure"} for episode in episodes)
                if any(episode["outcome"] in {"success", "failure"} for episode in episodes)
                else None
            ),
            "episode_controls": {
                "success_key": "s",
                "timeout_outcome": "failure",
                "reset_confirmation_key": "space",
                "reset_disables_follower_torque": True,
                "auto_execute_episodes": args.auto_execute_episodes,
                "execution_requires_space_release_and_repress_after_reset": (
                    not args.auto_execute_episodes
                ),
            },
            "long_duration_explicitly_authorized": args.allow_long_duration,
            "active_execution_s": sum(
                float(episode["active_execution_s"]) for episode in episodes
            ),
            "wall_time_since_first_execution_s": (
                time.perf_counter() - session_first_execution_wall_time
                if session_first_execution_wall_time is not None
                else 0.0
            ),
            "control_fps": args.fps,
            "policy_action_queue_steps": policy.config.n_action_steps,
            "replan_every_steps": args.replan_every_steps,
            "replan_count": sum(int(episode["replan_count"]) for episode in episodes),
            "policy_api": "select_action",
            "max_step_deg": args.max_step_deg,
            "max_target_outside_soft_limit_deg": args.max_target_outside_soft_limit_deg,
            "right_wrist_camera_enabled": args.use_right_wrist_camera,
            "soft_limits": {
                name: {"min": low, "max": high} for name, (low, high) in soft_limits.items()
            },
            "tracking_failure_cycles_compatibility_ignored": args.tracking_failure_cycles,
            "tracking_failure_timeout_s": args.tracking_failure_timeout_s,
            "commands_sent": len(all_records),
            "stop_reason": stop_reason,
            "episodes": episodes,
            "records": all_records,
            "follower_torque_off_attempted": connected,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            f"Stopped: {stop_reason}; episodes: {len(episodes)}/{args.num_episodes}; "
            f"success={report['success_count']} failure={report['failure_count']}; "
            f"commands sent: {len(all_records)}"
        )
        print("Follower torque-off disconnect attempted.")


if __name__ == "__main__":
    main()
