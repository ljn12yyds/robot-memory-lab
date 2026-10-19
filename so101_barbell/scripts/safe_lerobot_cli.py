#!/usr/bin/env python3
"""LeRobot 0.4.3 的本地安全与录制兼容层。

主要职责：
    1. 电机扭矩写入失败时进行有限次数重试；
    2. 人工遥操作采集时，第一条 Leader 动作与从臂姿态差异过大则拒绝执行；
    3. 数据集保存 send_action() 实际下发的动作，而非裁剪前目标；
    4. RL 采集时保存成功/失败标签，并对策略动作应用校准软限位；
    5. 修复停止采集时保存空 Episode 的竞态问题。
    6. 策略运行时让 Leader 跟随 Follower，Space 后切换到人工纠正。

该文件是底层公共入口，不建议直接运行；请使用 record_barbell_dataset.sh
或 record_barbell_policy_rollouts.sh。
"""

from __future__ import annotations

import atexit
import csv
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch

from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.robots.so101_follower.so101_follower import SO101Follower


# ---------- 1. 保存原始方法 ----------
# 后续 monkey patch 只影响当前 Python 进程，便于在不修改上游 API 的情况下修复行为。
_original_enable_torque = FeetechMotorsBus.enable_torque
_original_disable_torque = FeetechMotorsBus.disable_torque
_original_connect = SO101Follower.connect
_original_send_action = SO101Follower.send_action
_original_get_observation = SO101Follower.get_observation

# ---------- 2. 采集过程共享状态 ----------
# 这些变量在 recorder 主线程和键盘监听线程之间传递当前阶段、动作和标签。
_record_mode = False
_first_record_action = True
_last_sent_action: dict[str, Any] | None = None
_latest_follower_state: dict[str, float] | None = None
_recording_episode_active = False
_safe_stop_requested = False
_current_phase = "IDLE"
_current_episode_index: int | str = "?"
_rl_collection = os.environ.get("BARBELL_RL_COLLECTION") == "1"
_rl_episode_outcome: dict[str, Any] | None = None
# 混合控制状态：策略阶段主臂跟随从臂；按 Space 后切换为人工遥操作。
_hybrid_teleop = None
_human_intervention = False
_episode_had_intervention = False
_intervention_start_frame: int | None = None
_leader_follow_enabled = False
_current_dataset = None
_hybrid_lock = threading.RLock()
_sidecar_file = None
_sidecar_writer = None
_sidecar_rows_since_flush = 0

_JOINT_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


def _close_sidecar() -> None:
    """进程退出时刷新并关闭诊断 CSV，避免最后若干行丢失。"""
    global _sidecar_file
    if _sidecar_file is not None:
        _sidecar_file.flush()
        _sidecar_file.close()
        _sidecar_file = None


atexit.register(_close_sidecar)


def _save_rl_episode_label(episode_index: int, outcome: dict[str, Any]) -> None:
    """原子更新一条已经正式保存的策略 rollout 成败标签。"""
    path_string = os.environ.get("BARBELL_RL_LABELS")
    if not _rl_collection or not path_string:
        return

    path = Path(path_string)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {"schema_version": 1, "episodes": []}

    # 先移除同编号旧记录，使重试写入保持幂等，不会出现重复 Episode。
    episodes = [
        row for row in payload.get("episodes", []) if int(row["episode_index"]) != episode_index
    ]
    episodes.append(
        {
            "episode_index": episode_index,
            "success": bool(outcome["success"]),
            "termination": str(outcome["termination"]),
            "human_intervention": bool(outcome.get("human_intervention", False)),
            "intervention_start_frame": outcome.get("intervention_start_frame"),
            "recorded_at_unix_s": time.time(),
        }
    )
    payload["episodes"] = sorted(episodes, key=lambda row: int(row["episode_index"]))
    # 先写临时文件再 replace；即使突然断电，也不会留下半截 JSON。
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_sidecar_sample(
    raw_action: dict[str, Any],
    sent_action: dict[str, Any],
) -> None:
    """把原始动作、实际动作和状态写入独立 CSV，不污染训练特征。"""
    global _sidecar_file, _sidecar_writer, _sidecar_rows_since_flush
    path_string = os.environ.get("BARBELL_DIAGNOSTIC_LOG")
    if not _record_mode or not path_string:
        return

    # 第一个样本到来时再创建文件，避免连接失败也留下空日志。
    if _sidecar_file is None:
        path = Path(path_string)
        path.parent.mkdir(parents=True, exist_ok=True)
        _sidecar_file = path.open("w", newline="", encoding="utf-8")
        fieldnames = [
            "monotonic_ns",
            "phase",
            "episode_index",
            "control_source",
            "human_intervention",
        ]
        for prefix in ("raw", "sent", "state"):
            fieldnames.extend(f"{prefix}.{key}" for key in _JOINT_KEYS)
        _sidecar_writer = csv.DictWriter(_sidecar_file, fieldnames=fieldnames)
        _sidecar_writer.writeheader()

    row: dict[str, Any] = {
        "monotonic_ns": time.monotonic_ns(),
        "phase": _current_phase,
        "episode_index": _current_episode_index,
        "control_source": "human" if _human_intervention else "policy",
        "human_intervention": int(_human_intervention),
    }
    state = _latest_follower_state or {}
    for key in _JOINT_KEYS:
        row[f"raw.{key}"] = raw_action.get(key, "")
        row[f"sent.{key}"] = sent_action.get(key, "")
        row[f"state.{key}"] = state.get(key, "")
    _sidecar_writer.writerow(row)
    _sidecar_rows_since_flush += 1
    if _sidecar_rows_since_flush >= 30:
        _sidecar_file.flush()
        _sidecar_rows_since_flush = 0


def _read_position_dict(bus: FeetechMotorsBus) -> dict[str, float]:
    """以统一的浮点角度字典读取一套 SO101 的六个当前位置。"""
    return {
        name: float(value)
        for name, value in bus.sync_read("Present_Position").items()
    }


def _start_leader_following_follower(robot: SO101Follower, teleop: Any) -> None:
    """检查主从臂起点后，启用主臂扭矩并令其跟随从臂当前姿态。

    该操作只在策略 Episode 开始时执行。若主从差值过大则拒绝给主臂上扭矩，
    防止主臂突然追赶到从臂姿态。
    """
    global _leader_follow_enabled
    if teleop is None or not hasattr(teleop, "bus"):
        raise RuntimeError("Hybrid policy collection requires an SO101 Leader teleoperator.")

    follower = _read_position_dict(robot.bus)
    leader = _read_position_dict(teleop.bus)
    signed = {name: follower[name] - leader[name] for name in follower}
    absolute = {name: abs(value) for name, value in signed.items()}
    worst_name, worst_delta = max(absolute.items(), key=lambda item: item[1])
    threshold = float(os.environ.get("BARBELL_HYBRID_START_MAX_DELTA_DEG", "8.0"))
    details = ", ".join(f"{name}={value:+.2f}°" for name, value in signed.items())
    logging.info("Hybrid start deltas (follower - leader): %s", details)
    if worst_delta > threshold:
        raise RuntimeError(
            "HYBRID START GATE: Leader is not aligned with Follower. "
            f"Worst joint {worst_name} differs by {worst_delta:.2f}°, "
            f"limit is {threshold:.2f}°. Leader torque was not enabled. "
            f"Signed deltas (follower - leader): {details}."
        )

    # 先写入与当前位置一致的目标，再启用扭矩，可避免启用瞬间追赶旧目标。
    teleop.bus.sync_write("Goal_Position", follower)
    teleop.bus.enable_torque(num_retry=5)
    teleop.bus.sync_write("Goal_Position", follower)
    _leader_follow_enabled = True
    logging.info("Hybrid POLICY mode: Leader torque enabled and following Follower.")


def _sync_leader_from_follower_observation(observation: dict[str, Any]) -> None:
    """策略控制期间，把最新从臂关节状态同步写给主臂。"""
    if not _rl_collection:
        return
    with _hybrid_lock:
        if (
            not _leader_follow_enabled
            or _human_intervention
            or _hybrid_teleop is None
        ):
            return
        targets = {
            key.removesuffix(".pos"): float(observation[key])
            for key in _JOINT_KEYS
            if key in observation
        }
        _hybrid_teleop.bus.sync_write("Goal_Position", targets)


def _enter_human_intervention() -> None:
    """安全地停止主臂跟随、关闭其扭矩，并切换到人工遥操作。"""
    global _human_intervention, _episode_had_intervention, _intervention_start_frame
    global _leader_follow_enabled
    with _hybrid_lock:
        if not _rl_collection or not _recording_episode_active:
            return
        if _human_intervention:
            return
        if _hybrid_teleop is None:
            raise RuntimeError("Cannot enter intervention mode without a connected Leader.")

        # 先关闭同步标志，确保不再有新的主臂目标写入，然后释放主臂供人操作。
        _leader_follow_enabled = False
        try:
            _hybrid_teleop.bus.disable_torque(num_retry=5)
        except BaseException:
            _leader_follow_enabled = True
            raise
        _human_intervention = True
        _episode_had_intervention = True
        # 记录人工接管从当前 Episode 的第几帧开始。后续训练可分别降低
        # 接管前错误策略片段的权重、提高人工纠正片段的权重。
        episode_buffer = (
            None if _current_dataset is None else _current_dataset.episode_buffer
        )
        _intervention_start_frame = (
            0 if episode_buffer is None else int(episode_buffer.get("size", 0))
        )
        print(
            "\n\033[1;35mHUMAN INTERVENTION ACTIVE: Leader torque is off. "
            f"Correction begins at frame {_intervention_start_frame}. "
            "Use teleoperation, then press Right for success or Left for failure."
            "\033[0m\a",
            flush=True,
        )


def _release_leader_after_policy_phase() -> None:
    """结束策略阶段时关闭主臂扭矩，使 Reset 阶段可以安全人工摆放。"""
    global _leader_follow_enabled
    with _hybrid_lock:
        if _hybrid_teleop is None:
            return
        if _leader_follow_enabled:
            _leader_follow_enabled = False
            _hybrid_teleop.bus.disable_torque(num_retry=5)
            logging.info("Hybrid mode: Leader torque disabled for RESET.")


def enable_torque_with_retries(self, motors=None, num_retry=0):
    """扭矩启用至少重试 5 次，应对偶发串口状态包丢失。"""
    return _original_enable_torque(self, motors, num_retry=max(num_retry, 5))


def disable_torque_with_retries(self, motors=None, num_retry=0):
    """扭矩关闭至少重试 5 次，提升异常退出时的可靠性。"""
    return _original_disable_torque(self, motors, num_retry=max(num_retry, 5))


def connect_with_cleanup(self, calibrate: bool = True):
    """连接中途失败时，主动关闭已连接总线和相机后再抛出异常。"""
    try:
        return _original_connect(self, calibrate=calibrate)
    except BaseException:
        if self.bus.is_connected:
            try:
                self.bus.disable_torque(num_retry=5)
            finally:
                self.bus.disconnect(disable_torque=False)
        for camera in self.cameras.values():
            if camera.is_connected:
                camera.disconnect()
        raise


def send_action_with_recording_fix(self, action: dict[str, Any]) -> dict[str, Any]:
    """执行软限位/起点门禁，并记录 send_action 实际返回的目标。"""
    global _first_record_action, _last_sent_action

    raw_action = dict(action)
    if _rl_collection:
        # 把策略绝对目标限制在校准物理范围内。这与 max_relative_target 相互独立：
        # 前者限制绝对软限位，后者限制相对最新实测状态的单次变化。
        for key, value in list(action.items()):
            if not key.endswith(".pos"):
                continue
            name = key.removesuffix(".pos")
            if name == "gripper":
                low, high = 1.0, 99.0
            else:
                motor = self.bus.motors[name]
                calibration = self.calibration[name]
                resolution = self.bus.model_resolution_table[motor.model] - 1
                midpoint = (calibration.range_min + calibration.range_max) / 2
                low = (calibration.range_min - midpoint) * 360 / resolution + 0.5
                high = (calibration.range_max - midpoint) * 360 / resolution - 0.5
            action[key] = min(max(float(value), low), high)

    # ---------- 第一帧起点门禁 ----------
    # 这个门禁只适用于人工遥操作：此时 action 就是 Leader 当前姿态，
    # 可以用它判断 Leader 与 Follower 在接管瞬间是否对齐。
    #
    # RL rollout 的 action 来自策略，本来就可能与当前状态相差数度；
    # 把它解释成 “leader - follower” 会产生假阳性。策略模式不做主从对齐，
    # 但仍保留上面的校准绝对软限位，以及 SO101Follower.send_action 内置的
    # max_relative_target 单次相对动作限制。
    if _record_mode and _first_record_action:
        if _rl_collection:
            logging.info(
                "Policy rollout detected: skipping the Leader/Follower alignment gate; "
                "calibration soft limits and max_relative_target remain active."
            )
        else:
            threshold = float(os.environ.get("BARBELL_START_MAX_DELTA_DEG", "8.0"))
            present = self.bus.sync_read("Present_Position")
            requested = {
                key.removesuffix(".pos"): float(value)
                for key, value in action.items()
                if key.endswith(".pos")
            }
            signed_deltas = {
                motor: requested[motor] - float(present[motor]) for motor in requested
            }
            deltas = {motor: abs(value) for motor, value in signed_deltas.items()}
            worst_motor, worst_delta = max(deltas.items(), key=lambda item: item[1])
            details = ", ".join(
                f"{name}={value:+.2f}°" for name, value in signed_deltas.items()
            )
            logging.info("Recording start pose deltas (leader - follower): %s", details)
            if worst_delta > threshold:
                raise RuntimeError(
                    "RECORDING SAFETY GATE: leader/follower start poses are not aligned. "
                    f"Worst joint {worst_motor} differs by {worst_delta:.2f}°, "
                    f"limit is {threshold:.2f}°. No action was sent. "
                    f"Signed deltas (leader - follower): {details}. "
                    "Manually align both arms and start again."
                )
        _first_record_action = False

    # ---------- 真正发送，并保留经过 max_relative_target 裁剪后的结果 ----------
    sent = _original_send_action(self, action)
    _last_sent_action = dict(sent)
    _write_sidecar_sample(raw_action, sent)

    # LeRobot 默认把模型动作和待发送动作放在两个字典中。这里同步修改传入对象，
    # 而下方 recorder hook 会权威地用实际发送动作覆盖数据集 action 字段。
    action.clear()
    action.update(sent)
    return sent


def get_observation_with_sidecar_state(self) -> dict[str, Any]:
    """读取正常观测，同时缓存最新六关节状态供侧车日志使用。"""
    global _latest_follower_state
    observation = _original_get_observation(self)
    _latest_follower_state = {
        key: float(value)
        for key, value in observation.items()
        if key in _JOINT_KEYS
    }
    _sync_leader_from_follower_observation(observation)
    return observation


# ---------- 3. 安装进程内补丁 ----------
FeetechMotorsBus.enable_torque = enable_torque_with_retries
FeetechMotorsBus.disable_torque = disable_torque_with_retries
SO101Follower.connect = connect_with_cleanup
SO101Follower.send_action = send_action_with_recording_fix
SO101Follower.get_observation = get_observation_with_sidecar_state


def main() -> None:
    """选择 teleoperate/record，并为 record 注入保存、按键和阶段补丁。"""
    # ---------- 4. 选择上游 CLI ----------
    global _record_mode
    if len(sys.argv) < 2 or sys.argv[1] not in {"teleoperate", "record"}:
        raise SystemExit("usage: safe_lerobot_cli.py {teleoperate|record} [arguments...]")
    command = sys.argv.pop(1)
    if command == "teleoperate":
        from lerobot.scripts.lerobot_teleoperate import main as command_main
    else:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        import lerobot.scripts.lerobot_record as record_module
        from lerobot.utils.control_utils import is_headless

        # ---------- 5. 保存 recorder 原函数，随后用包装函数接管关键节点 ----------
        _record_mode = True
        original_build_dataset_frame = record_module.build_dataset_frame
        original_record_loop = record_module.record_loop
        original_save_episode = LeRobotDataset.save_episode
        original_predict_action = record_module.predict_action

        def predict_action_with_human_intervention(*args, **kwargs):
            """策略阶段正常推理；Space 接管后改为读取 Leader 的人工动作。"""
            if not (_rl_collection and _human_intervention):
                return original_predict_action(*args, **kwargs)
            if _hybrid_teleop is None:
                raise RuntimeError("Human intervention is active but Leader is unavailable.")
            action = _hybrid_teleop.get_action()
            values = [float(action[key]) for key in _JOINT_KEYS]
            # make_robot_action() 期望 [batch, action_dim] 的张量。
            return torch.tensor([values], dtype=torch.float32)

        def build_dataset_frame_with_sent_action(features, item, prefix):
            """action 字段强制使用电机实际收到的目标，观测字段保持原样。"""
            if prefix == "action":
                if _last_sent_action is None:
                    raise RuntimeError("No transmitted robot action is available for this frame.")
                item = _last_sent_action
            return original_build_dataset_frame(features, item, prefix)

        def record_loop_with_phase(*args, **kwargs):
            """标记 RECORDING/RESET 阶段，并处理 RL Episode 超时结果。"""
            global _recording_episode_active, _current_episode_index, _current_phase
            global _rl_episode_outcome, _safe_stop_requested
            global _hybrid_teleop, _human_intervention, _episode_had_intervention
            global _intervention_start_frame, _current_dataset
            dataset = kwargs.get("dataset")
            if dataset is None and len(args) > 9:
                dataset = args[9]
            events = kwargs.get("events")
            if events is None and len(args) > 1:
                events = args[1]
            _recording_episode_active = dataset is not None
            _current_phase = "RECORDING" if _recording_episode_active else "RESET"
            if dataset is not None:
                _current_episode_index = getattr(
                    getattr(dataset, "meta", None), "total_episodes", "?"
                )
            if _recording_episode_active:
                if _rl_collection:
                    _rl_episode_outcome = None
                    _hybrid_teleop = kwargs.get("teleop")
                    _human_intervention = False
                    _episode_had_intervention = False
                    _intervention_start_frame = None
                    _current_dataset = dataset
                    _start_leader_following_follower(
                        kwargs["robot"],
                        _hybrid_teleop,
                    )
                episode_index = getattr(getattr(dataset, "meta", None), "total_episodes", "?")
                print(
                    "\n\033[1;32m"
                    f"========== RECORDING EPISODE {episode_index} =========="
                    "\033[0m\n"
                    "Perform the task now. Tap Right on success or Left on failure.\a",
                    flush=True,
                )
            else:
                print(
                    "\n\033[1;33m"
                    "========== RESET THE ENVIRONMENT =========="
                    "\033[0m\n"
                    "Reposition the barbell/tray. When hands are clear, tap Right "
                    "to start the next episode; tap Esc to finish the batch.\a",
                    flush=True,
                )
            try:
                if _rl_collection and not _recording_episode_active:
                    # 官方 reset 会以 policy=None、teleop=None 调用 record_loop。
                    # 上游循环虽然不会发送动作，却会以 30 Hz 重复打印
                    # “No policy or teleoperator provided”。RL reset 只需要等待
                    # 操作者摆放物体，因此在这里使用安静的事件等待循环。
                    reset_seconds = float(kwargs.get("control_time_s", 0.0))
                    reset_deadline = time.perf_counter() + reset_seconds
                    while time.perf_counter() < reset_deadline:
                        if events is not None and events["exit_early"]:
                            events["exit_early"] = False
                            break
                        time.sleep(0.05)
                    result = None
                else:
                    result = original_record_loop(*args, **kwargs)
                if (
                    _rl_collection
                    and _recording_episode_active
                    and _rl_episode_outcome is None
                    and not _safe_stop_requested
                ):
                    _rl_episode_outcome = {
                        "success": False,
                        "termination": "episode_time_limit",
                    }
                    print(
                        "\n\033[1;31mEPISODE TIME LIMIT: saving as failure.\033[0m\a",
                        flush=True,
                    )
                return result
            except KeyboardInterrupt:
                # Ctrl+C 也转换成 recorder 能理解的停止事件，避免异常栈。
                # Recording 中断表示当前轨迹不完整，应清空；Reset 中断发生在
                # 一条完整轨迹之后，应允许上游继续执行 save_episode()。
                _safe_stop_requested = True
                if events is not None:
                    events["stop_recording"] = True
                    events["exit_early"] = True
                    events["rerecord_episode"] = _recording_episode_active
                if _recording_episode_active:
                    print(
                        "\n\033[1;31mCTRL+C RECEIVED DURING RECORDING: "
                        "discarding the partial episode and stopping safely.\033[0m\a",
                        flush=True,
                    )
                else:
                    print(
                        "\n\033[1;31mCTRL+C RECEIVED DURING RESET: "
                        "saving the completed episode and stopping safely.\033[0m\a",
                        flush=True,
                    )
                return None
            finally:
                if _recording_episode_active and _rl_collection:
                    _release_leader_after_policy_phase()
                _recording_episode_active = False
                _current_phase = "IDLE"

        def init_keyboard_listener_with_safe_stop():
            """定义方向键与 Esc 语义，避免停止时误删已完成 Episode。"""
            events = {
                "exit_early": False,
                "rerecord_episode": False,
                "stop_recording": False,
            }
            if is_headless():
                logging.warning(
                    "Headless environment detected. Keyboard controls are unavailable."
                )
                return None, events

            from pynput import keyboard

            pressed_keys = set()

            def on_press(key):
                """将物理按键转换为 LeRobot recorder 的事件标志。"""
                global _safe_stop_requested, _rl_episode_outcome
                # 长按方向键时 pynput 会重复触发。用 pressed_keys 去重，确保一次物理
                # 按键只产生一个事件，避免 Reset 的右键泄漏到下一条 Episode。
                if key in pressed_keys:
                    return
                pressed_keys.add(key)
                if key == keyboard.Key.right:
                    if _recording_episode_active:
                        if _rl_collection:
                            _rl_episode_outcome = {
                                "success": True,
                                "termination": "operator_success",
                            }
                        print(
                            "\n\033[1;36mRIGHT RECEIVED: ending the successful episode; "
                            "RESET will start next.\033[0m\a",
                            flush=True,
                        )
                    else:
                        print(
                            "\n\033[1;36mRIGHT RECEIVED: reset complete; "
                            "the next RECORDING phase will start.\033[0m\a",
                            flush=True,
                        )
                    events["exit_early"] = True
                elif key == keyboard.Key.left:
                    if _rl_collection and _recording_episode_active:
                        _rl_episode_outcome = {
                            "success": False,
                            "termination": "operator_failure",
                        }
                        print(
                            "\n\033[1;31mLEFT RECEIVED: saving this failed policy "
                            "rollout; RESET will start next.\033[0m\a",
                            flush=True,
                        )
                        events["exit_early"] = True
                    elif not _recording_episode_active:
                        print(
                            "\n\033[1;33mLEFT IGNORED DURING RESET: use Right when "
                            "the scene is ready.\033[0m\a",
                            flush=True,
                        )
                    else:
                        print(
                            "\n\033[1;31mLEFT RECEIVED: discard this episode and "
                            "re-record it.\033[0m\a",
                            flush=True,
                        )
                        events["rerecord_episode"] = True
                        events["exit_early"] = True
                elif key == keyboard.Key.space:
                    if _rl_collection and _recording_episode_active:
                        try:
                            _enter_human_intervention()
                        except BaseException as error:
                            logging.exception("Failed to enter human intervention mode: %s", error)
                            events["stop_recording"] = True
                            events["exit_early"] = True
                            events["rerecord_episode"] = True
                    elif _rl_collection:
                        print(
                            "\n\033[1;33mSPACE IGNORED DURING RESET.\033[0m\a",
                            flush=True,
                        )
                elif key == keyboard.Key.esc:
                    _safe_stop_requested = True
                    events["stop_recording"] = True
                    events["exit_early"] = True
                    events["rerecord_episode"] = _recording_episode_active
                    if _recording_episode_active:
                        print(
                            "\n\033[1;31mESC RECEIVED: discard the partial episode "
                            "and stop safely.\033[0m\a",
                            flush=True,
                        )
                    else:
                        print(
                            "\n\033[1;31mESC RECEIVED: preserve completed episodes "
                            "and stop safely.\033[0m\a",
                            flush=True,
                        )

            def on_release(key):
                """释放按键时允许下一次物理按压再次触发。"""
                pressed_keys.discard(key)

            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            listener.start()
            return listener, events

        def save_episode_without_empty_stop_race(
            self, episode_data=None, parallel_encoding=True
        ):
            """跳过空缓冲；正式保存成功后再原子写入 RL 成败标签。"""
            global _rl_episode_outcome
            episode_buffer = episode_data if episode_data is not None else self.episode_buffer
            size = 0 if episode_buffer is None else int(episode_buffer.get("size", 0))
            if size == 0:
                logging.info(
                    "Skipping an empty episode buffer (no frame was recorded)."
                )
                self.clear_episode_buffer()
                return None
            episode_index_value = episode_buffer.get("episode_index")
            if hasattr(episode_index_value, "item"):
                episode_index = int(episode_index_value.item())
            elif isinstance(episode_index_value, (list, tuple)):
                episode_index = int(episode_index_value[0])
            else:
                episode_index = int(episode_index_value)
            result = original_save_episode(
                self,
                episode_data=episode_data,
                parallel_encoding=parallel_encoding,
            )
            if _rl_collection:
                if _rl_episode_outcome is None:
                    raise RuntimeError(
                        f"RL episode {episode_index} was saved without an outcome label."
                    )
                _rl_episode_outcome["human_intervention"] = _episode_had_intervention
                _rl_episode_outcome["intervention_start_frame"] = _intervention_start_frame
                _save_rl_episode_label(episode_index, _rl_episode_outcome)
                print(
                    f"RL label saved: episode={episode_index} "
                    f"success={bool(_rl_episode_outcome['success'])} "
                    f"termination={_rl_episode_outcome['termination']}",
                    flush=True,
                )
                _rl_episode_outcome = None
            return result

        # ---------- 6. 把包装函数安装到上游 recorder 并启动 ----------
        record_module.build_dataset_frame = build_dataset_frame_with_sent_action
        record_module.record_loop = record_loop_with_phase
        record_module.predict_action = predict_action_with_human_intervention
        record_module.init_keyboard_listener = init_keyboard_listener_with_safe_stop
        LeRobotDataset.save_episode = save_episode_without_empty_stop_race
        command_main = record_module.main
    command_main()


if __name__ == "__main__":
    main()
