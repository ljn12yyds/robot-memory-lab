#!/usr/bin/env python3
"""为 Evo-RL 人工介入采集器增加 SO101 真实执行安全门。"""

from __future__ import annotations

import logging
import time

import numpy as np

from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.teleoperators.so_leader.so_leader import SOLeader


JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
RAW_TARGET_DELTA_LIMIT_DEG = 20.0
SOFT_LIMIT_TOLERANCE_DEG = 2.0
# SO101 在持位时常有约 1° 的齿隙、负载误差和编码器量化误差。该容差必须
# 大于近软限位使用的 1° 动态步长，否则“已足够接近”会被误判成持续卡滞。
# A commanded SO101 joint commonly holds 1-2 degrees away from its target under
# gravity/backlash.  Treat that as normal compliance; only a larger error with no
# measurable progress is an unsafe policy chunk.  Such a chunk is replanned by
# the HIL loop instead of terminating the complete collection process.
TRACKING_ERROR_DEG = 2.5
TRACKING_PROGRESS_DEG = 0.05
TRACKING_TIMEOUT_S = 3.0
MAX_POLICY_STEP_DEG = 5.0
# Manual teleoperation is allowed to have a large leader-to-follower gap because
# the follower advances in <=5 degree increments.  Diagnose the motor only when
# its *previously accepted motor goal* remains far from its measured position.
MANUAL_MOTOR_LAG_WARN_DEG = 3.0
MANUAL_MOTOR_LAG_CLEAR_DEG = 1.5
MANUAL_MOTOR_LAG_TIMEOUT_S = 1.0
MANUAL_MOTOR_LAG_REPORT_INTERVAL_S = 2.0

_original_enable_torque = FeetechMotorsBus.enable_torque
_original_disable_torque = FeetechMotorsBus.disable_torque
_original_connect = SOFollower.connect
_original_send_action = SOFollower.send_action
_original_predict_action_chunk = PI05Policy.predict_action_chunk
_original_set_leader_manual_control = SOLeader.set_manual_control


class HILPolicyReplanRequired(RuntimeError):
    """The policy chunk is unsafe; discard it and infer again from a fresh observation."""

    replan_policy = True


def _predict_action_chunk_bounded(self, batch, **kwargs):
    """Keep PI0.5 quantile-normalized actions inside the trained q01/q99 interval."""
    inference_start_ns = time.monotonic_ns()
    actions = _original_predict_action_chunk(self, batch, **kwargs)
    inference_end_ns = time.monotonic_ns()
    clipped = actions.clamp(-1.0, 1.0)
    serial = int(getattr(self, "_hil_chunk_serial", 0)) + 1
    self._hil_chunk_serial = serial
    # CPU clones preserve the immutable proposal evidence without retaining the
    # CUDA computation graph or allowing the action queue to mutate it.
    self._hil_last_chunk_capture = {
        "serial": serial,
        "inference_start_ns": inference_start_ns,
        "inference_end_ns": inference_end_ns,
        "raw": actions.detach().to(device="cpu").float().clone(),
        "clipped": clipped.detach().to(device="cpu").float().clone(),
    }
    return clipped


def _enable_torque_with_retries(self, motors=None, num_retry=0):
    return _original_enable_torque(self, motors, num_retry=max(num_retry, 5))


def _disable_torque_with_retries(self, motors=None, num_retry=0):
    """Disable torque, accepting a servo alarm only after safe readback.

    Feetech can apply the register write and still return an alarm bit in the
    same status packet.  In that case the stock driver raises even though the
    requested fail-safe state has already been reached.  Never suppress a
    communication/alarm error blindly: continue only when every requested
    motor independently reads back Torque_Enable=0.
    """
    motor_names = self._get_motors_list(motors)
    try:
        return _original_disable_torque(self, motors, num_retry=max(num_retry, 5))
    except (ConnectionError, RuntimeError) as error:
        try:
            torque = self.sync_read(
                "Torque_Enable", motors=motor_names, normalize=False, num_retry=5
            )
            voltage = self.sync_read(
                "Present_Voltage", motors=motor_names, normalize=False, num_retry=5
            )
        except Exception as readback_error:
            raise error from readback_error

        still_enabled = {
            name: int(torque[name]) for name in motor_names if int(torque[name]) != 0
        }
        if still_enabled:
            logging.error(
                "Leader torque-disable failed and readback is unsafe: enabled=%s "
                "voltage=%s original_error=%s",
                still_enabled,
                {name: float(voltage[name]) / 10.0 for name in motor_names},
                error,
            )
            raise error

        logging.warning(
            "Torque-disable returned a servo/communication error, but independent "
            "readback confirms Torque_Enable=0 for all requested motors; safe "
            "manual handover will continue. voltage=%s original_error=%s",
            {name: float(voltage[name]) / 10.0 for name in motor_names},
            error,
        )
        return None


def _set_leader_manual_control_with_readback(self, enabled: bool) -> None:
    """Switch leader mode and report torque readback without blocking on diagnostics."""
    if enabled:
        # Capture the electrically interesting instant while the leader is still
        # powered and holding its pose.  A later read after torque-off cannot
        # reveal a load-induced voltage sag.
        try:
            voltage_before = self.bus.sync_read("Present_Voltage", normalize=False)
            current_before = self.bus.sync_read("Present_Current", normalize=False)
            load_before = self.bus.sync_read("Present_Load", normalize=False)
            logging.info(
                "Leader pre-handover electrical snapshot (torque still ON): "
                "voltage_V=%s current_raw=%s load_raw=%s",
                {name: float(value) / 10.0 for name, value in voltage_before.items()},
                current_before,
                load_before,
            )
        except Exception as error:
            logging.warning("Could not capture leader pre-handover electrical snapshot: %s", error)

    _original_set_leader_manual_control(self, enabled)
    try:
        torque = self.bus.sync_read("Torque_Enable", normalize=False)
    except ConnectionError as error:
        logging.warning("Could not read leader torque state after mode switch: %s", error)
        return
    expected = 0 if enabled else 1
    mismatched = {name: value for name, value in torque.items() if value != expected}
    if mismatched:
        logging.warning(
            "Leader torque readback mismatch: manual_control=%s expected=%d actual=%s",
            enabled,
            expected,
            torque,
        )
    else:
        logging.info(
            "Leader torque readback: manual_control=%s Torque_Enable=%d on all motors.",
            enabled,
            expected,
        )


def _derive_soft_limits(robot: SOFollower) -> dict[str, tuple[float, float]]:
    limits = {}
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
        limits[f"{name}.pos"] = (low + margin, high - margin)
    return limits


def _connect_with_safety_state(self, calibrate=True):
    result = _original_connect(self, calibrate=calibrate)
    self._hil_soft_limits = _derive_soft_limits(self)
    self._hil_previous_command = None
    self._hil_previous_state = None
    self._hil_last_state = None
    self._hil_progress_time = None
    self._hil_error_sign = np.zeros(5)
    self._hil_tracking_enabled = True
    self._hil_last_goal_readback = None
    self._hil_manual_motor_lag_since = np.full(6, np.nan)
    self._hil_last_manual_motor_lag_report = float("-inf")
    print("Evo-RL HIL safety gates enabled:")
    for name in JOINT_NAMES:
        low, high = self._hil_soft_limits[name]
        print(f"  {name:22s} [{low:+8.2f}, {high:+8.2f}]")
    return result


def _reset_action_safety_state(self) -> None:
    """Clear command-tracking history at RECORDING/RESET phase boundaries."""
    self._hil_previous_command = None
    self._hil_previous_state = None
    self._hil_last_state = None
    self._hil_progress_time = None
    self._hil_error_sign = np.zeros(5)


def _set_action_safety_tracking_enabled(self, enabled: bool) -> None:
    """Apply stall detection to autonomous policy actions, not manual RESET/teleop."""
    enabled = bool(enabled)
    if getattr(self, "_hil_tracking_enabled", None) != enabled:
        _reset_action_safety_state(self)
        self._hil_manual_motor_lag_since = np.full(6, np.nan)
        self._hil_last_manual_motor_lag_report = float("-inf")
    self._hil_tracking_enabled = enabled


def _diagnose_manual_motor_tracking(
    self,
    current: np.ndarray,
    leader_gap: np.ndarray,
    now: float,
) -> None:
    """Warn about persistent follower motor lag without interrupting teleoperation."""
    previous_goal = getattr(self, "_hil_last_goal_readback", None)
    if previous_goal is None:
        return

    motor_error = np.abs(previous_goal - current)
    lag_since = self._hil_manual_motor_lag_since
    for index, error in enumerate(motor_error):
        if error <= MANUAL_MOTOR_LAG_CLEAR_DEG:
            lag_since[index] = np.nan
        elif error >= MANUAL_MOTOR_LAG_WARN_DEG and np.isnan(lag_since[index]):
            lag_since[index] = now

    sustained = (~np.isnan(lag_since)) & (
        now - lag_since >= MANUAL_MOTOR_LAG_TIMEOUT_S
    )
    if not bool(sustained.any()):
        return
    if now - self._hil_last_manual_motor_lag_report < MANUAL_MOTOR_LAG_REPORT_INTERVAL_S:
        return

    # These reads are diagnostic only and happen at most once per report interval.
    load = self.bus.sync_read("Present_Load", normalize=False)
    current_draw = self.bus.sync_read("Present_Current", normalize=False)
    voltage = self.bus.sync_read("Present_Voltage", normalize=False)
    temperature = self.bus.sync_read("Present_Temperature", normalize=False)
    details = []
    for index in np.flatnonzero(sustained):
        motor = JOINT_NAMES[index].removesuffix(".pos")
        details.append(
            f"{JOINT_NAMES[index]} motor_error={motor_error[index]:.2f}° "
            f"leader_gap={leader_gap[index]:+.2f}° load={load[motor]} "
            f"current={current_draw[motor]} voltage={voltage[motor] / 10:.1f}V "
            f"temp={temperature[motor]}°C"
        )
    logging.warning(
        "HIL TELEOP persistent follower motor lag (diagnostic only): %s",
        "; ".join(details),
    )
    self._hil_last_manual_motor_lag_report = now


def _prepare_policy_action(self, action: dict[str, float]) -> dict[str, float]:
    """Validate a fresh policy target and match rollout-time 5-degree limiting.

    This hook is called only for policy actions. Human teleoperation remains under
    the follower's calibration limits and built-in ``max_relative_target`` guard.
    """
    current_raw = self.bus.sync_read("Present_Position")
    current = np.asarray([float(current_raw[name.removesuffix(".pos")]) for name in JOINT_NAMES])
    target = current.copy()
    for index, name in enumerate(JOINT_NAMES):
        if name in action:
            target[index] = float(action[name])

    if not np.isfinite(target).all():
        raise HILPolicyReplanRequired("PI0.5 action contains NaN or Inf")

    raw_delta = target - current
    if float(np.abs(raw_delta).max()) > RAW_TARGET_DELTA_LIMIT_DEG:
        index = int(np.abs(raw_delta).argmax())
        raise HILPolicyReplanRequired(
            f"raw target jump: {JOINT_NAMES[index]} delta={raw_delta[index]:+.2f}°"
        )

    outside = []
    for index, name in enumerate(JOINT_NAMES):
        low, high = self._hil_soft_limits[name]
        if target[index] < low - SOFT_LIMIT_TOLERANCE_DEG:
            outside.append(f"{name}={target[index]:.2f}° below {low:.2f}°")
        elif target[index] > high + SOFT_LIMIT_TOLERANCE_DEG:
            outside.append(f"{name}={target[index]:.2f}° above {high:.2f}°")
    if outside:
        raise HILPolicyReplanRequired("; ".join(outside))

    limited = dict(action)
    for index, name in enumerate(JOINT_NAMES):
        if name not in action:
            continue
        low, high = self._hil_soft_limits[name]
        next_target = current[index] + np.clip(
            raw_delta[index], -MAX_POLICY_STEP_DEG, MAX_POLICY_STEP_DEG
        )
        limited[name] = float(np.clip(next_target, low, high))
    return limited


def _send_action_with_safety(self, action: dict[str, float]) -> dict[str, float]:
    current_raw = self.bus.sync_read("Present_Position")
    current = np.asarray([float(current_raw[name.removesuffix(".pos")]) for name in JOINT_NAMES])
    now = time.perf_counter()

    # 在发送下一条命令前检查上一条命令是否得到真实机械运动响应。
    previous_command = self._hil_previous_command
    previous_state = self._hil_previous_state
    tracking_enabled = bool(getattr(self, "_hil_tracking_enabled", True))
    if tracking_enabled and previous_command is not None and previous_state is not None:
        if self._hil_last_state is None:
            self._hil_last_state = current.copy()
            self._hil_progress_time = np.full(5, now)
        observed_step = np.abs(current[:5] - self._hil_last_state[:5])
        signed_error = previous_command[:5] - current[:5]
        error = np.abs(signed_error)
        error_sign = np.sign(signed_error)
        for index in range(5):
            direction_changed = (
                self._hil_error_sign[index] != 0
                and error_sign[index] != self._hil_error_sign[index]
            )
            if observed_step[index] >= TRACKING_PROGRESS_DEG or error[index] <= TRACKING_ERROR_DEG or direction_changed:
                self._hil_progress_time[index] = now
        stalled = (error > TRACKING_ERROR_DEG) & (
            now - self._hil_progress_time >= TRACKING_TIMEOUT_S
        )
        if bool(stalled.any()):
            details = ", ".join(
                f"{JOINT_NAMES[index]} error={error[index]:.2f}°"
                for index in np.flatnonzero(stalled)
            )
            raise HILPolicyReplanRequired(f"target tracking failure: {details}")
        self._hil_last_state = current.copy()
        self._hil_error_sign = error_sign
    elif tracking_enabled and self._hil_last_state is None:
        self._hil_last_state = current.copy()
        self._hil_progress_time = np.full(5, now)

    target = current.copy()
    for index, name in enumerate(JOINT_NAMES):
        if name in action:
            target[index] = float(action[name])
    if not np.isfinite(target).all():
        raise RuntimeError("HIL SAFETY GATE: action contains NaN or Inf.")
    raw_delta = target - current
    # The 20-degree rejection diagnoses model output only. During teleoperation
    # the leader may legitimately be much farther from a lagging follower; the
    # follower's max_relative_target then advances safely in <=5-degree steps.
    if tracking_enabled and float(np.abs(raw_delta).max()) > RAW_TARGET_DELTA_LIMIT_DEG:
        index = int(np.abs(raw_delta).argmax())
        raise RuntimeError(
            "HIL SAFETY GATE: raw target jump rejected: "
            f"{JOINT_NAMES[index]} delta={raw_delta[index]:+.2f}°."
        )

    safe_action = dict(action)
    if tracking_enabled:
        outside = []
        for index, name in enumerate(JOINT_NAMES):
            low, high = self._hil_soft_limits[name]
            if target[index] < low - SOFT_LIMIT_TOLERANCE_DEG:
                outside.append(f"{name}={target[index]:.2f}° below {low:.2f}°")
            elif target[index] > high + SOFT_LIMIT_TOLERANCE_DEG:
                outside.append(f"{name}={target[index]:.2f}° above {high:.2f}°")
        if outside:
            raise RuntimeError("HIL SAFETY GATE: target outside soft limits: " + "; ".join(outside))
    else:
        # Clip human targets to the mechanical limits and apply the same 5-degree
        # relative limiter here. This is equivalent to the follower's built-in
        # limiter, but avoids flooding the RESET console with one warning per
        # frame when the leader is temporarily far from the follower.
        for index, name in enumerate(JOINT_NAMES):
            if name in safe_action:
                low, high = self._hil_soft_limits[name]
                bounded_target = float(np.clip(target[index], low, high))
                safe_action[name] = float(
                    current[index]
                    + np.clip(
                        bounded_target - current[index],
                        -MAX_POLICY_STEP_DEG,
                        MAX_POLICY_STEP_DEG,
                    )
                )

        _diagnose_manual_motor_tracking(self, current, raw_delta, now)

    sent = _original_send_action(self, safe_action)
    goal_raw = self.bus.sync_read("Goal_Position")
    goal = np.asarray([float(goal_raw[name.removesuffix(".pos")]) for name in JOINT_NAMES])
    sent_vector = np.asarray([float(sent[name]) for name in JOINT_NAMES])
    register_error = np.abs(goal - sent_vector)
    if float(register_error.max()) > 0.5:
        raise RuntimeError(
            "HIL SAFETY GATE: Goal_Position readback mismatch: "
            f"maximum error={register_error.max():.2f}°."
        )
    self._hil_last_goal_readback = goal.copy()

    if tracking_enabled:
        self._hil_previous_state = current.copy()
        self._hil_previous_command = sent_vector
    return sent


def _describe_manual_reset_torque_mode(self) -> None:
    """Explain why the follower cannot be moved directly by hand during RESET."""
    logging.info(
        "RESET teleoperation torque mode: leader torque OFF (backdrivable); "
        "follower torque ON so it can follow the leader. Move/close the follower "
        "through the leader; do not force the powered follower by hand."
    )


FeetechMotorsBus.enable_torque = _enable_torque_with_retries
FeetechMotorsBus.disable_torque = _disable_torque_with_retries
SOFollower.connect = _connect_with_safety_state
SOFollower.send_action = _send_action_with_safety
SOFollower.prepare_policy_action = _prepare_policy_action
SOFollower.reset_action_safety_state = _reset_action_safety_state
SOFollower.set_action_safety_tracking_enabled = _set_action_safety_tracking_enabled
SOFollower.describe_manual_reset_torque_mode = _describe_manual_reset_torque_mode
SOLeader.set_manual_control = _set_leader_manual_control_with_readback
PI05Policy.predict_action_chunk = _predict_action_chunk_bounded


def main() -> None:
    from lerobot.scripts.lerobot_human_inloop_record import main as human_inloop_main

    human_inloop_main()


if __name__ == "__main__":
    main()
