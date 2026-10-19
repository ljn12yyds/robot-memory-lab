#!/usr/bin/env python3
"""只读检查左从臂通信、电压、温度、负载和位置反馈。

直接连接电机总线读取寄存器，但不启用扭矩、不写目标位置。
"""

from __future__ import annotations

import statistics
import time

from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower


PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_FOLLOWER_SERIAL-if00"


def main() -> None:
    """读取健康寄存器，并用 200 次位置读取测试串口稳定性。"""
    # ---------- 1. 构造无相机从臂对象 ----------
    robot = SO101Follower(
        SO101FollowerConfig(
            port=PORT,
            id="SO101_FOLLOWER_ID",
            use_degrees=True,
            cameras={},
        )
    )
    # ---------- 2. 只连接总线并读取健康信息 ----------
    robot.bus.connect()
    try:
        torque = robot.bus.sync_read("Torque_Enable", normalize=False)
        voltage = robot.bus.sync_read("Present_Voltage", normalize=False)
        temperature = robot.bus.sync_read("Present_Temperature", normalize=False)
        load = robot.bus.sync_read("Present_Load", normalize=False)
        current = robot.bus.sync_read("Present_Current", normalize=False)
        goal = robot.bus.sync_read("Goal_Position")
        present = robot.bus.sync_read("Present_Position")
        startup_force = robot.bus.sync_read("Minimum_Startup_Force", normalize=False)
        cw_dead_zone = robot.bus.sync_read("CW_Dead_Zone", normalize=False)
        ccw_dead_zone = robot.bus.sync_read("CCW_Dead_Zone", normalize=False)
        p_coefficient = robot.bus.sync_read("P_Coefficient", normalize=False)
        # ---------- 3. 连续读取位置，统计延迟与失败次数 ----------
        timings_ms = []
        failures = 0
        for _ in range(200):
            start = time.perf_counter()
            try:
                robot.bus.sync_read("Present_Position")
            except Exception:
                failures += 1
            timings_ms.append((time.perf_counter() - start) * 1000)
    finally:
        robot.bus.disconnect(disable_torque=False)

    # ---------- 4. 输出每个电机的遥测与总线统计 ----------
    print("Follower telemetry (raw register units where applicable):")
    for joint in torque:
        print(
            f"{joint:14s} torque={torque[joint]} "
            f"voltage={voltage[joint]} temp={temperature[joint]} "
            f"load={load[joint]} current={current[joint]} present={present[joint]:.2f} "
            f"goal={goal[joint]:.2f} startup_force={startup_force[joint]} "
            f"dead_zone={cw_dead_zone[joint]}/{ccw_dead_zone[joint]} "
            f"P={p_coefficient[joint]}"
        )
    print(
        "200 sync reads: "
        f"failures={failures}, mean={statistics.mean(timings_ms):.3f}ms, "
        f"p95={sorted(timings_ms)[189]:.3f}ms, max={max(timings_ms):.3f}ms"
    )
    print("READ ONLY: no torque or goal-position command was written.")
    raise SystemExit(0 if failures == 0 else 2)


if __name__ == "__main__":
    main()
