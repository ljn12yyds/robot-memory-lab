#!/usr/bin/env python3
"""只读比较左主臂与左从臂的六个校准关节角。

用于正式遥操作采集前确认两臂起始姿态接近；不会启用扭矩或写目标位置。
"""

from __future__ import annotations

from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader


FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_FOLLOWER_SERIAL-if00"
LEADER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_LEADER_SERIAL-if00"


def main() -> None:
    """只读比较主从臂角度，并用 8° 门限给出 PASS/FAIL。"""
    # ---------- 1. 构造不带相机的主从臂对象 ----------
    follower = SO101Follower(
        SO101FollowerConfig(
            port=FOLLOWER_PORT,
            id="SO101_FOLLOWER_ID",
            use_degrees=True,
            cameras={},
        )
    )
    leader = SO101Leader(
        SO101LeaderConfig(
            port=LEADER_PORT,
            id="SO101_LEADER_ID",
            use_degrees=True,
        )
    )

    # 只连接两条电机总线；不调用完整 connect/configure、扭矩方法或任何写方法。
    # ---------- 2. 只读连接并读取位置/扭矩标志 ----------
    follower.bus.connect()
    try:
        leader.bus.connect()
        try:
            follower_pos = follower.bus.sync_read("Present_Position")
            leader_pos = leader.bus.sync_read("Present_Position")
            follower_torque = follower.bus.sync_read("Torque_Enable", normalize=False)
            leader_torque = leader.bus.sync_read("Torque_Enable", normalize=False)
        finally:
            leader.bus.disconnect(disable_torque=False)
    finally:
        follower.bus.disconnect(disable_torque=False)

    # ---------- 3. 打印逐关节差值 ----------
    print("joint                 follower       leader    leader-follower   |delta|")
    print("-" * 76)
    passed = True
    for joint in follower_pos:
        delta = float(leader_pos[joint]) - float(follower_pos[joint])
        ok = abs(delta) <= 8.0
        passed = passed and ok
        print(
            f"{joint:20s} {float(follower_pos[joint]):9.2f}° "
            f"{float(leader_pos[joint]):11.2f}° {delta:+16.2f}° "
            f"{abs(delta):8.2f}° {'OK' if ok else 'FAIL'}"
        )
    print()
    print(f"START ALIGNMENT: {'PASS' if passed else 'FAIL'} (limit: 8.00°)")
    # 扭矩标志应该保持 0，以证明本脚本没有意外锁住主从臂。
    print(
        "Follower torque flags:",
        ", ".join(f"{joint}={value}" for joint, value in follower_torque.items()),
    )
    print(
        "Leader torque flags:",
        ", ".join(f"{joint}={value}" for joint, value in leader_torque.items()),
    )
    print("READ ONLY: no torque or goal-position command was written.")
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
