#!/usr/bin/env bash
# 左主臂→左从臂的 30 秒遥操作连通性测试。
# 会启用从臂扭矩并真实运动；测试期间必须清空工作区并守住物理急停。
set -euo pipefail

# ---------- 1. 固定环境与主从臂身份 ----------
# 本测试不打开相机，只确认串口、校准和电机跟随是否正常。
LEROBOT_BIN="/home/user/miniconda3/envs/lerobot/bin"
LEFT_FOLLOWER="/dev/serial/by-id/usb-1a86_USB_Single_Serial_FOLLOWER_SERIAL-if00"
LEFT_LEADER="/dev/serial/by-id/usb-1a86_USB_Single_Serial_LEADER_SERIAL-if00"

# ---------- 2. 启动安全包装后的 30 秒遥操作 ----------
# max_relative_target=5 表示单次目标相对当前实测角度最多变化 5°。
exec "${LEROBOT_BIN}/python" "$(dirname "$0")/safe_lerobot_cli.py" teleoperate \
  --robot.type=so101_follower \
  --robot.port="${LEFT_FOLLOWER}" \
  --robot.id=SO101_FOLLOWER_ID \
  --robot.use_degrees=true \
  --robot.disable_torque_on_disconnect=true \
  --robot.max_relative_target=5 \
  --teleop.type=so101_leader \
  --teleop.port="${LEFT_LEADER}" \
  --teleop.id=SO101_LEADER_ID \
  --teleop.use_degrees=true \
  --fps=30 \
  --teleop_time_s=30 \
  --display_data=false
