#!/usr/bin/env bash
# 只读检查：列出机械臂串口和相机稳定路径，不会向任何设备发送指令。
set -u

# ---------- 1. 串口 ----------
# by-id 名称包含 USB 芯片序列号，不受 ttyACM 编号变化影响。
echo "===== SO-ARM serial devices ====="
ls -l /dev/serial/by-id/ 2>&1 || true

# ---------- 2. V4L2 相机枚举 ----------
echo
echo "===== Camera devices ====="
v4l2-ctl --list-devices 2>&1 || true

# ---------- 3. 相机稳定路径 ----------
# 只列 video-index0；index1 通常是同一相机的另一视频接口。
echo
echo "===== Stable camera paths (index0 streams) ====="
find /dev/v4l/by-path -maxdepth 1 -type l -name '*video-index0' -printf '%p -> ' \
  -exec readlink -f {} \; 2>/dev/null | sort || true

# ---------- 4. 人工核对提示 ----------
echo
echo "Expected arm serial IDs:"
echo "  left follower: FOLLOWER_SERIAL"
echo "  left leader:   LEADER_SERIAL"
echo
echo "Known camera roles from the last confirmed mapping:"
echo "  top/front: 2M camera, software rotation 180 degrees"
echo "  left wrist: USB Camera (CN02KX4NLG0004ABK00)"
