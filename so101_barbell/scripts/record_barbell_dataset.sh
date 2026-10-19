#!/usr/bin/env bash
# 人工遥操作采集入口：左主臂控制左从臂，并录制两路相机、状态和实际动作。
# 右键保存成功 Episode，左键丢弃失败 Episode，Esc 按当前阶段安全结束。
# 本脚本会真实控制机械臂，运行期间必须清空工作区并守住物理急停。
set -euo pipefail

# ---------- 1. 软件环境与硬件稳定路径 ----------
# 路径可用环境变量覆盖，以应对 USB 拓扑变化。
LEROBOT_BIN="/home/user/miniconda3/envs/lerobot/bin"
LEFT_FOLLOWER="${LEFT_FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_FOLLOWER_SERIAL-if00}"
LEFT_LEADER="${LEFT_LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_LEADER_SERIAL-if00}"
TOP_CAMERA="${TOP_CAMERA_PATH:-/dev/v4l/by-path/pci-0000:05:00.0-usb-0:2:1.0-video-index0}"
LEFT_WRIST_CAMERA="${LEFT_WRIST_CAMERA_PATH:-/dev/v4l/by-path/pci-0000:00:14.0-usb-0:4.4:1.0-video-index0}"

# ---------- 2. 数据集名称与每阶段时长 ----------
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
REPO_ID="${BARBELL_REPO_ID:-barbell/so101_barbell_${RUN_STAMP}}"
DATA_ROOT="${BARBELL_DATA_ROOT:-/home/user/ljn/barbell_task/data}"
# 这里只设置一个较高的技术上限；正常采集由操作者在 Reset 阶段按 Esc 结束，
# 因此实际采集数量不必预先固定。
MAX_EPISODES="${BARBELL_MAX_EPISODES:-1000}"
EPISODE_SECONDS="${BARBELL_EPISODE_SECONDS:-30}"
RESET_SECONDS="${BARBELL_RESET_SECONDS:-45}"

# ---------- 3. 参数和设备存在性检查 ----------
if ! [[ "$MAX_EPISODES" =~ ^[1-9][0-9]*$ ]]; then
  echo "BARBELL_MAX_EPISODES must be a positive integer." >&2
  exit 2
fi

for path in "$LEFT_FOLLOWER" "$LEFT_LEADER" "$TOP_CAMERA" "$LEFT_WRIST_CAMERA"; do
  if [[ ! -e "$path" ]]; then
    echo "Required device is missing: $path" >&2
    exit 1
  fi
done

# 拒绝覆盖已有数据集，防止损坏已经采好的 Episode。
DATASET_DIR="${DATA_ROOT}/${REPO_ID}"
if [[ -e "$DATASET_DIR" ]]; then
  echo "Dataset already exists; choose a new BARBELL_REPO_ID: $DATASET_DIR" >&2
  exit 1
fi

# ---------- 4. 诊断日志和两路相机配置 ----------
REPORT_DIR="$(dirname "$0")/../reports/$(printf '%s' "$REPO_ID" | tr '/' '_')"
mkdir -p "$REPORT_DIR"
export BARBELL_DIAGNOSTIC_LOG="${BARBELL_DIAGNOSTIC_LOG:-${REPORT_DIR}/raw_leader_sidecar.csv}"

CAMERAS="{'front': {'type': 'opencv', 'index_or_path': '${TOP_CAMERA}', 'width': 640, 'height': 480, 'fps': 30, 'rotation': 180, 'fourcc': 'MJPG'}, 'wrist': {'type': 'opencv', 'index_or_path': '${LEFT_WRIST_CAMERA}', 'width': 640, 'height': 480, 'fps': 30, 'fourcc': 'MJPG'}}"

# ---------- 5. 显示本轮配置和按键语义 ----------
echo "Continuous collection dataset: $REPO_ID"
echo "Dataset directory: $DATASET_DIR"
echo "Per-episode maximum: ${EPISODE_SECONDS}s; reset maximum: ${RESET_SECONDS}s"
echo "Right = save successful episode / finish reset early"
echo "Left  = discard failed episode and re-record"
echo "Esc during recording = discard partial episode and stop"
echo "Esc during reset     = preserve completed episode and stop"
echo "Keep the physical E-stop ready throughout collection."
echo "Diagnostic sidecar (not used for training): $BARBELL_DIAGNOSTIC_LOG"
read -r -p "Align both arms, clear both camera views, then press Enter to connect..."

export BARBELL_START_MAX_DELTA_DEG="${BARBELL_START_MAX_DELTA_DEG:-8.0}"

# ---------- 6. 正式采集 ----------
# 暂时关闭 set -e，保证 recorder 报错后仍能执行审计和汇总。
set +e
"${LEROBOT_BIN}/python" "$(dirname "$0")/safe_lerobot_cli.py" record \
  --robot.type=so101_follower \
  --robot.port="${LEFT_FOLLOWER}" \
  --robot.id=SO101_FOLLOWER_ID \
  --robot.use_degrees=true \
  --robot.disable_torque_on_disconnect=true \
  --robot.max_relative_target=5 \
  --robot.cameras="${CAMERAS}" \
  --teleop.type=so101_leader \
  --teleop.port="${LEFT_LEADER}" \
  --teleop.id=SO101_LEADER_ID \
  --teleop.use_degrees=true \
  --display_data=true \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATASET_DIR" \
  --dataset.fps=30 \
  --dataset.num_episodes="$MAX_EPISODES" \
  --dataset.episode_time_s="$EPISODE_SECONDS" \
  --dataset.reset_time_s="$RESET_SECONDS" \
  --dataset.video=true \
  --dataset.push_to_hub=false \
  --dataset.single_task="Lift the blue barbell with the left arm and place it in the orange tray"
RECORD_STATUS=$?
set -e

# ---------- 7. 对落盘数据进行自动审计 ----------
if [[ -d "$DATASET_DIR/data" ]]; then
  set +e
  "${LEROBOT_BIN}/python" "$(dirname "$0")/audit_barbell_dataset.py" "$REPO_ID" \
    --root "$DATASET_DIR" \
    --output "$REPORT_DIR/audit.json"
  AUDIT_STATUS=$?
  set -e
else
  AUDIT_STATUS=2
  echo "No finalized episodes were found; automatic audit was skipped." >&2
fi

# ---------- 8. 分析 Leader→实际动作→Follower 的跟踪延迟 ----------
if [[ -s "$BARBELL_DIAGNOSTIC_LOG" ]]; then
  "${LEROBOT_BIN}/python" "$(dirname "$0")/analyze_raw_leader_sidecar.py" \
    "$BARBELL_DIAGNOSTIC_LOG" \
    --output "$REPORT_DIR/raw_leader_analysis.json"
else
  echo "No raw Leader sidecar samples were produced." >&2
fi

# ---------- 9. 输出结果路径并传播失败状态 ----------
echo "Dataset: $REPO_ID"
echo "Audit: $REPORT_DIR/audit.json"
echo "Raw Leader sidecar: $BARBELL_DIAGNOSTIC_LOG"
echo "Raw Leader analysis: $REPORT_DIR/raw_leader_analysis.json"
if (( RECORD_STATUS != 0 )); then
  echo "Recorder exited with error $RECORD_STATUS. Inspect the traceback before reuse." >&2
  exit "$RECORD_STATUS"
fi
if (( AUDIT_STATUS != 0 )); then
  echo "Automatic integrity audit failed. Do not train on this batch yet." >&2
  exit "$AUDIT_STATUS"
fi
echo "Collection batch finalized and automatic integrity audit passed."
