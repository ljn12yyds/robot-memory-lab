#!/usr/bin/env bash
# 强化学习 rollout 采集入口：SmolVLA 自动控制从臂并保存完整 LeRobot 数据。
# Recording 阶段右键保存成功、左键保存失败；失败数据不会被丢弃。
# 本脚本会真实控制机械臂，运行期间必须清空工作区并守住物理急停。
set -euo pipefail

# ---------- 1. 软件、硬件与基础 checkpoint ----------
# 路径均可用环境变量覆盖；默认使用已经验证过的左从臂和双相机。
LEROBOT_BIN="${LEROBOT_BIN:-/home/user/miniconda3/envs/lerobot/bin}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEFT_FOLLOWER="${LEFT_FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_FOLLOWER_SERIAL-if00}"
LEFT_LEADER="${LEFT_LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_LEADER_SERIAL-if00}"
TOP_CAMERA="${TOP_CAMERA_PATH:-/dev/v4l/by-path/pci-0000:05:00.0-usb-0:2:1.0-video-index0}"
LEFT_WRIST_CAMERA="${LEFT_WRIST_CAMERA_PATH:-/dev/v4l/by-path/pci-0000:00:14.0-usb-0:4.4:1.0-video-index0}"
CHECKPOINT="${BARBELL_POLICY_CHECKPOINT:-${PROJECT_ROOT}/outputs/smolvla_barbell_10k_v1/checkpoints/010000/pretrained_model}"
# ---------- 2. 本轮 rollout 的数据集参数 ----------
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
REPO_ID="${BARBELL_ROLLOUT_REPO_ID:-barbell/eval_so101_barbell_policy_rollouts_${RUN_STAMP}}"
# 所有强化学习 rollout 都集中到固定父目录。每轮仍使用独立子目录，
# 以保护各轮 LeRobot 的 Parquet、视频、Episode 索引和元数据不被覆盖。
RL_DATA_ROOT="${BARBELL_RL_DATA_ROOT:-${PROJECT_ROOT}/data/rl_rollouts}"
DATASET_DIR="${BARBELL_ROLLOUT_ROOT:-${RL_DATA_ROOT}/${REPO_ID##*/}}"
MAX_EPISODES="${BARBELL_ROLLOUT_EPISODES:-50}"
# 每条策略 rollout 最多运行 120 秒。操作者可在此之前用右键标记成功、
# 左键标记失败；若 120 秒内没有给出结果，则自动保存为超时失败。
EPISODE_SECONDS="${BARBELL_ROLLOUT_EPISODE_SECONDS:-120}"
RESET_SECONDS="${BARBELL_ROLLOUT_RESET_SECONDS:-45}"
MAX_STEP_DEG="${BARBELL_ROLLOUT_MAX_STEP_DEG:-5}"
# SmolVLA 每次会预测 50 步动作，但这里只执行前 10 步，然后丢弃剩余动作，
# 重新读取两路相机和关节状态进行下一次推理。这样可把视觉重规划间隔从
# 约 1.67 秒（50/30）缩短到约 0.33 秒（10/30，不含模型推理耗时）。
ACTION_STEPS="${BARBELL_ROLLOUT_ACTION_STEPS:-10}"
LABELS="${DATASET_DIR}/meta/rl_episode_labels.json"

# ---------- 3. 启动前检查 ----------
mkdir -p "$RL_DATA_ROOT"
for path in "$LEFT_FOLLOWER" "$LEFT_LEADER" "$TOP_CAMERA" "$LEFT_WRIST_CAMERA" "$CHECKPOINT"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path is missing: $path" >&2
    exit 1
  fi
done
if [[ -e "$DATASET_DIR" ]]; then
  echo "Dataset already exists; choose a new BARBELL_ROLLOUT_REPO_ID: $DATASET_DIR" >&2
  exit 1
fi
if ! [[ "$MAX_EPISODES" =~ ^[1-9][0-9]*$ ]]; then
  echo "BARBELL_ROLLOUT_EPISODES must be a positive integer." >&2
  exit 2
fi
if ! [[ "$ACTION_STEPS" =~ ^[1-9][0-9]*$ ]] || (( ACTION_STEPS > 50 )); then
  echo "BARBELL_ROLLOUT_ACTION_STEPS must be an integer from 1 to 50." >&2
  exit 2
fi
# LeRobot 官方要求由 policy 控制时，数据集短名称必须以 eval_ 开头。
DATASET_NAME="${REPO_ID##*/}"
if [[ "$DATASET_NAME" != eval_* ]]; then
  echo "Policy rollout dataset name must begin with eval_: ${DATASET_NAME}" >&2
  echo "Use for example: BARBELL_ROLLOUT_REPO_ID=barbell/eval_barbell_round1" >&2
  exit 2
fi

# ---------- 4. 相机、标签和诊断输出 ----------
# front 与训练时一致旋转 180°；wrist 保持原方向。
CAMERAS="{'front': {'type': 'opencv', 'index_or_path': '${TOP_CAMERA}', 'width': 640, 'height': 480, 'fps': 30, 'rotation': 180, 'fourcc': 'MJPG'}, 'wrist': {'type': 'opencv', 'index_or_path': '${LEFT_WRIST_CAMERA}', 'width': 640, 'height': 480, 'fps': 30, 'fourcc': 'MJPG'}}"
REPORT_DIR="${PROJECT_ROOT}/reports/$(printf '%s' "$REPO_ID" | tr '/' '_')"
mkdir -p "$REPORT_DIR"

# 这些环境变量通知 safe_lerobot_cli 切换到“失败也保存”的 RL 模式。
export BARBELL_RL_COLLECTION=1
export BARBELL_RL_LABELS="$LABELS"
export BARBELL_DIAGNOSTIC_LOG="${REPORT_DIR}/policy_rollout_sidecar.csv"
export BARBELL_START_MAX_DELTA_DEG="${BARBELL_START_MAX_DELTA_DEG:-8.0}"
export BARBELL_HYBRID_START_MAX_DELTA_DEG="${BARBELL_HYBRID_START_MAX_DELTA_DEG:-8.0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# 推理只使用已经下载到本机的 checkpoint 和 Hugging Face 缓存。
# 如果不设置离线模式，Transformers 即使发现了本地文件，也会先向
# huggingface.co 发送 HEAD 请求检查更新；网络不稳定时会在这里反复超时。
# 默认值设为 1 可保证现场采集不依赖外网。如确实需要联网更新模型，可在
# 命令前显式写 BARBELL_HF_OFFLINE=0。
BARBELL_HF_OFFLINE="${BARBELL_HF_OFFLINE:-1}"
if [[ "$BARBELL_HF_OFFLINE" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
else
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
fi

# ---------- 5. 显示配置与人工标注按键 ----------
echo "Policy rollout dataset: $REPO_ID"
echo "Fixed RL data root: $RL_DATA_ROOT"
echo "Dataset directory: $DATASET_DIR"
echo "Outcome labels: $LABELS"
echo "Checkpoint: $CHECKPOINT"
echo "Hugging Face offline mode: $BARBELL_HF_OFFLINE"
echo "Right during RECORDING = save success"
echo "Left during RECORDING  = save failure"
echo "Right during RESET     = begin next rollout"
echo "Esc during RECORDING   = discard partial rollout and stop"
echo "Esc during RESET       = preserve completed rollouts and stop"
echo "Episode timeout (${EPISODE_SECONDS}s) is saved as failure."
echo "Maximum relative command: ${MAX_STEP_DEG} degrees."
echo "Policy replanning: execute ${ACTION_STEPS}/50 predicted steps before using a fresh observation."
echo "Policy phase: Leader actively follows Follower."
echo "Space during RECORDING = release Leader torque and switch permanently to human teleoperation."
echo "Keep the physical E-stop ready throughout collection."
read -r -p "Clear the workspace, place the objects, then press Enter to load the policy..."

# ---------- 6. 模型连续控制并录制 ----------
# 暂时关闭 set -e，以便失败后仍然执行数据审计和错误汇总。
set +e
"${LEROBOT_BIN}/python" "${PROJECT_ROOT}/scripts/safe_lerobot_cli.py" record \
  --robot.type=so101_follower \
  --robot.port="${LEFT_FOLLOWER}" \
  --robot.id=SO101_FOLLOWER_ID \
  --robot.use_degrees=true \
  --robot.disable_torque_on_disconnect=true \
  --robot.max_relative_target="${MAX_STEP_DEG}" \
  --robot.cameras="${CAMERAS}" \
  --teleop.type=so101_leader \
  --teleop.port="${LEFT_LEADER}" \
  --teleop.id=SO101_LEADER_ID \
  --teleop.use_degrees=true \
  --policy.path="${CHECKPOINT}" \
  --policy.device=cuda \
  --policy.n_action_steps="${ACTION_STEPS}" \
  --display_data=false \
  --play_sounds=true \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_DIR}" \
  --dataset.fps=30 \
  --dataset.num_episodes="${MAX_EPISODES}" \
  --dataset.episode_time_s="${EPISODE_SECONDS}" \
  --dataset.reset_time_s="${RESET_SECONDS}" \
  --dataset.video=true \
  --dataset.push_to_hub=false \
  --dataset.single_task="Lift the blue barbell with the left arm and place it in the orange tray"
RECORD_STATUS=$?
set -e

# ---------- 7. 自动审计已经落盘的轨迹 ----------
if [[ -d "${DATASET_DIR}/data" ]]; then
  set +e
  "${LEROBOT_BIN}/python" "${PROJECT_ROOT}/scripts/audit_barbell_dataset.py" "$REPO_ID" \
    --root "$DATASET_DIR" \
    --output "${REPORT_DIR}/audit.json"
  AUDIT_STATUS=$?
  set -e
else
  AUDIT_STATUS=2
fi

# ---------- 8. 输出结果路径并传播错误 ----------
echo "Dataset: $DATASET_DIR"
echo "Labels: $LABELS"
echo "Audit: ${REPORT_DIR}/audit.json"
if (( RECORD_STATUS != 0 )); then
  echo "Policy recorder exited with error ${RECORD_STATUS}." >&2
  exit "$RECORD_STATUS"
fi
if (( AUDIT_STATUS != 0 )); then
  echo "Dataset integrity audit failed." >&2
  exit "$AUDIT_STATUS"
fi
