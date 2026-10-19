#!/usr/bin/env bash
# Evo-RL 原生 π0.5 人工介入数据采集入口（单左臂 SO101）。
# 策略阶段主臂跟随从臂；方向键标记恢复/成功/最终失败。
set -euo pipefail

EVO_ROOT="${EVO_ROOT:-/home/user/ljn/Evo-RL}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${EVO_ROOT}/.venv/bin/python"
CHECKPOINT="${PI05_HIL_CHECKPOINT:-${EVO_ROOT}/outputs/pi05_barbell_bc_continue_20k_v1/checkpoints/020000/pretrained_model}"

FOLLOWER_PORT="${LEFT_FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_FOLLOWER_SERIAL-if00}"
LEADER_PORT="${LEFT_LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_LEADER_SERIAL-if00}"
BASE_CAMERA="${TOP_CAMERA_PATH:-/dev/v4l/by-path/pci-0000:05:00.0-usb-0:2:1.0-video-index0}"
WRIST_CAMERA="${LEFT_WRIST_CAMERA_PATH:-/dev/v4l/by-path/pci-0000:00:14.0-usb-0:4.4:1.0-video-index0}"
CALIBRATION_DIR="${FOLLOWER_CALIBRATION_DIR:-/home/user/.cache/huggingface/lerobot/calibration/robots/so101_follower}"
LEADER_CALIBRATION_DIR="${LEADER_CALIBRATION_DIR:-/home/user/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
REPO_ID="${PI05_HIL_REPO_ID:-barbell/eval_pi05_barbell_hil_${RUN_STAMP}}"
DATA_PARENT="${PI05_HIL_DATA_ROOT:-${PROJECT_ROOT}/data/rl_rollouts/pi05_human_inloop}"
DATASET_ROOT="${PI05_HIL_DATASET_ROOT:-${DATA_PARENT}/${REPO_ID##*/}}"
EPISODES="${PI05_HIL_EPISODES:-20}"
REPLAN_STEPS="${PI05_HIL_REPLAN_STEPS:-20}"
MAX_STEP_DEG="${PI05_HIL_MAX_STEP_DEG:-5}"
ACP_ENABLE="${PI05_HIL_ACP_ENABLE:-false}"
ACP_USE_CFG="${PI05_HIL_ACP_USE_CFG:-false}"
ACP_CFG_BETA="${PI05_HIL_ACP_CFG_BETA:-1.0}"
DEVICE_WAIT_SECONDS="${PI05_HIL_DEVICE_WAIT_SECONDS:-15}"
ALLOW_UNVALIDATED_LIMITS="${PI05_HIL_ALLOW_UNVALIDATED_LIMITS:-0}"
OBJECT_ID="${PI05_HIL_OBJECT_ID:-blue_barbell_01}"
SETUP_ID="${PI05_HIL_SETUP_ID:-barbell_left_tray_right_v1}"

if [[ ! "$MAX_STEP_DEG" =~ ^5([.]0+)?$ && "$ALLOW_UNVALIDATED_LIMITS" != "1" ]]; then
  echo "Refusing unvalidated PI05_HIL_MAX_STEP_DEG=${MAX_STEP_DEG}." >&2
  echo "Validated HIL execution uses a constant 5-degree step limit." >&2
  echo "For an intentional experiment only, set PI05_HIL_ALLOW_UNVALIDATED_LIMITS=1." >&2
  exit 2
fi

for path in "$PYTHON_BIN" "$CHECKPOINT" "${LEADER_CALIBRATION_DIR}/SO101_LEADER_ID.json"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path is missing: $path" >&2
    exit 1
  fi
done

wait_for_device_path() {
  local path="$1"
  local deadline=$((SECONDS + DEVICE_WAIT_SECONDS))
  while [[ ! -e "$path" && $SECONDS -lt $deadline ]]; do
    echo "Waiting for USB device to enumerate: $path" >&2
    sleep 0.5
  done
  if [[ ! -e "$path" ]]; then
    echo "Required USB device did not appear within ${DEVICE_WAIT_SECONDS}s: $path" >&2
    echo "Check the powered USB hub/cable and install scripts/install_lerobot_usb_power_rule.sh." >&2
    exit 1
  fi
}

for path in "$FOLLOWER_PORT" "$LEADER_PORT" "$BASE_CAMERA" "$WRIST_CAMERA"; do
  wait_for_device_path "$path"
done
if [[ -e "$DATASET_ROOT" ]]; then
  echo "Dataset already exists: $DATASET_ROOT" >&2
  exit 1
fi
if ! [[ "$REPLAN_STEPS" =~ ^[1-9][0-9]*$ ]] || (( REPLAN_STEPS > 50 )); then
  echo "PI05_HIL_REPLAN_STEPS must be an integer from 1 to 50." >&2
  exit 2
fi
if [[ "$ACP_ENABLE" != "true" && "$ACP_ENABLE" != "false" ]]; then
  echo "PI05_HIL_ACP_ENABLE must be true or false." >&2
  exit 2
fi
if [[ "$ACP_USE_CFG" != "true" && "$ACP_USE_CFG" != "false" ]]; then
  echo "PI05_HIL_ACP_USE_CFG must be true or false." >&2
  exit 2
fi
if [[ "$ACP_USE_CFG" == "true" && "$ACP_ENABLE" != "true" ]]; then
  echo "PI05_HIL_ACP_USE_CFG=true requires PI05_HIL_ACP_ENABLE=true." >&2
  exit 2
fi

mkdir -p "$DATA_PARENT"
CAMERAS="{'base_0_rgb': {'type': 'opencv', 'index_or_path': '${BASE_CAMERA}', 'width': 640, 'height': 480, 'fps': 30, 'rotation': 180, 'fourcc': 'MJPG'}, 'left_wrist_0_rgb': {'type': 'opencv', 'index_or_path': '${WRIST_CAMERA}', 'width': 640, 'height': 480, 'fps': 30, 'fourcc': 'MJPG'}}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HIL_LAUNCHER_PATH="${PROJECT_ROOT}/scripts/record_pi05_human_inloop.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
# Use the foreground collection terminal instead of desktop-global pynput
# hotkeys. Set PI05_HIL_KEYBOARD_MODE=auto only to opt back into the old mode.
export LEROBOT_KEYBOARD_MODE="${PI05_HIL_KEYBOARD_MODE:-tty}"

echo "Evo-RL PI0.5 human-in-loop collection"
echo "Dataset: $DATASET_ROOT"
echo "Checkpoint: $CHECKPOINT"
echo "Right arrow = save success and enter RESET"
echo "Up arrow = preventive intervention when failure is likely but not yet confirmed"
echo "Left arrow = confirmed autonomous failure and human recovery"
echo "Down arrow = mark the current human recovery as terminal failure and enter RESET"
echo "After Up/Left, support the powered leader and press Enter to release its torque and start teleoperation."
echo "After preventive correction or human recovery, press Right arrow to save success."
echo "Space = pause/resume all policy and teleop execution"
echo "Esc = stop"
echo "Each policy episode starts only after you press Enter."
echo "There is no episode or RESET timeout; arrows/Enter/Space/Esc drive the workflow."
echo "Policy replans after ${REPLAN_STEPS} executed actions."
echo "Validated safety limits: constant ${MAX_STEP_DEG}° step and 2° soft-limit proposal tolerance."
echo "PI0.5 normalized actions are clipped to [-1,1]; unsafe chunks are resampled up to 3 times."
echo "Three consecutive unsafe chunks pause policy execution and enter human intervention."
echo "After Right Arrow, teleop takes over; manually restore the arm and scene, then press Enter."
echo "The first Enter prompt appears after the model, cameras, follower, and leader are ready."
echo "Keyboard mode: ${LEROBOT_KEYBOARD_MODE} (tty accepts keys only in the focused collection terminal)."
echo "HIL Dataset v1: object_id=${OBJECT_ID} setup_id=${SETUP_ID}"
echo "ACP inference: enable=${ACP_ENABLE} use_cfg=${ACP_USE_CFG} cfg_beta=${ACP_CFG_BETA}"

cd "$EVO_ROOT"
exec "$PYTHON_BIN" "${PROJECT_ROOT}/scripts/safe_evo_hil_cli.py" \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id=SO101_FOLLOWER_ID \
  --robot.use_degrees=true \
  --robot.disable_torque_on_disconnect=true \
  --robot.max_relative_target="$MAX_STEP_DEG" \
  --robot.calibration_dir="$CALIBRATION_DIR" \
  --robot.cameras="$CAMERAS" \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id=SO101_LEADER_ID \
  --teleop.calibration_dir="$LEADER_CALIBRATION_DIR" \
  --teleop.use_degrees=true \
  --policy.path="$CHECKPOINT" \
  --policy.device=cuda \
  --policy.n_action_steps="$REPLAN_STEPS" \
  --policy_sync_parallel=false \
  --acp_inference.enable="$ACP_ENABLE" \
  --acp_inference.use_cfg="$ACP_USE_CFG" \
  --acp_inference.cfg_beta="$ACP_CFG_BETA" \
  --intervention_toggle_key=i \
  --pause_toggle_key=space \
  --episode_success_key=right \
  --episode_failure_key=left \
  --episode_preventive_key=up \
  --episode_terminal_failure_key=down \
  --hil_dataset_v1_enabled=true \
  --hil_object_id="$OBJECT_ID" \
  --hil_setup_id="$SETUP_ID" \
  --display_data=false \
  --play_sounds=true \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.fps=30 \
  --dataset.num_episodes="$EPISODES" \
  --dataset.video=true \
  --dataset.push_to_hub=false \
  --dataset.single_task="Lift the blue barbell with the left arm and place it in the orange tray"
