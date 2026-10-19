#!/usr/bin/env bash
# Evaluate one SFT/ACP PI0.5 checkpoint with the same HIL and safety settings.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVO_ROOT="${EVO_ROOT:-/home/user/ljn/Evo-RL}"
LABEL="${1:-}"

case "$LABEL" in
  sft)
    CHECKPOINT="${EVO_ROOT}/outputs/pi05_barbell_expert_quantileclip_effb8_30k_v2/checkpoints/030000/pretrained_model"
    ACP_ENABLE=false
    OUTPUT_TAG=sft_control
    ;;
  005000|010000|015000|020000|025000|030000)
    CHECKPOINT="${EVO_ROOT}/outputs/pi05_barbell_hil_acp_30k_v1/checkpoints/${LABEL}/pretrained_model"
    ACP_ENABLE=true
    OUTPUT_TAG="acp_${LABEL}_positive"
    ;;
  *)
    echo "Usage: $0 {sft|005000|010000|015000|020000|025000|030000}" >&2
    exit 2
    ;;
esac

if [[ ! -f "${CHECKPOINT}/model.safetensors" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PI05_HIL_EPISODES="${PI05_EVAL_EPISODES:-5}"
export PI05_HIL_REPLAN_STEPS="${PI05_EVAL_REPLAN_STEPS:-20}"
export PI05_HIL_MAX_STEP_DEG="${PI05_EVAL_MAX_STEP_DEG:-5}"
export PI05_HIL_ACP_ENABLE="$ACP_ENABLE"
export PI05_HIL_ACP_USE_CFG=false
export PI05_HIL_ACP_CFG_BETA=1.0
export PI05_HIL_OBJECT_ID="${PI05_EVAL_OBJECT_ID:-blue_barbell_01}"
export PI05_HIL_SETUP_ID="${PI05_EVAL_SETUP_ID:-barbell_left_tray_right_v1}"
export PI05_HIL_CHECKPOINT="$CHECKPOINT"
export PI05_HIL_DATA_ROOT="${PI05_EVAL_DATA_ROOT:-${PROJECT_ROOT}/data/rl_rollouts/pi05_checkpoint_eval_v1/${OUTPUT_TAG}}"
export LEADER_CALIBRATION_DIR="${LEADER_CALIBRATION_DIR:-/home/user/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader}"
export FOLLOWER_CALIBRATION_DIR="${FOLLOWER_CALIBRATION_DIR:-/home/user/.cache/huggingface/lerobot/calibration/robots/so101_follower}"

echo "PI0.5 checkpoint evaluation"
echo "Label: ${LABEL}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Episodes: ${PI05_HIL_EPISODES}"
echo "ACP positive conditioning: ${ACP_ENABLE}"
echo "Output parent: ${PI05_HIL_DATA_ROOT}"

exec "${PROJECT_ROOT}/scripts/record_pi05_human_inloop.sh"
