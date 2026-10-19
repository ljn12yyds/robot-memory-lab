#!/usr/bin/env bash
# Evo-RL stage 3: update PI0.5 with positive/negative Advantage prompt tags.
set -euo pipefail

BARBELL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVORL_ROOT="${EVORL_ROOT:-/home/user/ljn/Evo-RL}"
DATASET_ROOT="${PI05_HIL_ACP_ROOT:-${BARBELL_ROOT}/data/rl_rollouts/pi05_hil_lerobot_v1_128ep_acp_20260814}"
DATASET_ID="${PI05_HIL_ACP_ID:-barbell/pi05_hil_lerobot_v1_128ep_acp_20260814}"
POLICY_CHECKPOINT="${PI05_ACP_BASE_MODEL:-${EVORL_ROOT}/outputs/pi05_barbell_expert_quantileclip_effb8_30k_v2/checkpoints/030000/pretrained_model}"
TAG="${PI05_ACP_TAG:-pi05_hil_pistar06_v1}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="${PI05_ACP_NUM_GPUS:-4}"
PER_GPU_BATCH="${PI05_ACP_PER_GPU_BATCH:-8}"
NUM_WORKERS="${PI05_ACP_NUM_WORKERS:-2}"
STEPS="${PI05_ACP_STEPS:-30000}"
SAVE_FREQ="${PI05_ACP_SAVE_FREQ:-5000}"
RUN_NAME="${PI05_ACP_RUN_NAME:-pi05_barbell_hil_acp_30k_v1}"
OUTPUT_DIR="${PI05_ACP_OUTPUT_DIR:-${EVORL_ROOT}/outputs/${RUN_NAME}}"
LOG_DIR="${BARBELL_ROOT}/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

[[ -d "$DATASET_ROOT" ]] || { echo "Missing ACP-annotated dataset: $DATASET_ROOT" >&2; exit 1; }
[[ -f "$POLICY_CHECKPOINT/model.safetensors" ]] || { echo "Missing PI0.5 checkpoint" >&2; exit 1; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Output already exists: $OUTPUT_DIR" >&2; exit 1; }
mkdir -p "$LOG_DIR" /home/user/ljn/.hf-cache/datasets

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
# PI0.5's PaliGemma tokenizer is already present in this cache.
export HF_HOME=/home/user/.cache/huggingface
export HF_DATASETS_CACHE=/home/user/ljn/.hf-cache/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$EVORL_ROOT"
"${EVORL_ROOT}/.venv/bin/accelerate" launch \
  --multi_gpu \
  --num_processes="$NUM_GPUS" \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  "${EVORL_ROOT}/.venv/bin/lerobot-train" \
  --dataset.repo_id="$DATASET_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=false \
  --policy.path="$POLICY_CHECKPOINT" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.push_to_hub=false \
  --acp.enable=true \
  --acp.indicator_field="complementary_info.acp_indicator_${TAG}" \
  --acp.indicator_dropout_prob=0.3 \
  --batch_size="$PER_GPU_BATCH" \
  --gradient_accumulation_steps=1 \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --log_freq=20 \
  --save_checkpoint=true \
  --save_freq="$SAVE_FREQ" \
  --eval_freq=0 \
  --seed=1000 \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$RUN_NAME" \
  --wandb.enable=false \
  2>&1 | tee "$LOG_FILE"
