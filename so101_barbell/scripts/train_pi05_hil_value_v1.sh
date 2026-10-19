#!/usr/bin/env bash
# Evo-RL stage 1: train the Pistar06 value model on the unified PI0.5 HIL dataset.
set -euo pipefail

BARBELL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVORL_ROOT="${EVORL_ROOT:-/home/user/ljn/Evo-RL}"
DATASET_ROOT="${PI05_HIL_LEROBOT_ROOT:-${BARBELL_ROOT}/data/rl_rollouts/pi05_hil_lerobot_v1_128ep_20260814}"
DATASET_ID="${PI05_HIL_LEROBOT_ID:-barbell/pi05_hil_lerobot_v1_128ep_20260814}"
VISION_MODEL="${PISTAR06_VISION_MODEL:-/home/user/ljn/evo_rl_deps/models/siglip-so400m-patch14-384}"
LANGUAGE_MODEL="${PISTAR06_LANGUAGE_MODEL:-/home/user/ljn/evo_rl_deps/models/gemma-3-270m}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="${PISTAR06_NUM_GPUS:-4}"
PER_GPU_BATCH="${PISTAR06_PER_GPU_BATCH:-4}"
GRAD_ACCUMULATION="${PISTAR06_GRAD_ACCUMULATION:-2}"
NUM_WORKERS="${PISTAR06_NUM_WORKERS:-2}"
STEPS="${PISTAR06_STEPS:-8000}"
SAVE_FREQ="${PISTAR06_SAVE_FREQ:-4000}"
RUN_NAME="${PISTAR06_RUN_NAME:-pi05_hil_pistar06_8k_v1}"
OUTPUT_DIR="${PISTAR06_OUTPUT_DIR:-${EVORL_ROOT}/outputs/value_train/${RUN_NAME}}"
LOG_DIR="${BARBELL_ROOT}/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

[[ -x "${EVORL_ROOT}/.venv/bin/accelerate" ]] || { echo "Missing accelerate" >&2; exit 1; }
[[ -d "$DATASET_ROOT" ]] || { echo "Missing dataset: $DATASET_ROOT" >&2; exit 1; }
[[ -f "$VISION_MODEL/model.safetensors" ]] || { echo "Missing SigLIP weights" >&2; exit 1; }
[[ -f "$LANGUAGE_MODEL/model.safetensors" ]] || { echo "Missing Gemma weights" >&2; exit 1; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Output already exists: $OUTPUT_DIR" >&2; exit 1; }
mkdir -p "$LOG_DIR" /home/user/ljn/.hf-cache/datasets

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export HF_HOME=/home/user/ljn/.hf-cache
export HF_DATASETS_CACHE=/home/user/ljn/.hf-cache/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Evo-RL Pistar06 value training"
echo "Dataset: $DATASET_ROOT"
echo "GPUs / per-GPU batch / accumulation / effective batch: $NUM_GPUS / $PER_GPU_BATCH / $GRAD_ACCUMULATION / $((NUM_GPUS * PER_GPU_BATCH * GRAD_ACCUMULATION))"
echo "Steps: $STEPS"
echo "Output: $OUTPUT_DIR"

cd "$EVORL_ROOT"
"${EVORL_ROOT}/.venv/bin/accelerate" launch \
  --multi_gpu \
  --num_processes="$NUM_GPUS" \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  "${EVORL_ROOT}/.venv/bin/lerobot-value-train" \
  --dataset.repo_id="$DATASET_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=false \
  --value.type=pistar06 \
  --value.vision_repo_id="$VISION_MODEL" \
  --value.language_repo_id="$LANGUAGE_MODEL" \
  --value.device=cuda \
  --value.dtype=bfloat16 \
  --value.freeze_vision_encoder=false \
  --value.freeze_language_model=false \
  --value.use_gradient_checkpointing=false \
  --value.push_to_hub=false \
  --targets.success_field=episode_success \
  --targets.default_success=failure \
  --targets.c_fail_coef=1.0 \
  --batch_size="$PER_GPU_BATCH" \
  --gradient_accumulation_steps="$GRAD_ACCUMULATION" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --log_freq=20 \
  --save_checkpoint=true \
  --save_freq="$SAVE_FREQ" \
  --seed=1000 \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$RUN_NAME" \
  --wandb.enable=false \
  2>&1 | tee "$LOG_FILE"
