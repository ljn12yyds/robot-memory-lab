#!/usr/bin/env bash
set -euo pipefail

EVORL_ROOT=${EVORL_ROOT:-/home/user/ljn/Evo-RL}
DATASET_ROOT=${PI05_SUBTASK_DATASET_ROOT:-/home/user/ljn/barbell_task/data/barbell/so101_barbell_subtask_v1}
DATASET_ID=${PI05_SUBTASK_DATASET_ID:-barbell/so101_barbell_subtask_v1}
BASE_MODEL=${PI05_BASE_MODEL:-$EVORL_ROOT/outputs/checkpoints/pi05_base}
RUN_NAME=${PI05_RUN_NAME:-pi05_barbell_subtask_sft_2k_v1}
OUTPUT_DIR=${PI05_OUTPUT_DIR:-$EVORL_ROOT/outputs/$RUN_NAME}
STEPS=${PI05_STEPS:-2000}
BATCH_SIZE=${PI05_BATCH_SIZE:-1}
GPU=${CUDA_VISIBLE_DEVICES:-2}
RENAME_MAP='{"observation.images.front":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}'

[[ -x "$EVORL_ROOT/.venv/bin/lerobot-train" ]] || { echo "missing lerobot-train" >&2; exit 1; }
[[ -d "$DATASET_ROOT" ]] || { echo "missing subtask dataset: $DATASET_ROOT" >&2; exit 1; }
[[ -f "$DATASET_ROOT/subtask_provenance.json" ]] || { echo "dataset lacks subtask provenance" >&2; exit 1; }
[[ -d "$BASE_MODEL" ]] || { echo "missing PI0.5 base model: $BASE_MODEL" >&2; exit 1; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "output already exists: $OUTPUT_DIR" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$EVORL_ROOT"
.venv/bin/lerobot-train \
  --dataset.repo_id="$DATASET_ID" \
  --dataset.root="$DATASET_ROOT" \
  --policy.path="$BASE_MODEL" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.push_to_hub=false \
  --rename_map="$RENAME_MAP" \
  --batch_size="$BATCH_SIZE" \
  --num_workers=2 \
  --steps="$STEPS" \
  --log_freq=20 \
  --save_checkpoint=true \
  --save_freq="$STEPS" \
  --eval_freq=0 \
  --wandb.enable=false \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$RUN_NAME"

