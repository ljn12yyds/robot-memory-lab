#!/usr/bin/env bash
# PI0.5 SO101 expert-only v2: bounded quantiles + true gradient accumulation.
# Offline training only; this script never connects to robot hardware.
set -euo pipefail

BARBELL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVORL_ROOT="${EVORL_ROOT:-/home/user/ljn/Evo-RL}"
TRAIN_BIN="${EVORL_ROOT}/.venv/bin/lerobot-train"
BASE_MODEL="${PI05_BASE_MODEL:-${EVORL_ROOT}/outputs/checkpoints/pi05_base}"
DATASET_ID="${PI05_DATASET_ID:-barbell/so101_barbell_100_merged_v1}"
DATASET_ROOT="${PI05_DATASET_ROOT:-${BARBELL_ROOT}/data/barbell/so101_barbell_100_merged_v1}"
RENAME_MAP='{"observation.images.front":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}'

GPU="${CUDA_VISIBLE_DEVICES:-2}"
STEPS="${PI05_STEPS:-30000}"
MICRO_BATCH="${PI05_MICRO_BATCH:-1}"
ACCUMULATION="${PI05_GRAD_ACCUMULATION:-8}"
NUM_WORKERS="${PI05_NUM_WORKERS:-2}"
SAVE_FREQ="${PI05_SAVE_FREQ:-5000}"
RUN_NAME="${PI05_RUN_NAME:-pi05_barbell_expert_quantileclip_effb8_30k_v2}"
OUTPUT_DIR="${PI05_OUTPUT_DIR:-${EVORL_ROOT}/outputs/${RUN_NAME}}"
LOG_DIR="${BARBELL_ROOT}/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

for value in "$STEPS" "$MICRO_BATCH" "$ACCUMULATION" "$SAVE_FREQ"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "训练步数和 batch 参数必须是正整数" >&2; exit 2; }
done
EFFECTIVE_BATCH=$((MICRO_BATCH * ACCUMULATION))
SAMPLE_EXPOSURES=$((STEPS * EFFECTIVE_BATCH))

[[ -x "$TRAIN_BIN" ]] || { echo "找不到训练程序: $TRAIN_BIN" >&2; exit 1; }
[[ -d "$BASE_MODEL" ]] || { echo "找不到基础权重: $BASE_MODEL" >&2; exit 1; }
[[ -d "$DATASET_ROOT" ]] || { echo "找不到数据集: $DATASET_ROOT" >&2; exit 1; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "输出目录已存在，拒绝覆盖: $OUTPUT_DIR" >&2; exit 1; }

mkdir -p "$LOG_DIR" "${EVORL_ROOT}/.cache/hf_datasets"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_DATASETS_CACHE="${EVORL_ROOT}/.cache/hf_datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "PI0.5 expert-only v2 training"
echo "Quantile contract: training/inference state and action clipped to [-1,1]"
echo "Optimizer updates: $STEPS"
echo "Micro batch / accumulation / effective batch: $MICRO_BATCH / $ACCUMULATION / $EFFECTIVE_BATCH"
echo "Sample exposures: $SAMPLE_EXPOSURES (about 3.0 epochs at default settings)"
echo "Output: $OUTPUT_DIR"

cd "$EVORL_ROOT"
"$TRAIN_BIN" \
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
    --policy.scheduler_warmup_steps=1000 \
    --policy.scheduler_decay_steps="$STEPS" \
    --rename_map="$RENAME_MAP" \
    --batch_size="$MICRO_BATCH" \
    --gradient_accumulation_steps="$ACCUMULATION" \
    --num_workers="$NUM_WORKERS" \
    --steps="$STEPS" \
    --log_freq=20 \
    --save_checkpoint=true \
    --save_freq="$SAVE_FREQ" \
    --eval_freq=0 \
    --wandb.enable=false \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$RUN_NAME" \
    2>&1 | tee "$LOG_FILE"
