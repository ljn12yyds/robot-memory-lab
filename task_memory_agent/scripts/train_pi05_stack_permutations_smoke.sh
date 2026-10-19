#!/usr/bin/env bash
set -euo pipefail

EVORL_ROOT="${EVORL_ROOT:-/home/user/ljn/Evo-RL}"
TRAIN_BIN="${EVORL_ROOT}/.venv/bin/lerobot-train"
START_MODEL="${PI05_START_MODEL:-${EVORL_ROOT}/outputs/checkpoints/pi05_base}"
DATASET_ID="robotwin_stack3_permutations30_subtasks_v1"
DATASET_ROOT="/home/user/.cache/huggingface/lerobot/${DATASET_ID}"
RUN_NAME="${PI05_RUN_NAME:-pi05_stack3_permutations30_smoke200_v1}"
OUTPUT_DIR="${EVORL_ROOT}/outputs/${RUN_NAME}"
LOG_FILE="/home/user/ljn/agent_iclr_robot/runs/${RUN_NAME}.log"
STEPS="${PI05_STEPS:-200}"
RENAME_MAP='{"observation.images.cam_high":"observation.images.base_0_rgb","observation.images.cam_left_wrist":"observation.images.left_wrist_0_rgb","observation.images.cam_right_wrist":"observation.images.right_wrist_0_rgb"}'

[[ -x "${TRAIN_BIN}" ]]
[[ -d "${START_MODEL}" ]]
[[ -d "${DATASET_ROOT}" ]]
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "output already exists: ${OUTPUT_DIR}" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export HF_DATASETS_CACHE="${EVORL_ROOT}/.cache/hf_datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "${EVORL_ROOT}"
"${TRAIN_BIN}" \
    --dataset.repo_id="${DATASET_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.video_backend=pyav \
    --policy.path="${START_MODEL}" \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.train_expert_only=true \
    --policy.freeze_vision_encoder=true \
    --policy.gradient_checkpointing=true \
    --policy.compile_model=false \
    --policy.push_to_hub=false \
    --rename_map="${RENAME_MAP}" \
    --batch_size=1 \
    --gradient_accumulation_steps=8 \
    --num_workers=0 \
    --steps="${STEPS}" \
    --log_freq=10 \
    --save_checkpoint=true \
    --save_freq=100 \
    --eval_freq=0 \
    --wandb.enable=false \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${RUN_NAME}" \
    2>&1 | tee "${LOG_FILE}"
