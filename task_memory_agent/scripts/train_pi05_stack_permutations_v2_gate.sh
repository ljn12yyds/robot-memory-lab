#!/usr/bin/env bash
set -euo pipefail

EVORL_ROOT="${EVORL_ROOT:-/home/user/ljn/Evo-RL}"
AGENT_ROOT="${AGENT_ROOT:-/home/user/ljn/agent_iclr_robot}"
DATASET_ID="robotwin_stack3_permutations30_subtasks_v3_trainstats"
DATASET_ROOT="/home/user/.cache/huggingface/lerobot/${DATASET_ID}"
RAW_ROOT="/home/user/ljn/RoboTwin/data/memory_pi05_permutations30/stack_blocks_three/aloha_agilex"
START_MODEL="${PI05_START_MODEL:-${EVORL_ROOT}/outputs/checkpoints/pi05_base}"
RUN_NAME="${PI05_RUN_NAME:-pi05_stack3_corrected_gate200_v1}"
OUTPUT_DIR="${EVORL_ROOT}/outputs/${RUN_NAME}"
LOG_FILE="${AGENT_ROOT}/runs/${RUN_NAME}.log"
PREFLIGHT="${AGENT_ROOT}/runs/pi05_dataset_v3_preflight.json"
STEPS="${PI05_STEPS:-200}"
SAVE_FREQ="${PI05_SAVE_FREQ:-${STEPS}}"
PYTHON="${EVORL_ROOT}/.venv/bin/python"
TRAIN_EPISODES="$(${PYTHON} -c 'import json; print(json.dumps(list(range(72))))')"
RENAME_MAP='{"observation.images.cam_high":"observation.images.base_0_rgb","observation.images.cam_left_wrist":"observation.images.left_wrist_0_rgb","observation.images.cam_right_wrist":"observation.images.right_wrist_0_rgb"}'

[[ -x "${EVORL_ROOT}/.venv/bin/lerobot-train" ]]
[[ -d "${START_MODEL}" ]]
[[ -d "${DATASET_ROOT}" ]]
[[ ! -e "${OUTPUT_DIR}" ]]

/home/user/ljn/RoboTwin/.venv_rt2/bin/python "${AGENT_ROOT}/scripts/audit_pi05_dataset.py" \
    --dataset-root "${DATASET_ROOT}" \
    --raw-data "${RAW_ROOT}/data" \
    --scene-info "${RAW_ROOT}/scene_info.json" \
    --stats-episode-max-exclusive 72 \
    --output "${PREFLIGHT}" >/dev/null
"${PYTHON}" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"] == "PASS", d' "${PREFLIGHT}"
grep -q action_is_pad "${EVORL_ROOT}/src/lerobot/policies/pi05/modeling_pi05.py"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export HF_DATASETS_CACHE="${EVORL_ROOT}/.cache/hf_datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MASK_SMOKE="${AGENT_ROOT}/runs/pi05_masked_forward_pretrain.json"
"${EVORL_ROOT}/.venv/bin/python" "${AGENT_ROOT}/scripts/smoke_pi05_masked_forward.py" \
    --checkpoint "${START_MODEL}" \
    --dataset-root "${DATASET_ROOT}" \
    --repo-id "${DATASET_ID}" \
    --output "${MASK_SMOKE}" \
    --device cuda >/dev/null
"${PYTHON}" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"] == "PASS", d' "${MASK_SMOKE}"

cd "${EVORL_ROOT}"
"${EVORL_ROOT}/.venv/bin/lerobot-train" \
    --dataset.repo_id="${DATASET_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.episodes="${TRAIN_EPISODES}" \
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
    --save_freq="${SAVE_FREQ}" \
    --eval_freq=0 \
    --wandb.enable=false \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${RUN_NAME}" \
    2>&1 | tee "${LOG_FILE}"
