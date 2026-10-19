#!/usr/bin/env bash
# 第一阶段行为克隆（BC）训练入口。
# 默认读取 100 条合并数据，在 GPU 1 上训练 10000 步并定期保存 checkpoint。
# 纯训练脚本，不连接机械臂；输出目录已存在时会拒绝覆盖。
set -euo pipefail

# ---------- 1. 路径与超参数 ----------
# 所有变量都可在命令前覆盖，便于开展新实验而不用修改脚本。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/user/ljn/lerobot}"
PY_ENV="${PY_ENV:-/home/user/miniconda3/envs/lerobot}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
DATASET_ID="${SMOLVLA_DATASET_ID:-barbell/so101_barbell_100_merged_v1}"
DATASET_ROOT="${SMOLVLA_DATASET_ROOT:-${PROJECT_ROOT}/data/barbell/so101_barbell_100_merged_v1}"
STEPS="${SMOLVLA_STEPS:-10000}"
BATCH_SIZE="${SMOLVLA_BATCH_SIZE:-4}"
RUN_NAME="${SMOLVLA_RUN_NAME:-smolvla_barbell_10k_v1}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${RUN_NAME}"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

# ---------- 2. 创建日志/缓存并防止覆盖旧实验 ----------
mkdir -p "${LOG_DIR}" "${PROJECT_ROOT}/.cache/huggingface/datasets"

if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Output already exists; refusing to overwrite: ${OUTPUT_DIR}" >&2
    exit 1
fi

# ---------- 3. 固定 GPU 和本地数据缓存 ----------
export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.cache/huggingface/datasets"
export TOKENIZERS_PARALLELISM=false

cd "${LEROBOT_ROOT}"

# ---------- 4. 打印最终生效配置 ----------
echo "Run: ${RUN_NAME}"
echo "GPU: ${GPU}"
echo "Dataset: ${DATASET_ID}"
echo "Dataset directory: ${DATASET_ROOT}"
echo "Steps: ${STEPS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"

# ---------- 5. 启动 LeRobot BC 训练 ----------
# exec 让训练进程直接接管当前 shell；标准输出和错误都写入日志。
exec "${PY_ENV}/bin/lerobot-train" \
    --policy.path="${PROJECT_ROOT}/models/smolvla_base_barbell_2cam" \
    --policy.push_to_hub=false \
    --dataset.repo_id="${DATASET_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${RUN_NAME}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers=4 \
    --steps="${STEPS}" \
    --log_freq=50 \
    --save_freq=2000 \
    --eval_freq=100000 \
    --wandb.enable=false \
    >>"${LOG_FILE}" 2>&1
