#!/usr/bin/env bash
# 第一轮 AWR/advantage 加权微调入口。
# 先根据 Episode 成败生成逐帧 advantage，再从当前 BC checkpoint 继续训练。
# 纯训练脚本，不连接机械臂；输入必须是 prepare_barbell_rl_dataset.py 的输出。
set -euo pipefail

# ---------- 1. 路径、数据集和微调超参数 ----------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/user/ljn/lerobot}"
PY_ENV="${PY_ENV:-/home/user/miniconda3/envs/lerobot}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
DATASET_ID="${BARBELL_RL_REPO_ID:-barbell/so101_barbell_rl_round1}"
DATASET_ROOT="${BARBELL_RL_DATASET_ROOT:-${PROJECT_ROOT}/data/${DATASET_ID}}"
LABELS="${BARBELL_RL_LABELS:-${DATASET_ROOT}/meta/rl_episode_labels.json}"
BASE_CHECKPOINT="${BARBELL_RL_BASE_CHECKPOINT:-${PROJECT_ROOT}/outputs/smolvla_barbell_10k_v1/checkpoints/010000/pretrained_model}"
STEPS="${BARBELL_RL_STEPS:-3000}"
BATCH_SIZE="${BARBELL_RL_BATCH_SIZE:-4}"
RUN_NAME="${BARBELL_RL_RUN_NAME:-smolvla_barbell_rl_round1}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${RUN_NAME}"
ADVANTAGES="${DATASET_ROOT}/segment_advantages.parquet"
# 成功人工接管轨迹采用三段式权重：早期自主段保留基础权重，只将接管前
# 一小段风险窗口降权，接管后人工纠正片段升权。数值是人工设定的
# advantage 分数，实际 loss 权重还会经过 exp(advantage/beta)。
INTERVENTION_POLICY_ADVANTAGE="${BARBELL_INTERVENTION_POLICY_ADVANTAGE:--0.5}"
INTERVENTION_HUMAN_ADVANTAGE="${BARBELL_INTERVENTION_HUMAN_ADVANTAGE:-1.0}"
INTERVENTION_RISK_WINDOW_FRAMES="${BARBELL_INTERVENTION_RISK_WINDOW_FRAMES:-90}"

# ---------- 2. 防止覆盖旧实验 ----------
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Output already exists; refusing to overwrite: ${OUTPUT_DIR}" >&2
    exit 1
fi

# ---------- 3. 根据 Episode 标签生成逐帧 advantage ----------
"${PY_ENV}/bin/python" "${PROJECT_ROOT}/scripts/compute_barbell_segment_advantages.py" \
    --dataset-root "${DATASET_ROOT}" \
    --labels "${LABELS}" \
    --output "${ADVANTAGES}" \
    --intervention-policy-advantage "${INTERVENTION_POLICY_ADVANTAGE}" \
    --intervention-human-advantage "${INTERVENTION_HUMAN_ADVANTAGE}" \
    --intervention-risk-window-frames "${INTERVENTION_RISK_WINDOW_FRAMES}"

# ---------- 4. 固定 GPU 并进入 LeRobot 源码目录 ----------
export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false
cd "${LEROBOT_ROOT}"

# ---------- 5. 从 BC checkpoint 做 advantage 加权微调 ----------
# use_rabc=true 复用逐样本 loss 权重接口；head_mode=advantage 表示直接读取
# segment_advantages.parquet，权重公式为 exp(advantage / beta)。
exec "${PY_ENV}/bin/lerobot-train" \
    --policy.path="${BASE_CHECKPOINT}" \
    --policy.push_to_hub=false \
    --dataset.repo_id="${DATASET_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${RUN_NAME}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers=4 \
    --steps="${STEPS}" \
    --use_rabc=true \
    --rabc_progress_path="${ADVANTAGES}" \
    --rabc_head_mode=advantage \
    --rabc_beta=1.0 \
    --log_freq=50 \
    --save_freq=1000 \
    --eval_freq=100000 \
    --wandb.enable=false
