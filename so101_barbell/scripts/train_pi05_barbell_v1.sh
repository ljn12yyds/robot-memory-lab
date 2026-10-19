#!/usr/bin/env bash
# π0.5 在 SO101 哑铃示范数据上的第一阶段行为克隆训练入口。
#
# 这份脚本只读取离线数据并使用 GPU 训练，不会连接相机、主臂或从臂，
# 因此运行期间机械臂不会运动。默认进行 2000 次参数更新，约相当于先做一轮
# 可评估的短训练；确认真实机械臂效果后，再通过环境变量增加训练步数。
set -euo pipefail

# ---------- 1. 项目路径 ----------
# BARBELL_ROOT：任务数据、日志和本脚本所在目录。
# EVORL_ROOT：Evo-RL 源码、Python 虚拟环境和 π0.5 权重所在目录。
BARBELL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVORL_ROOT="${EVORL_ROOT:-/home/user/ljn/Evo-RL}"
TRAIN_BIN="${EVORL_ROOT}/.venv/bin/lerobot-train"
BASE_MODEL="${PI05_BASE_MODEL:-${EVORL_ROOT}/outputs/checkpoints/pi05_base}"

# ---------- 2. 数据集配置 ----------
# 旧数据只有 front 和 wrist 两路相机。训练时把它们改名为 π0.5 约定的
# base_0_rgb 和 left_wrist_0_rgb；缺少的右腕相机会由 π0.5 处理器自动补成
# 带 mask 的空图，不会伪装成真实右腕画面。
DATASET_ID="${PI05_DATASET_ID:-barbell/so101_barbell_100_merged_v1}"
DATASET_ROOT="${PI05_DATASET_ROOT:-${BARBELL_ROOT}/data/barbell/so101_barbell_100_merged_v1}"
RENAME_MAP='{"observation.images.front":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}'

# ---------- 3. 首轮训练超参数 ----------
# 冒烟测试实测 batch_size=1 时每步约 1.06 秒。默认 2000 步约需 35 分钟，
# 具体速度会受 GPU 占用和磁盘读图速度影响。
# train_expert_only=true：只训练动作专家相关部分，保留 VLM 的通用视觉语言能力。
# freeze_vision_encoder=true：冻结视觉编码器，减少显存并降低小数据过拟合风险。
# gradient_checkpointing=true：用额外计算换显存，让 24GB 级显卡也能训练。
GPU="${CUDA_VISIBLE_DEVICES:-2}"
STEPS="${PI05_STEPS:-2000}"
BATCH_SIZE="${PI05_BATCH_SIZE:-1}"
NUM_WORKERS="${PI05_NUM_WORKERS:-2}"
SAVE_FREQ="${PI05_SAVE_FREQ:-2000}"
RUN_NAME="${PI05_RUN_NAME:-pi05_barbell_bc_2k_v1}"
OUTPUT_DIR="${PI05_OUTPUT_DIR:-${EVORL_ROOT}/outputs/${RUN_NAME}}"
LOG_DIR="${BARBELL_ROOT}/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

# ---------- 4. 启动前检查 ----------
[[ -x "${TRAIN_BIN}" ]] || { echo "找不到训练程序: ${TRAIN_BIN}" >&2; exit 1; }
[[ -d "${BASE_MODEL}" ]] || { echo "找不到 π0.5 基础权重: ${BASE_MODEL}" >&2; exit 1; }
[[ -d "${DATASET_ROOT}" ]] || { echo "找不到训练数据集: ${DATASET_ROOT}" >&2; exit 1; }

mkdir -p "${LOG_DIR}" "${EVORL_ROOT}/.cache/hf_datasets"
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "输出目录已经存在，为防止覆盖旧实验而停止: ${OUTPUT_DIR}" >&2
    exit 1
fi

# ---------- 5. 离线模式与运行摘要 ----------
# 权重已经下载到本机，强制离线可避免 Hugging Face 网络超时。
export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_DATASETS_CACHE="${EVORL_ROOT}/.cache/hf_datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "π0.5 SO101 哑铃训练"
echo "GPU: ${GPU}"
echo "基础权重: ${BASE_MODEL}"
echo "数据集: ${DATASET_ROOT}"
echo "训练步数 / batch size: ${STEPS} / ${BATCH_SIZE}"
echo "输出目录: ${OUTPUT_DIR}"
echo "日志文件: ${LOG_FILE}"

cd "${EVORL_ROOT}"

# ---------- 6. 执行训练 ----------
# tee 同时把训练过程显示在终端并保存到日志。管道失败也会因 pipefail 返回非零。
"${TRAIN_BIN}" \
    --dataset.repo_id="${DATASET_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --policy.path="${BASE_MODEL}" \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.train_expert_only=true \
    --policy.freeze_vision_encoder=true \
    --policy.gradient_checkpointing=true \
    --policy.compile_model=false \
    --policy.push_to_hub=false \
    --rename_map="${RENAME_MAP}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --steps="${STEPS}" \
    --log_freq=20 \
    --save_checkpoint=true \
    --save_freq="${SAVE_FREQ}" \
    --eval_freq=0 \
    --wandb.enable=false \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${RUN_NAME}" \
    2>&1 | tee "${LOG_FILE}"
