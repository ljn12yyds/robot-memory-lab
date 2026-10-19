#!/usr/bin/env bash
# Evo-RL stage 2: infer value/n-step advantage and write ACP tags to a derived dataset.
set -euo pipefail

BARBELL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVORL_ROOT="${EVORL_ROOT:-/home/user/ljn/Evo-RL}"
SOURCE_ROOT="${PI05_HIL_LEROBOT_ROOT:-${BARBELL_ROOT}/data/rl_rollouts/pi05_hil_lerobot_v1_128ep_20260814}"
DATASET_ROOT="${PI05_HIL_ACP_ROOT:-${BARBELL_ROOT}/data/rl_rollouts/pi05_hil_lerobot_v1_128ep_acp_20260814}"
DATASET_ID="${PI05_HIL_ACP_ID:-barbell/pi05_hil_lerobot_v1_128ep_acp_20260814}"
VALUE_RUN="${PISTAR06_RUN_NAME:-pi05_hil_pistar06_8k_v1}"
VALUE_CHECKPOINT="${PISTAR06_CHECKPOINT:-${EVORL_ROOT}/outputs/value_train/${VALUE_RUN}}"
TAG="${PI05_ACP_TAG:-pi05_hil_pistar06_v1}"
FORCE_INTERVENTION_POSITIVE="${PI05_ACP_FORCE_INTERVENTION_POSITIVE:-true}"
# Our HIL default: successful intervention frames are positive and failed
# intervention frames are negative. Set this to false to reproduce Evo-RL's
# original unconditional-positive intervention baseline.
GATE_INTERVENTION_BY_SUCCESS="${PI05_ACP_GATE_INTERVENTION_BY_SUCCESS:-true}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="${PISTAR06_NUM_GPUS:-4}"
PER_GPU_BATCH="${PISTAR06_INFER_PER_GPU_BATCH:-8}"
NUM_WORKERS="${PISTAR06_INFER_NUM_WORKERS:-2}"
RUN_NAME="${PISTAR06_INFER_RUN_NAME:-pi05_hil_pistar06_infer_v1}"
OUTPUT_DIR="${PISTAR06_INFER_OUTPUT_DIR:-${EVORL_ROOT}/outputs/value_infer/${RUN_NAME}}"
LOG_DIR="${BARBELL_ROOT}/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

[[ -d "$SOURCE_ROOT" ]] || { echo "Missing clean dataset: $SOURCE_ROOT" >&2; exit 1; }
[[ -d "$VALUE_CHECKPOINT" ]] || { echo "Missing value checkpoint: $VALUE_CHECKPOINT" >&2; exit 1; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Output already exists: $OUTPUT_DIR" >&2; exit 1; }
if [[ ! -e "$DATASET_ROOT" ]]; then
  mkdir -p "$DATASET_ROOT"
  cp -a "$SOURCE_ROOT/meta" "$DATASET_ROOT/meta"
  cp -a "$SOURCE_ROOT/data" "$DATASET_ROOT/data"
  ln -s "$SOURCE_ROOT/videos" "$DATASET_ROOT/videos"
  cp -a "$SOURCE_ROOT/training_view_provenance.json" "$DATASET_ROOT/"
else
  echo "Derived ACP dataset already exists; refusing an ambiguous overwrite: $DATASET_ROOT" >&2
  exit 1
fi
mkdir -p "$LOG_DIR" /home/user/ljn/.hf-cache/datasets

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export HF_HOME=/home/user/ljn/.hf-cache
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
  "${EVORL_ROOT}/.venv/bin/lerobot-value-infer" \
  --dataset.repo_id="$DATASET_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --inference.checkpoint_path="$VALUE_CHECKPOINT" \
  --inference.checkpoint_ref=last \
  --runtime.device=cuda \
  --runtime.batch_size="$PER_GPU_BATCH" \
  --runtime.num_workers="$NUM_WORKERS" \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.force_intervention_positive="$FORCE_INTERVENTION_POSITIVE" \
  --acp.gate_intervention_by_episode_success="$GATE_INTERVENTION_BY_SUCCESS" \
  --acp.intervention_field=complementary_info.is_intervention \
  --acp.value_field="complementary_info.value_${TAG}" \
  --acp.advantage_field="complementary_info.advantage_${TAG}" \
  --acp.indicator_field="complementary_info.acp_indicator_${TAG}" \
  --seed=1000 \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$RUN_NAME" \
  2>&1 | tee "$LOG_FILE"
