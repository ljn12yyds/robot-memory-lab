#!/usr/bin/env bash
set -euo pipefail

root="/home/user/ljn/agent_iclr_robot"
model="/home/user/ljn/models/Qwen3-VL-8B-Instruct"
download_log="${root}/logs/qwen_download.log"
test_log="${root}/logs/qwen_integration.log"

cd "${root}"
while ! grep -q 'MODEL_READY=' "${download_log}" 2>/dev/null; do
    if ! pgrep -f 'scripts/download_qwen.py' >/dev/null; then
        echo "Qwen download stopped before completion" >&2
        exit 1
    fi
    sleep 60
done

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="${root}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used \
    --format=csv,noheader > "${root}/logs/qwen_gpu_before.txt"

/home/user/ljn/LiLa-WAM/.venv/bin/python \
    scripts/test_qwen_integration.py \
    --model "${model}" \
    > "${test_log}" 2>&1

echo "QWEN_INTEGRATION_READY=${root}/logs/qwen_integration_result.json"
