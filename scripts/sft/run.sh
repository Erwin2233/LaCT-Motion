#!/bin/bash
# Coconut-style Latent SFT Training for Text-to-Motion
# Usage:
#   bash run.sh          # 自动后台运行 + 实时查看日志（Ctrl+C 仅退出日志，训练继续）
#   bash run.sh --fg     # 前台运行（不 setsid，Ctrl+C 会中断训练）

set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5
export TOKENIZERS_PARALLELISM=false
NPROC=2
PYTHON=${PYTHON:-python}
TORCHRUN=${TORCHRUN:-torchrun}
PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
CONFIG=${PROJECT_DIR}/options/sft/t2m_coconut.yaml
LOG_DIR=${PROJECT_DIR}/logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=${LOG_DIR}/train_${TIMESTAMP}.log

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}"

# ── Auto setsid: if not already detached and not --fg, re-exec under setsid ──
if [[ "${1:-}" != "--fg" && "${__RUN_DETACHED:-}" != "1" ]]; then
    # Re-exec self under setsid + nohup, detached from terminal
    # Pass LOG_FILE so the child uses the same path (avoids timestamp drift)
    __RUN_DETACHED=1 __RUN_LOG_FILE="${LOG_FILE}" \
        setsid nohup bash "$0" --fg >"${LOG_FILE}" 2>&1 &
    BG_PID=$!
    echo "Training launched in background."
    echo "  PID: ${BG_PID}"
    echo "  Log: ${LOG_FILE}"
    echo ""
    echo "To view logs:  tail -f ${LOG_FILE}"
    echo "To stop:       kill ${BG_PID}"
    exit 0
fi
# Strip --fg so it doesn't interfere below
shift 2>/dev/null || true
# Use parent's log path if available (avoids timestamp drift in re-exec)
LOG_FILE=${__RUN_LOG_FILE:-${LOG_FILE}}

# ── Main training logic ──────────────────────────────────────────────────────

echo "============================================"
echo "  Latent-CoT-Motion Training"
echo "  GPUs: ${CUDA_VISIBLE_DEVICES} (${NPROC} procs)"
echo "  Log:  ${LOG_FILE}"
echo "  PID:  $$"
echo "  Time: $(date)"
echo "============================================"

# Step 1: Build data (if not already built)
if [ ! -f "${PROJECT_DIR}/data/t2m_train.json" ]; then
    echo ""
    echo "[Step 1/2] Building training data with VQ-VAE encoding..."
    ${PYTHON} get_train_data.py --output-dir data \
        --think-steps ${PROJECT_DIR}/data/texts_think_steps.json \
        --device cuda:0
    echo "Data build complete."
else
    echo ""
    echo "[Step 1/2] Training data already exists, skipping build."
fi

# Step 2: Launch distributed training
echo ""
echo "[Step 2/2] Launching training..."
# Use MASTER_PORT env var if set; otherwise pick a random port in 29500-39500
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 10000))}
echo "  master_port=${MASTER_PORT}"
${TORCHRUN} \
    --nproc_per_node=${NPROC} \
    --master_port=${MASTER_PORT} \
    train_sft.py ${CONFIG}

echo ""
echo "Training finished at $(date)"
