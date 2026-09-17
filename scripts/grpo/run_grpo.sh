#!/bin/bash
# ==============================================================
# GRPO Training Launch Script for CoconutMotion
# Aligned with: trl/run_t2m_semantic_grpo_full.sh
# ==============================================================
# Usage:
#   bash run_grpo.sh              # default: 4 GPUs
#   bash run_grpo.sh 8            # use 8 GPUs
#   bash run_grpo.sh 2            # use 2 GPUs (quick test)
#   CUDA_VISIBLE_DEVICES=0,1 bash run_grpo.sh 2  # specific GPUs
# ==============================================================
export CUDA_VISIBLE_DEVICES=4,5,6,7
set -euo pipefail

# ======================== Config ========================
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG_TEMPLATE="${PROJECT_DIR}/options/grpo/t2m_grpo.yaml"
CONFIG="${CONFIG_TEMPLATE}"
NUM_GPUS="${1:-4}"

# 默认不使用 flash_attn，与 UniMo 运行方式保持一致。
# 可通过环境变量覆盖，例如:
#   ATTN_IMPL_OVERRIDE=sdpa bash run_grpo.sh 8
ATTN_IMPL_OVERRIDE="${ATTN_IMPL_OVERRIDE:-eager}"

# Create a runtime config so we don't mutate the source YAML.
RUNTIME_CONFIG_DIR="${PROJECT_DIR}/outputs/grpo/runtime_configs"
mkdir -p "${RUNTIME_CONFIG_DIR}"
RUNTIME_CONFIG="${RUNTIME_CONFIG_DIR}/t2m_grpo_$(date +%Y%m%d_%H%M%S).yaml"

python - <<PY
import yaml

src = r"${CONFIG_TEMPLATE}"
dst = r"${RUNTIME_CONFIG}"
override = r"${ATTN_IMPL_OVERRIDE}"

with open(src) as f:
    cfg = yaml.safe_load(f)

old_impl = cfg.get("attn_implementation", None)
cfg["attn_implementation"] = override

with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

print(f"[run_grpo.sh] attn_implementation override: {old_impl} -> {override}")
print(f"[run_grpo.sh] runtime config: {dst}")
PY

CONFIG="${RUNTIME_CONFIG}"

VISIBLE_GPU_COUNT=$(python - <<'PY'
import os

cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if cvd:
    ids = [x.strip() for x in cvd.split(",") if x.strip()]
    print(len(ids))
else:
    try:
        import torch
        print(torch.cuda.device_count())
    except Exception:
        print(0)
PY
)

if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] NUM_GPUS must be a positive integer, got: ${NUM_GPUS}"
    exit 1
fi
if [ "${NUM_GPUS}" -le 0 ]; then
    echo "[ERROR] NUM_GPUS must be >= 1, got: ${NUM_GPUS}"
    exit 1
fi
if [ "${VISIBLE_GPU_COUNT}" -le 0 ]; then
    echo "[ERROR] No visible GPUs detected."
    echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    exit 1
fi
if [ "${NUM_GPUS}" -gt "${VISIBLE_GPU_COUNT}" ]; then
    echo "[ERROR] Requested NUM_GPUS=${NUM_GPUS}, but only ${VISIBLE_GPU_COUNT} GPU(s) are visible."
    echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "  Please run: bash run_grpo.sh ${VISIBLE_GPU_COUNT}  (or fewer)"
    exit 1
fi

# Use python and torchrun from the active Python environment.

# NCCL / distributed settings
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM="false"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export PYTHONUNBUFFERED=1  # force unbuffered stdout so prints appear immediately in nohup log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduce fragmentation from variable-length sequences

# torch.compile settings
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1  # avoid graph break from .item() in flash attention
export TORCH_LOGS="-dynamo"                  # suppress verbose dynamo warnings

# Pick a free distributed master port unless MASTER_PORT is explicitly set.
pick_free_port() {
    python - <<'PY'
import socket

def pick(start=29500, end=29999):
    for p in range(start, end + 1):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", p))
        except OSError:
            s.close()
            continue
        s.close()
        return p
    raise RuntimeError("No free TCP port found in [29500, 29999]")

print(pick())
PY
}

if [ -n "${MASTER_PORT:-}" ]; then
    SELECTED_MASTER_PORT="${MASTER_PORT}"
else
    SELECTED_MASTER_PORT="$(pick_free_port)"
    export MASTER_PORT="${SELECTED_MASTER_PORT}"
fi

# T2M reward model environment (aligned with TRL reference)
export T2M_DEVICE="${T2M_DEVICE:-cuda:0}"
export T2M_USE_BATCH_EMBEDDING="${T2M_USE_BATCH_EMBEDDING:-0}"

# Reward logging
export T2M_VERBOSE_REWARD="${T2M_VERBOSE_REWARD:-1}"
export T2M_REWARD_LOG_INTERVAL="${T2M_REWARD_LOG_INTERVAL:-10}"
export T2M_PRINT_SAMPLE_DETAILS="${T2M_PRINT_SAMPLE_DETAILS:-0}"

# Executables come from the environment activated by the caller.
command -v python >/dev/null
command -v torchrun >/dev/null

# ======================== Pre-flight Checks ========================
cd "${PROJECT_DIR}"

echo "============================================"
echo " GRPO Training for CoconutMotion"
echo " (aligned with TRL semantic_grpo_full)"
echo "============================================"
echo " Project dir : ${PROJECT_DIR}"
echo " Config      : ${CONFIG}"
echo " Num GPUs    : ${NUM_GPUS}"
echo " Visible GPUs: ${VISIBLE_GPU_COUNT}"
echo " Master port : ${SELECTED_MASTER_PORT}"
echo " Python      : $(which python)"
echo " PyTorch     : $(python -c 'import torch; print(torch.__version__)')"
echo " CUDA devices: ${CUDA_VISIBLE_DEVICES:-all}"
echo " T2M_DEVICE  : ${T2M_DEVICE}"
echo "============================================"

# Verify critical paths
SFT_CKPT=$(python -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('sft_checkpoint', ''))
")

if [ ! -d "${SFT_CKPT}" ]; then
    echo "[ERROR] SFT checkpoint not found: ${SFT_CKPT}"
    echo "  Available checkpoints:"
    ls -d checkpoints/sft/checkpoint-* 2>/dev/null || echo "  (none)"
    exit 1
fi
echo " SFT ckpt    : ${SFT_CKPT}"

TRAIN_DATA=$(python -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('train_path', ''))
")
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "[ERROR] Training data not found: ${TRAIN_DATA}"
    exit 1
fi
echo " Train data   : ${TRAIN_DATA}"

# Print key GRPO hyperparameters from config
python -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print('--------------------------------------------')
print(' Key GRPO Hyperparameters:')
print(f'  num_generations (G) : {cfg.get(\"num_generations\", \"?\")}')
print(f'  loss_type           : {cfg.get(\"loss_type\", \"?\")}')
print(f'  attn_implementation : {cfg.get(\"attn_implementation\", \"?\")}')
print(f'  beta (KL)           : {cfg.get(\"beta\", \"?\")}')
print(f'  epsilon (clip)      : {cfg.get(\"epsilon\", \"?\")}')
print(f'  lr                  : {cfg.get(\"lr\", \"?\")}')
print(f'  batch_size          : {cfg.get(\"batch_size_training\", \"?\")}')
print(f'  grad_accum          : {cfg.get(\"gradient_accumulation_steps\", \"?\")}')
print(f'  max_new_tokens      : {cfg.get(\"max_new_tokens\", \"?\")}')
print(f'  reward_preset       : {cfg.get(\"reward_preset\", \"?\")}')
print(f'  reward_weights      : {cfg.get(\"reward_weights\", \"?\")}')
print(f'  gradient_ckpt       : {cfg.get(\"gradient_checkpointing\", False)}')
print(f'  use_lora            : {cfg.get(\"use_lora\", False)}')
eff = cfg.get('batch_size_training',2) * cfg.get('gradient_accumulation_steps',16) * ${NUM_GPUS}
print(f'  effective_batch     : {eff} prompts/step ({eff}*{cfg.get(\"num_generations\",8)} = {eff*cfg.get(\"num_generations\",8)} completions)')
print('--------------------------------------------')
"

# Create output directory
SAVE_PATH=$(python -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('save_path', './outputs/grpo'))
")
mkdir -p "${SAVE_PATH}"
echo " Save path    : ${SAVE_PATH}"
echo "============================================"

# ======================== Launch Training ========================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${SAVE_PATH}/train_${TIMESTAMP}.log"

echo "[$(date)] Starting GRPO training..."
echo "  Log file: ${LOG_FILE}"
echo ""

nohup torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${SELECTED_MASTER_PORT}" \
    train_grpo.py \
    --config "${CONFIG}" \
    > "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "[$(date)] Training launched in background (PID: ${TRAIN_PID})"
echo "  Log file : ${LOG_FILE}"
echo "  Monitor  : tail -f ${LOG_FILE}"
echo "  Stop     : kill ${TRAIN_PID}"
echo "${TRAIN_PID}" > "${SAVE_PATH}/train_${TIMESTAMP}.pid"
