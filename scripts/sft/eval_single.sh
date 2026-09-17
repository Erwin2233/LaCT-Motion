#!/bin/bash
# 单 epoch 评测脚本，可指定显卡
# 用法:
#   bash eval_single.sh --epoch 13 --gpus 4,5 --repeat 20
#   bash eval_single.sh --epoch 5  --gpus 4   --repeat 1
#   bash eval_single.sh --epoch 13                         # 默认 GPU 0, repeat 1
#   bash eval_single.sh --ckpt checkpoints/sft/checkpoint-epoch6 --tag c1_epoch6 --gpus 0
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
PYTHON=${PYTHON:-python}
PROJECT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
CONFIG=${PROJECT_DIR}/options/sft/t2m_coconut.yaml
cd "${PROJECT_DIR}"
RESULT_DIR=${PROJECT_DIR}/eval_results
LOG_DIR=${PROJECT_DIR}/logs

# ── 默认值 ──
EPOCH=""
GPU_IDS="0"
REPEAT=1
MULTIMODALITY=""
CKPT_OVERRIDE=""
TAG=""
FLOPS_MODE="estimate"

# ── 参数解析 ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epoch)   EPOCH="$2";   shift 2 ;;
        --gpus)    GPU_IDS="$2"; shift 2 ;;
        --repeat)  REPEAT="$2";  shift 2 ;;
        --ckpt)    CKPT_OVERRIDE="$2"; shift 2 ;;
        --tag)     TAG="$2";     shift 2 ;;
        --flops-mode) FLOPS_MODE="$2"; shift 2 ;;
        --multimodality) MULTIMODALITY="--multimodality"; shift ;;
        *) echo "未知参数: $1"; echo "用法: bash eval_single.sh --epoch <N> [--gpus <id,id,...>] [--repeat <N>] [--flops-mode estimate|profile|off] [--multimodality]"; echo "  或: bash eval_single.sh --ckpt <path> --tag <name> [--gpus <id,id,...>] [--repeat <N>] [--flops-mode estimate|profile|off]"; exit 1 ;;
    esac
done

if [[ -n "${CKPT_OVERRIDE}" ]]; then
    CKPT="${CKPT_OVERRIDE}"
    if [[ -z "${TAG}" ]]; then
        TAG=$(basename "${CKPT}")
    fi
    EPOCH="${TAG}"
elif [[ -n "${EPOCH}" ]]; then
    CKPT=${PROJECT_DIR}/checkpoints/sft/runs/checkpoint-epoch${EPOCH}
    TAG="checkpoint-epoch${EPOCH}"
else
    echo "错误: 必须指定 --epoch 或 --ckpt"
    echo "用法: bash eval_single.sh --epoch <N> [--gpus <id,id,...>] [--repeat <N>]"
    echo "  或: bash eval_single.sh --ckpt <path> --tag <name> [--gpus <id,id,...>] [--repeat <N>]"
    exit 1
fi

if [[ ! -d "${CKPT}" ]]; then
    echo "[ERROR] Checkpoint 不存在: ${CKPT}"
    exit 1
fi

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

# 将 gpu_ids 拆成数组, 例如 "4,5" -> (4 5)
IFS=',' read -ra GPUS <<< "${GPU_IDS}"
NUM_GPUS=${#GPUS[@]}

echo "============================================"
echo "  Evaluate checkpoint-epoch${EPOCH}"
echo "  GPUs: ${GPU_IDS} (${NUM_GPUS} 张卡)"
echo "  Repeat: ${REPEAT}"
echo "  Time: $(date)"
echo "============================================"

if [[ ${NUM_GPUS} -eq 1 || ${REPEAT} -le 1 ]]; then
    # ---- 单卡评测 ----
    OUT=${RESULT_DIR}/checkpoint-epoch${EPOCH}.json
    LOG=${LOG_DIR}/eval_epoch${EPOCH}.log
    echo "[RUN] GPU ${GPUS[0]} — repeat=${REPEAT} — log: ${LOG}"
    CUDA_VISIBLE_DEVICES=${GPUS[0]} ${PYTHON} "${PROJECT_DIR}/eval_t2m.py" \
        "${CONFIG}" \
        --checkpoint "${CKPT}" \
        --repeat "${REPEAT}" \
        --flops-mode "${FLOPS_MODE}" \
        --output "${OUT}" \
        --device cuda:0 \
        ${MULTIMODALITY} \
        2>&1 | tee "${LOG}"
    echo "[DONE] Result: ${OUT}"
else
    # ---- 多卡并行: 将 repeat 均分 ----
    PIDS=()
    OUTS=()
    LOGS=()
    COUNTS=()
    ASSIGNED=0

    for ((i = 0; i < NUM_GPUS; i++)); do
        if [[ $i -lt $((NUM_GPUS - 1)) ]]; then
            CNT=$((REPEAT / NUM_GPUS))
        else
            CNT=$((REPEAT - ASSIGNED))
        fi
        ASSIGNED=$((ASSIGNED + CNT))

        GPU=${GPUS[$i]}
        OUT_I=${RESULT_DIR}/checkpoint-epoch${EPOCH}_gpu${GPU}.json
        LOG_I=${LOG_DIR}/eval_epoch${EPOCH}_gpu${GPU}.log

        OUTS+=("${OUT_I}")
        LOGS+=("${LOG_I}")
        COUNTS+=("${CNT}")

        echo "[RUN] GPU ${GPU} — repeat=${CNT} — log: ${LOG_I}"
        CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} "${PROJECT_DIR}/eval_t2m.py" \
            "${CONFIG}" \
            --checkpoint "${CKPT}" \
            --repeat "${CNT}" \
            --flops-mode "${FLOPS_MODE}" \
            --output "${OUT_I}" \
            --device cuda:0 \
            ${MULTIMODALITY} \
            > "${LOG_I}" 2>&1 &
        PIDS+=($!)
    done

    # 等待所有任务完成
    FAIL=0
    for ((i = 0; i < NUM_GPUS; i++)); do
        wait ${PIDS[$i]} || { echo "[FAIL] GPU ${GPUS[$i]} — see ${LOGS[$i]}"; FAIL=1; }
    done

    if [[ ${FAIL} -ne 0 ]]; then
        echo "部分评测失败，请检查上方日志。"
        exit 1
    fi

    # 合并多卡结果取加权平均
    MERGED=${RESULT_DIR}/checkpoint-epoch${EPOCH}.json

    MERGE_ARGS=()
    for ((i = 0; i < NUM_GPUS; i++)); do
        MERGE_ARGS+=("${OUTS[$i]}" "${COUNTS[$i]}")
    done

    ${PYTHON} -c "
import json, math, sys

args = sys.argv[1:]
files  = args[0::2]
counts = [int(c) for c in args[1::2]]
total  = sum(counts)

avg_keys = (
    'fid', 'fid_ci',
    'top1', 'top1_ci',
    'top2', 'top2_ci',
    'top3', 'top3_ci',
    'diversity', 'diversity_ci',
    'matching_score', 'matching_score_ci',
    'multimodality', 'multimodality_ci',
    'motion_emb_cos', 'motion_emb_cos_ci',
    'semantic_cos', 'semantic_cos_ci',
    'flops_per_sample', 'flops_per_sample_ci',
    'generated_tokens_total', 'generated_tokens_total_ci',
    'generated_tokens_per_sample', 'generated_tokens_per_sample_ci',
    'per_sample_latency_ms', 'per_sample_latency_ms_ci',
    'peak_gpu_memory_bytes_mean',
)
max_keys = ('peak_gpu_memory_bytes_max', 'peak_gpu_memory_mb_max')

weighted_sum = {k: 0.0 for k in avg_keys}
weights = {k: 0 for k in avg_keys}
merged = {}

def _as_float(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None

for f, n in zip(files, counts):
    with open(f) as fp:
        data = json.load(fp)

    for k in avg_keys:
        v = _as_float(data.get(k))
        if v is None or math.isnan(v):
            continue
        weighted_sum[k] += v * n
        weights[k] += n

    for k in max_keys:
        v = _as_float(data.get(k))
        if v is None or math.isnan(v):
            continue
        if k not in merged or v > merged[k]:
            merged[k] = v

for k in avg_keys:
    if weights[k] > 0:
        merged[k] = weighted_sum[k] / weights[k]

if 'peak_gpu_memory_bytes_max' in merged and 'peak_gpu_memory_mb_max' not in merged:
    merged['peak_gpu_memory_mb_max'] = merged['peak_gpu_memory_bytes_max'] / (1024 ** 2)

merged['repeat'] = total
merged['note'] = f'weighted average from {len(files)}-GPU parallel eval'

with open('${MERGED}', 'w') as fp:
    json.dump(merged, fp, indent=2)

print('Merged result:')
for k, v in merged.items():
    if isinstance(v, float):
        if 'flops' in k:
            print(f'  {k:28s}: {v:.3e}')
        elif 'bytes' in k:
            print(f'  {k:28s}: {v:.0f}')
        elif 'latency' in k:
            print(f'  {k:28s}: {v:.3f}')
        else:
            print(f'  {k:28s}: {v:.4f}')
    else:
        print(f'  {k:28s}: {v}')
" "${MERGE_ARGS[@]}"

    echo ""
    echo "[DONE] Merged result: ${MERGED}"
fi

echo ""
echo "============================================"
echo "  Finished at $(date)"
echo "============================================"
