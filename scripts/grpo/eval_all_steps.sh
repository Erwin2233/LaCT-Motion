#!/bin/bash
# 批量评测 grpo outputs 下所有 step 的脚本
# 每张卡同时只评测一个 checkpoint，多张卡并行，跑完一个接下一个
# 用法:
#   bash eval_all_steps.sh --gpus 0,1,2,3
#   bash eval_all_steps.sh --gpus 4,5,6,7 --multimodality
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
EVAL_SCRIPT="${SCRIPT_DIR}/scripts/sft/eval_single.sh"
CKPT_ROOT=${SCRIPT_DIR}/checkpoints/grpo
LOG_DIR=${SCRIPT_DIR}/eval_logs
mkdir -p "${LOG_DIR}"

# ── 默认值 ──
GPU_IDS=""
EXTRA_ARGS=()

# ── 参数解析 ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)    GPU_IDS="$2"; shift 2 ;;
        --multimodality) EXTRA_ARGS+=("--multimodality"); shift ;;
        *) echo "未知参数: $1"; echo "用法: bash eval_all_steps.sh --gpus <id,id,...> [--multimodality]"; exit 1 ;;
    esac
done

if [[ -z "${GPU_IDS}" ]]; then
    echo "错误: 必须通过 --gpus 指定显卡，例如 --gpus 0,1,2,3"
    echo "用法: bash eval_all_steps.sh --gpus <id,id,...> [--multimodality]"
    exit 1
fi

# ── 拆分 GPU 列表 ──
IFS=',' read -ra GPUS <<< "${GPU_IDS}"
NUM_GPUS=${#GPUS[@]}

# ── 收集所有 step 目录，按 step 数字排序 ──
STEPS=()
for d in "${CKPT_ROOT}"/step-*; do
    [[ -d "$d" ]] && STEPS+=("$d")
done
IFS=$'\n' STEPS=($(for s in "${STEPS[@]}"; do echo "$s"; done | sort -t'-' -k2 -n)); unset IFS

# 如果 final 目录存在，追加到末尾
if [[ -d "${CKPT_ROOT}/final" ]]; then
    STEPS+=("${CKPT_ROOT}/final")
fi

TOTAL=${#STEPS[@]}

if [[ ${TOTAL} -eq 0 ]]; then
    echo "未找到任何 step 目录: ${CKPT_ROOT}"
    exit 1
fi

echo "============================================"
echo "  批量评测 GRPO checkpoints"
echo "  GPUs: ${GPU_IDS} (${NUM_GPUS} 张卡)"
echo "  每张卡同时评测 1 个 checkpoint, repeat=1"
echo "  共 ${TOTAL} 个 checkpoint:"
for s in "${STEPS[@]}"; do
    echo "    - $(basename "$s")"
done
echo "  开始时间: $(date)"
echo "============================================"
echo ""

# ── 将 checkpoint 轮询分配给各 GPU ──
# GPU_QUEUE[i] 存放分配给第 i 张卡的 checkpoint 索引列表（空格分隔）
declare -a GPU_QUEUE
for ((i = 0; i < NUM_GPUS; i++)); do
    GPU_QUEUE[$i]=""
done

for ((j = 0; j < TOTAL; j++)); do
    gpu_idx=$((j % NUM_GPUS))
    GPU_QUEUE[$gpu_idx]="${GPU_QUEUE[$gpu_idx]} $j"
done

# 打印分配方案
echo "分配方案:"
for ((i = 0; i < NUM_GPUS; i++)); do
    names=""
    for idx in ${GPU_QUEUE[$i]}; do
        names="${names} $(basename "${STEPS[$idx]}")"
    done
    echo "  GPU ${GPUS[$i]}:${names}"
done
echo ""

# ── 每张卡启动一个 worker，串行跑分配到的 checkpoint ──
WORKER_PIDS=()
WORKER_LOGS=()

for ((i = 0; i < NUM_GPUS; i++)); do
    GPU=${GPUS[$i]}
    QUEUE="${GPU_QUEUE[$i]}"
    WORKER_LOG="${LOG_DIR}/worker_gpu${GPU}.log"
    WORKER_LOGS+=("${WORKER_LOG}")

    (
        for idx in ${QUEUE}; do
            CKPT="${STEPS[$idx]}"
            NAME=$(basename "${CKPT}")
            TAG="grpo_${NAME}"

            echo ""
            echo "[GPU ${GPU}] ======== 开始评测 ${NAME} (tag: ${TAG}) ======== $(date)"

            if bash "${EVAL_SCRIPT}" \
                --ckpt "${CKPT}" \
                --tag "${TAG}" \
                --gpus "${GPU}" \
                --repeat 1 \
                "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; then
                echo "[GPU ${GPU}] [OK] ${NAME} 评测完成 $(date)"
            else
                echo "[GPU ${GPU}] [FAIL] ${NAME} 评测失败 $(date)"
            fi
        done
    ) > "${WORKER_LOG}" 2>&1 &

    WORKER_PIDS+=($!)
    echo "Worker GPU ${GPU} 已启动 (PID ${!}), 日志: ${WORKER_LOG}"
done

echo ""
echo "所有 worker 已启动，等待完成..."
echo "可用 tail -f ${LOG_DIR}/worker_gpu*.log 查看实时进度"
echo ""

# ── 等待所有 worker 完成 ──
FAIL=0
for ((i = 0; i < NUM_GPUS; i++)); do
    if wait ${WORKER_PIDS[$i]}; then
        echo "[OK] GPU ${GPUS[$i]} worker 全部完成"
    else
        echo "[FAIL] GPU ${GPUS[$i]} worker 存在失败任务，请查看: ${WORKER_LOGS[$i]}"
        FAIL=1
    fi
done

echo ""
echo "============================================"
echo "  全部评测结束"
echo "  完成时间: $(date)"
echo "  总计: ${TOTAL} 个 checkpoint, ${NUM_GPUS} 张卡"

# 汇总各 worker 的成功/失败
OK_COUNT=0
FAIL_COUNT=0
for log in "${WORKER_LOGS[@]}"; do
    ok=$(grep -c '\[OK\]' "$log" 2>/dev/null || true)
    fail=$(grep -c '\[FAIL\]' "$log" 2>/dev/null || true)
    OK_COUNT=$((OK_COUNT + ok))
    FAIL_COUNT=$((FAIL_COUNT + fail))
done
echo "  成功: ${OK_COUNT} 个"
echo "  失败: ${FAIL_COUNT} 个"

if [[ ${FAIL_COUNT} -gt 0 ]]; then
    echo ""
    echo "  失败详情:"
    for log in "${WORKER_LOGS[@]}"; do
        grep '\[FAIL\]' "$log" 2>/dev/null | sed 's/^/    /' || true
    done
fi
echo "============================================"
