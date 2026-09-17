#!/bin/bash
# Batch evaluation on GPUs 4 & 5
# Usage:
#   bash eval.sh                          # 评估所有 checkpoint（GPU 4,5 并行，每次跑两个）
#   bash eval.sh checkpoint-epoch19       # 只评估指定 checkpoint
#   bash eval.sh --repeat 20             # 所有 checkpoint，每个重复 20 次算置信区间

set -euo pipefail

export TOKENIZERS_PARALLELISM=false
PYTHON=${PYTHON:-python}
PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
CONFIG=${PROJECT_DIR}/options/sft/t2m_coconut.yaml
cd "${PROJECT_DIR}"
CKPT_DIR=${PROJECT_DIR}/checkpoints/sft/runs
RESULT_DIR=${PROJECT_DIR}/eval_results
LOG_DIR=${PROJECT_DIR}/logs

GPU0=4
GPU1=5
REPEAT=1

# ── 解析参数 ──────────────────────────────────────────────────────────────────
SPECIFIC_CKPT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repeat)
            REPEAT="$2"; shift 2 ;;
        checkpoint-*)
            SPECIFIC_CKPT="$1"; shift ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

# ── 收集要评估的 checkpoint ──────────────────────────────────────────────────
if [[ -n "${SPECIFIC_CKPT}" ]]; then
    CKPTS=("${SPECIFIC_CKPT}")
else
    # 按 epoch 数字排序
    CKPTS=($(ls -d "${CKPT_DIR}"/checkpoint-epoch* 2>/dev/null \
        | xargs -I{} basename {} \
        | sort -t 'h' -k2 -n))
fi

if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "No checkpoints found in ${CKPT_DIR}"
    exit 1
fi

echo "============================================"
echo "  Latent-CoT-Motion Evaluation"
echo "  GPUs: ${GPU0}, ${GPU1}"
echo "  Repeat: ${REPEAT}"
echo "  Checkpoints (${#CKPTS[@]}):"
for c in "${CKPTS[@]}"; do echo "    - ${c}"; done
echo "  Results: ${RESULT_DIR}/"
echo "  Time: $(date)"
echo "============================================"

# ── 单个 checkpoint 评估函数 ─────────────────────────────────────────────────
run_eval() {
    local ckpt_name=$1
    local gpu=$2
    local ckpt_path="${CKPT_DIR}/${ckpt_name}"
    local out_json="${RESULT_DIR}/${ckpt_name}.json"
    local log_file="${LOG_DIR}/eval_${ckpt_name}.log"

    if [[ -f "${out_json}" ]]; then
        echo "[SKIP] ${ckpt_name} — result already exists: ${out_json}"
        return 0
    fi

    echo "[START] ${ckpt_name} on GPU ${gpu} — log: ${log_file}"
    CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} "${PROJECT_DIR}/eval_t2m.py" \
        "${CONFIG}" \
        --checkpoint "${ckpt_path}" \
        --repeat "${REPEAT}" \
        --output "${out_json}" \
        --device cuda:0 \
        > "${log_file}" 2>&1
    local rc=$?

    if [[ ${rc} -eq 0 ]]; then
        echo "[DONE]  ${ckpt_name} on GPU ${gpu} — result: ${out_json}"
    else
        echo "[FAIL]  ${ckpt_name} on GPU ${gpu} — see ${log_file}"
    fi
    return ${rc}
}

# ── 两张卡并行评估 ───────────────────────────────────────────────────────────
i=0
total=${#CKPTS[@]}
fail_count=0

while [[ ${i} -lt ${total} ]]; do
    pids=()

    # GPU0 任务
    if [[ ${i} -lt ${total} ]]; then
        run_eval "${CKPTS[$i]}" "${GPU0}" &
        pids+=($!)
        ((i++))
    fi

    # GPU1 任务（如果还有剩余 checkpoint）
    if [[ ${i} -lt ${total} ]]; then
        run_eval "${CKPTS[$i]}" "${GPU1}" &
        pids+=($!)
        ((i++))
    fi

    # 等待这一轮完成
    for pid in "${pids[@]}"; do
        wait "${pid}" || ((fail_count++))
    done
done

# ── 汇总结果 ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  All evaluations finished at $(date)"
echo "  Failed: ${fail_count} / ${total}"
echo "============================================"
echo ""

# 打印汇总表格
printf "%-25s %10s %10s %10s %10s %10s %15s\n" \
    "Checkpoint" "FID" "Top-1" "Top-2" "Top-3" "Diversity" "Matching"
printf "%-25s %10s %10s %10s %10s %10s %15s\n" \
    "-------------------------" "----------" "----------" "----------" "----------" "----------" "---------------"

for c in "${CKPTS[@]}"; do
    json="${RESULT_DIR}/${c}.json"
    if [[ -f "${json}" ]]; then
        ${PYTHON} -c "
import json, sys
with open('${json}') as f:
    r = json.load(f)
print(f\"${c:<25s} {r['fid']:10.4f} {r['top1']:10.4f} {r['top2']:10.4f} {r['top3']:10.4f} {r['diversity']:10.4f} {r['matching_score']:15.4f}\")
"
    else
        printf "%-25s %10s\n" "${c}" "(no result)"
    fi
done

echo ""
echo "JSON results saved in: ${RESULT_DIR}/"
