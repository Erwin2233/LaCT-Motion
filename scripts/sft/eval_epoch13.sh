#!/bin/bash
# 评测 checkpoint-epoch8，使用 GPU 4,5
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
PYTHON=python
PROJECT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
CONFIG=${PROJECT_DIR}/options/sft/t2m_coconut.yaml
cd "${PROJECT_DIR}"
CKPT=${PROJECT_DIR}/checkpoints/sft/runs/checkpoint-epoch13
RESULT_DIR=${PROJECT_DIR}/eval_results
LOG_DIR=${PROJECT_DIR}/logs

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

REPEAT=${1:-1}   # 默认 1 次，可传参: bash eval_epoch8.sh 20

echo "============================================"
echo "  Evaluate checkpoint-epoch8"
echo "  GPUs: 4, 5"
echo "  Repeat: ${REPEAT}"
echo "  Time: $(date)"
echo "============================================"

# GPU 4: repeat 前半
# GPU 5: repeat 后半
# 单次评测只用一张卡即可；多次 repeat 时两卡各跑一半再合并
if [[ ${REPEAT} -le 1 ]]; then
    # 单次评测，用 GPU 4
    OUT=${RESULT_DIR}/checkpoint-epoch8.json
    LOG=${LOG_DIR}/eval_checkpoint-epoch8.log
    echo "[RUN] GPU 4 — repeat=${REPEAT} — log: ${LOG}"
    CUDA_VISIBLE_DEVICES=4 ${PYTHON} "${PROJECT_DIR}/eval_t2m.py" \
        "${CONFIG}" \
        --checkpoint "${CKPT}" \
        --repeat "${REPEAT}" \
        --output "${OUT}" \
        --device cuda:0 \
        2>&1 | tee "${LOG}"
    echo "[DONE] Result: ${OUT}"
else
    # 多次 repeat: 两张卡各跑一半
    HALF=$((REPEAT / 2))
    REST=$((REPEAT - HALF))

    OUT_A=${RESULT_DIR}/checkpoint-epoch8_gpuA.json
    OUT_B=${RESULT_DIR}/checkpoint-epoch8_gpuB.json
    LOG_A=${LOG_DIR}/eval_checkpoint-epoch8_gpuA.log
    LOG_B=${LOG_DIR}/eval_checkpoint-epoch8_gpuB.log

    echo "[RUN] GPU 4 — repeat=${HALF} — log: ${LOG_A}"
    echo "[RUN] GPU 5 — repeat=${REST} — log: ${LOG_B}"

    CUDA_VISIBLE_DEVICES=4 ${PYTHON} "${PROJECT_DIR}/eval_t2m.py" \
        "${CONFIG}" \
        --checkpoint "${CKPT}" \
        --repeat "${HALF}" \
        --output "${OUT_A}" \
        --device cuda:0 \
        > "${LOG_A}" 2>&1 &
    PID_A=$!

    CUDA_VISIBLE_DEVICES=5 ${PYTHON} "${PROJECT_DIR}/eval_t2m.py" \
        "${CONFIG}" \
        --checkpoint "${CKPT}" \
        --repeat "${REST}" \
        --output "${OUT_B}" \
        --device cuda:0 \
        > "${LOG_B}" 2>&1 &
    PID_B=$!

    # 等待两个任务完成
    FAIL=0
    wait ${PID_A} || { echo "[FAIL] GPU 4 — see ${LOG_A}"; FAIL=1; }
    wait ${PID_B} || { echo "[FAIL] GPU 5 — see ${LOG_B}"; FAIL=1; }

    if [[ ${FAIL} -ne 0 ]]; then
        echo "Some evaluations failed, check logs above."
        exit 1
    fi

    # 合并两张卡的结果取平均
    MERGED=${RESULT_DIR}/checkpoint-epoch8.json
    ${PYTHON} -c "
import json, sys

with open('${OUT_A}') as f: a = json.load(f)
with open('${OUT_B}') as f: b = json.load(f)

n_a, n_b = ${HALF}, ${REST}
total = n_a + n_b
merged = {}
for k in ('fid', 'top1', 'top2', 'top3', 'diversity', 'matching_score', 'motion_emb_cos', 'semantic_cos'):
    merged[k] = (a[k] * n_a + b[k] * n_b) / total
merged['repeat'] = total
merged['note'] = 'weighted average from 2-GPU parallel eval'

with open('${MERGED}', 'w') as f:
    json.dump(merged, f, indent=2)

print('Merged result:')
for k, v in merged.items():
    if isinstance(v, float):
        print(f'  {k:20s}: {v:.4f}')
    else:
        print(f'  {k:20s}: {v}')
"
    echo ""
    echo "[DONE] Merged result: ${MERGED}"
fi

echo ""
echo "============================================"
echo "  Finished at $(date)"
echo "============================================"
