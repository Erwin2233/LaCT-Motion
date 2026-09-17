#!/usr/bin/env bash
# Run c_thought ablation experiments sequentially with isolated configs/checkpoints.
#
# Example:
#   bash run_ablation_c_thought.sh
#   bash run_ablation_c_thought.sh --gpus 4,5 --values 1,4
#   bash run_ablation_c_thought.sh --run-root ./ablation_runs/cthought_custom
#   bash run_ablation_c_thought.sh --bg

set -euo pipefail

ORIG_ARGS=("$@")

PROJECT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
BASE_CONFIG="${PROJECT_DIR}/options/sft/t2m_coconut.yaml"
PYTHON="${PYTHON:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"
THINK_STEPS="${THINK_STEPS:-${PROJECT_DIR}/data/texts_think_steps.json}"

GPUS="${GPUS:-4,5}"
VALUES_CSV="${VALUES_CSV:-1,4}"
RUN_ROOT=""
SKIP_BUILD=0
CONTINUE_ON_ERROR=1
RUN_BG=0

usage() {
  cat <<'EOF'
Usage:
  bash run_ablation_c_thought.sh [options]

Options:
  --gpus ID,ID,...            Physical GPU ids to use. Default: 4,5
  --values V1,V2,...          c_thought list. Default: 1,4
  --base-config PATH          Base YAML config. Default: options/sft/t2m_coconut.yaml
  --run-root DIR              Output root for ablation artifacts
  --bg                        Run in background via setsid+nohup
  --fg                        Force foreground run
  --skip-build                Skip get_train_data.py step even if data is missing
  --stop-on-error             Stop immediately if any run fails
  -h, --help                  Show this help

Env overrides:
  PYTHON, TORCHRUN, THINK_STEPS, GPUS, VALUES_CSV
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --values) VALUES_CSV="$2"; shift 2 ;;
    --base-config) BASE_CONFIG="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --bg) RUN_BG=1; shift ;;
    --fg) RUN_BG=0; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --stop-on-error) CONTINUE_ON_ERROR=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "[ERROR] Base config not found: ${BASE_CONFIG}"
  exit 1
fi

if [[ -z "${RUN_ROOT}" ]]; then
  TS=$(date +%Y%m%d_%H%M%S)
  RUN_ROOT="${PROJECT_DIR}/ablation_runs/c_thought_${TS}"
else
  TS=$(date +%Y%m%d_%H%M%S)
fi

LOG_DIR="${RUN_ROOT}/logs"
CFG_DIR="${RUN_ROOT}/configs"
mkdir -p "${RUN_ROOT}" "${LOG_DIR}" "${CFG_DIR}"

# Optional detached mode using setsid + nohup.
if [[ "${RUN_BG}" -eq 1 && "${__ABL_DETACHED:-0}" != "1" ]]; then
  DETACH_LOG="${LOG_DIR}/ablation_${TS}.log"
  __ABL_DETACHED=1 \
    setsid nohup bash "$0" "${ORIG_ARGS[@]}" >"${DETACH_LOG}" 2>&1 &
  BG_PID=$!
  echo "Ablation launched in background."
  echo "  PID: ${BG_PID}"
  echo "  Log: ${DETACH_LOG}"
  echo ""
  echo "To view logs:  tail -f ${DETACH_LOG}"
  echo "To stop:       kill ${BG_PID}"
  exit 0
fi

IFS=',' read -ra GPU_ARR <<< "${GPUS}"
if [[ ${#GPU_ARR[@]} -eq 0 ]]; then
  echo "[ERROR] --gpus is empty"
  exit 1
fi
NPROC="${NPROC:-${#GPU_ARR[@]}}"
if [[ "${NPROC}" -lt 1 ]]; then
  echo "[ERROR] NPROC must be >= 1, got ${NPROC}"
  exit 1
fi

IFS=',' read -ra VALUE_ARR <<< "${VALUES_CSV}"
if [[ ${#VALUE_ARR[@]} -eq 0 ]]; then
  echo "[ERROR] --values is empty"
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPUS}"

cd "${PROJECT_DIR}"

if [[ "${SKIP_BUILD}" -ne 1 && ! -f "${PROJECT_DIR}/data/t2m_train.json" ]]; then
  echo "[INFO] Building training data..."
  "${PYTHON}" get_train_data.py --output-dir data \
    --think-steps "${THINK_STEPS}" \
    --device cuda:0
  echo "[INFO] Data build complete."
fi

echo "============================================"
echo "  Latent-CoT-Motion c_thought Ablation"
echo "  Base config: ${BASE_CONFIG}"
echo "  c_thought:   ${VALUES_CSV}"
echo "  GPUs:        ${CUDA_VISIBLE_DEVICES} (nproc=${NPROC})"
echo "  Run root:    ${RUN_ROOT}"
echo "  Time:        $(date)"
echo "============================================"

FAIL_COUNT=0
RESULT_SUMMARY="${RUN_ROOT}/summary.txt"
: > "${RESULT_SUMMARY}"

for raw_v in "${VALUE_ARR[@]}"; do
  c=$(echo "${raw_v}" | xargs)
  if [[ ! "${c}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Invalid c_thought value: ${c}" | tee -a "${RESULT_SUMMARY}"
    ((FAIL_COUNT++))
    if [[ "${CONTINUE_ON_ERROR}" -eq 0 ]]; then
      exit 1
    fi
    continue
  fi

  CFG_PATH="${CFG_DIR}/t2m_coconut_c${c}.yaml"
  SAVE_PATH="${RUN_ROOT}/checkpoints_c${c}"
  RUN_NAME="t2m-coconut-qwen3b-ablation-c${c}"
  LOG_PATH="${LOG_DIR}/train_c${c}_${TS}.log"

  "${PYTHON}" - "${BASE_CONFIG}" "${CFG_PATH}" "${c}" "${SAVE_PATH}" "${RUN_NAME}" <<'PY'
import sys
import yaml

base_cfg, out_cfg, c_thought, save_path, run_name = sys.argv[1:]
c = int(c_thought)
with open(base_cfg, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["c_thought"] = c
cfg["save_path"] = save_path
cfg["name"] = run_name
cfg["resume"] = 0
cfg["load_model_path"] = "None"

with open(out_cfg, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

  MASTER_PORT=$((29500 + RANDOM % 10000))
  echo ""
  echo "[START] c_thought=${c}"
  echo "  config: ${CFG_PATH}"
  echo "  save:   ${SAVE_PATH}"
  echo "  log:    ${LOG_PATH}"
  echo "  port:   ${MASTER_PORT}"

  set +e
  "${TORCHRUN}" \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT}" \
    train_sft.py "${CFG_PATH}" 2>&1 | tee "${LOG_PATH}"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ ${rc} -eq 0 ]]; then
    echo "[DONE] c_thought=${c}" | tee -a "${RESULT_SUMMARY}"
    echo "c_thought=${c} status=success log=${LOG_PATH} ckpt=${SAVE_PATH}" >> "${RESULT_SUMMARY}"
  else
    echo "[FAIL] c_thought=${c}, rc=${rc}" | tee -a "${RESULT_SUMMARY}"
    echo "c_thought=${c} status=fail rc=${rc} log=${LOG_PATH} ckpt=${SAVE_PATH}" >> "${RESULT_SUMMARY}"
    ((FAIL_COUNT++))
    if [[ "${CONTINUE_ON_ERROR}" -eq 0 ]]; then
      echo "[STOP] stop-on-error enabled."
      exit "${rc}"
    fi
  fi
done

echo ""
echo "============================================"
echo "  Ablation finished at $(date)"
echo "  Failed runs: ${FAIL_COUNT}"
echo "  Summary: ${RESULT_SUMMARY}"
echo "============================================"

if [[ ${FAIL_COUNT} -ne 0 ]]; then
  exit 1
fi
