#!/usr/bin/env bash
# run_lora_auto_eval.sh
#
# 用途 / Purpose:
#   Polls a LoRA training run dir for manifest.json (training completion signal),
#   then automatically launches ProcessBench MATH + GSM8K evaluations.
#   适用于 phase_e_train_value_lora.py 产生的运行目录，训练结束后自动触发评测。
#
# Usage:
#   CUDA_DEVICE=<gpu> RUN_DIR=<run_dir> PYTHON_BIN=<python> EVAL_ROOT=<eval_root>
#   BACKBONE_FAMILY=<causal_lm|process_reward_model> bash run_lora_auto_eval.sh
#
# Required env vars:
#   CUDA_DEVICE     -- GPU index to use for eval
#   RUN_DIR         -- path to the training run dir (must contain manifest.json when done)
#   PYTHON_BIN      -- python binary (default: python3)
#   EVAL_ROOT       -- output root for eval results (default: assets/artifacts/phase_e_eval)
#   POLL_INTERVAL   -- seconds between manifest checks (default: 60)
#   MAX_WAIT_SEC    -- give up after this many seconds (default: 14400 = 4h)

set -euo pipefail

CUDA_DEVICE="${CUDA_DEVICE:-0}"
RUN_DIR="${RUN_DIR:?RUN_DIR must be set}"
PYTHON_BIN="${PYTHON_BIN:-/home/zling/anaconda3/bin/python3}"
EVAL_ROOT="${EVAL_ROOT:-assets/artifacts/phase_e_eval}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-14400}"
LOGFILE="${LOGFILE:-/dev/stderr}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOGFILE}"; }

MANIFEST="${RUN_DIR}/manifest.json"
ELAPSED=0

log "Waiting for manifest: ${MANIFEST} (poll every ${POLL_INTERVAL}s, max ${MAX_WAIT_SEC}s)"
while [ ! -f "${MANIFEST}" ]; do
    if [ "${ELAPSED}" -ge "${MAX_WAIT_SEC}" ]; then
        log "ERROR: timed out waiting for manifest after ${MAX_WAIT_SEC}s"
        exit 1
    fi
    sleep "${POLL_INTERVAL}"
    ELAPSED=$((ELAPSED + POLL_INTERVAL))
done
log "Manifest found after ${ELAPSED}s. Starting evals."

# Derive run name from dir basename for eval naming
RUN_BASENAME="$(basename "${RUN_DIR}")"

# MATH eval
log "Launching ProcessBench MATH eval on GPU ${CUDA_DEVICE} ..."
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" -u scripts/phase_e_eval_benchmark.py \
    --value-run-dir "${RUN_DIR}" \
    --benchmark-id processbench_math \
    --run-name "${RUN_BASENAME}_pb_math" \
    --output-root "${EVAL_ROOT}" \
    --batch-size 16 \
    --feature-cache-mode off \
    --require-cuda \
    | tee -a "${LOGFILE}"

log "MATH eval done. Launching ProcessBench GSM8K eval ..."

# GSM8K eval
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" -u scripts/phase_e_eval_benchmark.py \
    --value-run-dir "${RUN_DIR}" \
    --benchmark-id processbench_gsm8k \
    --run-name "${RUN_BASENAME}_pb_gsm" \
    --output-root "${EVAL_ROOT}" \
    --batch-size 16 \
    --feature-cache-mode off \
    --require-cuda \
    | tee -a "${LOGFILE}"

log "All evals complete for ${RUN_BASENAME}."
