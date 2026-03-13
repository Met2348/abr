#!/usr/bin/env bash
# Follow one or two Phase E candidate run prefixes until completion, then run
# benchmark eval plus Phase F preflight automatically.
#
# 中文：
# 等待指定的 Phase E 候选训练完成，然后自动串起：
# 1. ProcessBench GSM / Math 评测；
# 2. Phase F modern preflight；
# 这样新候选不会停留在“训练跑完但没人继续验证”的状态。
#
# English:
# Wait for named Phase E candidate prefixes to finish, then chain:
# 1. ProcessBench GSM / Math evaluation,
# 2. Phase F modern preflight,
# so new candidates do not stall at raw training artifacts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_phase_f_followup_$(date +%m%d_%H%M)}"
GPU_DEVICE="${GPU_DEVICE:-2}"
POLL_SECONDS="${POLL_SECONDS:-300}"
PHASE_E_RUN_ROOT="${PHASE_E_RUN_ROOT:-assets/artifacts/phase_e_runs}"
PHASE_E_EVAL_ROOT="${PHASE_E_EVAL_ROOT:-assets/artifacts/phase_e_eval}"
PROCESSBENCH_GSM_PATH="${PROCESSBENCH_GSM_PATH:-assets/external_datasets/qwen_processbench/gsm8k.json}"
PROCESSBENCH_MATH_PATH="${PROCESSBENCH_MATH_PATH:-assets/external_datasets/qwen_processbench/math.json}"
BENCH_BATCH_SIZE="${BENCH_BATCH_SIZE:-96}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_GPU_MEMORY_GIB="${MAX_GPU_MEMORY_GIB:-60}"
FIXED_THRESHOLD="${FIXED_THRESHOLD:-0.5}"
CAND_A_ID="${CAND_A_ID:-pbr44}"
CAND_B_ID="${CAND_B_ID:-pbr45}"
CAND_A_RUN_PREFIX="${CAND_A_RUN_PREFIX:-phase_e_pbr44_lora_semcons_v1_s42}"
CAND_B_RUN_PREFIX="${CAND_B_RUN_PREFIX:-phase_e_pbr45_lora_pbr12_l2style_s42}"

LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"
mkdir -p "$LOG_ROOT"

log_line() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1" | tee -a "$SUITE_LOG_FILE" >&2
}

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    exit 1
  fi
}

latest_dir_by_prefix_with_marker() {
  local root="$1"
  local prefix="$2"
  local marker="$3"
  "$PYTHON_BIN" - "$root" "$prefix" "$marker" <<'PY_HELPER'
from pathlib import Path
import sys
root = Path(sys.argv[1])
prefix = sys.argv[2]
marker = sys.argv[3]
matches = []
for path in root.glob(f"{prefix}_*"):
    if not path.is_dir():
        continue
    if not (path / marker).exists():
        continue
    matches.append(path)
matches.sort(key=lambda item: item.stat().st_mtime, reverse=True)
print(matches[0] if matches else "")
PY_HELPER
}

latest_completed_run_dir() {
  local root="$1"
  local prefix="$2"
  "$PYTHON_BIN" - "$root" "$prefix" <<'PY_HELPER'
from pathlib import Path
import sys
root = Path(sys.argv[1])
prefix = sys.argv[2]
matches = []
for path in root.glob(f"{prefix}_*"):
    if not path.is_dir():
        continue
    if not (path / 'manifest.json').exists():
        continue
    if not (path / 'summary.json').exists():
        continue
    if not (path / 'best_value_head.pt').exists():
        continue
    matches.append(path)
matches.sort(key=lambda item: item.stat().st_mtime, reverse=True)
print(matches[0] if matches else "")
PY_HELPER
}

wait_for_completed_run_dir() {
  local root="$1"
  local prefix="$2"
  local label="$3"
  local run_dir=""
  while true; do
    run_dir="$(latest_completed_run_dir "$root" "$prefix")"
    if [[ -n "$run_dir" ]]; then
      log_line "completed ${label}: ${run_dir}"
      printf '%s\n' "$run_dir"
      return 0
    fi
    log_line "waiting for ${label} completion: prefix=${prefix} root=${root}"
    sleep "$POLL_SECONDS"
  done
}

run_processbench_eval() {
  local candidate_id="$1"
  local run_dir="$2"
  local benchmark_id="$3"
  local benchmark_path="$4"
  local eval_run_name="${RUN_PREFIX}_${candidate_id}_${benchmark_id}"
  log_line "benchmark eval: candidate=${candidate_id} benchmark=${benchmark_id} run_dir=${run_dir}"
  CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
    "$PYTHON_BIN" scripts/phase_e_eval_benchmark.py \
      --value-run-dir "$run_dir" \
      --benchmark-id "$benchmark_id" \
      --benchmark-path "$benchmark_path" \
      --run-name "$eval_run_name" \
      --batch-size "$BENCH_BATCH_SIZE" \
      --max-length "$MAX_LENGTH" \
      --processbench-f1-threshold-policy fixed \
      --processbench-f1-threshold "$FIXED_THRESHOLD" \
      --max-gpu-memory-gib "$MAX_GPU_MEMORY_GIB" \
      --feature-cache-mode read_write \
      --require-cuda \
      2>&1 | tee -a "$SUITE_LOG_FILE" >&2
  latest_dir_by_prefix_with_marker "$PHASE_E_EVAL_ROOT" "$eval_run_name" "metrics.json"
}

render_summary() {
  local cand_a_run_dir="$1"
  local cand_b_run_dir="$2"
  local cand_a_gsm_eval="$3"
  local cand_a_math_eval="$4"
  local cand_b_gsm_eval="$5"
  local cand_b_math_eval="$6"
  local preflight_summary="$7"
  {
    echo "# Phase E -> Phase F Follow-Up Summary"
    echo
    echo "- run_prefix: \`${RUN_PREFIX}\`"
    echo "- gpu_device: \`${GPU_DEVICE}\`"
    echo "- candidate_a: \`${CAND_A_ID}\`"
    echo "- candidate_b: \`${CAND_B_ID}\`"
    echo "- candidate_a_run_dir: \`${cand_a_run_dir}\`"
    echo "- candidate_b_run_dir: \`${cand_b_run_dir}\`"
    echo
    echo "## ProcessBench eval artifacts"
    echo
    echo "1. ${CAND_A_ID} gsm: \`${cand_a_gsm_eval}\`"
    echo "2. ${CAND_A_ID} math: \`${cand_a_math_eval}\`"
    echo "3. ${CAND_B_ID} gsm: \`${cand_b_gsm_eval}\`"
    echo "4. ${CAND_B_ID} math: \`${cand_b_math_eval}\`"
    echo
    echo "## Phase F preflight"
    echo
    echo "- summary: \`${preflight_summary}\`"
    echo
    echo "## Reading guide"
    echo
    echo "- If one candidate wins both fixed-threshold F1 and reward-hacking stability, it becomes the stronger promotion target."
    echo "- If neither candidate beats the current reference under preflight, treat the training result as data insight, not deployment promotion."
  } > "$SUMMARY_FILE"
}

require_path "$PROCESSBENCH_GSM_PATH" "ProcessBench GSM path"
require_path "$PROCESSBENCH_MATH_PATH" "ProcessBench Math path"

log_line "waiting for completed Phase E candidates"
CAND_A_RUN_DIR="$(wait_for_completed_run_dir "$PHASE_E_RUN_ROOT" "$CAND_A_RUN_PREFIX" "$CAND_A_ID")"
CAND_B_RUN_DIR="$(wait_for_completed_run_dir "$PHASE_E_RUN_ROOT" "$CAND_B_RUN_PREFIX" "$CAND_B_ID")"

CAND_A_GSM_EVAL_DIR="$(run_processbench_eval "$CAND_A_ID" "$CAND_A_RUN_DIR" processbench_gsm8k "$PROCESSBENCH_GSM_PATH")"
CAND_A_MATH_EVAL_DIR="$(run_processbench_eval "$CAND_A_ID" "$CAND_A_RUN_DIR" processbench_math "$PROCESSBENCH_MATH_PATH")"
CAND_B_GSM_EVAL_DIR="$(run_processbench_eval "$CAND_B_ID" "$CAND_B_RUN_DIR" processbench_gsm8k "$PROCESSBENCH_GSM_PATH")"
CAND_B_MATH_EVAL_DIR="$(run_processbench_eval "$CAND_B_ID" "$CAND_B_RUN_DIR" processbench_math "$PROCESSBENCH_MATH_PATH")"

log_line "launching Phase F modern preflight for ${CAND_A_ID} vs ${CAND_B_ID}"
RUN_PREFIX_PHASE_F="${RUN_PREFIX}_modern_preflight"
CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
GPU_DEVICE="$GPU_DEVICE" \
RUN_PREFIX="$RUN_PREFIX_PHASE_F" \
CAND_A_ID="$CAND_A_ID" \
CAND_B_ID="$CAND_B_ID" \
PBR26_RUN_DIR="$CAND_A_RUN_DIR" \
PBR31_RUN_DIR="$CAND_B_RUN_DIR" \
PBR26_GSM_EVAL="$CAND_A_GSM_EVAL_DIR" \
PBR26_MATH_EVAL="$CAND_A_MATH_EVAL_DIR" \
PBR31_GSM_EVAL="$CAND_B_GSM_EVAL_DIR" \
PBR31_MATH_EVAL="$CAND_B_MATH_EVAL_DIR" \
PROCESSBENCH_GSM_PATH="$PROCESSBENCH_GSM_PATH" \
PROCESSBENCH_MATH_PATH="$PROCESSBENCH_MATH_PATH" \
FIXED_THRESHOLD="$FIXED_THRESHOLD" \
THRESHOLD_BATCH_SIZE="$BENCH_BATCH_SIZE" \
  bash scripts/run_phase_f_modern_preflight_suite.sh 2>&1 | tee -a "$SUITE_LOG_FILE" >&2

PREFLIGHT_SUMMARY="assets/artifacts/phase_f_logs/${RUN_PREFIX_PHASE_F}/final_summary.md"
render_summary \
  "$CAND_A_RUN_DIR" \
  "$CAND_B_RUN_DIR" \
  "$CAND_A_GSM_EVAL_DIR" \
  "$CAND_A_MATH_EVAL_DIR" \
  "$CAND_B_GSM_EVAL_DIR" \
  "$CAND_B_MATH_EVAL_DIR" \
  "$PREFLIGHT_SUMMARY"

log_line "summary_file: ${SUMMARY_FILE}"
