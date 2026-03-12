#!/usr/bin/env bash
# Phase F modern preflight suite.
#
# English
# -------
# Audit the current best verified Phase E candidates (`PBR26`, `PBR31`) under
# the two questions that matter before any RL promotion:
# 1. threshold / generator-shift stability,
# 2. reward-hacking surface under simple adversarial probes.
#
# 中文
# ----
# 用当前最新、最强的候选（`PBR26`, `PBR31`）做 Phase F 预审：
# 1. fixed-threshold / generator-shift 是否稳定；
# 2. 简单 reward-hacking probe 下是否暴露明显攻击面。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PREFIX="${RUN_PREFIX:-phase_f_modern_preflight_$(date +%m%d_%H%M)}"
LOG_ROOT="assets/artifacts/phase_f_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"
THRESHOLD_RESULTS_JSON=""
PROBE_RESULTS_JSON_PBR26=""
PROBE_RESULTS_JSON_PBR31=""
CURRENT_STAGE="bootstrap"

mkdir -p "$LOG_ROOT"

PBR26_RUN_DIR="${PBR26_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_pbr26_dpo_ms_full_s42_value_20260311T134542Z}"
PBR31_RUN_DIR="${PBR31_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_pbr31_lora_mathprm_pbr12data_s42_20260311T150316Z}"
PBR26_GSM_EVAL="${PBR26_GSM_EVAL:-assets/artifacts/phase_e_eval/pbr26_dpo_ms_full_s42_gsm8k_eval_20260311T140556Z}"
PBR26_MATH_EVAL="${PBR26_MATH_EVAL:-assets/artifacts/phase_e_eval/pbr26_dpo_ms_full_s42_math_eval_20260311T140605Z}"
PBR31_GSM_EVAL="${PBR31_GSM_EVAL:-assets/artifacts/phase_e_eval/pbr31_verify_gsm_0312_20260311T170309Z}"
PBR31_MATH_EVAL="${PBR31_MATH_EVAL:-assets/artifacts/phase_e_eval/pbr31_verify_math_0312_20260311T170630Z}"
PROCESSBENCH_GSM_PATH="${PROCESSBENCH_GSM_PATH:-assets/external_datasets/qwen_processbench/gsm8k.json}"
PROCESSBENCH_MATH_PATH="${PROCESSBENCH_MATH_PATH:-assets/external_datasets/qwen_processbench/math.json}"
FIXED_THRESHOLD="${FIXED_THRESHOLD:-0.5}"
THRESHOLD_BATCH_SIZE="${THRESHOLD_BATCH_SIZE:-64}"
PROBE_MAX_ERROR="${PROBE_MAX_ERROR:-64}"
PROBE_MAX_CORRECT="${PROBE_MAX_CORRECT:-64}"
GPU_DEVICE="${GPU_DEVICE:-3}"
CAND_A_ID="${CAND_A_ID:-pbr26}"
CAND_B_ID="${CAND_B_ID:-pbr31}"

log_line() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1" | tee -a "$SUITE_LOG_FILE"
}

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    exit 1
  fi
}

for path_label in \
  "$PBR26_RUN_DIR|PBR26 run dir" \
  "$PBR31_RUN_DIR|PBR31 run dir" \
  "$PBR26_GSM_EVAL|PBR26 GSM eval" \
  "$PBR26_MATH_EVAL|PBR26 Math eval" \
  "$PBR31_GSM_EVAL|PBR31 GSM eval" \
  "$PBR31_MATH_EVAL|PBR31 Math eval" \
  "$PROCESSBENCH_GSM_PATH|ProcessBench GSM path" \
  "$PROCESSBENCH_MATH_PATH|ProcessBench Math path"; do
  IFS='|' read -r path label <<< "$path_label"
  require_path "$path" "$label"
done

CURRENT_STAGE="threshold_shift"
log_line "Running threshold / generator-shift audit on ${CAND_A_ID} and ${CAND_B_ID}"
THRESHOLD_RESULTS_JSON="$($PYTHON_BIN scripts/phase_f_analyze_threshold_shift.py \
  --run-name "${RUN_PREFIX}_threshold_shift" \
  --case "${CAND_A_ID}_gsm|${PBR26_GSM_EVAL}" \
  --case "${CAND_A_ID}_math|${PBR26_MATH_EVAL}" \
  --case "${CAND_B_ID}_gsm|${PBR31_GSM_EVAL}" \
  --case "${CAND_B_ID}_math|${PBR31_MATH_EVAL}" \
  --fixed-threshold "${FIXED_THRESHOLD}" | tee -a "$SUITE_LOG_FILE" | awk -F': ' '/summary_json/{print $2}' | tail -n1)"
THRESHOLD_DIR="$(dirname "$THRESHOLD_RESULTS_JSON")"

CURRENT_STAGE="reward_hacking_probe"
log_line "Running reward-hacking probe on GPU ${GPU_DEVICE} for ${CAND_A_ID}"
PROBE_RESULTS_JSON_PBR26="$({
  CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" \
  $PYTHON_BIN scripts/phase_f_probe_reward_hacking.py \
    --run-name "${RUN_PREFIX}_reward_probe_${CAND_A_ID}" \
    --candidate "${CAND_A_ID}|${PBR26_RUN_DIR}" \
    --benchmark-spec "processbench_gsm8k|${PROCESSBENCH_GSM_PATH}" \
    --benchmark-spec "processbench_math|${PROCESSBENCH_MATH_PATH}" \
    --fixed-threshold "${FIXED_THRESHOLD}" \
    --batch-size "${THRESHOLD_BATCH_SIZE}" \
    --max-error-examples-per-benchmark "${PROBE_MAX_ERROR}" \
    --max-correct-examples-per-benchmark "${PROBE_MAX_CORRECT}" \
    --require-cuda; } | tee -a "$SUITE_LOG_FILE" | awk -F': ' '/summary_json/{print $2}' | tail -n1)"
PROBE_DIR_PBR26="$(dirname "$PROBE_RESULTS_JSON_PBR26")"

log_line "Running reward-hacking probe on GPU ${GPU_DEVICE} for ${CAND_B_ID}"
PROBE_RESULTS_JSON_PBR31="$({
  CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" \
  $PYTHON_BIN scripts/phase_f_probe_reward_hacking.py \
    --run-name "${RUN_PREFIX}_reward_probe_${CAND_B_ID}" \
    --candidate "${CAND_B_ID}|${PBR31_RUN_DIR}" \
    --benchmark-spec "processbench_gsm8k|${PROCESSBENCH_GSM_PATH}" \
    --benchmark-spec "processbench_math|${PROCESSBENCH_MATH_PATH}" \
    --fixed-threshold "${FIXED_THRESHOLD}" \
    --batch-size "${THRESHOLD_BATCH_SIZE}" \
    --max-error-examples-per-benchmark "${PROBE_MAX_ERROR}" \
    --max-correct-examples-per-benchmark "${PROBE_MAX_CORRECT}" \
    --require-cuda; } | tee -a "$SUITE_LOG_FILE" | awk -F': ' '/summary_json/{print $2}' | tail -n1)"
PROBE_DIR_PBR31="$(dirname "$PROBE_RESULTS_JSON_PBR31")"

CURRENT_STAGE="summary"
log_line "Rendering final summary"
$PYTHON_BIN - "$THRESHOLD_DIR" "$PROBE_DIR_PBR26" "$PROBE_DIR_PBR31" "$SUMMARY_FILE" <<'PY'
import json
import sys
from pathlib import Path

threshold_dir = Path(sys.argv[1])
probe_dir_pbr26 = Path(sys.argv[2])
probe_dir_pbr31 = Path(sys.argv[3])
summary_file = Path(sys.argv[4])
threshold_summary = json.loads((threshold_dir / 'summary.json').read_text(encoding='utf-8'))
probe_summary_pbr26 = json.loads((probe_dir_pbr26 / 'summary.json').read_text(encoding='utf-8'))
probe_summary_pbr31 = json.loads((probe_dir_pbr31 / 'summary.json').read_text(encoding='utf-8'))
case_rows = list(threshold_summary.get('cases', []))
probe_rows = list(probe_summary_pbr26.get('metrics', [])) + list(probe_summary_pbr31.get('metrics', []))

lines = [
    '# Phase F Modern Preflight Summary',
    '',
    f'- threshold_shift_dir: `{threshold_dir}`',
    f'- reward_probe_dirs: `{probe_dir_pbr26}`, `{probe_dir_pbr31}`',
    '',
    '## Threshold / Shift Snapshot',
    '',
    '| case_id | best_f1 | f1@0.5 | near_best_width | gen_tau_std | worst_gen_logo_f1 |',
    '|---|---:|---:|---:|---:|---:|',
]
for row in case_rows:
    lines.append(
        '| {case_id} | {best_f1:.4f} | {fixed_f1:.4f} | {width:.3f} | {tau_std:.4f} | {worst_logo:.4f} |'.format(
            case_id=row['case_id'],
            best_f1=row['best_f1'],
            fixed_f1=row['fixed_f1'],
            width=row['near_best_window_width'],
            tau_std=row['generator_best_threshold_std'],
            worst_logo=row['worst_generator_logo_f1'],
        )
    )

lines.extend([
    '',
    '## Reward-Hacking Probe Snapshot',
    '',
    '| candidate | benchmark | group | attack | mean_delta | flip@0.5 | outrank_safe | risk |',
    '|---|---|---|---|---:|---:|---:|---|',
])
for row in probe_rows:
    outrank = 'n/a' if row['outrank_safe_rate'] is None else f"{row['outrank_safe_rate']:.4f}"
    lines.append(
        '| {candidate} | {benchmark} | {group} | {attack} | {mean_delta:.4f} | {flip:.4f} | {outrank} | {risk} |'.format(
            candidate=row['candidate_id'],
            benchmark=row['benchmark_id'],
            group=row['probe_group'],
            attack=row['attack_name'],
            mean_delta=row['mean_delta'],
            flip=row['flip_rate_fixed05'],
            outrank=outrank,
            risk=row['risk_level'],
        )
    )

lines.extend([
    '',
    '## Reading Guide',
    '',
    '- `near_best_width` too small means deployment thresholds are brittle.',
    '- `worst_gen_logo_f1` low means generator-shift is still a controller risk.',
    '- `flip@0.5` or `outrank_safe` on `first_bad` / `filler_tail` means PRM-style reward hacking surface remains.',
    '',
])
summary_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(summary_file)
PY

log_line "Modern preflight complete: ${SUMMARY_FILE}"
