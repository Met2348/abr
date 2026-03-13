#!/usr/bin/env bash
# Parallel overnight package for the new Math-PRM hybrid frontier.
#
# Runs three complementary lines:
# 1. Phase E PH3 frontier run on GPU1
# 2. Phase F hybrid preflight on GPU3 using already-finished PH1/PH2 candidates
# 3. Phase F hybrid usability suite on CPU using already-finished PH1/PH2 scored rows

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUN_PREFIX="${RUN_PREFIX:-phase_e_phase_f_hybrid_overnight_$(date +%m%d_%H%M)}"
GPU_E="${GPU_E:-1}"
GPU_F="${GPU_F:-3}"
LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
mkdir -p "$LOG_ROOT"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"

log_line() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1" | tee -a "$SUITE_LOG_FILE"
}

log_line "Launching Phase E PH3 frontier on GPU${GPU_E}"
(
  CUDA_VISIBLE_DEVICES="$GPU_E" \
  RUN_PREFIX="${RUN_PREFIX}_ph3" \
  ACTIVE_PHASE_E_PB_HYBRID_GROUP=PH3_PRM_LOCAL_TA15_MSGRID5_ARCH_SWEEP_SMOKE \
  bash scripts/run_phase_e_prmbackbone_hybrid_frontier_suite.sh
) > "${LOG_ROOT}/ph3.log" 2>&1 &
PID_PH3=$!

log_line "Launching Phase F hybrid preflight on GPU${GPU_F} using PH1/PH2 candidates"
(
  CUDA_VISIBLE_DEVICES="$GPU_F" \
  GPU_DEVICE="$GPU_F" \
  RUN_PREFIX="${RUN_PREFIX}_hybrid_preflight" \
  bash scripts/run_phase_f_hybrid_preflight_suite.sh
) > "${LOG_ROOT}/hybrid_preflight.log" 2>&1 &
PID_PREFLIGHT=$!

log_line "Launching Phase F hybrid usability suite on CPU"
(
  RUN_PREFIX="${RUN_PREFIX}_hybrid_usability" \
  bash scripts/run_phase_f_hybrid_usability_suite.sh
) > "${LOG_ROOT}/hybrid_usability.log" 2>&1 &
PID_USABILITY=$!

STATUS_PH3="running"
STATUS_PREFLIGHT="running"
STATUS_USABILITY="running"

wait "$PID_PH3" || STATUS_PH3="failed"
wait "$PID_PREFLIGHT" || STATUS_PREFLIGHT="failed"
wait "$PID_USABILITY" || STATUS_USABILITY="failed"

if [[ "$STATUS_PH3" == "running" ]]; then STATUS_PH3="ok"; fi
if [[ "$STATUS_PREFLIGHT" == "running" ]]; then STATUS_PREFLIGHT="ok"; fi
if [[ "$STATUS_USABILITY" == "running" ]]; then STATUS_USABILITY="ok"; fi

cat > "$SUMMARY_FILE" <<EOF
# Phase E/F Hybrid Overnight v2

- run_prefix: ${RUN_PREFIX}
- suite_log: ${SUITE_LOG_FILE}

## Component Status

- PH3 frontier: ${STATUS_PH3}
- hybrid preflight: ${STATUS_PREFLIGHT}
- hybrid usability: ${STATUS_USABILITY}

## Logs

1. ${LOG_ROOT}/ph3.log
2. ${LOG_ROOT}/hybrid_preflight.log
3. ${LOG_ROOT}/hybrid_usability.log
EOF

log_line "summary_file: ${SUMMARY_FILE}"
