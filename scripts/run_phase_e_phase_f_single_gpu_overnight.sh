#!/usr/bin/env bash
# Single-GPU overnight launcher for Phase E improvement + Phase F usability audit.
#
# Design principles from recent literature and local evidence:
# 1. benchmark-oriented data geometry beats naive source concat;
# 2. judge / consensus filtering is worth testing on ambiguous low-margin pairs;
# 3. RL promotion should stay behind threshold/shift + reward-hacking audits;
# 4. on crowded servers, sequential single-GPU execution is safer than multi-job VRAM contention.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_phase_f_overnight_$(date +%m%d_%H%M)}"
GPU_ID="${GPU_ID:-3}"
LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
LAUNCH_LOG="${LOG_ROOT}/launch.log"
MANIFEST_JSONL="${LOG_ROOT}/launch_manifest.jsonl"
PLAN_MD="${LOG_ROOT}/overnight_plan.md"
CURRENT_STAGE="bootstrap"
ENABLE_CR1="${ENABLE_CR1:-0}"
E_MODEL_PATH="${E_MODEL_PATH:-assets/models/Qwen2.5-Math-PRM-7B}"

mkdir -p "$LOG_ROOT"
: > "$LAUNCH_LOG"
: > "$MANIFEST_JSONL"

log_line() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1" | tee -a "$LAUNCH_LOG"
}

append_manifest() {
  "$PYTHON_BIN" - "$MANIFEST_JSONL" "$1" "$2" "$3" "$4" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
out_path, step_name, stage, status, detail = sys.argv[1:]
row = {
    'generated_at': datetime.now(timezone.utc).isoformat(),
    'step_name': step_name,
    'stage': stage,
    'status': status,
    'detail': detail,
}
with Path(out_path).open('a', encoding='utf-8') as f:
    f.write(json.dumps(row, ensure_ascii=False) + '\n')
PY
}

run_step() {
  local step_name="$1"
  shift
  local cmd="$*"
  CURRENT_STAGE="$step_name"
  log_line "START ${step_name}"
  log_line "CMD   ${cmd}"
  append_manifest "$step_name" "$CURRENT_STAGE" "started" "$cmd"
  if bash -lc "cd '$REPO_ROOT' && export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' && ${cmd}" >>"$LAUNCH_LOG" 2>&1; then
    log_line "DONE  ${step_name}"
    append_manifest "$step_name" "$CURRENT_STAGE" "ok" "$cmd"
  else
    log_line "FAIL  ${step_name}"
    append_manifest "$step_name" "$CURRENT_STAGE" "failed" "$cmd"
    return 1
  fi
}

cat > "$PLAN_MD" <<EOF
# Phase E / F Single-GPU Overnight Plan

- run_prefix: ${RUN_PREFIX}
- gpu_id: ${GPU_ID}
- launch_log: ${LAUNCH_LOG}
- manifest_jsonl: ${MANIFEST_JSONL}

## Best-Practice Basis

1. VerifyBench: verifier behavior is input-structure-sensitive, so prioritize benchmark-oriented supervision.
2. AbstentionBench: controller usefulness depends on abstain/reject behavior, not just verifier AUC.
3. PURE / Stop Summation: naive PRM reward use is reward-hack-prone, so keep RL behind preflight audits.
4. Qwen-style consensus filtering intuition: low-margin ambiguous pairs are the best place for selective judge relabel.

## Planned Steps on GPU${GPU_ID}

1. Phase E PH1_PRM_LOCAL_TA15_ARCH_SWEEP_SMOKE with Math-PRM backbone
2. Phase E PH2_PRM_LOCAL_TA10_MSGRID10_ARCH_SWEEP_SMOKE with Math-PRM backbone
3. Phase E CR1_CURATED_CENTER_GATE_SMOKE (optional; default off because current recipe guard blocks it)
4. Phase F modern preflight on PBR26 / PBR31

## Morning Checks

1. cat ${MANIFEST_JSONL}
2. tail -n 120 ${LAUNCH_LOG}
3. ls -dt assets/artifacts/phase_e_logs/${RUN_PREFIX}_* 2>/dev/null || true
4. ls -dt assets/artifacts/phase_f_logs/${RUN_PREFIX}_pf* 2>/dev/null || true
EOF

log_line "GPU snapshot at launch:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | tee -a "$LAUNCH_LOG"

run_step ph1_hybrid_prmbackbone \
  "CUDA_VISIBLE_DEVICES=${GPU_ID} RUN_PREFIX=${RUN_PREFIX}_ph1 MODEL_PATH=${E_MODEL_PATH} ACTIVE_PHASE_E_PB_HYBRID_GROUP=PH1_PRM_LOCAL_TA15_ARCH_SWEEP_SMOKE TRAIN_BATCH_SIZE=16 EVAL_BATCH_SIZE=24 BENCH_MAX_SAMPLES=128 MAX_GPU_MEMORY_GIB=36 MAX_CPU_MEMORY_GIB=96 FEATURE_CACHE_MODE=read_write RECIPE_RISK_POLICY=error bash scripts/run_phase_e_processbench_hybrid_suite.sh"

run_step ph2_hybrid_prmbackbone \
  "CUDA_VISIBLE_DEVICES=${GPU_ID} RUN_PREFIX=${RUN_PREFIX}_ph2 MODEL_PATH=${E_MODEL_PATH} ACTIVE_PHASE_E_PB_HYBRID_GROUP=PH2_PRM_LOCAL_TA10_MSGRID10_ARCH_SWEEP_SMOKE TRAIN_BATCH_SIZE=16 EVAL_BATCH_SIZE=24 BENCH_MAX_SAMPLES=128 MAX_GPU_MEMORY_GIB=36 MAX_CPU_MEMORY_GIB=96 FEATURE_CACHE_MODE=read_write RECIPE_RISK_POLICY=error bash scripts/run_phase_e_processbench_hybrid_suite.sh"

if [[ "$ENABLE_CR1" == "1" ]]; then
  run_step cr1_curated \
    "CUDA_VISIBLE_DEVICES=${GPU_ID} RUN_PREFIX=${RUN_PREFIX}_cr1 ACTIVE_PHASE_E_CURATED_GROUP=CR1_CURATED_CENTER_GATE_SMOKE TRAIN_BATCH_SIZE=64 EVAL_BATCH_SIZE=80 BENCH_MAX_SAMPLES=128 MAX_GPU_MEMORY_GIB=36 MAX_CPU_MEMORY_GIB=96 FEATURE_CACHE_MODE=read_write RECIPE_RISK_POLICY=error bash scripts/run_phase_e_curated_rlready_suite.sh"
else
  log_line "SKIP cr1_curated (ENABLE_CR1=0)"
  append_manifest cr1_curated skipped ok "disabled by default; current recipe guard blocks this group"
fi

run_step pf_modern_preflight \
  "GPU_DEVICE=${GPU_ID} RUN_PREFIX=${RUN_PREFIX}_pf FIXED_THRESHOLD=0.5 THRESHOLD_BATCH_SIZE=48 PROBE_MAX_ERROR=64 PROBE_MAX_CORRECT=64 bash scripts/run_phase_f_modern_preflight_suite.sh"

log_line "Overnight single-GPU package complete."
append_manifest all final ok "${RUN_PREFIX}"
