#!/usr/bin/env bash
# Offline controller/usability study on the stronger PH1/PH2 hybrid verifier slices.
#
# 中文：
# 目标是回答：
# 1. 新 hybrid verifier 比旧 PBR26/PBR31 更适合做 controller 吗？
# 2. 哪个 slice 更适合 Math / GSM 的 heuristic family？
# 3. BC 和 BC->RL 在这批更强 verifier 上会不会比之前更稳？

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PREFIX="${RUN_PREFIX:-phase_f_hybrid_usability_$(date +%m%d_%H%M)}"
ACTIVE_GROUP="${ACTIVE_GROUP:-HALL_HYBRID_USABILITY}"
PROCESSBENCH_GSM="${PROCESSBENCH_GSM:-assets/external_datasets/qwen_processbench/gsm8k.json}"
PROCESSBENCH_MATH="${PROCESSBENCH_MATH:-assets/external_datasets/qwen_processbench/math.json}"

PH1_MLP_GSM_ROWS="${PH1_MLP_GSM_ROWS:-assets/artifacts/phase_e_eval/phase_e_prmbackbone_hybrid_ph1_live_0312_ph1_prm_local_ta15_arch_sweep_smoke_mlp_processbench_gsm8k_20260312T083610Z/scored_rows.jsonl}"
PH1_MLP_MATH_ROWS="${PH1_MLP_MATH_ROWS:-assets/artifacts/phase_e_eval/phase_e_prmbackbone_hybrid_ph1_live_0312_ph1_prm_local_ta15_arch_sweep_smoke_mlp_processbench_math_20260312T083645Z/scored_rows.jsonl}"
PH1_GATED_GSM_ROWS="${PH1_GATED_GSM_ROWS:-assets/artifacts/phase_e_eval/phase_e_prmbackbone_hybrid_ph1_live_0312_ph1_prm_local_ta15_arch_sweep_smoke_gated_mlp_processbench_gsm8k_20260312T083808Z/scored_rows.jsonl}"
PH1_GATED_MATH_ROWS="${PH1_GATED_MATH_ROWS:-assets/artifacts/phase_e_eval/phase_e_prmbackbone_hybrid_ph1_live_0312_ph1_prm_local_ta15_arch_sweep_smoke_gated_mlp_processbench_math_20260312T083817Z/scored_rows.jsonl}"
PH2_MLP_GSM_ROWS="${PH2_MLP_GSM_ROWS:-assets/artifacts/phase_e_eval/phase_e_prmbackbone_hybrid_ph2_live_0312_ph2_prm_local_ta10_msgrid10_arch_sweep_smoke_mlp_processbench_gsm8k_20260312T083713Z/scored_rows.jsonl}"
PH2_MLP_MATH_ROWS="${PH2_MLP_MATH_ROWS:-assets/artifacts/phase_e_eval/phase_e_prmbackbone_hybrid_ph2_live_0312_ph2_prm_local_ta10_msgrid10_arch_sweep_smoke_mlp_processbench_math_20260312T083722Z/scored_rows.jsonl}"
PH2_GATED_GSM_ROWS="${PH2_GATED_GSM_ROWS:-assets/artifacts/phase_e_eval/phase_e_prmbackbone_hybrid_ph2_live_0312_ph2_prm_local_ta10_msgrid10_arch_sweep_smoke_gated_mlp_processbench_gsm8k_20260312T083847Z/scored_rows.jsonl}"
PH2_GATED_MATH_ROWS="${PH2_GATED_MATH_ROWS:-assets/artifacts/phase_e_eval/phase_e_prmbackbone_hybrid_ph2_live_0312_ph2_prm_local_ta10_msgrid10_arch_sweep_smoke_gated_mlp_processbench_math_20260312T083856Z/scored_rows.jsonl}"

LOG_ROOT="assets/artifacts/phase_f_logs/${RUN_PREFIX}_${ACTIVE_GROUP}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"
mkdir -p "$LOG_ROOT"

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

latest_dir_by_prefix_or_empty() {
  local root="$1"
  local prefix="$2"
  python - "$root" "$prefix" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
prefix = sys.argv[2]
matches = sorted(root.glob(f"{prefix}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
print(matches[0] if matches else "")
PY
}

for path_label in \
  "$PROCESSBENCH_GSM|ProcessBench GSM" \
  "$PROCESSBENCH_MATH|ProcessBench Math" \
  "$PH1_MLP_GSM_ROWS|PH1 MLP GSM rows" \
  "$PH1_MLP_MATH_ROWS|PH1 MLP Math rows" \
  "$PH1_GATED_GSM_ROWS|PH1 gated GSM rows" \
  "$PH1_GATED_MATH_ROWS|PH1 gated Math rows" \
  "$PH2_MLP_GSM_ROWS|PH2 MLP GSM rows" \
  "$PH2_MLP_MATH_ROWS|PH2 MLP Math rows" \
  "$PH2_GATED_GSM_ROWS|PH2 gated GSM rows" \
  "$PH2_GATED_MATH_ROWS|PH2 gated Math rows"; do
  IFS='|' read -r path label <<< "$path_label"
  require_path "$path" "$label"
done

run_policy_sweep() {
  log_line "R1 policy sweep on PH1/PH2 hybrid slices"
  "$PYTHON_BIN" scripts/phase_f_controller_policy_sweep.py \
    --run-name "${RUN_PREFIX}_controller_sweep" \
    --case "ph1_mlp_math|${PH1_MLP_MATH_ROWS}" \
    --case "ph1_gated_math|${PH1_GATED_MATH_ROWS}" \
    --case "ph2_mlp_math|${PH2_MLP_MATH_ROWS}" \
    --case "ph2_gated_math|${PH2_GATED_MATH_ROWS}" \
    --case "ph1_mlp_gsm|${PH1_MLP_GSM_ROWS}" \
    --case "ph1_gated_gsm|${PH1_GATED_GSM_ROWS}" \
    --case "ph2_mlp_gsm|${PH2_MLP_GSM_ROWS}" \
    --case "ph2_gated_gsm|${PH2_GATED_GSM_ROWS}" \
    | tee -a "$SUITE_LOG_FILE"
}

run_generator_robustness() {
  log_line "R2 generator-robust policy search on PH1/PH2 hybrid slices"
  "$PYTHON_BIN" scripts/phase_f_controller_generator_robustness.py \
    --run-name "${RUN_PREFIX}_generator_robustness" \
    --case "ph1_mlp_math|${PH1_MLP_MATH_ROWS}|${PROCESSBENCH_MATH}" \
    --case "ph1_gated_math|${PH1_GATED_MATH_ROWS}|${PROCESSBENCH_MATH}" \
    --case "ph2_mlp_math|${PH2_MLP_MATH_ROWS}|${PROCESSBENCH_MATH}" \
    --case "ph2_gated_math|${PH2_GATED_MATH_ROWS}|${PROCESSBENCH_MATH}" \
    --case "ph1_mlp_gsm|${PH1_MLP_GSM_ROWS}|${PROCESSBENCH_GSM}" \
    --case "ph1_gated_gsm|${PH1_GATED_GSM_ROWS}|${PROCESSBENCH_GSM}" \
    --case "ph2_mlp_gsm|${PH2_MLP_GSM_ROWS}|${PROCESSBENCH_GSM}" \
    --case "ph2_gated_gsm|${PH2_GATED_GSM_ROWS}|${PROCESSBENCH_GSM}" \
    | tee -a "$SUITE_LOG_FILE"
}

run_ensemble_eval() {
  log_line "R3 ensemble evaluation on PH1/PH2 hybrid pairs"
  "$PYTHON_BIN" scripts/phase_f_controller_ensemble_eval.py \
    --run-name "${RUN_PREFIX}_ensemble_eval" \
    --case "math_ph1mlp_ph2mlp|${PH1_MLP_MATH_ROWS}|${PH2_MLP_MATH_ROWS}" \
    --case "math_ph1gated_ph2gated|${PH1_GATED_MATH_ROWS}|${PH2_GATED_MATH_ROWS}" \
    --case "math_ph1mlp_ph1gated|${PH1_MLP_MATH_ROWS}|${PH1_GATED_MATH_ROWS}" \
    --case "math_ph2mlp_ph2gated|${PH2_MLP_MATH_ROWS}|${PH2_GATED_MATH_ROWS}" \
    --case "gsm_ph1mlp_ph2mlp|${PH1_MLP_GSM_ROWS}|${PH2_MLP_GSM_ROWS}" \
    --case "gsm_ph1gated_ph2gated|${PH1_GATED_GSM_ROWS}|${PH2_GATED_GSM_ROWS}" \
    --case "gsm_ph1mlp_ph1gated|${PH1_MLP_GSM_ROWS}|${PH1_GATED_GSM_ROWS}" \
    --case "gsm_ph2mlp_ph2gated|${PH2_MLP_GSM_ROWS}|${PH2_GATED_GSM_ROWS}" \
    | tee -a "$SUITE_LOG_FILE"
}

run_bc_only() {
  log_line "R4 BC on strongest domain-specialized hybrid slices"
  "$PYTHON_BIN" scripts/phase_f_behavior_clone_controller.py \
    --run-name "${RUN_PREFIX}_bc_only" \
    --case "ph1_gated_math|${PH1_GATED_MATH_ROWS}|${PROCESSBENCH_MATH}|threshold_only|{\"tau\": 0.42}" \
    --case "ph2_mlp_math|${PH2_MLP_MATH_ROWS}|${PROCESSBENCH_MATH}|threshold_only|{\"tau\": 0.42}" \
    --case "ph1_mlp_gsm|${PH1_MLP_GSM_ROWS}|${PROCESSBENCH_GSM}|delayed_drop|{\"tau\": 0.42, \"delta\": 0.25, \"min_step\": 4}" \
    --case "ph2_mlp_gsm|${PH2_MLP_GSM_ROWS}|${PROCESSBENCH_GSM}|delayed_drop|{\"tau\": 0.42, \"delta\": 0.25, \"min_step\": 4}" \
    --seed 42 \
    --hidden-dim 16 \
    --bc-epochs 50 \
    --bc-learning-rate 0.003 \
    | tee -a "$SUITE_LOG_FILE"
}

run_bc_then_rl() {
  log_line "R5 BC->RL on strongest domain-specialized hybrid slices"
  "$PYTHON_BIN" scripts/phase_f_behavior_clone_controller.py \
    --run-name "${RUN_PREFIX}_bc_then_rl" \
    --case "ph1_gated_math|${PH1_GATED_MATH_ROWS}|${PROCESSBENCH_MATH}|threshold_only|{\"tau\": 0.42}" \
    --case "ph2_mlp_math|${PH2_MLP_MATH_ROWS}|${PROCESSBENCH_MATH}|threshold_only|{\"tau\": 0.42}" \
    --case "ph1_mlp_gsm|${PH1_MLP_GSM_ROWS}|${PROCESSBENCH_GSM}|delayed_drop|{\"tau\": 0.42, \"delta\": 0.25, \"min_step\": 4}" \
    --case "ph2_mlp_gsm|${PH2_MLP_GSM_ROWS}|${PROCESSBENCH_GSM}|delayed_drop|{\"tau\": 0.42, \"delta\": 0.25, \"min_step\": 4}" \
    --seed 42 \
    --hidden-dim 16 \
    --bc-epochs 50 \
    --bc-learning-rate 0.003 \
    --do-rl-finetune \
    --rl-epochs 40 \
    --rl-learning-rate 0.001 \
    --robust-lambda 0.5 \
    | tee -a "$SUITE_LOG_FILE"
}

run_rl_like_probe() {
  log_line "R6 robust-from-scratch RL-like probe on best-guess hybrid math/gsm slices"
  "$PYTHON_BIN" scripts/phase_f_train_trainable_controller.py \
    --run-name "${RUN_PREFIX}_rl_like_robust" \
    --case "ph1_gated_math|${PH1_GATED_MATH_ROWS}|${PROCESSBENCH_MATH}" \
    --case "ph2_mlp_gsm|${PH2_MLP_GSM_ROWS}|${PROCESSBENCH_GSM}" \
    --seed 42 \
    --epochs 80 \
    --learning-rate 0.001 \
    --hidden-dim 16 \
    --robust-lambda 0.5 \
    --selection-metric worst_generator_balanced_f1 \
    --reward-mode balanced \
    | tee -a "$SUITE_LOG_FILE"
}

render_summary() {
  local sweep_dir robust_dir ensemble_dir bc_only_dir bc_then_rl_dir rl_like_dir
  sweep_dir="$(latest_dir_by_prefix_or_empty assets/artifacts/phase_f_controller_sweep "${RUN_PREFIX}_controller_sweep")"
  robust_dir="$(latest_dir_by_prefix_or_empty assets/artifacts/phase_f_controller_robustness "${RUN_PREFIX}_generator_robustness")"
  ensemble_dir="$(latest_dir_by_prefix_or_empty assets/artifacts/phase_f_controller_ensemble "${RUN_PREFIX}_ensemble_eval")"
  bc_only_dir="$(latest_dir_by_prefix_or_empty assets/artifacts/phase_f_bc "${RUN_PREFIX}_bc_only")"
  bc_then_rl_dir="$(latest_dir_by_prefix_or_empty assets/artifacts/phase_f_bc "${RUN_PREFIX}_bc_then_rl")"
  rl_like_dir="$(latest_dir_by_prefix_or_empty assets/artifacts/phase_f_rl_like "${RUN_PREFIX}_rl_like_robust")"
  cat > "$SUMMARY_FILE" <<EOF
# Phase F Hybrid Usability Suite

- run_prefix: ${RUN_PREFIX}
- active_group: ${ACTIVE_GROUP}
- suite_log: ${SUITE_LOG_FILE}

## Artifacts

1. controller_sweep: ${sweep_dir:-N/A}
2. generator_robustness: ${robust_dir:-N/A}
3. ensemble_eval: ${ensemble_dir:-N/A}
4. bc_only: ${bc_only_dir:-N/A}
5. bc_then_rl: ${bc_then_rl_dir:-N/A}
6. rl_like_robust: ${rl_like_dir:-N/A}

## Reading Guide

- This suite asks whether the stronger PH1/PH2 hybrid verifiers are already better controller candidates than the older PBR26/PBR31 family.
- Read controller sweep / generator robustness first for stable heuristics.
- Then read BC / BC->RL to see whether those heuristics can be distilled into a trainable controller.
EOF
}

case "$ACTIVE_GROUP" in
  H1_OFFLINE_ONLY)
    run_policy_sweep
    run_generator_robustness
    run_ensemble_eval
    ;;
  H2_DISTILL_ONLY)
    run_bc_only
    run_bc_then_rl
    run_rl_like_probe
    ;;
  HALL_HYBRID_USABILITY)
    run_policy_sweep
    run_generator_robustness
    run_ensemble_eval
    run_bc_only
    run_bc_then_rl
    run_rl_like_probe
    ;;
  *)
    echo "ERROR: unknown ACTIVE_GROUP=${ACTIVE_GROUP}" >&2
    exit 1
    ;;
esac

render_summary
log_line "summary_file: ${SUMMARY_FILE}"
