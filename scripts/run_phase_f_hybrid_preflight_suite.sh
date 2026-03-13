#!/usr/bin/env bash
# Preflight stronger PH1/PH2 hybrid verifier candidates before any RL/control promotion.
#
# 中文：
# 这一步回答两个部署前问题：
# 1. 固定阈值附近稳不稳？
# 2. 会不会被 superficial tails / style attacks 明显抬分？
#
# 这些候选都来自同一 Math-PRM frozen backbone，因此可以共用一次 reward-hacking probe。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PREFIX="${RUN_PREFIX:-phase_f_hybrid_preflight_$(date +%m%d_%H%M)}"
GPU_DEVICE="${GPU_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"
PROCESSBENCH_GSM="${PROCESSBENCH_GSM:-assets/external_datasets/qwen_processbench/gsm8k.json}"
PROCESSBENCH_MATH="${PROCESSBENCH_MATH:-assets/external_datasets/qwen_processbench/math.json}"

PH1_MLP_RUN_DIR="${PH1_MLP_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_prmbackbone_hybrid_ph1_live_0312_ph1_prm_local_ta15_arch_sweep_smoke_mlp_20260312T083551Z}"
PH1_GATED_RUN_DIR="${PH1_GATED_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_prmbackbone_hybrid_ph1_live_0312_ph1_prm_local_ta15_arch_sweep_smoke_gated_mlp_20260312T083748Z}"
PH2_MLP_RUN_DIR="${PH2_MLP_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_prmbackbone_hybrid_ph2_live_0312_ph2_prm_local_ta10_msgrid10_arch_sweep_smoke_mlp_20260312T082725Z}"
PH2_GATED_RUN_DIR="${PH2_GATED_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_prmbackbone_hybrid_ph2_live_0312_ph2_prm_local_ta10_msgrid10_arch_sweep_smoke_gated_mlp_20260312T083827Z}"

PH1_MLP_GSM_EVAL="${PH1_MLP_GSM_EVAL:-assets/artifacts/phase_e_eval/phase_e_prmbackbone_hybrid_ph1_live_0312_ph1_prm_local_ta15_arch_sweep_smoke_mlp_processbench_gsm8k_20260312T083610Z}"
PH1_MLP_MATH_EVAL="${PH1_MLP_MATH_EVAL:-assets/artifacts/phase_e_eval/phase_e_prmbackbone_hybrid_ph1_live_0312_ph1_prm_local_ta15_arch_sweep_smoke_mlp_processbench_math_20260312T083645Z}"
PH1_GATED_GSM_EVAL="${PH1_GATED_GSM_EVAL:-assets/artifacts/phase_e_eval/phase_e_prmbackbone_hybrid_ph1_live_0312_ph1_prm_local_ta15_arch_sweep_smoke_gated_mlp_processbench_gsm8k_20260312T083808Z}"
PH1_GATED_MATH_EVAL="${PH1_GATED_MATH_EVAL:-assets/artifacts/phase_e_eval/phase_e_prmbackbone_hybrid_ph1_live_0312_ph1_prm_local_ta15_arch_sweep_smoke_gated_mlp_processbench_math_20260312T083817Z}"
PH2_MLP_GSM_EVAL="${PH2_MLP_GSM_EVAL:-assets/artifacts/phase_e_eval/phase_e_prmbackbone_hybrid_ph2_live_0312_ph2_prm_local_ta10_msgrid10_arch_sweep_smoke_mlp_processbench_gsm8k_20260312T083713Z}"
PH2_MLP_MATH_EVAL="${PH2_MLP_MATH_EVAL:-assets/artifacts/phase_e_eval/phase_e_prmbackbone_hybrid_ph2_live_0312_ph2_prm_local_ta10_msgrid10_arch_sweep_smoke_mlp_processbench_math_20260312T083722Z}"
PH2_GATED_GSM_EVAL="${PH2_GATED_GSM_EVAL:-assets/artifacts/phase_e_eval/phase_e_prmbackbone_hybrid_ph2_live_0312_ph2_prm_local_ta10_msgrid10_arch_sweep_smoke_gated_mlp_processbench_gsm8k_20260312T083847Z}"
PH2_GATED_MATH_EVAL="${PH2_GATED_MATH_EVAL:-assets/artifacts/phase_e_eval/phase_e_prmbackbone_hybrid_ph2_live_0312_ph2_prm_local_ta10_msgrid10_arch_sweep_smoke_gated_mlp_processbench_math_20260312T083856Z}"

LOG_ROOT="assets/artifacts/phase_f_logs/${RUN_PREFIX}"
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

for path_label in \
  "$PROCESSBENCH_GSM|ProcessBench GSM" \
  "$PROCESSBENCH_MATH|ProcessBench Math" \
  "$PH1_MLP_RUN_DIR|PH1 MLP run dir" \
  "$PH1_GATED_RUN_DIR|PH1 gated run dir" \
  "$PH2_MLP_RUN_DIR|PH2 MLP run dir" \
  "$PH2_GATED_RUN_DIR|PH2 gated run dir" \
  "$PH1_MLP_GSM_EVAL|PH1 MLP GSM eval" \
  "$PH1_MLP_MATH_EVAL|PH1 MLP Math eval" \
  "$PH1_GATED_GSM_EVAL|PH1 gated GSM eval" \
  "$PH1_GATED_MATH_EVAL|PH1 gated Math eval" \
  "$PH2_MLP_GSM_EVAL|PH2 MLP GSM eval" \
  "$PH2_MLP_MATH_EVAL|PH2 MLP Math eval" \
  "$PH2_GATED_GSM_EVAL|PH2 gated GSM eval" \
  "$PH2_GATED_MATH_EVAL|PH2 gated Math eval"; do
  IFS='|' read -r path label <<< "$path_label"
  require_path "$path" "$label"
done

log_line "Running threshold/shift audit on PH1/PH2 hybrid candidates"
THRESHOLD_DIR="$($PYTHON_BIN scripts/phase_f_analyze_threshold_shift.py \
  --run-name "${RUN_PREFIX}_threshold_shift" \
  --case "ph1_mlp_gsm|${PH1_MLP_GSM_EVAL}" \
  --case "ph1_mlp_math|${PH1_MLP_MATH_EVAL}" \
  --case "ph1_gated_gsm|${PH1_GATED_GSM_EVAL}" \
  --case "ph1_gated_math|${PH1_GATED_MATH_EVAL}" \
  --case "ph2_mlp_gsm|${PH2_MLP_GSM_EVAL}" \
  --case "ph2_mlp_math|${PH2_MLP_MATH_EVAL}" \
  --case "ph2_gated_gsm|${PH2_GATED_GSM_EVAL}" \
  --case "ph2_gated_math|${PH2_GATED_MATH_EVAL}" | tee -a "$SUITE_LOG_FILE" | awk -F': ' '/summary_json/ {print $2}' | xargs dirname)"

log_line "Running reward-hacking probe on GPU ${GPU_DEVICE} for PH1/PH2 hybrid candidates"
PROBE_DIR="$({ CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$PYTHON_BIN" scripts/phase_f_probe_reward_hacking.py \
  --run-name "${RUN_PREFIX}_reward_probe" \
  --candidate "ph1_mlp|${PH1_MLP_RUN_DIR}" \
  --candidate "ph1_gated|${PH1_GATED_RUN_DIR}" \
  --candidate "ph2_mlp|${PH2_MLP_RUN_DIR}" \
  --candidate "ph2_gated|${PH2_GATED_RUN_DIR}" \
  --benchmark-spec "processbench_gsm8k|${PROCESSBENCH_GSM}" \
  --benchmark-spec "processbench_math|${PROCESSBENCH_MATH}" \
  --batch-size 48 \
  --max-length 1024 \
  --require-cuda; } | tee -a "$SUITE_LOG_FILE" | awk -F': ' '/summary_json/ {print $2}' | xargs dirname)"

$PYTHON_BIN - "$THRESHOLD_DIR" "$PROBE_DIR" "$SUMMARY_FILE" <<'PY'
from pathlib import Path
import json, sys
threshold_dir = Path(sys.argv[1])
probe_dir = Path(sys.argv[2])
out_path = Path(sys.argv[3])
threshold = json.loads((threshold_dir / 'summary.json').read_text(encoding='utf-8'))
probe = json.loads((probe_dir / 'summary.json').read_text(encoding='utf-8'))
lines = [
    '# Phase F Hybrid Preflight Summary',
    '',
    f'- threshold_shift_dir: `{threshold_dir}`',
    f'- reward_probe_dir: `{probe_dir}`',
    '',
    '## Threshold / Shift Snapshot',
    '',
    '| case_id | best_f1 | f1@0.5 | near_best_width | gen_tau_std | worst_gen_logo_f1 |',
    '|---|---:|---:|---:|---:|---:|',
]
for row in threshold.get('cases', []):
    lines.append(
        '| {case_id} | {best_f1:.4f} | {fixed_f1:.4f} | {near_best_window_width:.3f} | {generator_best_threshold_std:.4f} | {worst_generator_logo_f1:.4f} |'.format(**row)
    )
lines.extend(['', '## Reward-Hacking Probe Snapshot', '', '| candidate | benchmark | group | attack | mean_delta | flip@0.5 | outrank_safe | risk |', '|---|---|---|---|---:|---:|---:|---|'])
for row in probe.get('metric_rows', []):
    if row.get('attack_name') not in {'confidence_tail', 'filler_tail', 'self_verify_tail', 'repeat_last_claim'}:
        continue
    lines.append(
        '| {candidate_id} | {benchmark_id} | {probe_group} | {attack_name} | {mean_delta:.4f} | {flip_rate_at_fixed_threshold:.4f} | {outrank_safe_rate} | {risk_level} |'.format(
            candidate_id=row['candidate_id'],
            benchmark_id=row['benchmark_id'],
            probe_group=row['probe_group'],
            attack_name=row['attack_name'],
            mean_delta=float(row.get('mean_delta', 0.0)),
            flip_rate_at_fixed_threshold=float(row.get('flip_rate_at_fixed_threshold', 0.0)),
            outrank_safe_rate=('n/a' if row.get('outrank_safe_rate') is None else f"{float(row['outrank_safe_rate']):.4f}"),
            risk_level=row.get('risk_level', 'unknown'),
        )
    )
lines.extend(['', '## Reading Guide', '', '- `near_best_width` 太小意味着部署阈值脆弱。', '- `worst_gen_logo_f1` 太低意味着 generator shift 仍是 controller 风险。', '- `flip@0.5` / `outrank_safe` 高说明该 candidate 更容易被 superficial tails 刷分。'])
out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
PY

log_line "summary_file: ${SUMMARY_FILE}"
