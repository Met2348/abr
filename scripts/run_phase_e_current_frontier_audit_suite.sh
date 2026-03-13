#!/usr/bin/env bash
# CPU-only Phase E current-frontier audit suite.
#
# English
# -------
# This suite consolidates the current frontier into two reusable diagnostics:
# 1. compare the training pair pools that produced the current key runs,
# 2. compare the held-out/same-family/ProcessBench metrics side by side.
#
# 中文
# ----
# 这个 suite 把当前主线候选统一审计成两类可复用诊断：
# 1. 对比这些关键 run 背后的训练 pair pool，
# 2. 把 held-out / same-family / ProcessBench 指标并排放在一起。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_current_frontier_audit}"
LOG_DIR="assets/artifacts/phase_e_analysis/${RUN_PREFIX}"
LOG_FILE="${LOG_DIR}/suite.log"
SUMMARY_FILE="${LOG_DIR}/final_summary.md"
mkdir -p "$LOG_DIR"
: > "$LOG_FILE"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1" | tee -a "$LOG_FILE"
}

log "Compare key pair pools"
"$PYTHON_BIN" -u scripts/phase_e_compare_pair_pools.py \
  --run-name "${RUN_PREFIX}_pair_pools" \
  --output-root assets/artifacts/phase_e_analysis \
  --pair-dir PBR12 assets/artifacts/phase_e_pairs/phase_e_pbr12_dpo_plus_mathms_pairs__70b9f8db5f31 \
  --pair-dir PBR26 assets/artifacts/phase_e_pairs/phase_e_pbr26_dpo_plus_ms_full_pairs__b17437d10dfc \
  --pair-dir PBR38B assets/artifacts/phase_e_pairs/phase_e_pbr38b_consensus_filtered_pairs | tee -a "$LOG_FILE"

log "Build current frontier matrix"
"$PYTHON_BIN" -u scripts/phase_e_analyze_frontier_matrix.py \
  --run-name "${RUN_PREFIX}_frontier" \
  --output-root assets/artifacts/phase_e_analysis \
  --entry PBR26 assets/artifacts/phase_e_runs/phase_e_pbr26_dpo_ms_full_s42_value_20260311T134542Z assets/artifacts/phase_e_samefamily_eval/pbr26_samefamily_verify_0312_20260311T165945Z assets/artifacts/phase_e_eval/pbr26_dpo_ms_full_gsm8k_eval_20260311T140510Z assets/artifacts/phase_e_eval/pbr26_dpo_ms_full_math_eval_20260311T140419Z \
  --entry PBR31 assets/artifacts/phase_e_runs/phase_e_pbr31_lora_mathprm_pbr12data_s42_20260311T150316Z assets/artifacts/phase_e_samefamily_eval/pbr31_samefamily_verify_0312_20260311T170244Z assets/artifacts/phase_e_eval/pbr31_verify_gsm_0312_20260311T170309Z assets/artifacts/phase_e_eval/pbr31_verify_math_0312_20260311T170630Z \
  --entry PBR32 assets/artifacts/phase_e_runs/phase_e_pbr32_lora_mathprm_alllayers_pbr12data_s42_20260311T152656Z - assets/artifacts/phase_e_eval/pbr32_lora_mathprm_alllayers_pb_gsm_20260311T171442Z assets/artifacts/phase_e_eval/pbr32_lora_mathprm_alllayers_pb_math_20260311T171442Z \
  --entry PBR33 assets/artifacts/phase_e_runs/phase_e_pbr33_lora_mathprm_top4_pbr26data_s42_20260311T162439Z - assets/artifacts/phase_e_eval/phase_e_pbr33_lora_mathprm_top4_pbr26data_s42_20260311T162439Z_pb_gsm_20260311T184356Z assets/artifacts/phase_e_eval/phase_e_pbr33_lora_mathprm_top4_pbr26data_s42_20260311T162439Z_pb_math_20260311T183625Z \
  --entry PBR35 assets/artifacts/phase_e_runs/phase_e_pbr35_lora_mathprm_all28_pbr26data_s42_20260311T200442Z - assets/artifacts/phase_e_eval/phase_e_pbr35_lora_mathprm_all28_pbr26data_s42_20260311T200442Z_pb_gsm_20260311T215400Z assets/artifacts/phase_e_eval/phase_e_pbr35_lora_mathprm_all28_pbr26data_s42_20260311T200442Z_pb_math_20260311T213039Z \
  --entry PBR37 assets/artifacts/phase_e_runs/phase_e_pbr37_lora_r8_contrastive02_pbr26data_s42_20260311T204010Z - assets/artifacts/phase_e_eval/phase_e_pbr37_lora_r8_contrastive02_pbr26data_s42_20260311T204010Z_pb_gsm_20260312T005005Z assets/artifacts/phase_e_eval/phase_e_pbr37_lora_r8_contrastive02_pbr26data_s42_20260311T204010Z_pb_math_20260312T004005Z \
  --entry PBR38B assets/artifacts/phase_e_runs/phase_e_pbr38b_consensus820_frozen_mlp_s42_20260312T085318Z - - - | tee -a "$LOG_FILE"

PAIR_SUMMARY="$(ls -1dt assets/artifacts/phase_e_analysis/${RUN_PREFIX}_pair_pools_* | head -n 1)"
FRONTIER_SUMMARY="$(ls -1dt assets/artifacts/phase_e_analysis/${RUN_PREFIX}_frontier_* | head -n 1)"

cat > "$SUMMARY_FILE" <<EOF
# Phase E Current Frontier Audit

- pair_pool_summary: \`${PAIR_SUMMARY}/summary.md\`
- frontier_summary: \`${FRONTIER_SUMMARY}/summary.md\`
- suite_log: \`${LOG_FILE}\`

## Current Read

1. The frontier matrix compares the current main scalar candidates side by side.
2. The pair-pool comparison explains whether a negative run came from architecture or from a broken training pool.
3. If a curated pool drops \`local_first_bad_edge\`, it should not be promoted to benchmark training.
EOF

log "Summary written to ${SUMMARY_FILE}"
