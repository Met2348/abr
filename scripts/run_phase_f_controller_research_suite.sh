#!/usr/bin/env bash
# Phase F controller research suite.
#
# 这套脚本把当前最有信息增益的 `Phase F` 研究实验组织成参数组，
# 重点回答三个问题：
# 1) 旧 controller 失败到底是不是 rule design 问题；
# 2) 在 generator / policy-shift 下，哪些 controller family 更稳；
# 3) 多个强 verifier 的离线 score ensemble 能否进一步抬 controller 表现。
#
# This suite bundles the highest-information-gain Phase F experiments into
# reproducible parameter groups. It focuses on controller design, robustness,
# and weak-verifier score ensembling, all using existing scored artifacts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PREFIX="${RUN_PREFIX:-phase_f_controller_research}"
ACTIVE_PHASE_F_CONTROLLER_GROUP="${ACTIVE_PHASE_F_CONTROLLER_GROUP:-RALL_PHASEF_CONTROLLER_RESEARCH}"

PROCESSBENCH_GSM="${PROCESSBENCH_GSM:-assets/external_datasets/qwen_processbench/gsm8k.json}"
PROCESSBENCH_MATH="${PROCESSBENCH_MATH:-assets/external_datasets/qwen_processbench/math.json}"

PBR12_GSM_ROWS="${PBR12_GSM_ROWS:-assets/artifacts/phase_e_eval/pbr12_dpo_mathms_gsm_fulleval_20260311T114553Z/scored_rows.jsonl}"
PBR12_MATH_ROWS="${PBR12_MATH_ROWS:-assets/artifacts/phase_e_eval/pbr12_dpo_mathms_math_fulleval_20260311T114553Z/scored_rows.jsonl}"
PBR19_GSM_ROWS="${PBR19_GSM_ROWS:-assets/artifacts/phase_e_eval/pbr19_dpo_mathms_joint_gsm_fulleval_20260311T123421Z/scored_rows.jsonl}"
PBR19_MATH_ROWS="${PBR19_MATH_ROWS:-assets/artifacts/phase_e_eval/pbr19_dpo_mathms_joint_math_fulleval_20260311T123421Z/scored_rows.jsonl}"
PBR21_GSM_ROWS="${PBR21_GSM_ROWS:-assets/artifacts/phase_e_eval/pbr21_joint_10ep_gsm_fulleval_20260311T124739Z/scored_rows.jsonl}"
PBR21_MATH_ROWS="${PBR21_MATH_ROWS:-assets/artifacts/phase_e_eval/pbr21_joint_10ep_math_fulleval_20260311T124740Z/scored_rows.jsonl}"
PBR26_GSM_ROWS="${PBR26_GSM_ROWS:-assets/artifacts/phase_e_eval/pbr26_dpo_ms_full_gsm8k_eval_20260311T140510Z/scored_rows.jsonl}"
PBR26_MATH_ROWS="${PBR26_MATH_ROWS:-assets/artifacts/phase_e_eval/pbr26_dpo_ms_full_math_eval_20260311T140419Z/scored_rows.jsonl}"
PBR31_GSM_ROWS="${PBR31_GSM_ROWS:-assets/artifacts/phase_e_eval/pbr31_verify_gsm_0312_20260311T170309Z/scored_rows.jsonl}"
PBR31_MATH_ROWS="${PBR31_MATH_ROWS:-assets/artifacts/phase_e_eval/pbr31_verify_math_0312_20260311T170630Z/scored_rows.jsonl}"

LOG_ROOT="assets/artifacts/phase_f_logs/${RUN_PREFIX}_${ACTIVE_PHASE_F_CONTROLLER_GROUP}"
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
  "$PBR12_GSM_ROWS|PBR12 GSM rows" \
  "$PBR12_MATH_ROWS|PBR12 Math rows" \
  "$PBR19_GSM_ROWS|PBR19 GSM rows" \
  "$PBR19_MATH_ROWS|PBR19 Math rows" \
  "$PBR21_GSM_ROWS|PBR21 GSM rows" \
  "$PBR21_MATH_ROWS|PBR21 Math rows" \
  "$PBR26_GSM_ROWS|PBR26 GSM rows" \
  "$PBR26_MATH_ROWS|PBR26 Math rows" \
  "$PBR31_GSM_ROWS|PBR31 GSM rows" \
  "$PBR31_MATH_ROWS|PBR31 Math rows"; do
  IFS='|' read -r path label <<< "$path_label"
  require_path "$path" "$label"
done

run_controller_sweep() {
  log_line "R1 controller sweep: broad policy-family search on major candidates"
  $PYTHON_BIN scripts/phase_f_controller_policy_sweep.py \
    --run-name "${RUN_PREFIX}_controller_sweep" \
    --case "pbr12_math|${PBR12_MATH_ROWS}" \
    --case "pbr12_gsm|${PBR12_GSM_ROWS}" \
    --case "pbr19_math|${PBR19_MATH_ROWS}" \
    --case "pbr19_gsm|${PBR19_GSM_ROWS}" \
    --case "pbr21_math|${PBR21_MATH_ROWS}" \
    --case "pbr21_gsm|${PBR21_GSM_ROWS}" \
    --case "pbr26_math|${PBR26_MATH_ROWS}" \
    --case "pbr26_gsm|${PBR26_GSM_ROWS}" \
    | tee -a "$SUITE_LOG_FILE"
}

run_generator_robustness() {
  log_line "R2 generator robustness: select policies by worst-generator balanced_f1"
  $PYTHON_BIN scripts/phase_f_controller_generator_robustness.py \
    --run-name "${RUN_PREFIX}_generator_robustness" \
    --case "pbr26_math|${PBR26_MATH_ROWS}|${PROCESSBENCH_MATH}" \
    --case "pbr26_gsm|${PBR26_GSM_ROWS}|${PROCESSBENCH_GSM}" \
    --case "pbr31_math|${PBR31_MATH_ROWS}|${PROCESSBENCH_MATH}" \
    --case "pbr31_gsm|${PBR31_GSM_ROWS}|${PROCESSBENCH_GSM}" \
    | tee -a "$SUITE_LOG_FILE"
}

run_ensemble_eval() {
  log_line "R3 weak-verifier ensemble: combine PBR26 and PBR31 offline scores"
  $PYTHON_BIN scripts/phase_f_controller_ensemble_eval.py \
    --run-name "${RUN_PREFIX}_ensemble_eval" \
    --case "pbr26_pbr31_math|${PBR26_MATH_ROWS}|${PBR31_MATH_ROWS}" \
    --case "pbr26_pbr31_gsm|${PBR26_GSM_ROWS}|${PBR31_GSM_ROWS}" \
    --case "pbr19_pbr31_math|${PBR19_MATH_ROWS}|${PBR31_MATH_ROWS}" \
    --case "pbr19_pbr31_gsm|${PBR19_GSM_ROWS}|${PBR31_GSM_ROWS}" \
    | tee -a "$SUITE_LOG_FILE"
}

case "$ACTIVE_PHASE_F_CONTROLLER_GROUP" in
  R1_CONTROLLER_SWEEP)
    # 目的：确认旧 controller 失败是不是 rule design 问题
    # 观察：best policy family 与 baseline 的差距
    # 预期：threshold-only / guarded / delayed 应显著优于 baseline_immediate
    run_controller_sweep
    ;;
  R2_GENERATOR_ROBUSTNESS)
    # 目的：看 policy-shift / generator-shift 下哪些 controller 更稳
    # 观察：worst-generator balanced_f1
    # 预期：robust best family 会偏向更保守的 threshold-only / guarded
    run_generator_robustness
    ;;
  R3_ENSEMBLE_EVAL)
    # 目的：验证 weak-verifier ensemble 是否能进一步提升 controller
    # 观察：best ensemble mode 与单模型 best policy 的差距
    # 预期：mean/min 型 ensemble 可能提升稳健性，但不一定提升 raw headline
    run_ensemble_eval
    ;;
  RALL_PHASEF_CONTROLLER_RESEARCH)
    # 目的：一夜跑完 controller 研究主线
    # 观察：
    # - broad controller family frontier
    # - worst-generator robustness
    # - weak-verifier ensemble frontier
    # 预期：
    # - baseline_immediate 被系统性淘汰
    # - Math 更偏 threshold-only
    # - GSM 更偏 delayed/guarded
    # - ensemble 可能提高 robustness，但不一定赢过最强单模型
    run_controller_sweep
    run_generator_robustness
    run_ensemble_eval
    ;;
  *)
    echo "ERROR: unknown ACTIVE_PHASE_F_CONTROLLER_GROUP=${ACTIVE_PHASE_F_CONTROLLER_GROUP}" >&2
    exit 1
    ;;
esac

{
  echo "# Phase F Controller Research Suite"
  echo
  echo "- group: \`${ACTIVE_PHASE_F_CONTROLLER_GROUP}\`"
  echo "- log: \`${SUITE_LOG_FILE}\`"
  echo "- completed_at: \`$(date -u '+%Y-%m-%dT%H:%M:%SZ')\`"
} > "$SUMMARY_FILE"

log_line "Suite complete. Summary -> ${SUMMARY_FILE}"
