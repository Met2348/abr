#!/usr/bin/env bash
# Phase F overnight usability suite.
#
# 这个脚本把 Phase F 从“只有零散研究脚本”推进到“夜间可连续产出 usable controller artifact”的程度。
# 它做的事情分成三层：
# 1. 先用更强的 Phase E verifier 切片重跑 offline controller 研究；
# 2. 再用共享 heuristic teacher 训练 BC / BC->RL controller；
# 3. 最后补一小组 robust-from-scratch RL-like 对照，确认 RL 现在到底值不值得继续押注。
#
# This suite turns Phase F from a collection of isolated research utilities into
# an overnight-ready pipeline:
# 1. refresh controller research on stronger verifier slices,
# 2. train BC / BC->RL controllers from shared heuristic teachers,
# 3. keep one compact robust-from-scratch RL-like control arm as a reality check.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PREFIX="${RUN_PREFIX:-phase_f_usability_overnight_$(date +%m%d_%H%M)}"
ACTIVE_PHASE_F_USABILITY_GROUP="${ACTIVE_PHASE_F_USABILITY_GROUP:-UALL_PHASEF_USABILITY}"

PROCESSBENCH_GSM="${PROCESSBENCH_GSM:-assets/external_datasets/qwen_processbench/gsm8k.json}"
PROCESSBENCH_MATH="${PROCESSBENCH_MATH:-assets/external_datasets/qwen_processbench/math.json}"

PBR26_MATH_ROWS="${PBR26_MATH_ROWS:-assets/artifacts/phase_e_eval/pbr26_dpo_ms_full_math_fulleval_0312_20260311T140557Z/scored_rows.jsonl}"
PBR31_MATH_ROWS="${PBR31_MATH_ROWS:-assets/artifacts/phase_e_eval/pbr31_verify_math_0312_20260311T170630Z/scored_rows.jsonl}"
PBR32_MATH_ROWS="${PBR32_MATH_ROWS:-assets/artifacts/phase_e_eval/pbr32_lora_mathprm_alllayers_pb_math_20260311T171442Z/scored_rows.jsonl}"
PBR33_MATH_ROWS="${PBR33_MATH_ROWS:-assets/artifacts/phase_e_eval/phase_e_pbr33_lora_mathprm_top4_pbr26data_s42_20260311T162439Z_pb_math_20260311T183625Z/scored_rows.jsonl}"

PBR19_GSM_ROWS="${PBR19_GSM_ROWS:-assets/artifacts/phase_e_eval/pbr19_dpo_mathms_joint_gsm_fulleval_20260311T123421Z/scored_rows.jsonl}"
PBR31_GSM_ROWS="${PBR31_GSM_ROWS:-assets/artifacts/phase_e_eval/pbr31_verify_gsm_0312_20260311T170309Z/scored_rows.jsonl}"
PBR32_GSM_ROWS="${PBR32_GSM_ROWS:-assets/artifacts/phase_e_eval/pbr32_lora_mathprm_alllayers_pb_gsm_20260311T171442Z/scored_rows.jsonl}"
PBR33_GSM_ROWS="${PBR33_GSM_ROWS:-assets/artifacts/phase_e_eval/phase_e_pbr33_lora_mathprm_top4_pbr26data_s42_20260311T162439Z_pb_gsm_20260311T184356Z/scored_rows.jsonl}"

LOG_ROOT="assets/artifacts/phase_f_logs/${RUN_PREFIX}_${ACTIVE_PHASE_F_USABILITY_GROUP}"
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

latest_dir_by_prefix() {
  local root="$1"
  local prefix="$2"
  python - "$root" "$prefix" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
prefix = sys.argv[2]
matches = sorted(root.glob(f"{prefix}_*"), key=lambda path: path.stat().st_mtime, reverse=True)
if not matches:
    raise SystemExit(f"No artifact directory matches prefix={prefix!r} under {root}")
print(matches[0])
PY
}

latest_dir_by_prefix_or_empty() {
  local root="$1"
  local prefix="$2"
  python - "$root" "$prefix" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
prefix = sys.argv[2]
matches = sorted(root.glob(f"{prefix}_*"), key=lambda path: path.stat().st_mtime, reverse=True)
print(matches[0] if matches else "")
PY
}

run_policy_sweep() {
  log_line "R1 policy sweep on refreshed verifier slices"
  "$PYTHON_BIN" scripts/phase_f_controller_policy_sweep.py \
    --run-name "${RUN_PREFIX}_controller_sweep" \
    --case "pbr26_math|${PBR26_MATH_ROWS}" \
    --case "pbr31_math|${PBR31_MATH_ROWS}" \
    --case "pbr32_math|${PBR32_MATH_ROWS}" \
    --case "pbr33_math|${PBR33_MATH_ROWS}" \
    --case "pbr19_gsm|${PBR19_GSM_ROWS}" \
    --case "pbr31_gsm|${PBR31_GSM_ROWS}" \
    --case "pbr32_gsm|${PBR32_GSM_ROWS}" \
    --case "pbr33_gsm|${PBR33_GSM_ROWS}" \
    | tee -a "$SUITE_LOG_FILE"
}

run_generator_robustness() {
  log_line "R2 generator-robust policy search on refreshed verifier slices"
  "$PYTHON_BIN" scripts/phase_f_controller_generator_robustness.py \
    --run-name "${RUN_PREFIX}_generator_robustness" \
    --case "pbr26_math|${PBR26_MATH_ROWS}|${PROCESSBENCH_MATH}" \
    --case "pbr31_math|${PBR31_MATH_ROWS}|${PROCESSBENCH_MATH}" \
    --case "pbr32_math|${PBR32_MATH_ROWS}|${PROCESSBENCH_MATH}" \
    --case "pbr33_math|${PBR33_MATH_ROWS}|${PROCESSBENCH_MATH}" \
    --case "pbr19_gsm|${PBR19_GSM_ROWS}|${PROCESSBENCH_GSM}" \
    --case "pbr31_gsm|${PBR31_GSM_ROWS}|${PROCESSBENCH_GSM}" \
    --case "pbr32_gsm|${PBR32_GSM_ROWS}|${PROCESSBENCH_GSM}" \
    --case "pbr33_gsm|${PBR33_GSM_ROWS}|${PROCESSBENCH_GSM}" \
    | tee -a "$SUITE_LOG_FILE"
}

run_ensemble_eval() {
  log_line "R3 weak-verifier ensemble evaluation on complementary strong candidates"
  "$PYTHON_BIN" scripts/phase_f_controller_ensemble_eval.py \
    --run-name "${RUN_PREFIX}_ensemble_eval" \
    --case "math_p26_p32|${PBR26_MATH_ROWS}|${PBR32_MATH_ROWS}" \
    --case "math_p26_p33|${PBR26_MATH_ROWS}|${PBR33_MATH_ROWS}" \
    --case "math_p31_p32|${PBR31_MATH_ROWS}|${PBR32_MATH_ROWS}" \
    --case "gsm_p19_p31|${PBR19_GSM_ROWS}|${PBR31_GSM_ROWS}" \
    --case "gsm_p19_p33|${PBR19_GSM_ROWS}|${PBR33_GSM_ROWS}" \
    --case "gsm_p31_p33|${PBR31_GSM_ROWS}|${PBR33_GSM_ROWS}" \
    | tee -a "$SUITE_LOG_FILE"
}

run_bc_only() {
  log_line "R4 behavior cloning from shared strong heuristic teachers"
  "$PYTHON_BIN" scripts/phase_f_behavior_clone_controller.py \
    --run-name "${RUN_PREFIX}_bc_only" \
    --case "pbr26_math|${PBR26_MATH_ROWS}|${PROCESSBENCH_MATH}|threshold_only|{\"tau\": 0.42}" \
    --case "pbr32_math|${PBR32_MATH_ROWS}|${PROCESSBENCH_MATH}|threshold_only|{\"tau\": 0.42}" \
    --case "pbr31_gsm|${PBR31_GSM_ROWS}|${PROCESSBENCH_GSM}|delayed_drop|{\"tau\": 0.42, \"delta\": 0.25, \"min_step\": 4}" \
    --case "pbr33_gsm|${PBR33_GSM_ROWS}|${PROCESSBENCH_GSM}|delayed_drop|{\"tau\": 0.42, \"delta\": 0.25, \"min_step\": 4}" \
    --seed 42 \
    --hidden-dim 16 \
    --bc-epochs 50 \
    --bc-learning-rate 0.003 \
    | tee -a "$SUITE_LOG_FILE"
}

run_bc_then_rl() {
  log_line "R5 BC warm start followed by robust RL-like fine-tune"
  "$PYTHON_BIN" scripts/phase_f_behavior_clone_controller.py \
    --run-name "${RUN_PREFIX}_bc_then_rl" \
    --case "pbr26_math|${PBR26_MATH_ROWS}|${PROCESSBENCH_MATH}|threshold_only|{\"tau\": 0.42}" \
    --case "pbr32_math|${PBR32_MATH_ROWS}|${PROCESSBENCH_MATH}|threshold_only|{\"tau\": 0.42}" \
    --case "pbr31_gsm|${PBR31_GSM_ROWS}|${PROCESSBENCH_GSM}|delayed_drop|{\"tau\": 0.42, \"delta\": 0.25, \"min_step\": 4}" \
    --case "pbr33_gsm|${PBR33_GSM_ROWS}|${PROCESSBENCH_GSM}|delayed_drop|{\"tau\": 0.42, \"delta\": 0.25, \"min_step\": 4}" \
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
  log_line "R6 robust-from-scratch RL-like probe on strongest Math / GSM slices"
  "$PYTHON_BIN" scripts/phase_f_train_trainable_controller.py \
    --run-name "${RUN_PREFIX}_rl_like_robust" \
    --case "pbr32_math|${PBR32_MATH_ROWS}|${PROCESSBENCH_MATH}" \
    --case "pbr33_gsm|${PBR33_GSM_ROWS}|${PROCESSBENCH_GSM}" \
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
  local sweep_dir
  local robust_dir
  local ensemble_dir
  local bc_only_dir
  local bc_then_rl_dir
  local rl_like_dir
  sweep_dir="$(latest_dir_by_prefix_or_empty "assets/artifacts/phase_f_controller_sweep" "${RUN_PREFIX}_controller_sweep")"
  robust_dir="$(latest_dir_by_prefix_or_empty "assets/artifacts/phase_f_controller_robustness" "${RUN_PREFIX}_generator_robustness")"
  ensemble_dir="$(latest_dir_by_prefix_or_empty "assets/artifacts/phase_f_controller_ensemble" "${RUN_PREFIX}_ensemble_eval")"
  bc_only_dir="$(latest_dir_by_prefix_or_empty "assets/artifacts/phase_f_bc" "${RUN_PREFIX}_bc_only")"
  bc_then_rl_dir="$(latest_dir_by_prefix_or_empty "assets/artifacts/phase_f_bc" "${RUN_PREFIX}_bc_then_rl")"
  rl_like_dir="$(latest_dir_by_prefix_or_empty "assets/artifacts/phase_f_rl_like" "${RUN_PREFIX}_rl_like_robust")"
  {
    echo "# Phase F Overnight Usability Suite"
    echo
    echo "- run_prefix: \`${RUN_PREFIX}\`"
    echo "- group_id: \`${ACTIVE_PHASE_F_USABILITY_GROUP}\`"
    echo "- suite_log: \`${SUITE_LOG_FILE}\`"
    echo
    echo "## Artifacts"
    echo
    echo "1. controller_sweep: \`${sweep_dir:-N/A}\`"
    echo "2. generator_robustness: \`${robust_dir:-N/A}\`"
    echo "3. ensemble_eval: \`${ensemble_dir:-N/A}\`"
    echo "4. bc_only: \`${bc_only_dir:-N/A}\`"
    echo "5. bc_then_rl: \`${bc_then_rl_dir:-N/A}\`"
    echo "6. rl_like_robust: \`${rl_like_dir:-N/A}\`"
    echo
    echo "## Reading Guide"
    echo
    echo "- R1-R3 answer whether strong verifiers already support usable heuristics."
    echo "- R4-R5 answer whether BC warm-start is enough to distill those heuristics."
    echo "- R6 keeps one robust-from-scratch control arm so RL hype stays grounded."
  } > "$SUMMARY_FILE"
}

for path_label in \
  "${PROCESSBENCH_GSM}|ProcessBench GSM" \
  "${PROCESSBENCH_MATH}|ProcessBench Math" \
  "${PBR26_MATH_ROWS}|PBR26 Math rows" \
  "${PBR31_MATH_ROWS}|PBR31 Math rows" \
  "${PBR32_MATH_ROWS}|PBR32 Math rows" \
  "${PBR33_MATH_ROWS}|PBR33 Math rows" \
  "${PBR19_GSM_ROWS}|PBR19 GSM rows" \
  "${PBR31_GSM_ROWS}|PBR31 GSM rows" \
  "${PBR32_GSM_ROWS}|PBR32 GSM rows" \
  "${PBR33_GSM_ROWS}|PBR33 GSM rows"; do
  IFS='|' read -r path label <<< "$path_label"
  require_path "$path" "$label"
done

case "$ACTIVE_PHASE_F_USABILITY_GROUP" in
  U1_OFFLINE_RESEARCH_REFRESH)
    # 目的：只刷新 stronger verifiers 上的 offline controller 结论。
    # 观察：Math / GSM 上的 best policy family 是否延续“threshold_only vs delayed_drop”分化。
    # 预期：旧 `baseline_immediate` 会继续被淘汰，且 p32/p33 会成为更高价值的 controller slices。
    run_policy_sweep
    run_generator_robustness
    run_ensemble_eval
    ;;
  U2_CONTROLLER_DISTILL)
    # 目的：把 strongest heuristic teacher 压缩成可部署的 BC / BC->RL controller。
    # 观察：BC 是否已经足够接近 heuristic；RL fine-tune 是否再次拖后腿。
    # 预期：`bc_only` 依然比 `bc_then_rl` 更稳，说明当前 RL 不是第一优先级。
    run_bc_only
    run_bc_then_rl
    ;;
  U3_RL_STRESS)
    # 目的：把 RL-like 方向保留为严格对照，而不是默认 promotion path。
    # 观察：worst-generator 指标是否真的超过 BC / heuristic。
    # 预期：robust-from-scratch 仍难稳定赢过 BC teacher。
    run_rl_like_probe
    ;;
  UALL_PHASEF_USABILITY)
    # 目的：一夜串行跑完“研究 -> 蒸馏 -> RL 对照”三层链路。
    # 观察：
    # - 哪些 verifier slice 真正适合作 controller；
    # - heuristic 是否已经足够；
    # - RL 是否仍应降级为对照路线。
    # 预期：
    # - offline heuristic / BC 会继续领先；
    # - RL-like 结果更适合作为 stress test，而非当前主线。
    run_policy_sweep
    run_generator_robustness
    run_ensemble_eval
    run_bc_only
    run_bc_then_rl
    run_rl_like_probe
    ;;
  *)
    echo "ERROR: unknown ACTIVE_PHASE_F_USABILITY_GROUP=${ACTIVE_PHASE_F_USABILITY_GROUP}" >&2
    exit 1
    ;;
esac

render_summary
log_line "Suite complete. Summary -> ${SUMMARY_FILE}"
