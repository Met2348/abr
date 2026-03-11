#!/usr/bin/env bash
# Phase F preflight suite.
#
# 目标：
# 1) 在真正进入 Phase F controller/RL 之前，先对当前强候选做三类更贴近部署的离线检查；
# 2) 避免把“benchmark AUC 很高”误读成“可以放心拿去做 RL 主奖励”。
#
# 这套 suite 当前固定检查：
# - fixed-threshold stability
# - generator / policy-shift robustness
# - superficial reward-hacking probe
#
# 当前默认候选：
# - PBR12
# - PBR21
# - PRX1
#
# 它们共享 `Qwen2.5-Math-PRM-7B` backbone，因此 reward-hacking probe 可以复用同一 backbone，
# 减少显存搬运和重复加载。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PREFIX="${RUN_PREFIX:-phase_f_preflight}"
LOG_ROOT="assets/artifacts/phase_f_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"
CURRENT_STAGE="bootstrap"

mkdir -p "$LOG_ROOT"

PBR12_RUN_DIR="${PBR12_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_pbr12_dpo_plus_mathms_s42_value_20260311T112809Z}"
PBR21_RUN_DIR="${PBR21_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_pbr21_dpo_mathms_joint_10ep_s42_value_20260311T123656Z}"
PRX1_RUN_DIR="${PRX1_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_prx1_pbr10core_term1024_mlp_0311_20260311T123100Z}"

PBR12_GSM_EVAL="${PBR12_GSM_EVAL:-assets/artifacts/phase_e_eval/pbr12_dpo_mathms_gsm_fulleval_20260311T114553Z}"
PBR12_MATH_EVAL="${PBR12_MATH_EVAL:-assets/artifacts/phase_e_eval/pbr12_dpo_mathms_math_fulleval_20260311T114553Z}"
PBR21_GSM_EVAL="${PBR21_GSM_EVAL:-assets/artifacts/phase_e_eval/pbr21_joint_10ep_gsm_fulleval_20260311T124739Z}"
PBR21_MATH_EVAL="${PBR21_MATH_EVAL:-assets/artifacts/phase_e_eval/pbr21_joint_10ep_math_fulleval_20260311T124740Z}"
PRX1_GSM_EVAL="${PRX1_GSM_EVAL:-assets/artifacts/phase_e_eval/phase_e_prx1_gsm_fulleval_0311_20260311T130544Z}"
PRX1_MATH_EVAL="${PRX1_MATH_EVAL:-assets/artifacts/phase_e_eval/phase_e_prx1_math_fulleval_0311_verify_20260311T132702Z}"

PROCESSBENCH_GSM_PATH="${PROCESSBENCH_GSM_PATH:-assets/external_datasets/qwen_processbench/gsm8k.json}"
PROCESSBENCH_MATH_PATH="${PROCESSBENCH_MATH_PATH:-assets/external_datasets/qwen_processbench/math.json}"

FIXED_THRESHOLD="${FIXED_THRESHOLD:-0.5}"
THRESHOLD_BATCH_SIZE="${THRESHOLD_BATCH_SIZE:-48}"
PROBE_MAX_ERROR="${PROBE_MAX_ERROR:-48}"
PROBE_MAX_CORRECT="${PROBE_MAX_CORRECT:-48}"
GPU_DEVICE="${GPU_DEVICE:-3}"

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
  "$PBR12_RUN_DIR|PBR12 run dir" \
  "$PBR21_RUN_DIR|PBR21 run dir" \
  "$PRX1_RUN_DIR|PRX1 run dir" \
  "$PBR12_GSM_EVAL|PBR12 GSM eval" \
  "$PBR12_MATH_EVAL|PBR12 Math eval" \
  "$PBR21_GSM_EVAL|PBR21 GSM eval" \
  "$PBR21_MATH_EVAL|PBR21 Math eval" \
  "$PRX1_GSM_EVAL|PRX1 GSM eval" \
  "$PRX1_MATH_EVAL|PRX1 Math eval" \
  "$PROCESSBENCH_GSM_PATH|ProcessBench GSM path" \
  "$PROCESSBENCH_MATH_PATH|ProcessBench Math path"; do
  IFS='|' read -r path label <<< "$path_label"
  require_path "$path" "$label"
done

CURRENT_STAGE="threshold_shift"
log_line "Running threshold / policy-shift audit"
THRESHOLD_SHIFT_RUN="$($PYTHON_BIN scripts/phase_f_analyze_threshold_shift.py \
  --run-name "${RUN_PREFIX}_threshold_shift" \
  --case "pbr12_gsm|${PBR12_GSM_EVAL}" \
  --case "pbr12_math|${PBR12_MATH_EVAL}" \
  --case "pbr21_gsm|${PBR21_GSM_EVAL}" \
  --case "pbr21_math|${PBR21_MATH_EVAL}" \
  --case "prx1_gsm|${PRX1_GSM_EVAL}" \
  --case "prx1_math|${PRX1_MATH_EVAL}" \
  --fixed-threshold "${FIXED_THRESHOLD}" | tee -a "$SUITE_LOG_FILE" | awk -F': ' '/summary_json/{print $2}' | tail -n1)"
THRESHOLD_SHIFT_DIR="$(dirname "$THRESHOLD_SHIFT_RUN")"

CURRENT_STAGE="reward_hacking_probe"
log_line "Running reward-hacking probe on GPU ${GPU_DEVICE}"
PROBE_RUN="$(
  CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" \
  $PYTHON_BIN scripts/phase_f_probe_reward_hacking.py \
    --run-name "${RUN_PREFIX}_reward_probe" \
    --candidate "pbr12|${PBR12_RUN_DIR}" \
    --candidate "pbr21|${PBR21_RUN_DIR}" \
    --candidate "prx1|${PRX1_RUN_DIR}" \
    --benchmark-spec "processbench_gsm8k|${PROCESSBENCH_GSM_PATH}" \
    --benchmark-spec "processbench_math|${PROCESSBENCH_MATH_PATH}" \
    --fixed-threshold "${FIXED_THRESHOLD}" \
    --batch-size "${THRESHOLD_BATCH_SIZE}" \
    --max-error-examples-per-benchmark "${PROBE_MAX_ERROR}" \
    --max-correct-examples-per-benchmark "${PROBE_MAX_CORRECT}" \
    --require-cuda \
    | tee -a "$SUITE_LOG_FILE" | awk -F': ' '/summary_json/{print $2}' | tail -n1
)"
PROBE_DIR="$(dirname "$PROBE_RUN")"

CURRENT_STAGE="final_summary"
log_line "Rendering final summary"
$PYTHON_BIN - "$THRESHOLD_SHIFT_DIR" "$PROBE_DIR" "$SUMMARY_FILE" <<'PY'
import json
import sys
from pathlib import Path

threshold_dir = Path(sys.argv[1])
probe_dir = Path(sys.argv[2])
summary_file = Path(sys.argv[3])

threshold_summary = json.loads((threshold_dir / "summary.json").read_text(encoding="utf-8"))
probe_summary = json.loads((probe_dir / "summary.json").read_text(encoding="utf-8"))

case_rows = list(threshold_summary.get("cases", []))
probe_rows = list(probe_summary.get("metrics", []))

lines = [
    "# Phase F Preflight Suite Summary",
    "",
    f"- threshold_shift_dir: `{threshold_dir}`",
    f"- reward_probe_dir: `{probe_dir}`",
    "",
    "## Threshold / Shift Snapshot",
    "",
    "| case_id | best_f1 | f1@0.5 | near_best_width | gen_tau_std | worst_gen_logo_f1 |",
    "|---|---:|---:|---:|---:|---:|",
]
for row in case_rows:
    lines.append(
        "| {case_id} | {best_f1:.4f} | {fixed_f1:.4f} | {width:.3f} | {tau_std:.4f} | {worst_logo:.4f} |".format(
            case_id=row["case_id"],
            best_f1=row["best_f1"],
            fixed_f1=row["fixed_f1"],
            width=row["near_best_window_width"],
            tau_std=row["generator_best_threshold_std"],
            worst_logo=row["worst_generator_logo_f1"],
        )
    )

lines.extend(
    [
        "",
        "## Reward-Hacking Probe Snapshot",
        "",
        "| candidate | benchmark | group | attack | mean_delta | flip@0.5 | outrank_safe | risk |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
)
for row in probe_rows:
    outrank = "n/a" if row["outrank_safe_rate"] is None else f"{row['outrank_safe_rate']:.4f}"
    lines.append(
        "| {candidate} | {benchmark} | {group} | {attack} | {mean_delta:.4f} | {flip:.4f} | {outrank} | {risk} |".format(
            candidate=row["candidate_id"],
            benchmark=row["benchmark_id"],
            group=row["probe_group"],
            attack=row["attack_name"],
            mean_delta=row["mean_delta"],
            flip=row["flip_rate_fixed05"],
            outrank=outrank,
            risk=row["risk_level"],
        )
    )

lines.extend(
    [
        "",
        "## Reading Guide",
        "",
        "- `near_best_width` very small => deployment threshold is brittle.",
        "- `worst_gen_logo_f1` low => threshold learned on one generator subset does not transfer well to another policy slice.",
        "- `first_bad` probe with non-trivial `flip@0.5` or `outrank_safe` => obvious reward-hacking surface remains.",
        "",
    ]
)
summary_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_file)
PY

log_line "Phase F preflight complete. Summary -> ${SUMMARY_FILE}"
