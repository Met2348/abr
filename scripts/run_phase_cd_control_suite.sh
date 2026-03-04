#!/usr/bin/env bash
# Phase C/D unified control suite.
#
# Why this file exists:
# - We need one-click, reproducible control experiments spanning:
#   1) Phase C comparable baseline,
#   2) Phase D teacher-fusion ablations,
#   3) Phase D4 external-pair chain.
# - Existing suites are strong but isolated; this script orchestrates them and
#   always emits a cross-stage diagnosis report.
#
# What this file does:
# 1. Resolve one control bundle (`ACTIVE_PHASE_CD_GROUP`).
# 2. Run C baseline -> D teacher groups -> D4 external group (best effort).
# 3. Continue on stage failure by default (`FAIL_FAST=0`) and record statuses.
# 4. Run `scripts/phase_cd_compare_report.py` to summarize all historical D and
#    comparable C runs in one report.
#
# Quick start (light):
#   ACTIVE_PHASE_CD_GROUP=CD_LIGHT \
#   RUN_PREFIX=phase_cd_light \
#   CUDA_PHASE_C=1 CUDA_PHASE_D=2 CUDA_PHASE_D4=3 \
#   bash scripts/run_phase_cd_control_suite.sh
#
# Quick start (full):
#   ACTIVE_PHASE_CD_GROUP=CD_FULL \
#   RUN_PREFIX=phase_cd_full \
#   CUDA_PHASE_C=1 CUDA_PHASE_D=2 CUDA_PHASE_D4=3 \
#   bash scripts/run_phase_cd_control_suite.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACTIVE_PHASE_CD_GROUP="${ACTIVE_PHASE_CD_GROUP:-CD_LIGHT}"
RUN_PREFIX="${RUN_PREFIX:-phase_cd_control}"
FAIL_FAST="${FAIL_FAST:-0}" # 0: continue on failure, 1: stop immediately.
DRY_RUN="${DRY_RUN:-0}"     # 1: print commands and mark stages as skipped.

# GPU assignment per stage (can be changed by env).
CUDA_PHASE_C="${CUDA_PHASE_C:-1}"
CUDA_PHASE_D="${CUDA_PHASE_D:-2}"
CUDA_PHASE_D4="${CUDA_PHASE_D4:-3}"

# Shared batch controls (safe defaults for crowded servers).
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-192}"
C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-192}"
C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-192}"

ANALYSIS_DATASET="${ANALYSIS_DATASET:-strategyqa}"
ANALYSIS_OUT_DIR="${ANALYSIS_OUT_DIR:-assets/artifacts/phase_cd_reports/${RUN_PREFIX}}"

LOG_ROOT="assets/artifacts/phase_cd_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"
STAGE_RESULTS_JSONL="${LOG_ROOT}/stage_results.jsonl"
CURRENT_STAGE="init"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S %z"
}

log_line() {
  local msg="$1"
  echo "[$(timestamp)] $msg"
}

write_failure_summary() {
  local exit_code="$1"
  mkdir -p "$LOG_ROOT"
  cat > "$SUMMARY_FILE" <<EOF
# Phase C/D Control Suite Summary

- generated_at: $(date --iso-8601=seconds)
- group_id: ${ACTIVE_PHASE_CD_GROUP:-N/A}
- run_prefix: ${RUN_PREFIX:-N/A}
- status: failed
- exit_code: ${exit_code}
- failed_stage: ${CURRENT_STAGE:-unknown}
- suite_log_file: ${SUITE_LOG_FILE}
- stage_results_jsonl: ${STAGE_RESULTS_JSONL}
EOF
}

on_exit() {
  local exit_code="$1"
  if [[ "$exit_code" -ne 0 ]]; then
    {
      log_line "Failure stage: ${CURRENT_STAGE}"
      log_line "Exit code: ${exit_code}"
    } | tee -a "$SUITE_LOG_FILE" >/dev/null
    write_failure_summary "$exit_code"
  fi
}

trap 'on_exit $?' EXIT

append_stage_row() {
  local stage="$1"
  local status="$2"
  local rc="$3"
  local group_id="$4"
  local run_prefix="$5"
  local summary_path="$6"
  local notes="$7"
  local command="$8"
  "$PYTHON_BIN" - "$STAGE_RESULTS_JSONL" "$stage" "$status" "$rc" "$group_id" "$run_prefix" "$summary_path" "$notes" "$command" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
row = {
    "stage": sys.argv[2],
    "status": sys.argv[3],
    "exit_code": None if sys.argv[4] in {"", "null", "None"} else int(sys.argv[4]),
    "group_id": sys.argv[5],
    "run_prefix": sys.argv[6],
    "summary_path": sys.argv[7] or None,
    "notes": sys.argv[8] or None,
    "command": sys.argv[9] or None,
}
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

run_stage_command() {
  local stage="$1"
  local group_id="$2"
  local run_prefix="$3"
  local summary_path="$4"
  local command="$5"

  CURRENT_STAGE="$stage"
  {
    log_line "Stage       : $stage"
    log_line "Group       : $group_id"
    log_line "Run prefix  : $run_prefix"
    log_line "Command     : $command"
  } | tee -a "$SUITE_LOG_FILE"

  if [[ "$DRY_RUN" == "1" ]]; then
    append_stage_row "$stage" "skipped" "null" "$group_id" "$run_prefix" "$summary_path" "dry_run" "$command"
    return 0
  fi

  set +e
  /bin/bash -lc "$command" 2>&1 | tee -a "$SUITE_LOG_FILE"
  local rc=${PIPESTATUS[0]}
  set -e

  if [[ "$rc" -eq 0 ]]; then
    append_stage_row "$stage" "ok" "$rc" "$group_id" "$run_prefix" "$summary_path" "" "$command"
    return 0
  fi

  append_stage_row "$stage" "failed" "$rc" "$group_id" "$run_prefix" "$summary_path" "command_failed" "$command"
  if [[ "$FAIL_FAST" == "1" ]]; then
    return "$rc"
  fi
  return 0
}

extract_summary_field() {
  local summary_path="$1"
  local key="$2"
  if [[ ! -f "$summary_path" ]]; then
    return 1
  fi
  grep -E "^- ${key}: " "$summary_path" | tail -n 1 | sed -E "s/^- ${key}: //"
}

render_final_summary() {
  CURRENT_STAGE="final_summary"
  "$PYTHON_BIN" - "$STAGE_RESULTS_JSONL" "$SUMMARY_FILE" "$ACTIVE_PHASE_CD_GROUP" "$RUN_PREFIX" "$SUITE_LOG_FILE" "$ANALYSIS_OUT_DIR" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

rows_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
group_id = sys.argv[3]
run_prefix = sys.argv[4]
suite_log_file = sys.argv[5]
analysis_out_dir = Path(sys.argv[6])

rows = []
if rows_path.exists():
    for raw in rows_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            rows.append(json.loads(raw))

failed = [r for r in rows if r.get("status") == "failed"]
status = "ok" if not failed else "partial_failed"

lines = []
lines.append("# Phase C/D Control Suite Summary")
lines.append("")
lines.append(f"- generated_at: {datetime.now(timezone.utc).isoformat()}")
lines.append(f"- group_id: {group_id}")
lines.append(f"- run_prefix: {run_prefix}")
lines.append(f"- status: {status}")
lines.append(f"- suite_log_file: {suite_log_file}")
lines.append(f"- stage_results_jsonl: {rows_path}")
lines.append(f"- analysis_summary: {analysis_out_dir / 'summary.md'}")
lines.append("")
lines.append("## Stage Results")
lines.append("")
lines.append("| Stage | Status | Exit | Group | Run Prefix | Summary | Notes |")
lines.append("|---|---|---:|---|---|---|---|")
for row in rows:
    lines.append(
        "| {stage} | {status} | {exit_code} | {group_id} | {run_prefix} | {summary_path} | {notes} |".format(
            stage=row.get("stage", ""),
            status=row.get("status", ""),
            exit_code=row.get("exit_code", ""),
            group_id=row.get("group_id", ""),
            run_prefix=row.get("run_prefix", ""),
            summary_path=row.get("summary_path", ""),
            notes=row.get("notes", ""),
        )
    )

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
}

resolve_group() {
  case "$ACTIVE_PHASE_CD_GROUP" in
    CD_LIGHT)
      GROUP_TITLE="Phase C/D Light Control Bundle"
      C_GROUP="C2_STRATEGYQA_CQR_SMOKE"
      D_GROUPS=("D4_STRATEGYQA_SMOKE_3WAY_HQ")
      D4_GROUP="D4ABC_STRATEGYQA_SMOKE"
      FORCE_FULL_SAMPLE="0"
      ;;
    CD_FULL)
      GROUP_TITLE="Phase C/D Full Control Bundle"
      C_GROUP="C2_STRATEGYQA_CQR_FULL"
      # Full bundle includes both legacy and HQ teacher paths for comparability.
      D_GROUPS=("D4_STRATEGYQA_SMOKE_3WAY" "D4_STRATEGYQA_FULL_3WAY_HQ")
      D4_GROUP="D4ABC_STRATEGYQA_FULL"
      FORCE_FULL_SAMPLE="1"
      ;;
    CD_METHOD_FIX_LIGHT)
      GROUP_TITLE="Phase C/D Method-Fix Light Bundle (PRM as Pair Gate)"
      C_GROUP="C2_STRATEGYQA_CQR_SMOKE"
      D_GROUPS=("D5_STRATEGYQA_SMOKE_MC_CTRL" "D5_STRATEGYQA_SMOKE_MC_PRM_PAIR_GATE")
      D4_GROUP=""
      FORCE_FULL_SAMPLE="0"
      ;;
    CD_METHOD_FIX_FULL)
      GROUP_TITLE="Phase C/D Method-Fix Full Bundle (PRM as Pair Gate)"
      C_GROUP="C2_STRATEGYQA_CQR_FULL"
      D_GROUPS=("D5_STRATEGYQA_FULL_MC_CTRL" "D5_STRATEGYQA_FULL_MC_PRM_PAIR_GATE")
      D4_GROUP=""
      FORCE_FULL_SAMPLE="1"
      ;;
    CD_D6_RANKING_LIGHT)
      GROUP_TITLE="Phase C/D D6 Ranking-First Light Bundle"
      C_GROUP="C2_STRATEGYQA_CQR_SMOKE"
      D_GROUPS=("D6_STRATEGYQA_SMOKE_RANKING_CTRL" "D6_STRATEGYQA_SMOKE_RANKING_PRM_GATE")
      D4_GROUP=""
      FORCE_FULL_SAMPLE="0"
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_CD_GROUP=$ACTIVE_PHASE_CD_GROUP" >&2
      echo "Supported groups: CD_LIGHT, CD_FULL, CD_METHOD_FIX_LIGHT, CD_METHOD_FIX_FULL, CD_D6_RANKING_LIGHT" >&2
      exit 1
      ;;
  esac
}

main() {
  mkdir -p "$LOG_ROOT"
  : > "$STAGE_RESULTS_JSONL"
  {
    log_line "Repo root       : $REPO_ROOT"
    log_line "Python          : $PYTHON_BIN"
    log_line "Group           : $ACTIVE_PHASE_CD_GROUP"
    log_line "Run prefix      : $RUN_PREFIX"
    log_line "GPU alloc       : C=$CUDA_PHASE_C D=$CUDA_PHASE_D D4=$CUDA_PHASE_D4"
    log_line "Batch controls  : rollout=$ROLLOUT_BATCH_SIZE c2_train=$C2_TRAIN_BATCH_SIZE c2_eval=$C2_EVAL_BATCH_SIZE"
  } | tee -a "$SUITE_LOG_FILE"

  resolve_group
  log_line "Group title      : $GROUP_TITLE" | tee -a "$SUITE_LOG_FILE"

  # Preflight checks to catch script syntax/import issues early.
  CURRENT_STAGE="preflight"
  /bin/bash -lc "bash -n scripts/run_phase_c_value_suite.sh scripts/run_phase_d_teacher_suite.sh scripts/run_phase_d_external_pair_suite.sh scripts/run_phase_cd_control_suite.sh" 2>&1 | tee -a "$SUITE_LOG_FILE"
  "$PYTHON_BIN" -m py_compile scripts/phase_cd_compare_report.py 2>&1 | tee -a "$SUITE_LOG_FILE"

  # Stage 1: run comparable Phase C baseline.
  local c_run_prefix="${RUN_PREFIX}_c"
  local c_summary="assets/artifacts/phase_c_logs/${c_run_prefix}/final_summary.md"
  local c_cmd="ACTIVE_PHASE_C_GROUP=${C_GROUP} RUN_PREFIX=${c_run_prefix} ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE} C2_TRAIN_BATCH_SIZE=${C2_TRAIN_BATCH_SIZE} C2_EVAL_BATCH_SIZE=${C2_EVAL_BATCH_SIZE} CUDA_VISIBLE_DEVICES=${CUDA_PHASE_C}"
  if [[ "$FORCE_FULL_SAMPLE" == "1" ]]; then
    c_cmd+=" TRAIN_MAX_SAMPLES= EVAL_MAX_SAMPLES="
  fi
  c_cmd+=" bash scripts/run_phase_c_value_suite.sh"
  run_stage_command "c_baseline" "$C_GROUP" "$c_run_prefix" "$c_summary" "$c_cmd"

  # Attempt to resolve C1 dirs from C summary for D4 external-pair chain.
  local phase_c_train_dir=""
  local phase_c_eval_dir=""
  if [[ -f "$c_summary" ]]; then
    phase_c_train_dir="$(extract_summary_field "$c_summary" "c1_train_dir" || true)"
    phase_c_eval_dir="$(extract_summary_field "$c_summary" "c1_eval_dir" || true)"
  fi

  # Stage 2: run one or more D teacher groups.
  local d_group
  for d_group in "${D_GROUPS[@]}"; do
    local d_prefix="${RUN_PREFIX}_${d_group,,}"
    local d_summary="assets/artifacts/phase_d_logs/${d_prefix}/final_summary.md"
    local d_cmd="ACTIVE_PHASE_D_GROUP=${d_group} RUN_PREFIX=${d_prefix} ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE} C2_TRAIN_BATCH_SIZE=${C2_TRAIN_BATCH_SIZE} C2_EVAL_BATCH_SIZE=${C2_EVAL_BATCH_SIZE} CUDA_VISIBLE_DEVICES=${CUDA_PHASE_D}"
    if [[ "$FORCE_FULL_SAMPLE" == "1" ]]; then
      d_cmd+=" TRAIN_MAX_SAMPLES= EVAL_MAX_SAMPLES="
    fi
    d_cmd+=" bash scripts/run_phase_d_teacher_suite.sh"
    run_stage_command "d_teacher_${d_group,,}" "$d_group" "$d_prefix" "$d_summary" "$d_cmd"
  done

  # Stage 3: run D4 external-pair chain, preferring C1 dirs from stage 1.
  local d4_prefix=""
  local d4_summary=""
  if [[ -n "${D4_GROUP:-}" ]]; then
    d4_prefix="${RUN_PREFIX}_${D4_GROUP,,}"
    d4_summary="assets/artifacts/phase_d_logs/${d4_prefix}/final_summary.md"
    if [[ -n "$phase_c_train_dir" && -n "$phase_c_eval_dir" ]]; then
      local d4_cmd="ACTIVE_PHASE_D4_GROUP=${D4_GROUP} RUN_PREFIX=${d4_prefix} PHASE_C_TRAIN_DIR=${phase_c_train_dir} PHASE_C_EVAL_DIR=${phase_c_eval_dir} C2_TRAIN_BATCH_SIZE=${C2_TRAIN_BATCH_SIZE} C2_EVAL_BATCH_SIZE=${C2_EVAL_BATCH_SIZE} CUDA_VISIBLE_DEVICES=${CUDA_PHASE_D4} bash scripts/run_phase_d_external_pair_suite.sh"
      run_stage_command "d_external_${D4_GROUP,,}" "$D4_GROUP" "$d4_prefix" "$d4_summary" "$d4_cmd"
    else
      local skip_note="missing_c1_dirs_from_c_baseline"
      append_stage_row "d_external_${D4_GROUP,,}" "skipped" "null" "$D4_GROUP" "$d4_prefix" "$d4_summary" "$skip_note" ""
      log_line "Skip D4 external stage: $skip_note" | tee -a "$SUITE_LOG_FILE"
    fi
  else
    append_stage_row "d_external" "skipped" "null" "<disabled>" "<disabled>" "<none>" "disabled_for_group" ""
    log_line "Skip D4 external stage: disabled_for_group" | tee -a "$SUITE_LOG_FILE"
  fi

  # Stage 4: always run global C/D diagnosis across historical logs.
  local diag_cmd="$PYTHON_BIN -u scripts/phase_cd_compare_report.py --phase-c-logs-root assets/artifacts/phase_c_logs --phase-d-logs-root assets/artifacts/phase_d_logs --output-dir ${ANALYSIS_OUT_DIR} --dataset ${ANALYSIS_DATASET}"
  run_stage_command "diagnose_all_history" "PHASE_CD_COMPARE" "$RUN_PREFIX" "${ANALYSIS_OUT_DIR}/summary.md" "$diag_cmd"

  render_final_summary
  log_line "Final summary: $SUMMARY_FILE" | tee -a "$SUITE_LOG_FILE"
}

main "$@"
