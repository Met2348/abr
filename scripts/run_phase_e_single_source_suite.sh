#!/usr/bin/env bash
# Phase E single-source learnability bundle.
#
# Why this file exists:
# - The main `run_phase_e_suite.sh` script runs one experiment group at a time.
# - After the Phase E strategy refresh, we now want a one-command way to answer
#   a narrower scientific question:
#     "On one high-quality pair dataset by itself, is the value/ranking head
#      learnable?"
# - This wrapper bundles several single-source groups together and renders a
#   compact comparison summary.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ACTIVE_PHASE_E_SINGLE_GROUP="${ACTIVE_PHASE_E_SINGLE_GROUP:-S1_SINGLE_SOURCE_SMOKE_MATRIX}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_single_source}"

LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"
GROUP_RESULTS_JSONL="${LOG_ROOT}/group_results.jsonl"
CURRENT_STAGE="bootstrap"

GROUP_TITLE=""
GROUP_INTENTION=""
GROUP_OBSERVE=""
GROUP_EXPECT=""
MATRIX_GROUPS=()
OPTIONAL_NOTES=()

log_line() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1"
}

on_exit() {
  local exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    return
  fi
  mkdir -p "$LOG_ROOT"
  {
    echo "# Phase E Single-Source Suite Summary"
    echo
    echo "- group_id: ${ACTIVE_PHASE_E_SINGLE_GROUP}"
    echo "- group_title: ${GROUP_TITLE}"
    echo "- run_prefix: ${RUN_PREFIX}"
    echo "- status: failed"
    echo "- failed_stage: ${CURRENT_STAGE}"
    echo "- suite_log_file: ${SUITE_LOG_FILE}"
  } > "$SUMMARY_FILE"
}
trap on_exit EXIT

resolve_group() {
  case "$ACTIVE_PHASE_E_SINGLE_GROUP" in
    S1_SINGLE_SOURCE_SMOKE_MATRIX)
      GROUP_TITLE="S1 Single-Source Smoke Matrix"
      GROUP_INTENTION="Quickly compare whether currently runnable single high-quality pair datasets can each teach a learnable ranking signal on their own."
      GROUP_OBSERVE="Judge only held-out same-source pair metrics, without cross-benchmark pressure."
      GROUP_EXPECT="At least one single-source family should show clearly positive held-out pair ranking above random."
      MATRIX_GROUPS=(
        E6_MATH_SHEPHERD_SAME_SOURCE_SMOKE
        E8_PRMBENCH_PREVIEW_SAME_SOURCE_SMOKE
      )
      ;;
    S2_SINGLE_SOURCE_SEED3_MATRIX)
      GROUP_TITLE="S2 Single-Source Seed3 Matrix"
      GROUP_INTENTION="Run the first official same-source learnability matrix with 3 seeds for each currently runnable single-source dataset."
      GROUP_OBSERVE="Compare held-out pair accuracy, AUC, ranking score, and seed variance across the single-source families."
      GROUP_EXPECT="The most compatible source should show both positive held-out ranking and low seed variance."
      MATRIX_GROUPS=(
        E7_MATH_SHEPHERD_SAME_SOURCE_SEED3
        E9_PRMBENCH_PREVIEW_SAME_SOURCE_SEED3
      )
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_E_SINGLE_GROUP=$ACTIVE_PHASE_E_SINGLE_GROUP" >&2
      exit 1
      ;;
  esac

  if [[ "${INCLUDE_RPRM:-0}" == "1" ]]; then
    if python - <<'PY' >/dev/null 2>&1
import pyarrow.parquet  # noqa: F401
PY
    then
      if [[ "$ACTIVE_PHASE_E_SINGLE_GROUP" == "S1_SINGLE_SOURCE_SMOKE_MATRIX" ]]; then
        MATRIX_GROUPS+=(E10_RPRM_SAME_SOURCE_SMOKE)
      else
        MATRIX_GROUPS+=(E11_RPRM_SAME_SOURCE_SEED3)
      fi
    else
      OPTIONAL_NOTES+=("R-PRM omitted because pyarrow.parquet is not importable in the current environment.")
    fi
  fi
}

append_group_result() {
  local group_id="$1"
  local summary_path="$2"
  local seed_results_path="$3"
  python - "$group_id" "$summary_path" "$seed_results_path" "$GROUP_RESULTS_JSONL" <<'PY'
import json
import statistics
import sys
from pathlib import Path

group_id = sys.argv[1]
summary_path = Path(sys.argv[2])
seed_results_path = Path(sys.argv[3])
out_path = Path(sys.argv[4])

rows = []
for raw in seed_results_path.read_text(encoding="utf-8").splitlines():
    raw = raw.strip()
    if raw:
        rows.append(json.loads(raw))
if not rows:
    raise ValueError(f"No seed rows found in {seed_results_path}")

def mean(values):
    return float(statistics.mean(values)) if values else None

def std(values):
    return float(statistics.pstdev(values)) if len(values) > 1 else None

row = {
    "group_id": group_id,
    "summary_path": str(summary_path),
    "seed_results_path": str(seed_results_path),
    "mean_heldout_pair_acc": mean([float(item["heldout_pair_acc"]) for item in rows]),
    "mean_heldout_auc": mean([float(item["heldout_auc"]) for item in rows]),
    "mean_heldout_ranking_score": mean([float(item.get("heldout_ranking_score", 0.0)) for item in rows]),
    "std_heldout_pair_acc": std([float(item["heldout_pair_acc"]) for item in rows]),
    "std_heldout_auc": std([float(item["heldout_auc"]) for item in rows]),
    "std_heldout_ranking_score": std([float(item.get("heldout_ranking_score", 0.0)) for item in rows]),
}
with out_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

render_final_summary() {
  python - "$GROUP_RESULTS_JSONL" "$SUMMARY_FILE" "$ACTIVE_PHASE_E_SINGLE_GROUP" "$GROUP_TITLE" "$RUN_PREFIX" "$SUITE_LOG_FILE" "$GROUP_INTENTION" "$GROUP_OBSERVE" "$GROUP_EXPECT" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

rows_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
group_id = sys.argv[3]
group_title = sys.argv[4]
run_prefix = sys.argv[5]
suite_log_file = sys.argv[6]
group_intention = sys.argv[7]
group_observe = sys.argv[8]
group_expect = sys.argv[9]

rows = []
if rows_path.exists():
    for raw in rows_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            rows.append(json.loads(raw))

lines = [
    "# Phase E Single-Source Suite Summary",
    "",
    f"- generated_at: {datetime.now(timezone.utc).isoformat()}",
    f"- group_id: {group_id}",
    f"- group_title: {group_title}",
    f"- run_prefix: {run_prefix}",
    f"- status: {'ok' if rows else 'empty'}",
    f"- suite_log_file: {suite_log_file}",
    f"- group_intention: {group_intention}",
    f"- observe: {group_observe}",
    f"- expect: {group_expect}",
    "",
    "## Group Comparison",
    "",
    "| group_id | mean_pair_acc | mean_auc | mean_ranking_score | std_pair_acc | std_auc | std_ranking_score |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    def fmt(value):
        if value is None:
            return "N/A"
        return f"{float(value):.4f}"
    lines.append(
        "| "
        + " | ".join(
            [
                str(row["group_id"]),
                fmt(row["mean_heldout_pair_acc"]),
                fmt(row["mean_heldout_auc"]),
                fmt(row["mean_heldout_ranking_score"]),
                fmt(row["std_heldout_pair_acc"]),
                fmt(row["std_heldout_auc"]),
                fmt(row["std_heldout_ranking_score"]),
            ]
        )
        + " |"
    )
    lines.append(f"Path: `{row['summary_path']}`")
summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

resolve_group
mkdir -p "$LOG_ROOT"
: > "$SUITE_LOG_FILE"
: > "$GROUP_RESULTS_JSONL"

{
  log_line "Phase E Single-Source Suite"
  log_line "group_id=${ACTIVE_PHASE_E_SINGLE_GROUP}"
  log_line "group_title=${GROUP_TITLE}"
  log_line "group_intention=${GROUP_INTENTION}"
  log_line "group_observe=${GROUP_OBSERVE}"
  log_line "group_expect=${GROUP_EXPECT}"
  log_line "run_prefix=${RUN_PREFIX}"
  log_line "groups=${MATRIX_GROUPS[*]}"
  if [[ ${#OPTIONAL_NOTES[@]} -gt 0 ]]; then
    for note in "${OPTIONAL_NOTES[@]}"; do
      log_line "note=${note}"
    done
  fi
} | tee -a "$SUITE_LOG_FILE"

for group_id in "${MATRIX_GROUPS[@]}"; do
  CURRENT_STAGE="${group_id}"
  sub_prefix="${RUN_PREFIX}_${group_id,,}"
  log_line "Launching sub-suite ${group_id} with RUN_PREFIX=${sub_prefix}" | tee -a "$SUITE_LOG_FILE"
  ACTIVE_PHASE_E_GROUP="$group_id" RUN_PREFIX="$sub_prefix" bash scripts/run_phase_e_suite.sh | tee -a "$SUITE_LOG_FILE"
  sub_summary="assets/artifacts/phase_e_logs/${sub_prefix}/final_summary.md"
  sub_seed_results="assets/artifacts/phase_e_logs/${sub_prefix}/seed_results.jsonl"
  if [[ ! -f "$sub_summary" ]]; then
    echo "ERROR: Missing sub-suite summary: $sub_summary" >&2
    exit 1
  fi
  if [[ ! -f "$sub_seed_results" ]]; then
    echo "ERROR: Missing sub-suite seed results: $sub_seed_results" >&2
    exit 1
  fi
  append_group_result "$group_id" "$sub_summary" "$sub_seed_results"
done

render_final_summary
log_line "Summary file   : ${SUMMARY_FILE}" | tee -a "$SUITE_LOG_FILE"
log_line "Group complete" | tee -a "$SUITE_LOG_FILE"
