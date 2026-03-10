#!/usr/bin/env bash
# Phase E intradataset ACC90 suite.
#
# Why this file exists:
# - The current Phase E pivot is no longer about cross-dataset transfer.
# - We need one wrapper that asks a much narrower question:
#   "Can a value head fit one dataset's own held-out pairs above 90% ACC?"
# - Each dataset therefore gets its own recipe matrix, and the suite should
#   automatically select the strongest same-source candidate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ACTIVE_PHASE_E_INTRADATASET_GROUP="${ACTIVE_PHASE_E_INTRADATASET_GROUP:-I1_INTRADATASET_SMOKE}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_intradataset}"
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
    echo "# Phase E Intradataset Suite Summary"
    echo
    echo "- group_id: ${ACTIVE_PHASE_E_INTRADATASET_GROUP}"
    echo "- group_title: ${GROUP_TITLE}"
    echo "- run_prefix: ${RUN_PREFIX}"
    echo "- status: failed"
    echo "- failed_stage: ${CURRENT_STAGE}"
    echo "- suite_log_file: ${SUITE_LOG_FILE}"
  } > "$SUMMARY_FILE"
}
trap on_exit EXIT

resolve_group() {
  case "$ACTIVE_PHASE_E_INTRADATASET_GROUP" in
    I1_INTRADATASET_SMOKE)
      GROUP_TITLE="I1 Intradataset ACC90 Smoke"
      GROUP_INTENTION="Cheap sanity check for the new ACC90-only same-source recipe family."
      GROUP_OBSERVE="Run one representative MLP recipe per high-quality dataset and inspect only held-out pair metrics."
      GROUP_EXPECT="At least one source should show a clearly positive held-out same-source signal before the full matrix is launched."
      MATRIX_GROUPS=(
        E41_MS_ACC90_MLP_RANK_SEED3
        E45_PRMBENCH_ACC90_MLP_RANK_SEED3
        E48_RPRM_ACC90_MLP_RANK_SEED3
      )
      ;;
    I2_MS_ACC90_MATRIX)
      GROUP_TITLE="I2 Math-Shepherd ACC90 Matrix"
      GROUP_INTENTION="Exhaust the main same-source recipes on Math-Shepherd and test whether any of them can clear 90% held-out ACC."
      GROUP_OBSERVE="Compare linear, MLP, joint, and high-confidence-denoised variants only on Math-Shepherd held-out pairs."
      GROUP_EXPECT="One Math-Shepherd-specific recipe should emerge as the strongest same-source checkpoint family, though 90% is intentionally demanding."
      MATRIX_GROUPS=(
        E40_MS_ACC90_LINEAR_ROBUST_SEED3
        E41_MS_ACC90_MLP_RANK_SEED3
        E42_MS_ACC90_MLP_JOINT_SEED3
        E43_MS_ACC90_MLP_HIGHCONF_SEED3
      )
      ;;
    I3_PRMBENCH_ACC90_MATRIX)
      GROUP_TITLE="I3 PRMBench Preview ACC90 Matrix"
      GROUP_INTENTION="Test whether a direct, high-quality process-pair dataset can support 90%+ same-source held-out accuracy."
      GROUP_OBSERVE="Only held-out PRMBench Preview pair accuracy/AUC matter."
      GROUP_EXPECT="PRMBench Preview is the strongest candidate to cross 90% ACC cleanly."
      MATRIX_GROUPS=(
        E44_PRMBENCH_ACC90_LINEAR_SEED3
        E45_PRMBENCH_ACC90_MLP_RANK_SEED3
        E46_PRMBENCH_ACC90_MLP_JOINT_SEED3
      )
      ;;
    I4_RPRM_ACC90_MATRIX)
      GROUP_TITLE="I4 R-PRM ACC90 Matrix"
      GROUP_INTENTION="Test whether direct chosen/rejected supervision can support 90%+ same-source held-out accuracy."
      GROUP_OBSERVE="Only held-out R-PRM pair accuracy/AUC matter."
      GROUP_EXPECT="R-PRM should be one of the strongest direct same-source candidates."
      MATRIX_GROUPS=(
        E47_RPRM_ACC90_LINEAR_SEED3
        E48_RPRM_ACC90_MLP_RANK_SEED3
        E49_RPRM_ACC90_MLP_JOINT_SEED3
      )
      ;;
    I5_ALL_ACC90_MATRIX)
      GROUP_TITLE="I5 All Intradataset ACC90 Matrix"
      GROUP_INTENTION="Run the full high-quality same-source matrix and pick one checkpoint family per dataset under an ACC90 objective."
      GROUP_OBSERVE="Ignore all transfer and benchmark signals; only same-source held-out pair metrics are used for selection."
      GROUP_EXPECT="This should tell us which datasets are actually strong enough to produce RL-candidate value heads under a strict same-source target."
      MATRIX_GROUPS=(
        E40_MS_ACC90_LINEAR_ROBUST_SEED3
        E41_MS_ACC90_MLP_RANK_SEED3
        E42_MS_ACC90_MLP_JOINT_SEED3
        E43_MS_ACC90_MLP_HIGHCONF_SEED3
        E44_PRMBENCH_ACC90_LINEAR_SEED3
        E45_PRMBENCH_ACC90_MLP_RANK_SEED3
        E46_PRMBENCH_ACC90_MLP_JOINT_SEED3
        E47_RPRM_ACC90_LINEAR_SEED3
        E48_RPRM_ACC90_MLP_RANK_SEED3
        E49_RPRM_ACC90_MLP_JOINT_SEED3
      )
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_E_INTRADATASET_GROUP=$ACTIVE_PHASE_E_INTRADATASET_GROUP" >&2
      exit 1
      ;;
  esac
  if [[ -n "${PHASE_E_INTRADATASET_GROUPS_OVERRIDE:-}" ]]; then
    # shellcheck disable=SC2206
    MATRIX_GROUPS=(${PHASE_E_INTRADATASET_GROUPS_OVERRIDE})
  fi
}

append_group_result() {
  local group_id="$1"
  local summary_path="$2"
  python - "$group_id" "$summary_path" "$GROUP_RESULTS_JSONL" <<'PY'
import json
import re
import sys
from pathlib import Path

group_id = sys.argv[1]
summary_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
text = summary_path.read_text(encoding='utf-8')

def grab(name: str):
    m = re.search(rf"- {re.escape(name)}: `([^`]+)`", text)
    if not m:
        return None
    return float(m.group(1))

row = {
    'group_id': group_id,
    'summary_path': str(summary_path),
    'mean_heldout_pair_acc': grab('mean_heldout_pair_acc'),
    'mean_heldout_auc': grab('mean_heldout_auc'),
    'mean_heldout_ranking_score': grab('mean_heldout_ranking_score'),
    'std_heldout_pair_acc': grab('std_heldout_pair_acc'),
    'std_heldout_auc': grab('std_heldout_auc'),
}
with out_path.open('a', encoding='utf-8') as f:
    f.write(json.dumps(row, ensure_ascii=False) + '\n')
PY
}

render_final_summary() {
  python - "$GROUP_RESULTS_JSONL" "$SUMMARY_FILE" "$ACTIVE_PHASE_E_INTRADATASET_GROUP" "$GROUP_TITLE" "$RUN_PREFIX" "$SUITE_LOG_FILE" "$GROUP_INTENTION" "$GROUP_OBSERVE" "$GROUP_EXPECT" <<'PY'
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
    for raw in rows_path.read_text(encoding='utf-8').splitlines():
        raw = raw.strip()
        if raw:
            rows.append(json.loads(raw))

lines = [
    '# Phase E Intradataset Suite Summary',
    '',
    f'- generated_at: {datetime.now(timezone.utc).isoformat()}',
    f'- group_id: {group_id}',
    f'- group_title: {group_title}',
    f'- run_prefix: {run_prefix}',
    f'- status: {"ok" if rows else "empty"}',
    f'- suite_log_file: {suite_log_file}',
    f'- group_intention: {group_intention}',
    f'- observe: {group_observe}',
    f'- expect: {group_expect}',
    f'- candidate_report_json: `assets/artifacts/phase_e_candidates/{run_prefix}_candidate/candidate_report.json`',
    f'- candidate_report_md: `assets/artifacts/phase_e_candidates/{run_prefix}_candidate/candidate_report.md`',
    '',
    '## Group Comparison',
    '',
    '| group_id | mean_pair_acc | mean_auc | mean_ranking_score | std_pair_acc | std_auc |',
    '|---|---:|---:|---:|---:|---:|',
]
for row in rows:
    def fmt(value):
        if value is None:
            return 'N/A'
        return f'{float(value):.4f}'
    lines.append(
        '| ' + ' | '.join([
            str(row['group_id']),
            fmt(row['mean_heldout_pair_acc']),
            fmt(row['mean_heldout_auc']),
            fmt(row['mean_heldout_ranking_score']),
            fmt(row['std_heldout_pair_acc']),
            fmt(row['std_heldout_auc']),
        ]) + ' |'
    )
    lines.append(f"Path: `{row['summary_path']}`")
summary_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
PY
}

resolve_group
mkdir -p "$LOG_ROOT"
: > "$SUITE_LOG_FILE"
: > "$GROUP_RESULTS_JSONL"

{
  log_line "Phase E Intradataset ACC90 Suite"
  log_line "group_id=${ACTIVE_PHASE_E_INTRADATASET_GROUP}"
  log_line "group_title=${GROUP_TITLE}"
  log_line "group_intention=${GROUP_INTENTION}"
  log_line "group_observe=${GROUP_OBSERVE}"
  log_line "group_expect=${GROUP_EXPECT}"
  log_line "run_prefix=${RUN_PREFIX}"
  log_line "groups=${MATRIX_GROUPS[*]}"
} | tee -a "$SUITE_LOG_FILE"

for group_id in "${MATRIX_GROUPS[@]}"; do
  CURRENT_STAGE="$group_id"
  sub_prefix="${RUN_PREFIX}_${group_id,,}"
  log_line "Launching ${group_id} with RUN_PREFIX=${sub_prefix}" | tee -a "$SUITE_LOG_FILE"

  extra_env=()
  if [[ "$ACTIVE_PHASE_E_INTRADATASET_GROUP" == "I1_INTRADATASET_SMOKE" ]]; then
    extra_env+=(
      SEEDS_OVERRIDE=42
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-3000}"
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-3000}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-4}"
      PHASE_E_DEFAULT_BATCH_SIZE="${PHASE_E_DEFAULT_BATCH_SIZE:-128}"
    )
  fi

  env "${extra_env[@]}" ACTIVE_PHASE_E_GROUP="$group_id" RUN_PREFIX="$sub_prefix" bash scripts/run_phase_e_suite.sh | tee -a "$SUITE_LOG_FILE"
  sub_summary="assets/artifacts/phase_e_logs/${sub_prefix}/final_summary.md"
  if [[ ! -f "$sub_summary" ]]; then
    echo "ERROR: Missing sub-suite summary: $sub_summary" >&2
    exit 1
  fi
  append_group_result "$group_id" "$sub_summary"
done

CURRENT_STAGE="select_candidate"
selector_args=(
  python -u scripts/phase_e_select_intradataset_candidate.py
  --run-name "${RUN_PREFIX}_candidate"
  --output-root assets/artifacts/phase_e_candidates
)
for group_id in "${MATRIX_GROUPS[@]}"; do
  selector_args+=(--suite-log-dirs "assets/artifacts/phase_e_logs/${RUN_PREFIX}_${group_id,,}")
done
log_line "RUN: ${selector_args[*]}" | tee -a "$SUITE_LOG_FILE"
"${selector_args[@]}" | tee -a "$SUITE_LOG_FILE"

render_final_summary
log_line "Summary file   : ${SUMMARY_FILE}" | tee -a "$SUITE_LOG_FILE"
log_line "Group complete" | tee -a "$SUITE_LOG_FILE"
