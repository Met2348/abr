#!/usr/bin/env bash
# Phase E Math-Shepherd trust bundle.
#
# Why this file exists:
# - `run_phase_e_suite.sh` runs exactly one Phase E group.
# - For later RL-heavy stages, we need a higher-level bundle that asks:
#     "Which Math-Shepherd recipe gives the most trustworthy value head,
#      not merely the highest single-run score?"
# - This wrapper launches a matrix of Math-Shepherd groups, then calls a
#   dedicated candidate selector to recommend the best `best_value_head.pt`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ACTIVE_PHASE_E_MS_GROUP="${ACTIVE_PHASE_E_MS_GROUP:-MS1_MATH_SHEPHERD_TRUST_SMOKE}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_math_trust}"

LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"
CURRENT_STAGE="bootstrap"

GROUP_TITLE=""
GROUP_INTENTION=""
GROUP_OBSERVE=""
GROUP_EXPECT=""
MATRIX_GROUPS=()

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
    echo "# Phase E Math-Shepherd Trust Suite Summary"
    echo
    echo "- group_id: ${ACTIVE_PHASE_E_MS_GROUP}"
    echo "- group_title: ${GROUP_TITLE}"
    echo "- run_prefix: ${RUN_PREFIX}"
    echo "- status: failed"
    echo "- failed_stage: ${CURRENT_STAGE}"
    echo "- suite_log_file: ${SUITE_LOG_FILE}"
  } > "$SUMMARY_FILE"
}
trap on_exit EXIT

resolve_group() {
  case "$ACTIVE_PHASE_E_MS_GROUP" in
    MS1_MATH_SHEPHERD_TRUST_SMOKE)
      GROUP_TITLE="MS1 Math-Shepherd Trust Smoke"
      GROUP_INTENTION="Quickly compare several Math-Shepherd training recipes before committing overnight GPU time."
      GROUP_OBSERVE="Use one seed, reduced pair counts, and smaller benchmark slices to surface obvious instability or configuration mistakes."
      GROUP_EXPECT="At least one recipe should stay positive on held-out Math-Shepherd pairs and avoid obvious benchmark collapse."
      MATRIX_GROUPS=(
        E6_MATH_SHEPHERD_SAME_SOURCE_SMOKE
        E1_MATH_SHEPHERD_PAIR_LEARN_SMOKE
        E12_MATH_SHEPHERD_TRUST_LOWLR_SEED3
        E13_MATH_SHEPHERD_TRUST_UNWEIGHTED_SEED3
        E14_MATH_SHEPHERD_TRUST_ANTISAT_SEED3
        E15_MATH_SHEPHERD_TRUST_ROBUST_SEED3
      )
      SUITE_SEEDS_OVERRIDE="${SUITE_SEEDS_OVERRIDE:-42}"
      SUITE_MAX_PAIRS_TOTAL="${SUITE_MAX_PAIRS_TOTAL:-2000}"
      SUITE_MAX_PAIRS_PER_SOURCE="${SUITE_MAX_PAIRS_PER_SOURCE:-2000}"
      SUITE_BENCH_MAX_SAMPLES="${SUITE_BENCH_MAX_SAMPLES:-128}"
      SUITE_TRAIN_BATCH_SIZE="${SUITE_TRAIN_BATCH_SIZE:-192}"
      SUITE_EVAL_BATCH_SIZE="${SUITE_EVAL_BATCH_SIZE:-192}"
      ;;
    MS2_MATH_SHEPHERD_TRUST_SEED3_MATRIX)
      GROUP_TITLE="MS2 Math-Shepherd Trust Seed3 Matrix"
      GROUP_INTENTION="Run the official Math-Shepherd trust matrix and pick the strongest checkpoint family for later RL-facing stages."
      GROUP_OBSERVE="Compare same-source learnability, benchmark-native behavior, and seed stability across several ranking-first recipes."
      GROUP_EXPECT="One conservative recipe should emerge as the best compromise between held-out strength, benchmark behavior, and low seed fragility."
      MATRIX_GROUPS=(
        E7_MATH_SHEPHERD_SAME_SOURCE_SEED3
        E2_MATH_SHEPHERD_PAIR_LEARN_SEED3
        E12_MATH_SHEPHERD_TRUST_LOWLR_SEED3
        E13_MATH_SHEPHERD_TRUST_UNWEIGHTED_SEED3
        E14_MATH_SHEPHERD_TRUST_ANTISAT_SEED3
        E15_MATH_SHEPHERD_TRUST_ROBUST_SEED3
      )
      SUITE_SEEDS_OVERRIDE="${SUITE_SEEDS_OVERRIDE:-}"
      SUITE_MAX_PAIRS_TOTAL="${SUITE_MAX_PAIRS_TOTAL:-}"
      SUITE_MAX_PAIRS_PER_SOURCE="${SUITE_MAX_PAIRS_PER_SOURCE:-}"
      SUITE_BENCH_MAX_SAMPLES="${SUITE_BENCH_MAX_SAMPLES:-}"
      SUITE_TRAIN_BATCH_SIZE="${SUITE_TRAIN_BATCH_SIZE:-}"
      SUITE_EVAL_BATCH_SIZE="${SUITE_EVAL_BATCH_SIZE:-}"
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_E_MS_GROUP=$ACTIVE_PHASE_E_MS_GROUP" >&2
      exit 1
      ;;
  esac

  if [[ -n "${PHASE_E_MS_GROUPS_OVERRIDE:-}" ]]; then
    # shellcheck disable=SC2206
    MATRIX_GROUPS=(${PHASE_E_MS_GROUPS_OVERRIDE})
  fi
}

render_final_summary() {
  python - "$SUMMARY_FILE" "$ACTIVE_PHASE_E_MS_GROUP" "$GROUP_TITLE" "$RUN_PREFIX" "$SUITE_LOG_FILE" "$GROUP_INTENTION" "$GROUP_OBSERVE" "$GROUP_EXPECT" "$CANDIDATE_REPORT_JSON" "$CANDIDATE_REPORT_MD" "$SUBSUITES_JSON" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path = Path(sys.argv[1])
group_id = sys.argv[2]
group_title = sys.argv[3]
run_prefix = sys.argv[4]
suite_log_file = sys.argv[5]
group_intention = sys.argv[6]
group_observe = sys.argv[7]
group_expect = sys.argv[8]
candidate_json = Path(sys.argv[9])
candidate_md = Path(sys.argv[10])
sub_json = Path(sys.argv[11])

candidate = json.loads(candidate_json.read_text(encoding="utf-8")) if candidate_json.exists() else {}
sub_suites = json.loads(sub_json.read_text(encoding="utf-8")) if sub_json.exists() else []
selected = candidate.get("selected_group")
selected_mode = candidate.get("selected_mode", "none")

lines = [
    "# Phase E Math-Shepherd Trust Suite Summary",
    "",
    f"- generated_at: {datetime.now(timezone.utc).isoformat()}",
    f"- group_id: {group_id}",
    f"- group_title: {group_title}",
    f"- run_prefix: {run_prefix}",
    f"- status: {'ok' if candidate_json.exists() else 'empty'}",
    f"- suite_log_file: {suite_log_file}",
    f"- group_intention: {group_intention}",
    f"- observe: {group_observe}",
    f"- expect: {group_expect}",
    f"- candidate_report_json: `{candidate_json}`",
    f"- candidate_report_md: `{candidate_md}`",
    "",
    "## Sub-Suites",
    "",
]
for item in sub_suites:
    lines.append(f"- `{item['group_id']}` -> `{item['summary_path']}`")

if selected:
    lines.extend(
        [
            "",
            "## Recommended Checkpoint",
            "",
            f"- selected_mode: `{selected_mode}`",
            f"- group_id: `{selected['group_id']}`",
            f"- best_seed: `{selected['best_seed']}`",
            f"- best_checkpoint_path: `{selected['best_checkpoint_path']}`",
            f"- trust_score: `{float(selected.get('trust_score') or 0.0):.6f}`",
            "",
        ]
    )

groups = candidate.get("groups", [])
if groups:
    lines.extend(
        [
            "## Group Comparison",
            "",
            "| group_id | gate_pass | mean_hold_pair | mean_hold_auc | mean_rank | pb_gsm_auc | pb_math_auc | std_hold_pair | std_hold_auc | trust_score |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in groups:
        pb_gsm = dict(group.get("benchmark_means", {})).get("processbench_gsm8k", {})
        pb_math = dict(group.get("benchmark_means", {})).get("processbench_math", {})
        def fmt(value):
            if value is None:
                return "N/A"
            return f"{float(value):.4f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(group["group_id"]),
                    "1" if bool(group.get("gate_pass")) else "0",
                    fmt(group.get("mean_heldout_pair_acc")),
                    fmt(group.get("mean_heldout_auc")),
                    fmt(group.get("mean_heldout_ranking_score")),
                    fmt(pb_gsm.get("auc")),
                    fmt(pb_math.get("auc")),
                    fmt(group.get("std_heldout_pair_acc")),
                    fmt(group.get("std_heldout_auc")),
                    fmt(group.get("trust_score")),
                ]
            )
            + " |"
        )

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

resolve_group
mkdir -p "$LOG_ROOT"
: > "$SUITE_LOG_FILE"
SUBSUITES_JSON="${LOG_ROOT}/sub_suites.json"
: > "$SUBSUITES_JSON"

{
  log_line "Phase E Math-Shepherd Trust Suite"
  log_line "group_id=${ACTIVE_PHASE_E_MS_GROUP}"
  log_line "group_title=${GROUP_TITLE}"
  log_line "group_intention=${GROUP_INTENTION}"
  log_line "group_observe=${GROUP_OBSERVE}"
  log_line "group_expect=${GROUP_EXPECT}"
  log_line "run_prefix=${RUN_PREFIX}"
  log_line "groups=${MATRIX_GROUPS[*]}"
} | tee -a "$SUITE_LOG_FILE"

for group_id in "${MATRIX_GROUPS[@]}"; do
  CURRENT_STAGE="${group_id}"
  sub_prefix="${RUN_PREFIX}_${group_id,,}"
  log_line "Launching sub-suite ${group_id} with RUN_PREFIX=${sub_prefix}" | tee -a "$SUITE_LOG_FILE"

  env_cmd=(
    env
    ACTIVE_PHASE_E_GROUP="$group_id"
    RUN_PREFIX="$sub_prefix"
  )
  if [[ -n "${SUITE_SEEDS_OVERRIDE:-}" ]]; then
    env_cmd+=(SEEDS_OVERRIDE="$SUITE_SEEDS_OVERRIDE")
  fi
  if [[ -n "${SUITE_MAX_PAIRS_TOTAL:-}" ]]; then
    env_cmd+=(MAX_PAIRS_TOTAL="$SUITE_MAX_PAIRS_TOTAL")
  fi
  if [[ -n "${SUITE_MAX_PAIRS_PER_SOURCE:-}" ]]; then
    env_cmd+=(MAX_PAIRS_PER_SOURCE="$SUITE_MAX_PAIRS_PER_SOURCE")
  fi
  if [[ -n "${SUITE_BENCH_MAX_SAMPLES:-}" ]]; then
    env_cmd+=(BENCH_MAX_SAMPLES="$SUITE_BENCH_MAX_SAMPLES")
  fi
  if [[ -n "${SUITE_TRAIN_BATCH_SIZE:-}" ]]; then
    env_cmd+=(TRAIN_BATCH_SIZE="$SUITE_TRAIN_BATCH_SIZE")
  fi
  if [[ -n "${SUITE_EVAL_BATCH_SIZE:-}" ]]; then
    env_cmd+=(EVAL_BATCH_SIZE="$SUITE_EVAL_BATCH_SIZE")
  fi
  "${env_cmd[@]}" bash scripts/run_phase_e_suite.sh | tee -a "$SUITE_LOG_FILE"

  sub_summary="assets/artifacts/phase_e_logs/${sub_prefix}/final_summary.md"
  if [[ ! -f "$sub_summary" ]]; then
    echo "ERROR: Missing sub-suite summary: $sub_summary" >&2
    exit 1
  fi
  python - "$group_id" "$sub_summary" "$SUBSUITES_JSON" <<'PY'
import json
import sys
from pathlib import Path

group_id = sys.argv[1]
summary_path = sys.argv[2]
out_path = Path(sys.argv[3])
rows = []
if out_path.exists() and out_path.read_text(encoding="utf-8").strip():
    rows = json.loads(out_path.read_text(encoding="utf-8"))
rows.append({"group_id": group_id, "summary_path": summary_path})
out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
done

CANDIDATE_RUN_NAME="${RUN_PREFIX}_candidate"
CANDIDATE_OUTPUT_ROOT="${CANDIDATE_OUTPUT_ROOT:-assets/artifacts/phase_e_candidates}"
suite_dirs=()
for group_id in "${MATRIX_GROUPS[@]}"; do
  suite_dirs+=("assets/artifacts/phase_e_logs/${RUN_PREFIX}_${group_id,,}")
done

CURRENT_STAGE="candidate_selection"
python -u scripts/phase_e_select_candidate.py \
  --suite-log-dirs "${suite_dirs[@]}" \
  --required-benchmark-ids processbench_gsm8k processbench_math \
  --run-name "$CANDIDATE_RUN_NAME" \
  --output-root "$CANDIDATE_OUTPUT_ROOT" | tee -a "$SUITE_LOG_FILE"

CANDIDATE_REPORT_JSON="${CANDIDATE_OUTPUT_ROOT}/${CANDIDATE_RUN_NAME}/candidate_report.json"
CANDIDATE_REPORT_MD="${CANDIDATE_OUTPUT_ROOT}/${CANDIDATE_RUN_NAME}/candidate_report.md"
render_final_summary
log_line "Summary file   : ${SUMMARY_FILE}" | tee -a "$SUITE_LOG_FILE"
log_line "Group complete" | tee -a "$SUITE_LOG_FILE"
