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
PHASE_E_AUTO_PICK_GPU="${PHASE_E_AUTO_PICK_GPU:-1}"
PHASE_E_AUTO_SAFE_BATCH="${PHASE_E_AUTO_SAFE_BATCH:-1}"
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
AUTO_PHASE_E_DEFAULT_BATCH_SIZE=""

log_line() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1"
}

select_gpu_if_needed() {
  if [[ "${PHASE_E_AUTO_PICK_GPU}" != "1" ]]; then
    return
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return
  fi
  local best_gpu
  best_gpu="$(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | sort -t',' -k2 -nr \
      | head -n 1 \
      | awk -F',' '{gsub(/ /, "", $1); print $1}'
  )"
  if [[ -n "$best_gpu" ]]; then
    export CUDA_VISIBLE_DEVICES="$best_gpu"
  fi
}

resolve_primary_gpu_index() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    printf '%s\n' "${CUDA_VISIBLE_DEVICES%%,*}"
    return
  fi
  printf '%s\n' ""
}

configure_safe_batch_defaults() {
  if [[ "${PHASE_E_AUTO_SAFE_BATCH}" != "1" ]]; then
    return
  fi
  if [[ -n "${PHASE_E_DEFAULT_BATCH_SIZE:-}" ]]; then
    AUTO_PHASE_E_DEFAULT_BATCH_SIZE="${PHASE_E_DEFAULT_BATCH_SIZE}"
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    AUTO_PHASE_E_DEFAULT_BATCH_SIZE="64"
    return
  fi
  local gpu_index
  gpu_index="$(resolve_primary_gpu_index)"
  local free_mem
  free_mem="$(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | awk -F',' -v gpu="$gpu_index" '{
          idx=$1
          mem=$2
          gsub(/ /, "", idx)
          gsub(/ /, "", mem)
          if (gpu == "" || idx == gpu) {
            print mem
            exit
          }
        }'
  )"
  if [[ -z "$free_mem" ]]; then
    AUTO_PHASE_E_DEFAULT_BATCH_SIZE="64"
  elif (( free_mem >= 70000 )); then
    AUTO_PHASE_E_DEFAULT_BATCH_SIZE="96"
  elif (( free_mem >= 40000 )); then
    AUTO_PHASE_E_DEFAULT_BATCH_SIZE="64"
  elif (( free_mem >= 20000 )); then
    AUTO_PHASE_E_DEFAULT_BATCH_SIZE="32"
  else
    AUTO_PHASE_E_DEFAULT_BATCH_SIZE="16"
  fi
}

log_gpu_snapshot() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log_line "gpu_status=nvidia-smi unavailable"
    return
  fi
  while IFS= read -r line; do
    log_line "gpu_status=${line}"
  done < <(
    nvidia-smi --query-gpu=index,name,memory.free,memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )
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
    I6_MS_ACC95_PUSH_MATRIX)
      GROUP_TITLE="I6 Math-Shepherd ACC95 Push Matrix"
      GROUP_INTENTION="Use the already-strong Math-Shepherd same-source regime as a controlled playground for the final push above 95% held-out ACC under the current code state."
      GROUP_OBSERVE="Compare one full-data verify control against two low-risk single-seed tweaks: logit-space ranking geometry and a more aggressive overfit-oriented recipe."
      GROUP_EXPECT="The verify control should stay above 95%, and at least one tweak may improve on it slightly without introducing instability."
      MATRIX_GROUPS=(
        E67_MS_ACC95_JOINT_VERIFY_SEED42
        E68_MS_ACC95_JOINT_LOGIT_SEED42
        E69_MS_ACC95_JOINT_OVERFIT_SEED42
      )
      ;;
    I7_REPAIRED_SMOKE)
      GROUP_TITLE="I7 Repaired All-Dataset Smoke"
      GROUP_INTENTION="Cheap end-to-end smoke for the current repaired same-source branch across all datasets, including the newly added R-PRM compact and PRM800K phase2-only recipes."
      GROUP_OBSERVE="Run one representative strong recipe per dataset under conservative batch defaults and inspect only held-out pair metrics."
      GROUP_EXPECT="Math-Shepherd and PRMBench should stay strong, while R-PRM compact and PRM800K repairs should at least beat their older baselines before we launch full seed-3 matrices."
      MATRIX_GROUPS=(
        E41_MS_ACC90_MLP_RANK_SEED3
        E46_PRMBENCH_ACC90_MLP_JOINT_SEED3
        E71_RPRM_COMPACT_ACC90_MLP_LOGIT_SEED3
        E74_PRM800K_ACC90_PHASE2_MLP_LOGIT_SEED3
      )
      ;;
    I8_RPRM_COMPACT_REPAIR_MATRIX)
      GROUP_TITLE="I8 R-PRM Compact Repair Matrix"
      GROUP_INTENTION="Systematically test whether compact-verdict R-PRM can become a viable same-source ACC90 candidate once the old starvation and score-space issues are removed."
      GROUP_OBSERVE="Compare repaired linear, repaired MLP logit, and repaired MLP joint recipes only on held-out R-PRM compact pairs."
      GROUP_EXPECT="At least one repaired compact recipe should clearly beat both the historical direct-pair matrix and the early compact baseline."
      MATRIX_GROUPS=(
        E70_RPRM_COMPACT_ACC90_LINEAR_ROBUST_SEED3
        E71_RPRM_COMPACT_ACC90_MLP_LOGIT_SEED3
        E72_RPRM_COMPACT_ACC90_MLP_JOINT_WIDE_SEED3
      )
      ;;
    I9_PRM800K_REPAIR_MATRIX)
      GROUP_TITLE="I9 PRM800K Repair Matrix"
      GROUP_INTENTION="Test whether PRM800K weakness is mainly caused by mixed-file ingestion and underpowered recipe choices rather than by the source itself."
      GROUP_OBSERVE="Compare three phase2-train-only same-source recipes with stricter split hygiene and higher-confidence completion-rating pairs."
      GROUP_EXPECT="If PRM800K is salvageable for same-source fitting, one repaired recipe should show a clean positive delta over the historical control; otherwise the source should be downgraded with more confidence."
      MATRIX_GROUPS=(
        E73_PRM800K_ACC90_PHASE2_LINEAR_LOGIT_SEED3
        E74_PRM800K_ACC90_PHASE2_MLP_LOGIT_SEED3
        E75_PRM800K_ACC90_PHASE2_MLP_JOINT_SEED3
      )
      ;;
    I10_ALL_DATASETS_REPAIRED_MATRIX)
      GROUP_TITLE="I10 All Datasets Repaired Matrix"
      GROUP_INTENTION="Run one repaired same-source matrix that covers every currently supported dataset and summarizes where ACC90 is genuinely achievable under the newest code path."
      GROUP_OBSERVE="Use the already strong Math-Shepherd/PRMBench anchors plus the repaired R-PRM compact and PRM800K phase2-only sweeps."
      GROUP_EXPECT="This matrix should separate datasets that are truly ACC90-capable from those that remain structurally weak even after the current engineering repairs."
      MATRIX_GROUPS=(
        E41_MS_ACC90_MLP_RANK_SEED3
        E42_MS_ACC90_MLP_JOINT_SEED3
        E46_PRMBENCH_ACC90_MLP_JOINT_SEED3
        E70_RPRM_COMPACT_ACC90_LINEAR_ROBUST_SEED3
        E71_RPRM_COMPACT_ACC90_MLP_LOGIT_SEED3
        E72_RPRM_COMPACT_ACC90_MLP_JOINT_WIDE_SEED3
        E73_PRM800K_ACC90_PHASE2_LINEAR_LOGIT_SEED3
        E74_PRM800K_ACC90_PHASE2_MLP_LOGIT_SEED3
        E75_PRM800K_ACC90_PHASE2_MLP_JOINT_SEED3
      )
      ;;
    I11_PRMBENCH_ACC95_PUSH_MATRIX)
      GROUP_TITLE="I11 PRMBench Preview ACC95 Push Matrix"
      GROUP_INTENTION="Use PRMBench Preview's explicit local-error pairs as a narrow same-source target and test a final last-mile push toward 95% held-out ACC."
      GROUP_OBSERVE="Compare one verify control against two PRMBench-specific tweaks: logit-space ranking geometry and a lightly overfit-oriented variant."
      GROUP_EXPECT="If the current head family can reach 95% on PRMBench Preview, one of these single-seed variants should beat the verify control cleanly."
      MATRIX_GROUPS=(
        E76_PRMBENCH_ACC95_JOINT_VERIFY_SEED42
        E77_PRMBENCH_ACC95_JOINT_LOGIT_SEED42
        E78_PRMBENCH_ACC95_JOINT_OVERFIT_SEED42
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
for raw in seed_results_path.read_text(encoding='utf-8').splitlines():
    raw = raw.strip()
    if raw:
        rows.append(json.loads(raw))
if not rows:
    raise ValueError(f'No seed rows found in {seed_results_path}')

def mean(values):
    return float(statistics.mean(values)) if values else None

def std(values):
    return float(statistics.pstdev(values)) if len(values) > 1 else None

row = {
    'group_id': group_id,
    'summary_path': str(summary_path),
    'seed_results_path': str(seed_results_path),
    'mean_heldout_pair_acc': mean([float(item['heldout_pair_acc']) for item in rows]),
    'mean_heldout_auc': mean([float(item['heldout_auc']) for item in rows]),
    'mean_heldout_ranking_score': mean([float(item.get('heldout_ranking_score', 0.0)) for item in rows]),
    'std_heldout_pair_acc': std([float(item['heldout_pair_acc']) for item in rows]),
    'std_heldout_auc': std([float(item['heldout_auc']) for item in rows]),
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


def dataset_family(group_id: str) -> str:
    group_text = str(group_id).upper()
    if "PRM800K" in group_text:
        return "prm800k"
    if "RPRM" in group_text or "R_PRM" in group_text:
        return "r_prm"
    if "PRMBENCH" in group_text:
        return "prmbench_preview"
    if "_MS_" in group_text or "MATH_SHEPHERD" in group_text or "MATH-SHEPHERD" in group_text:
        return "math_shepherd"
    return "other"


def sort_key(row: dict[str, object]) -> tuple[float, float, float]:
    return (
        float(row.get('mean_heldout_pair_acc') or 0.0),
        float(row.get('mean_heldout_auc') or 0.0),
        float(row.get('mean_heldout_ranking_score') or 0.0),
    )

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

if rows:
    lines.extend(
        [
            '',
            '## Best By Dataset',
            '',
            '| dataset | best_group | mean_pair_acc | mean_auc | mean_ranking_score |',
            '|---|---|---:|---:|---:|',
        ]
    )
    best_by_dataset = {}
    for row in rows:
        family = dataset_family(str(row.get('group_id', '')))
        current = best_by_dataset.get(family)
        if current is None or sort_key(row) > sort_key(current):
            best_by_dataset[family] = row
    for family in sorted(best_by_dataset):
        row = best_by_dataset[family]
        lines.append(
            '| ' + ' | '.join(
                [
                    family,
                    str(row['group_id']),
                    fmt(row['mean_heldout_pair_acc']),
                    fmt(row['mean_heldout_auc']),
                    fmt(row['mean_heldout_ranking_score']),
                ]
            ) + ' |'
        )
summary_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
PY
}

resolve_group
select_gpu_if_needed
configure_safe_batch_defaults
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
  log_line "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
  log_line "auto_phase_e_default_batch_size=${AUTO_PHASE_E_DEFAULT_BATCH_SIZE:-<unset>}"
  log_gpu_snapshot
  log_line "groups=${MATRIX_GROUPS[*]}"
} | tee -a "$SUITE_LOG_FILE"

for group_id in "${MATRIX_GROUPS[@]}"; do
  CURRENT_STAGE="$group_id"
  sub_prefix="${RUN_PREFIX}_${group_id,,}"
  log_line "Launching ${group_id} with RUN_PREFIX=${sub_prefix}" | tee -a "$SUITE_LOG_FILE"

  extra_env=()
  if [[ -n "${AUTO_PHASE_E_DEFAULT_BATCH_SIZE:-}" ]]; then
    extra_env+=(
      PHASE_E_DEFAULT_BATCH_SIZE="${AUTO_PHASE_E_DEFAULT_BATCH_SIZE}"
    )
  fi
  if [[ "$ACTIVE_PHASE_E_INTRADATASET_GROUP" == "I1_INTRADATASET_SMOKE" || "$ACTIVE_PHASE_E_INTRADATASET_GROUP" == "I7_REPAIRED_SMOKE" ]]; then
    extra_env+=(
      SEEDS_OVERRIDE=42
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-3000}"
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-12000}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-4}"
    )
  fi

  env "${extra_env[@]}" ACTIVE_PHASE_E_GROUP="$group_id" RUN_PREFIX="$sub_prefix" bash scripts/run_phase_e_suite.sh | tee -a "$SUITE_LOG_FILE"
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
