#!/usr/bin/env bash
# Phase E repair-diagnostics suite.
#
# English
# -------
# This wrapper exists to answer a narrower engineering question than the main
# Phase E science suites:
# 1. which recently identified defects have actually been repaired,
# 2. which repairs move the metrics in the expected direction,
# 3. and whether the repaired two-source mix still underperforms the single-source anchors.
#
# 中文
# ----
# 这个 wrapper 的问题比主线 Phase E science suite 更窄，也更工程化：
# 1. 最近识别出的缺陷到底修没修好，
# 2. 每个修复是否真的把指标往预期方向推了，
# 3. 修完之后的两源 mixture 是否仍然输给单源 anchor。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ACTIVE_PHASE_E_REPAIR_GROUP="${ACTIVE_PHASE_E_REPAIR_GROUP:-R1_REPAIR_SMOKE}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_repair_diag}"

LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"
REPORT_JSON="${LOG_ROOT}/repair_report.json"
GROUP_RESULTS_JSONL="${LOG_ROOT}/group_results.jsonl"
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
    echo "# Phase E Repair Diagnostics Summary"
    echo
    echo "- group_id: ${ACTIVE_PHASE_E_REPAIR_GROUP}"
    echo "- group_title: ${GROUP_TITLE}"
    echo "- run_prefix: ${RUN_PREFIX}"
    echo "- status: failed"
    echo "- failed_stage: ${CURRENT_STAGE}"
    echo "- suite_log_file: ${SUITE_LOG_FILE}"
  } > "$SUMMARY_FILE"
}
trap on_exit EXIT

resolve_group() {
  case "$ACTIVE_PHASE_E_REPAIR_GROUP" in
    R1_REPAIR_SMOKE)
      GROUP_TITLE="R1 Phase E Repair Smoke"
      GROUP_INTENTION="Cheap one-seed smoke that checks whether the repair plumbing is wired correctly and whether the repaired recipes move in the expected direction."
      GROUP_OBSERVE="Compare single-source anchors, the old Stage B baseline, and several repaired Stage B variants under a reduced budget."
      GROUP_EXPECT="The repaired variants should at least avoid obvious collapse relative to the legacy Stage B baseline."
      MATRIX_GROUPS=(
        E20_STAGEA_MS_ANCHOR_SEED3
        E21_STAGEA_RPRM_ANCHOR_SEED3
        E24_STAGEB_MS_RPRM_MIX_SEED3
        E60_STAGEB_MS_RPRM_LOGITONLY_SEED3
        E61_STAGEB_MS_RPRM_LOGIT_CONFWT_SEED3
        E62_STAGEB_MS_RPRM_LOGIT_CONFWT_SPLIT_SEED3
        E63_STAGEB_MS_RPRM_REPAIRED_LINEAR_SEED3
        E64_STAGEB_MS_RPRM_REPAIRED_MLP_SEED3
      )
      SUITE_SEEDS_OVERRIDE="${SUITE_SEEDS_OVERRIDE:-42}"
      SUITE_MAX_PAIRS_TOTAL="${SUITE_MAX_PAIRS_TOTAL:-2500}"
      SUITE_MAX_PAIRS_PER_SOURCE="${SUITE_MAX_PAIRS_PER_SOURCE:-2500}"
      SUITE_BENCH_MAX_SAMPLES="${SUITE_BENCH_MAX_SAMPLES:-128}"
      SUITE_PHASE_E_DEFAULT_BATCH_SIZE="${SUITE_PHASE_E_DEFAULT_BATCH_SIZE:-128}"
      ;;
    R2_REPAIR_SEED3_MATRIX)
      GROUP_TITLE="R2 Phase E Repair Seed3 Matrix"
      GROUP_INTENTION="Run the official repair matrix that isolates objective, weighting, split hygiene, step-budget, and head-capacity changes."
      GROUP_OBSERVE="This suite is the main post-fix diagnosis surface for the Math-Shepherd + R-PRM mixture."
      GROUP_EXPECT="If the old failure was mostly engineering-induced, the repaired Stage B recipes should beat the legacy baseline and narrow the gap to the anchors."
      MATRIX_GROUPS=(
        E20_STAGEA_MS_ANCHOR_SEED3
        E21_STAGEA_RPRM_ANCHOR_SEED3
        E24_STAGEB_MS_RPRM_MIX_SEED3
        E60_STAGEB_MS_RPRM_LOGITONLY_SEED3
        E61_STAGEB_MS_RPRM_LOGIT_CONFWT_SEED3
        E62_STAGEB_MS_RPRM_LOGIT_CONFWT_SPLIT_SEED3
        E63_STAGEB_MS_RPRM_REPAIRED_LINEAR_SEED3
        E64_STAGEB_MS_RPRM_REPAIRED_MLP_SEED3
      )
      SUITE_SEEDS_OVERRIDE="${SUITE_SEEDS_OVERRIDE:-}"
      SUITE_MAX_PAIRS_TOTAL="${SUITE_MAX_PAIRS_TOTAL:-}"
      SUITE_MAX_PAIRS_PER_SOURCE="${SUITE_MAX_PAIRS_PER_SOURCE:-}"
      SUITE_BENCH_MAX_SAMPLES="${SUITE_BENCH_MAX_SAMPLES:-}"
      SUITE_PHASE_E_DEFAULT_BATCH_SIZE="${SUITE_PHASE_E_DEFAULT_BATCH_SIZE:-}"
      ;;
    R3_RPRM_DATAFIX_SMOKE)
      GROUP_TITLE="R3 R-PRM Data-Fix Smoke"
      GROUP_INTENTION="Cheap smoke that verifies the repaired compact-verdict R-PRM path and compares it against the strongest non-R-PRM anchor."
      GROUP_OBSERVE="Run only executable fixed groups by default, so the official smoke is not blocked by the known-bad legacy truncation contract."
      GROUP_EXPECT="If the data-contract repair is real, the compact R-PRM groups should run cleanly and the repaired Stage B mixes should produce non-collapsed summaries."
      MATRIX_GROUPS=(
        E20_STAGEA_MS_ANCHOR_SEED3
        E12_RPRM_COMPACT_VERDICT_SEED3
        E65_STAGEB_MS_RPRM_COMPACT_REPAIRED_LINEAR_SEED3
        E66_STAGEB_MS_RPRM_COMPACT_REPAIRED_MLP_SEED3
      )
      SUITE_SEEDS_OVERRIDE="${SUITE_SEEDS_OVERRIDE:-42}"
      SUITE_MAX_PAIRS_TOTAL="${SUITE_MAX_PAIRS_TOTAL:-2500}"
      SUITE_MAX_PAIRS_PER_SOURCE="${SUITE_MAX_PAIRS_PER_SOURCE:-2500}"
      SUITE_BENCH_MAX_SAMPLES="${SUITE_BENCH_MAX_SAMPLES:-128}"
      SUITE_PHASE_E_DEFAULT_BATCH_SIZE="${SUITE_PHASE_E_DEFAULT_BATCH_SIZE:-128}"
      ;;
    R4_RPRM_DATAFIX_SEED3)
      GROUP_TITLE="R4 R-PRM Data-Fix Seed3 Matrix"
      GROUP_INTENTION="Official seed-3 diagnosis of whether compact-verdict R-PRM is now usable in same-source and mixed-source Phase E runs."
      GROUP_OBSERVE="This official matrix excludes the legacy long-analysis groups because they fail the truncation gate by design and would otherwise abort the suite before the fixed groups run."
      GROUP_EXPECT="If the old R-PRM failure was mostly contract/truncation driven, the compact groups should now run end-to-end and produce stable non-random held-out metrics."
      MATRIX_GROUPS=(
        E20_STAGEA_MS_ANCHOR_SEED3
        E12_RPRM_COMPACT_VERDICT_SEED3
        E65_STAGEB_MS_RPRM_COMPACT_REPAIRED_LINEAR_SEED3
        E66_STAGEB_MS_RPRM_COMPACT_REPAIRED_MLP_SEED3
      )
      SUITE_SEEDS_OVERRIDE="${SUITE_SEEDS_OVERRIDE:-}"
      SUITE_MAX_PAIRS_TOTAL="${SUITE_MAX_PAIRS_TOTAL:-}"
      SUITE_MAX_PAIRS_PER_SOURCE="${SUITE_MAX_PAIRS_PER_SOURCE:-}"
      SUITE_BENCH_MAX_SAMPLES="${SUITE_BENCH_MAX_SAMPLES:-}"
      SUITE_PHASE_E_DEFAULT_BATCH_SIZE="${SUITE_PHASE_E_DEFAULT_BATCH_SIZE:-}"
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_E_REPAIR_GROUP=$ACTIVE_PHASE_E_REPAIR_GROUP" >&2
      exit 1
      ;;
  esac
  if [[ -n "${PHASE_E_REPAIR_GROUPS_OVERRIDE:-}" ]]; then
    # shellcheck disable=SC2206
    MATRIX_GROUPS=(${PHASE_E_REPAIR_GROUPS_OVERRIDE})
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

bench_ids = sorted({bench_id for row in rows for bench_id in row.get("benchmarks", {})})

def mean(values):
    return float(statistics.mean(values)) if values else 0.0

def std(values):
    return float(statistics.pstdev(values)) if len(values) > 1 else None

bench_means = {}
for bench_id in bench_ids:
    pair_values = [float(row.get("benchmarks", {}).get(bench_id, {}).get("pair_acc", 0.0)) for row in rows]
    auc_values = [float(row.get("benchmarks", {}).get(bench_id, {}).get("auc", 0.0)) for row in rows]
    bench_means[bench_id] = {
        "mean_pair_acc": mean(pair_values),
        "mean_auc": mean(auc_values),
        "std_pair_acc": std(pair_values),
        "std_auc": std(auc_values),
    }

first = rows[0]
train_cfg = dict(first.get("train_config", {}) or {})
row = {
    "group_id": group_id,
    "summary_path": str(summary_path),
    "seed_results_path": str(seed_results_path),
    "num_seeds": int(len(rows)),
    "mean_heldout_pair_acc": mean([float(item["heldout_pair_acc"]) for item in rows]),
    "mean_heldout_auc": mean([float(item["heldout_auc"]) for item in rows]),
    "mean_heldout_ranking_score": mean([float(item["heldout_ranking_score"]) for item in rows]),
    "std_heldout_pair_acc": std([float(item["heldout_pair_acc"]) for item in rows]),
    "std_heldout_auc": std([float(item["heldout_auc"]) for item in rows]),
    "pair_split_granularity": str(first.get("pair_split_granularity", "pair_id")),
    "train_config": {
        "objective_mode": str(train_cfg.get("objective_mode", "")),
        "ranking_target_space": str(train_cfg.get("ranking_target_space", "score")),
        "pair_weight_mode": str(train_cfg.get("pair_weight_mode", "")),
        "source_balance": str(train_cfg.get("source_balance", "")),
        "checkpoint_selection_metric": str(train_cfg.get("checkpoint_selection_metric", "")),
        "learning_rate": float(train_cfg.get("learning_rate", 0.0) or 0.0),
        "num_train_epochs": int(train_cfg.get("num_train_epochs", 0) or 0),
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 0) or 0),
        "head_architecture": str(train_cfg.get("head_architecture", "")),
    },
    "benchmark_means": bench_means,
}
with out_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

render_final_summary() {
  python - "$GROUP_RESULTS_JSONL" "$SUMMARY_FILE" "$REPORT_JSON" "$ACTIVE_PHASE_E_REPAIR_GROUP" "$GROUP_TITLE" "$RUN_PREFIX" "$SUITE_LOG_FILE" "$GROUP_INTENTION" "$GROUP_OBSERVE" "$GROUP_EXPECT" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

rows_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
report_json = Path(sys.argv[3])
group_id = sys.argv[4]
group_title = sys.argv[5]
run_prefix = sys.argv[6]
suite_log_file = sys.argv[7]
group_intention = sys.argv[8]
group_observe = sys.argv[9]
group_expect = sys.argv[10]

rows = []
if rows_path.exists():
    for raw in rows_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            rows.append(json.loads(raw))

baseline = next((row for row in rows if row["group_id"] == "E24_STAGEB_MS_RPRM_MIX_SEED3"), None)
anchors = {
    row["group_id"]: row
    for row in rows
    if row["group_id"] in {"E20_STAGEA_MS_ANCHOR_SEED3", "E21_STAGEA_RPRM_ANCHOR_SEED3"}
}
repaired_rows = [row for row in rows if row["group_id"] in {
    "E60_STAGEB_MS_RPRM_LOGITONLY_SEED3",
    "E61_STAGEB_MS_RPRM_LOGIT_CONFWT_SEED3",
    "E62_STAGEB_MS_RPRM_LOGIT_CONFWT_SPLIT_SEED3",
    "E63_STAGEB_MS_RPRM_REPAIRED_LINEAR_SEED3",
    "E64_STAGEB_MS_RPRM_REPAIRED_MLP_SEED3",
    "E65_STAGEB_MS_RPRM_COMPACT_REPAIRED_LINEAR_SEED3",
    "E66_STAGEB_MS_RPRM_COMPACT_REPAIRED_MLP_SEED3",
}]

def bench_auc(row, bench_id):
    return float(row.get("benchmark_means", {}).get(bench_id, {}).get("mean_auc", 0.0))

def fmt(value):
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"

best_repaired = None
if repaired_rows:
    def repaired_score(row):
        return float(
            0.45 * float(row.get("mean_heldout_ranking_score", 0.0))
            + 0.20 * bench_auc(row, "prmbench_preview")
            + 0.20 * bench_auc(row, "processbench_math")
            + 0.15 * bench_auc(row, "processbench_gsm8k")
        )
    best_repaired = max(repaired_rows, key=repaired_score)

payload = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "group_id": group_id,
    "group_title": group_title,
    "run_prefix": run_prefix,
    "rows": rows,
    "baseline_group_id": baseline["group_id"] if baseline else None,
    "best_repaired_group_id": best_repaired["group_id"] if best_repaired else None,
}
report_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

lines = [
    "# Phase E Repair Diagnostics Summary",
    "",
    f"- generated_at: {payload['generated_at']}",
    f"- group_id: {group_id}",
    f"- group_title: {group_title}",
    f"- run_prefix: {run_prefix}",
    f"- status: {'ok' if rows else 'empty'}",
    f"- suite_log_file: {suite_log_file}",
    f"- group_intention: {group_intention}",
    f"- observe: {group_observe}",
    f"- expect: {group_expect}",
    f"- report_json: `{report_json}`",
    "",
    "## Group Comparison",
    "",
    "| group_id | seeds | rank_space | pair_weight | split | epochs | batch | head | held_pair | held_auc | held_rank | pb_gsm_auc | pb_math_auc | prmbench_auc |",
    "|---|---:|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    train_cfg = dict(row.get("train_config", {}) or {})
    lines.append(
        "| "
        + " | ".join(
            [
                str(row["group_id"]),
                str(row.get("num_seeds", 0)),
                str(train_cfg.get("ranking_target_space", "")),
                str(train_cfg.get("pair_weight_mode", "")),
                str(row.get("pair_split_granularity", "")),
                str(train_cfg.get("num_train_epochs", 0)),
                str(train_cfg.get("per_device_train_batch_size", 0)),
                str(train_cfg.get("head_architecture", "")),
                fmt(row.get("mean_heldout_pair_acc")),
                fmt(row.get("mean_heldout_auc")),
                fmt(row.get("mean_heldout_ranking_score")),
                fmt(bench_auc(row, "processbench_gsm8k")),
                fmt(bench_auc(row, "processbench_math")),
                fmt(bench_auc(row, "prmbench_preview")),
            ]
        )
        + " |"
    )
    lines.append(f"Path: `{row['summary_path']}`")

if baseline and repaired_rows:
    lines.extend(
        [
            "",
            "## Repair Deltas vs Legacy Stage B Baseline",
            "",
            "| group_id | d_held_pair | d_held_auc | d_held_rank | d_pb_math_auc | d_prmbench_auc |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in repaired_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["group_id"]),
                    fmt(float(row["mean_heldout_pair_acc"]) - float(baseline["mean_heldout_pair_acc"])),
                    fmt(float(row["mean_heldout_auc"]) - float(baseline["mean_heldout_auc"])),
                    fmt(float(row["mean_heldout_ranking_score"]) - float(baseline["mean_heldout_ranking_score"])),
                    fmt(bench_auc(row, "processbench_math") - bench_auc(baseline, "processbench_math")),
                    fmt(bench_auc(row, "prmbench_preview") - bench_auc(baseline, "prmbench_preview")),
                ]
            )
            + " |"
        )

if best_repaired is not None:
    lines.extend(
        [
            "",
            "## Best Repaired Mix",
            "",
            f"- group_id: `{best_repaired['group_id']}`",
            f"- mean_heldout_pair_acc: `{float(best_repaired['mean_heldout_pair_acc']):.6f}`",
            f"- mean_heldout_auc: `{float(best_repaired['mean_heldout_auc']):.6f}`",
            f"- mean_processbench_math_auc: `{bench_auc(best_repaired, 'processbench_math'):.6f}`",
            f"- mean_prmbench_preview_auc: `{bench_auc(best_repaired, 'prmbench_preview'):.6f}`",
        ]
    )
    if "E20_STAGEA_MS_ANCHOR_SEED3" in anchors:
        lines.append(
            f"- delta_vs_E20_heldout_pair_acc: `{float(best_repaired['mean_heldout_pair_acc']) - float(anchors['E20_STAGEA_MS_ANCHOR_SEED3']['mean_heldout_pair_acc']):.6f}`"
        )
        lines.append(
            f"- delta_vs_E20_prmbench_auc: `{bench_auc(best_repaired, 'prmbench_preview') - bench_auc(anchors['E20_STAGEA_MS_ANCHOR_SEED3'], 'prmbench_preview'):.6f}`"
        )
    if "E21_STAGEA_RPRM_ANCHOR_SEED3" in anchors:
        lines.append(
            f"- delta_vs_E21_heldout_pair_acc: `{float(best_repaired['mean_heldout_pair_acc']) - float(anchors['E21_STAGEA_RPRM_ANCHOR_SEED3']['mean_heldout_pair_acc']):.6f}`"
        )
        lines.append(
            f"- delta_vs_E21_prmbench_auc: `{bench_auc(best_repaired, 'prmbench_preview') - bench_auc(anchors['E21_STAGEA_RPRM_ANCHOR_SEED3'], 'prmbench_preview'):.6f}`"
        )

lines.append("")
summary_path.write_text("\n".join(lines), encoding="utf-8")
PY
}

resolve_group
mkdir -p "$LOG_ROOT"
: > "$SUITE_LOG_FILE"
: > "$GROUP_RESULTS_JSONL"

{
  log_line "Phase E Repair Diagnostics Suite"
  log_line "group_id=${ACTIVE_PHASE_E_REPAIR_GROUP}"
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
  if [[ -n "${SUITE_PHASE_E_DEFAULT_BATCH_SIZE:-}" ]]; then
    env_cmd+=(PHASE_E_DEFAULT_BATCH_SIZE="$SUITE_PHASE_E_DEFAULT_BATCH_SIZE")
  fi

  "${env_cmd[@]}" bash scripts/run_phase_e_suite.sh | tee -a "$SUITE_LOG_FILE"

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
log_line "Report json    : ${REPORT_JSON}" | tee -a "$SUITE_LOG_FILE"
log_line "Group complete" | tee -a "$SUITE_LOG_FILE"
