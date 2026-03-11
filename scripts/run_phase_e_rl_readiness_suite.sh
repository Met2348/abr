#!/usr/bin/env bash
# Phase E RL-readiness audit suite.
#
# 这个脚本存在的原因：
# - 当前仓库已经证明某些 source 上 same-source held-out pair 可以很高，
#   但这还不足以说 value head 已经达到 RL 可用水平。
# - RL 前至少还要补三类更贴近“决策使用”的离线检查：
#   1. same-family prompt-level rerank 是否靠谱，
#   2. rejection / abstention 是否真的能提高可靠性，
#   3. benchmark-native ProcessBench 复评是否至少没有明显塌掉。
# - 因此这里把这些检查固定成 one-click 审计，避免手工拼多条命令后再人工抄结果。
#
# 它负责的内容：
# - 选择当前仓库里最值得审计的已有 checkpoint；
# - 对每个 checkpoint 运行 same-family trust eval；
# - 对每个 checkpoint 运行 ProcessBench GSM8K / Math benchmark eval；
# - 汇总出一个统一 Markdown / JSONL 报告，方便判断“是否接近 RL 可用”。
#
# 控制流：
# - 先按 group 解析要审计的 checkpoint 列表；
# - 再逐个运行 same-family eval 和 benchmark eval；
# - 最后从产物里抽关键指标，做一个保守的启发式 gate 摘要。
#
# 它与其他文件的关系：
# - 复用 `scripts/phase_e_eval_samefamily_trust.py` 计算 rerank / rejection / pressure 指标；
# - 复用 `scripts/phase_e_eval_benchmark.py` 计算 benchmark-native 指标；
# - 自己不训练模型，只消费已有 `assets/artifacts/phase_e_runs/...` checkpoint。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ACTIVE_PHASE_E_RL_GROUP="${ACTIVE_PHASE_E_RL_GROUP:-RR4_COMPARE_CURRENT_TOPS}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_rl_readiness}"
RL_AUDIT_BATCH_SIZE="${RL_AUDIT_BATCH_SIZE:-96}"

LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"
AUDIT_RESULTS_JSONL="${LOG_ROOT}/audit_results.jsonl"
CURRENT_STAGE="bootstrap"

GROUP_TITLE=""
GROUP_INTENTION=""
GROUP_OBSERVE=""
GROUP_EXPECT=""
AUDIT_TARGETS=()

RL_AUDIT_MS_E68_RUN_DIR="${RL_AUDIT_MS_E68_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_ms_acc95_push_0310_2146_e68_ms_acc95_joint_logit_seed42_e68_ms_acc95_joint_logit_seed42_s42_value_20260310T151651Z}"
RL_AUDIT_MS_E14_RUN_DIR="${RL_AUDIT_MS_E14_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_ms_trust_seed3_fix_0310_1659_e14_math_shepherd_trust_antisat_seed3_e14_math_shepherd_trust_antisat_seed3_s43_value_20260310T091353Z}"
RL_AUDIT_PRMBENCH_E46_RUN_DIR="${RL_AUDIT_PRMBENCH_E46_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_prmbench_acc90_full_0310_1914_e46_prmbench_acc90_mlp_joint_seed3_e46_prmbench_acc90_mlp_joint_seed3_s43_value_20260310T113737Z}"
RL_AUDIT_MS_E43_RUN_DIR="${RL_AUDIT_MS_E43_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_all_acc90_0310_1915_e43_ms_acc90_mlp_highconf_seed3_e43_ms_acc90_mlp_highconf_seed3_s43_value_20260310T113946Z}"
RL_AUDIT_PRMBENCH_E78_RUN_DIR="${RL_AUDIT_PRMBENCH_E78_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_prmbench_acc95_push_0310_2359_e78_prmbench_acc95_joint_overfit_seed42_e78_prmbench_acc95_joint_overfit_seed42_s42_value_20260310T153050Z}"
RL_AUDIT_MS_GRID_MICRO_RUN_DIR="${RL_AUDIT_MS_GRID_MICRO_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_msgrid_warm_e68_micro_0311_20260310T163400Z}"
RL_AUDIT_MS_TA_MICRO_RUN_DIR="${RL_AUDIT_MS_TA_MICRO_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_msta_warm_e68_micro_0311_20260310T163102Z}"
RL_AUDIT_PRMBENCH_TA_SMOKE_RUN_DIR="${RL_AUDIT_PRMBENCH_TA_SMOKE_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_prmbench_terminal_anchor_joint_logit_smoke_0311_0025_20260310T155356Z}"

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
    echo "# Phase E RL-Readiness Audit Summary"
    echo
    echo "- group_id: ${ACTIVE_PHASE_E_RL_GROUP}"
    echo "- group_title: ${GROUP_TITLE}"
    echo "- run_prefix: ${RUN_PREFIX}"
    echo "- status: failed"
    echo "- failed_stage: ${CURRENT_STAGE}"
    echo "- suite_log_file: ${SUITE_LOG_FILE}"
  } > "$SUMMARY_FILE"
}
trap on_exit EXIT

resolve_group() {
  case "$ACTIVE_PHASE_E_RL_GROUP" in
    RR1_MS_E68_AUDIT)
      GROUP_TITLE="RR1 Math-Shepherd E68 RL Audit"
      GROUP_INTENTION="Audit the current strongest same-source Math-Shepherd checkpoint under RL-adjacent offline utility checks."
      GROUP_OBSERVE="Look for strong prompt-level rerank utility, useful rejection gains, and whether ProcessBench still collapses."
      GROUP_EXPECT="Same-family utility should be strong, but benchmark-native faithfulness is still likely too weak for RL-ready claims."
      AUDIT_TARGETS=(
        "ms_e68|math_shepherd|${RL_AUDIT_MS_E68_RUN_DIR}"
      )
      ;;
    RR2_MS_E14_AUDIT)
      GROUP_TITLE="RR2 Math-Shepherd E14 Trust Audit"
      GROUP_INTENTION="Audit the benchmark-aware Math-Shepherd trust candidate to see whether benchmark-friendly tuning also improves RL-adjacent utility."
      GROUP_OBSERVE="Compare same-family rerank quality against the more conservative ProcessBench-facing recipe."
      GROUP_EXPECT="This run may trade some same-source peak accuracy for slightly better benchmark behavior, but it is still unlikely to clear a conservative RL gate."
      AUDIT_TARGETS=(
        "ms_e14|math_shepherd|${RL_AUDIT_MS_E14_RUN_DIR}"
      )
      ;;
    RR3_PRMBENCH_E46_AUDIT)
      GROUP_TITLE="RR3 PRMBench E46 RL Audit"
      GROUP_INTENTION="Audit the strongest current PRMBench same-source checkpoint under the same RL-adjacent utility checks."
      GROUP_OBSERVE="Check whether a direct process-pair source gives more decision-useful prompt-level behavior than Math-Shepherd."
      GROUP_EXPECT="Same-family utility should be clearly positive, but benchmark-native robustness may still remain below RL-comfort level."
      AUDIT_TARGETS=(
        "prm_e46|prmbench_preview|${RL_AUDIT_PRMBENCH_E46_RUN_DIR}"
      )
      ;;
    RR4_COMPARE_CURRENT_TOPS)
      GROUP_TITLE="RR4 Compare Current Top Candidates"
      GROUP_INTENTION="Compare the repository's strongest current Math-Shepherd same-source winner, benchmark-aware Math-Shepherd trust candidate, and strongest PRMBench same-source winner under one RL-readiness audit."
      GROUP_OBSERVE="Judge whether any current checkpoint survives both same-family decision pressure and ProcessBench re-evaluation strongly enough to be called RL-usable."
      GROUP_EXPECT="At least one checkpoint should look very good inside its own family, but none should yet survive the full audit cleanly enough for an RL-ready claim."
      AUDIT_TARGETS=(
        "ms_e68|math_shepherd|${RL_AUDIT_MS_E68_RUN_DIR}"
        "ms_e14|math_shepherd|${RL_AUDIT_MS_E14_RUN_DIR}"
        "prm_e46|prmbench_preview|${RL_AUDIT_PRMBENCH_E46_RUN_DIR}"
      )
      ;;
    RR5_COMPARE_INTRADATASET_TOPS)
      GROUP_TITLE="RR5 Compare Intradataset Top Candidates"
      GROUP_INTENTION="Audit the strongest current same-source Math-Shepherd and PRMBench candidates under one shared RL-readiness protocol, using the highest-accuracy intradataset checkpoints rather than earlier bridge-era references."
      GROUP_OBSERVE="Check whether stronger same-source winners become more trustworthy under same-family rerank, rejection, pressure, and ProcessBench re-evaluation."
      GROUP_EXPECT="Math-Shepherd and PRMBench same-source winners should dominate same-family metrics, but benchmark behavior may still separate the truly safer RL prior from the merely overfit same-source classifier."
      AUDIT_TARGETS=(
        "ms_e68|math_shepherd|${RL_AUDIT_MS_E68_RUN_DIR}"
        "ms_e43|math_shepherd|${RL_AUDIT_MS_E43_RUN_DIR}"
        "prm_e46|prmbench_preview|${RL_AUDIT_PRMBENCH_E46_RUN_DIR}"
        "prm_e78|prmbench_preview|${RL_AUDIT_PRMBENCH_E78_RUN_DIR}"
      )
      ;;
    RR6_COMPARE_REPAIR_PILOTS)
      GROUP_TITLE="RR6 Compare ProcessBench Repair Pilots"
      GROUP_INTENTION="Audit whether the recent local-geometry and terminal-anchor repair pilots improve RL-adjacent behavior enough to justify scaling them beyond smoke size."
      GROUP_OBSERVE="Separate local-first-bad improvements from terminal-completion improvements under the same-family and ProcessBench audit lens."
      GROUP_EXPECT="Repair pilots should improve one slice of the audit but will likely remain incomplete candidates; this suite is intended to expose which slice each repair actually fixes."
      AUDIT_TARGETS=(
        "ms_grid_micro|math_shepherd|${RL_AUDIT_MS_GRID_MICRO_RUN_DIR}"
        "ms_ta_micro|math_shepherd|${RL_AUDIT_MS_TA_MICRO_RUN_DIR}"
        "prm_ta_smoke|prmbench_preview|${RL_AUDIT_PRMBENCH_TA_SMOKE_RUN_DIR}"
      )
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_E_RL_GROUP=$ACTIVE_PHASE_E_RL_GROUP" >&2
      exit 1
      ;;
  esac

  if [[ -n "${PHASE_E_RL_TARGETS_OVERRIDE:-}" ]]; then
    # 允许开发时临时覆写审计目标，格式保持 `label|source|run_dir`。
    # Allow temporary target overrides during development; keep the same `label|source|run_dir` format.
    # shellcheck disable=SC2206
    AUDIT_TARGETS=(${PHASE_E_RL_TARGETS_OVERRIDE})
  fi
}

latest_run_dir() {
  local prefix="$1"
  python - "$prefix" <<'PY'
from pathlib import Path
import sys

prefix = sys.argv[1]
matches = sorted(Path(prefix).parent.glob(Path(prefix).name + "_*"))
if not matches:
    raise SystemExit(f"No artifact directory matches prefix: {prefix}")
matches = sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)
print(matches[0])
PY
}

append_audit_result() {
  local audit_id="$1"
  local source_family="$2"
  local value_run_dir="$3"
  local samefamily_run_dir="$4"
  local pb_gsm_run_dir="$5"
  local pb_math_run_dir="$6"
  local output_path="$7"
  python - "$audit_id" "$source_family" "$value_run_dir" "$samefamily_run_dir" "$pb_gsm_run_dir" "$pb_math_run_dir" "$output_path" <<'PY'
import json
import sys
from pathlib import Path

audit_id = sys.argv[1]
source_family = sys.argv[2]
value_run_dir = Path(sys.argv[3])
samefamily_run_dir = Path(sys.argv[4])
pb_gsm_run_dir = Path(sys.argv[5])
pb_math_run_dir = Path(sys.argv[6])
output_path = Path(sys.argv[7])

samefamily_metrics = json.loads((samefamily_run_dir / "metrics.json").read_text(encoding="utf-8"))
pb_gsm_summary = json.loads((pb_gsm_run_dir / "summary.json").read_text(encoding="utf-8"))
pb_math_summary = json.loads((pb_math_run_dir / "summary.json").read_text(encoding="utf-8"))
prompt_rows_path = samefamily_run_dir / "prompt_rows.jsonl"

prompt_rows = []
for raw in prompt_rows_path.read_text(encoding="utf-8").splitlines():
    raw = raw.strip()
    if raw:
        prompt_rows.append(json.loads(raw))

random_top1 = 0.0
if prompt_rows:
    random_top1 = sum(
        (len(row.get("gold_top_candidate_ids", [])) / max(int(row.get("num_candidates", 1)), 1))
        for row in prompt_rows
    ) / len(prompt_rows)

rejection_map = {
    round(float(point["target_coverage"]), 2): point
    for point in samefamily_metrics.get("rejection_curve", [])
}
pressure_map = {
    int(point["subset_size"]): point
    for point in samefamily_metrics.get("pressure_curve", [])
}

base_top1 = float(samefamily_metrics.get("prompt_pool_top1_accuracy", 0.0))
rej40 = rejection_map.get(0.40)
rej40_top1 = float(rej40["top1_accuracy"]) if rej40 else None
rej40_gain = (float(rej40_top1) - base_top1) if rej40_top1 is not None else None
pressure8 = pressure_map.get(8)
pressure8_top1 = float(pressure8["top1_accuracy"]) if pressure8 else None

local_first_bad = samefamily_metrics.get("local_first_bad_edge_accuracy")
local_safe_bad = samefamily_metrics.get("local_safe_vs_bad_pair_accuracy")

def resolve_auc(summary_payload):
    metrics = dict(summary_payload.get("metrics", {}))
    if "pair_auc_good_vs_bad" in metrics:
        return float(metrics["pair_auc_good_vs_bad"])
    if "auc" in metrics:
        return float(metrics["auc"])
    return 0.0

pb_gsm_auc = resolve_auc(pb_gsm_summary)
pb_math_auc = resolve_auc(pb_math_summary)

# 这里的 gate 只是仓库内部的保守操作阈值，不是论文共识中的“普适 RL-ready 定义”。
# This gate is only a conservative internal operating threshold, not a universal paper-backed RL-ready definition.
samefamily_green = (
    base_top1 >= 0.80
    and (local_first_bad is None or float(local_first_bad) >= 0.75)
    and (rej40_gain is not None and float(rej40_gain) >= 0.03)
    and (pressure8_top1 is None or float(pressure8_top1) >= 0.70)
)
benchmark_green = pb_gsm_auc >= 0.55 and pb_math_auc >= 0.55
rl_ready_heuristic = bool(samefamily_green and benchmark_green)

if rl_ready_heuristic:
    assessment = "provisionally_rl_ready"
elif samefamily_green and not benchmark_green:
    assessment = "samefamily_only_not_benchmark_safe"
elif (base_top1 >= 0.65) or (local_first_bad is not None and float(local_first_bad) >= 0.65):
    assessment = "useful_signal_but_not_rl_ready"
else:
    assessment = "not_ready"

row = {
    "audit_id": audit_id,
    "source_family": source_family,
    "value_run_dir": str(value_run_dir),
    "samefamily_run_dir": str(samefamily_run_dir),
    "processbench_gsm8k_run_dir": str(pb_gsm_run_dir),
    "processbench_math_run_dir": str(pb_math_run_dir),
    "prompt_pool_top1_accuracy": base_top1,
    "prompt_pool_mean_regret": float(samefamily_metrics.get("prompt_pool_mean_regret", 0.0)),
    "random_top1_baseline": float(random_top1),
    "top1_lift_over_random": float(base_top1 - random_top1),
    "local_last_safe_top1_accuracy": samefamily_metrics.get("local_last_safe_top1_accuracy"),
    "local_first_bad_edge_accuracy": local_first_bad,
    "local_safe_vs_bad_pair_accuracy": local_safe_bad,
    "rejection_040_top1_accuracy": rej40_top1,
    "rejection_040_top1_gain": rej40_gain,
    "pressure_008_top1_accuracy": pressure8_top1,
    "processbench_gsm8k_auc": pb_gsm_auc,
    "processbench_math_auc": pb_math_auc,
    "samefamily_green": bool(samefamily_green),
    "benchmark_green": bool(benchmark_green),
    "rl_ready_heuristic": bool(rl_ready_heuristic),
    "assessment": assessment,
}

with output_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

render_final_summary() {
  python - "$AUDIT_RESULTS_JSONL" "$SUMMARY_FILE" "$ACTIVE_PHASE_E_RL_GROUP" "$GROUP_TITLE" "$RUN_PREFIX" "$SUITE_LOG_FILE" "$GROUP_INTENTION" "$GROUP_OBSERVE" "$GROUP_EXPECT" <<'PY'
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
    "# Phase E RL-Readiness Audit Summary",
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
    "## Audit Rule",
    "",
    "- This suite uses a conservative internal heuristic, not a universal literature threshold.",
    "- `samefamily_green` requires strong prompt-pool top1, positive rejection gain at 40% coverage, and no obvious collapse at best-of-8 pressure.",
    "- `benchmark_green` requires both `ProcessBench GSM8K` and `ProcessBench Math` AUC to reach at least `0.55`.",
    "- `rl_ready_heuristic = samefamily_green AND benchmark_green`.",
    "",
    "## Candidate Comparison",
    "",
    "| audit_id | source | pool_top1 | random_top1 | lift | first_bad_acc | rej40_gain | p8_top1 | pb_gsm_auc | pb_math_auc | samefamily_green | benchmark_green | rl_ready | assessment |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]

def fmt(value):
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"

for row in rows:
    lines.append(
        "| " + " | ".join(
            [
                str(row["audit_id"]),
                str(row["source_family"]),
                fmt(row.get("prompt_pool_top1_accuracy")),
                fmt(row.get("random_top1_baseline")),
                fmt(row.get("top1_lift_over_random")),
                fmt(row.get("local_first_bad_edge_accuracy")),
                fmt(row.get("rejection_040_top1_gain")),
                fmt(row.get("pressure_008_top1_accuracy")),
                fmt(row.get("processbench_gsm8k_auc")),
                fmt(row.get("processbench_math_auc")),
                "1" if bool(row.get("samefamily_green")) else "0",
                "1" if bool(row.get("benchmark_green")) else "0",
                "1" if bool(row.get("rl_ready_heuristic")) else "0",
                str(row.get("assessment", "")),
            ]
        ) + " |"
    )
    lines.extend(
        [
            f"  value_run_dir: `{row['value_run_dir']}`",
            f"  samefamily_run_dir: `{row['samefamily_run_dir']}`",
            f"  processbench_gsm8k_run_dir: `{row['processbench_gsm8k_run_dir']}`",
            f"  processbench_math_run_dir: `{row['processbench_math_run_dir']}`",
        ]
    )

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

run_one_audit() {
  local audit_id="$1"
  local source_family="$2"
  local value_run_dir="$3"
  local samefamily_prefix="${RUN_PREFIX}_${audit_id}_samefamily"
  local pb_gsm_prefix="${RUN_PREFIX}_${audit_id}_processbench_gsm8k"
  local pb_math_prefix="${RUN_PREFIX}_${audit_id}_processbench_math"
  local samefamily_run_dir=""
  local pb_gsm_run_dir=""
  local pb_math_run_dir=""

  if [[ ! -d "$value_run_dir" ]]; then
    echo "ERROR: Missing value_run_dir: $value_run_dir" >&2
    exit 1
  fi

  CURRENT_STAGE="${audit_id}:samefamily"
  log_line "Running same-family trust audit for ${audit_id} on ${value_run_dir}" | tee -a "$SUITE_LOG_FILE"
  python -u scripts/phase_e_eval_samefamily_trust.py \
    --value-run-dir "$value_run_dir" \
    --run-name "$samefamily_prefix" \
    --output-root assets/artifacts/phase_e_samefamily_eval \
    --batch-size "$RL_AUDIT_BATCH_SIZE" \
    --feature-cache-mode read_write \
    --edge-weight-mode confidence | tee -a "$SUITE_LOG_FILE"
  samefamily_run_dir="$(latest_run_dir "assets/artifacts/phase_e_samefamily_eval/${samefamily_prefix}")"

  CURRENT_STAGE="${audit_id}:processbench_gsm8k"
  log_line "Running ProcessBench GSM8K benchmark audit for ${audit_id}" | tee -a "$SUITE_LOG_FILE"
  python -u scripts/phase_e_eval_benchmark.py \
    --value-run-dir "$value_run_dir" \
    --benchmark-id processbench_gsm8k \
    --run-name "$pb_gsm_prefix" \
    --output-root assets/artifacts/phase_e_eval \
    --batch-size "$RL_AUDIT_BATCH_SIZE" \
    --feature-cache-mode read_write | tee -a "$SUITE_LOG_FILE"
  pb_gsm_run_dir="$(latest_run_dir "assets/artifacts/phase_e_eval/${pb_gsm_prefix}")"

  CURRENT_STAGE="${audit_id}:processbench_math"
  log_line "Running ProcessBench Math benchmark audit for ${audit_id}" | tee -a "$SUITE_LOG_FILE"
  python -u scripts/phase_e_eval_benchmark.py \
    --value-run-dir "$value_run_dir" \
    --benchmark-id processbench_math \
    --run-name "$pb_math_prefix" \
    --output-root assets/artifacts/phase_e_eval \
    --batch-size "$RL_AUDIT_BATCH_SIZE" \
    --feature-cache-mode read_write | tee -a "$SUITE_LOG_FILE"
  pb_math_run_dir="$(latest_run_dir "assets/artifacts/phase_e_eval/${pb_math_prefix}")"

  append_audit_result \
    "$audit_id" \
    "$source_family" \
    "$value_run_dir" \
    "$samefamily_run_dir" \
    "$pb_gsm_run_dir" \
    "$pb_math_run_dir" \
    "$AUDIT_RESULTS_JSONL"
}

resolve_group
mkdir -p "$LOG_ROOT"
: > "$SUITE_LOG_FILE"
: > "$AUDIT_RESULTS_JSONL"

{
  log_line "Phase E RL-Readiness Audit Suite"
  log_line "group_id=${ACTIVE_PHASE_E_RL_GROUP}"
  log_line "group_title=${GROUP_TITLE}"
  log_line "group_intention=${GROUP_INTENTION}"
  log_line "group_observe=${GROUP_OBSERVE}"
  log_line "group_expect=${GROUP_EXPECT}"
  log_line "run_prefix=${RUN_PREFIX}"
  log_line "rl_audit_batch_size=${RL_AUDIT_BATCH_SIZE}"
  log_line "audit_targets=${AUDIT_TARGETS[*]}"
} | tee -a "$SUITE_LOG_FILE"

for item in "${AUDIT_TARGETS[@]}"; do
  IFS='|' read -r audit_id source_family value_run_dir <<<"$item"
  run_one_audit "$audit_id" "$source_family" "$value_run_dir"
done

CURRENT_STAGE="render_summary"
render_final_summary
log_line "RL-readiness audit complete. Summary -> ${SUMMARY_FILE}" | tee -a "$SUITE_LOG_FILE"
