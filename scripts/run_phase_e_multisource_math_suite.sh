#!/usr/bin/env bash
# Phase E multi-source math suite.
#
# English
# -------
# `run_phase_e_suite.sh` runs exactly one Phase E group.  This wrapper lives one
# level above it and compares *families* of groups:
# A. single-source anchors
# B. balanced two-source mixtures
# C. main three-source mixture
# D. weak-source ablations
# E. staged curricula
#
# 中文
# ----
# `run_phase_e_suite.sh` 一次只跑一个 Phase E group。
# 本脚本比它再高一层，负责比较“实验家族”而不是单个 group：
# A. 单源 anchor
# B. 平衡双源混合
# C. 主三源混合
# D. 弱源消融
# E. 分阶段 curriculum
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ACTIVE_PHASE_E_MM_GROUP="${ACTIVE_PHASE_E_MM_GROUP:-MM1_MULTISOURCE_MATH_SMOKE}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_multisource_math}"

LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"
DIRECT_RESULTS_JSONL="${LOG_ROOT}/direct_results.jsonl"
CURRICULUM_RESULTS_JSONL="${LOG_ROOT}/curriculum_results.jsonl"
CURRENT_STAGE="bootstrap"

GROUP_TITLE=""
GROUP_INTENTION=""
GROUP_OBSERVE=""
GROUP_EXPECT=""
DIRECT_GROUPS=()
CURRICULA=()

log_line() {
  # Keep timestamped logs so a failed overnight bundle can be reconstructed
  # without guessing execution order.
  # 所有日志都带时间戳，方便排查 overnight 任务失败时的执行顺序。
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1"
}

on_exit() {
  local exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    return
  fi
  # Leave a minimal failure report behind even if the suite crashes halfway.
  # 即使中途失败，也要留下最小失败 summary，保证排查入口稳定。
  mkdir -p "$LOG_ROOT"
  {
    echo "# Phase E Multi-Source Math Suite Summary"
    echo
    echo "- group_id: ${ACTIVE_PHASE_E_MM_GROUP}"
    echo "- group_title: ${GROUP_TITLE}"
    echo "- run_prefix: ${RUN_PREFIX}"
    echo "- status: failed"
    echo "- failed_stage: ${CURRENT_STAGE}"
    echo "- suite_log_file: ${SUITE_LOG_FILE}"
  } > "$SUMMARY_FILE"
}
trap on_exit EXIT

resolve_group() {
  # This registry defines the higher-level experiment programs.
  # Each entry is not a single run but a bundle of direct groups and/or curricula.
  # 这里注册的是“更高层的实验程序”，每个条目不是单次运行，而是一组 direct group
  # 和/或一组 curriculum。
  case "$ACTIVE_PHASE_E_MM_GROUP" in
    MM1_MULTISOURCE_MATH_SMOKE)
      GROUP_TITLE="MM1 Multi-Source Math Smoke"
      GROUP_INTENTION="Quickly validate the new multi-source math mainline with one seed, small pair caps, and one staged curriculum run."
      GROUP_OBSERVE="This smoke is not for scientific claims; it is for proving that Stage A/B/C/D direct mixtures and Stage E staged continuation all execute end-to-end."
      GROUP_EXPECT="Every selected direct group and the staged curriculum should finish cleanly and emit comparable summaries."
      DIRECT_GROUPS=(
        E20_STAGEA_MS_ANCHOR_SEED3
        E24_STAGEB_MS_RPRM_MIX_SEED3
        E27_STAGEC_MS_RPRM_PRMBENCH_MIX_SEED3
        E28_STAGED_TRIMIX_PLUS_PRM800K_LOWWT_SEED3
      )
      CURRICULA=(
        CUR1_STAGEE_MS_TO_MSRPRM
      )
      SUITE_SEEDS_OVERRIDE="${SUITE_SEEDS_OVERRIDE:-42}"
      SUITE_MAX_PAIRS_PER_SOURCE="${SUITE_MAX_PAIRS_PER_SOURCE:-1000}"
      SUITE_BENCH_MAX_SAMPLES="${SUITE_BENCH_MAX_SAMPLES:-128}"
      SUITE_TRAIN_BATCH_SIZE="${SUITE_TRAIN_BATCH_SIZE:-128}"
      SUITE_EVAL_BATCH_SIZE="${SUITE_EVAL_BATCH_SIZE:-128}"
      ;;
    MM2_MULTISOURCE_MATH_STAGE_ABCD_SEED3)
      GROUP_TITLE="MM2 Multi-Source Math Stage A-D Seed3"
      GROUP_INTENTION="Run the official direct multi-source math matrix covering Stage A anchors, Stage B two-source mixes, Stage C tri-mix, and Stage D weak-source ablations."
      GROUP_OBSERVE="This matrix answers which direct mixture family is strongest before any staged warm-start logic is introduced."
      GROUP_EXPECT="At least one Stage B/C direct mixture should improve benchmark-facing behavior over the single-source anchors without collapsing held-out quality."
      DIRECT_GROUPS=(
        E20_STAGEA_MS_ANCHOR_SEED3
        E21_STAGEA_RPRM_ANCHOR_SEED3
        E22_STAGEA_PRMBENCH_PREVIEW_ANCHOR_SEED3
        E23_STAGEA_PRM800K_CTRL_SEED3
        E24_STAGEB_MS_RPRM_MIX_SEED3
        E25_STAGEB_MS_PRMBENCH_MIX_SEED3
        E26_STAGEB_RPRM_PRMBENCH_MIX_SEED3
        E27_STAGEC_MS_RPRM_PRMBENCH_MIX_SEED3
        E28_STAGED_TRIMIX_PLUS_PRM800K_LOWWT_SEED3
        E29_STAGED_MS_PLUS_PRM800K_LOWWT_SEED3
      )
      CURRICULA=()
      SUITE_SEEDS_OVERRIDE="${SUITE_SEEDS_OVERRIDE:-}"
      SUITE_MAX_PAIRS_PER_SOURCE="${SUITE_MAX_PAIRS_PER_SOURCE:-}"
      SUITE_BENCH_MAX_SAMPLES="${SUITE_BENCH_MAX_SAMPLES:-}"
      SUITE_TRAIN_BATCH_SIZE="${SUITE_TRAIN_BATCH_SIZE:-}"
      SUITE_EVAL_BATCH_SIZE="${SUITE_EVAL_BATCH_SIZE:-}"
      ;;
    MM3_MULTISOURCE_MATH_STAGEE_CURRICULUM_SEED3)
      GROUP_TITLE="MM3 Multi-Source Math Stage E Curriculum Seed3"
      GROUP_INTENTION="Run the staged curriculum hypotheses after the direct mixtures are defined."
      GROUP_OBSERVE="This suite asks whether warm-starting from the Math-Shepherd anchor improves mixture training beyond one-shot direct concatenation."
      GROUP_EXPECT="At least one curriculum should match or beat the corresponding direct mixture while reducing seed fragility."
      DIRECT_GROUPS=()
      CURRICULA=(
        CUR1_STAGEE_MS_TO_MSRPRM
        CUR2_STAGEE_MS_TO_TRIMIX
        CUR3_STAGEE_MS_TO_MSRPRM_TO_TRIMIX
      )
      SUITE_SEEDS_OVERRIDE="${SUITE_SEEDS_OVERRIDE:-}"
      SUITE_MAX_PAIRS_PER_SOURCE="${SUITE_MAX_PAIRS_PER_SOURCE:-}"
      SUITE_BENCH_MAX_SAMPLES="${SUITE_BENCH_MAX_SAMPLES:-}"
      SUITE_TRAIN_BATCH_SIZE="${SUITE_TRAIN_BATCH_SIZE:-}"
      SUITE_EVAL_BATCH_SIZE="${SUITE_EVAL_BATCH_SIZE:-}"
      ;;
    MM4_MULTISOURCE_MATH_FULL_PROGRAM)
      GROUP_TITLE="MM4 Multi-Source Math Full Program"
      GROUP_INTENTION="Run both the direct Stage A-D matrix and the Stage E curricula in one overnight bundle."
      GROUP_OBSERVE="This is the main comprehensive suite for deciding which math-family training path is trustworthy enough to promote."
      GROUP_EXPECT="The suite should identify whether the best path is a direct mixture or a staged curriculum, with clear per-family summaries."
      DIRECT_GROUPS=(
        E20_STAGEA_MS_ANCHOR_SEED3
        E21_STAGEA_RPRM_ANCHOR_SEED3
        E22_STAGEA_PRMBENCH_PREVIEW_ANCHOR_SEED3
        E23_STAGEA_PRM800K_CTRL_SEED3
        E24_STAGEB_MS_RPRM_MIX_SEED3
        E25_STAGEB_MS_PRMBENCH_MIX_SEED3
        E26_STAGEB_RPRM_PRMBENCH_MIX_SEED3
        E27_STAGEC_MS_RPRM_PRMBENCH_MIX_SEED3
        E28_STAGED_TRIMIX_PLUS_PRM800K_LOWWT_SEED3
        E29_STAGED_MS_PLUS_PRM800K_LOWWT_SEED3
      )
      CURRICULA=(
        CUR1_STAGEE_MS_TO_MSRPRM
        CUR2_STAGEE_MS_TO_TRIMIX
        CUR3_STAGEE_MS_TO_MSRPRM_TO_TRIMIX
      )
      SUITE_SEEDS_OVERRIDE="${SUITE_SEEDS_OVERRIDE:-}"
      SUITE_MAX_PAIRS_PER_SOURCE="${SUITE_MAX_PAIRS_PER_SOURCE:-}"
      SUITE_BENCH_MAX_SAMPLES="${SUITE_BENCH_MAX_SAMPLES:-}"
      SUITE_TRAIN_BATCH_SIZE="${SUITE_TRAIN_BATCH_SIZE:-}"
      SUITE_EVAL_BATCH_SIZE="${SUITE_EVAL_BATCH_SIZE:-}"
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_E_MM_GROUP=$ACTIVE_PHASE_E_MM_GROUP" >&2
      exit 1
      ;;
  esac

  if [[ -n "${PHASE_E_MM_DIRECT_GROUPS_OVERRIDE:-}" ]]; then
    # shellcheck disable=SC2206
    DIRECT_GROUPS=(${PHASE_E_MM_DIRECT_GROUPS_OVERRIDE})
  fi
  if [[ -n "${PHASE_E_MM_CURRICULA_OVERRIDE:-}" ]]; then
    # shellcheck disable=SC2206
    CURRICULA=(${PHASE_E_MM_CURRICULA_OVERRIDE})
  fi
}

stage_label_for_group() {
  # Map direct-group ids back to the conceptual Stage A/B/C/D labels used in docs.
  # 把 direct-group id 映射回文档里的 Stage A/B/C/D 语义标签。
  case "$1" in
    E20_*|E21_*|E22_*|E23_*) printf 'A' ;;
    E24_*|E25_*|E26_*) printf 'B' ;;
    E27_*) printf 'C' ;;
    E28_*|E29_*) printf 'D' ;;
    *) printf '?' ;;
  esac
}

run_phase_e_subsuite() {
  local group_id="$1"
  local sub_prefix="$2"
  shift 2
  local extra_env=("$@")
  # This helper delegates one concrete group to `run_phase_e_suite.sh` while
  # injecting top-level overrides such as smaller smoke seeds or batch sizes.
  # 这个 helper 会把一个具体 group 委托给 `run_phase_e_suite.sh`，
  # 并注入上层 suite 想施加的覆盖参数（例如 smoke 用更少 seed、更小 batch）。
  local env_cmd=(
    env
    ACTIVE_PHASE_E_GROUP="$group_id"
    RUN_PREFIX="$sub_prefix"
  )
  if [[ -n "${SUITE_SEEDS_OVERRIDE:-}" ]]; then
    env_cmd+=(SEEDS_OVERRIDE="$SUITE_SEEDS_OVERRIDE")
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
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    env_cmd+=(CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES")
  fi
  env_cmd+=("${extra_env[@]}")
  "${env_cmd[@]}" bash scripts/run_phase_e_suite.sh | tee -a "$SUITE_LOG_FILE"
}

append_summary_result() {
  local result_id="$1"
  local stage_label="$2"
  local summary_path="$3"
  local out_jsonl="$4"
  # Summarize one completed direct/curriculum sub-suite into a compact JSONL row
  # so the final top-level summary can compare them uniformly.
  # 把一个子 suite 的结果压缩成统一 JSONL 行，方便最外层 summary 一视同仁地比较。
  python - "$result_id" "$stage_label" "$summary_path" "$out_jsonl" <<'PY'
import json
import re
import sys
from pathlib import Path

result_id = sys.argv[1]
stage_label = sys.argv[2]
summary_path = Path(sys.argv[3])
out_path = Path(sys.argv[4])
text = summary_path.read_text(encoding="utf-8")

def grab(name: str) -> float | None:
    m = re.search(rf"- {re.escape(name)}: `([^`]+)`", text)
    if not m:
        return None
    return float(m.group(1))

row = {
    "result_id": result_id,
    "stage_label": stage_label,
    "summary_path": str(summary_path),
    "mean_heldout_pair_acc": grab("mean_heldout_pair_acc"),
    "mean_heldout_auc": grab("mean_heldout_auc"),
    "mean_heldout_ranking_score": grab("mean_heldout_ranking_score"),
    "std_heldout_pair_acc": grab("std_heldout_pair_acc"),
    "std_heldout_auc": grab("std_heldout_auc"),
    "mean_processbench_gsm8k_auc": grab("mean_processbench_gsm8k_auc"),
    "mean_processbench_math_auc": grab("mean_processbench_math_auc"),
    "mean_prmbench_preview_auc": grab("mean_prmbench_preview_auc"),
}
with out_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

lookup_best_checkpoint() {
  local suite_log_dir="$1"
  local seed="$2"
  # Curriculum runs need to know "the best checkpoint produced by the previous
  # stage for this exact seed".  That information lives in the seed results file.
  # curriculum 需要知道“上一阶段这个 seed 的最佳 checkpoint 是哪个”，
  # 信息来源就是对应 suite 的 seed_results.jsonl。
  python - "$suite_log_dir" "$seed" <<'PY'
import json
import sys
from pathlib import Path

suite_log_dir = Path(sys.argv[1])
seed = int(sys.argv[2])
rows_path = suite_log_dir / "seed_results.jsonl"
if not rows_path.exists():
    raise SystemExit(f"Missing seed_results.jsonl: {rows_path}")
for raw in rows_path.read_text(encoding="utf-8").splitlines():
    raw = raw.strip()
    if not raw:
        continue
    row = json.loads(raw)
    if int(row["seed"]) == seed:
        value_run_dir = Path(row["value_run_dir"])
        print(value_run_dir / "best_value_head.pt")
        raise SystemExit(0)
raise SystemExit(f"Seed {seed} not found in {rows_path}")
PY
}

render_phase_e_like_summary() {
  local rows_path="$1"
  local summary_path="$2"
  local group_id="$3"
  local group_title="$4"
  local run_prefix="$5"
  local suite_log_file="$6"
  local group_intention="$7"
  local group_observe="$8"
  local group_expect="$9"
  local extra_header="${10}"
  # Reuse the same Markdown style as the single-group Phase E suite so readers
  # do not need to learn two different summary formats.
  # 这里复用单组 Phase E suite 的 Markdown 风格，避免读者面对两套不同 summary 格式。
  python - "$rows_path" "$summary_path" "$group_id" "$group_title" "$run_prefix" "$suite_log_file" "$group_intention" "$group_observe" "$group_expect" "$extra_header" <<'PY'
import json
import statistics
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
extra_header = sys.argv[10]

rows = []
if rows_path.exists():
    for raw in rows_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            rows.append(json.loads(raw))

benchmark_ids = []
for row in rows:
    for benchmark_id in row.get("benchmarks", {}):
        if benchmark_id not in benchmark_ids:
            benchmark_ids.append(benchmark_id)
benchmark_ids.sort()

lines = [
    "# Phase E Suite Summary",
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
]
if extra_header:
    lines.append(f"- stage_sequence: {extra_header}")
lines.extend(["", "## Per-Seed Metrics", ""])
header = ["seed", "train_pairs", "val_pairs", "heldout_pair_acc", "heldout_auc", "heldout_ranking_score"]
for benchmark_id in benchmark_ids:
    header.append(f"{benchmark_id}_pair_acc")
    header.append(f"{benchmark_id}_auc")
lines.append("| " + " | ".join(header) + " |")
lines.append("|" + "|".join(["---"] * len(header)) + "|")
for row in rows:
    values = [
        str(row["seed"]),
        str(row["train_pairs"]),
        str(row["val_pairs"]),
        f"{float(row['heldout_pair_acc']):.4f}",
        f"{float(row['heldout_auc']):.4f}",
        f"{float(row['heldout_ranking_score']):.4f}",
    ]
    benchmarks = row.get("benchmarks", {})
    for benchmark_id in benchmark_ids:
        item = benchmarks.get(benchmark_id, {})
        values.append(f"{float(item.get('pair_acc', 0.0)):.4f}")
        values.append(f"{float(item.get('auc', 0.0)):.4f}")
    lines.append("| " + " | ".join(values) + " |")

if rows:
    lines.extend(["", "## Aggregated", ""])
    heldout_pair_accs = [float(row["heldout_pair_acc"]) for row in rows]
    heldout_aucs = [float(row["heldout_auc"]) for row in rows]
    heldout_ranking_scores = [float(row["heldout_ranking_score"]) for row in rows]
    lines.append(f"- mean_heldout_pair_acc: `{statistics.mean(heldout_pair_accs):.6f}`")
    lines.append(f"- mean_heldout_auc: `{statistics.mean(heldout_aucs):.6f}`")
    lines.append(f"- mean_heldout_ranking_score: `{statistics.mean(heldout_ranking_scores):.6f}`")
    if len(rows) > 1:
        lines.append(f"- std_heldout_pair_acc: `{statistics.pstdev(heldout_pair_accs):.6f}`")
        lines.append(f"- std_heldout_auc: `{statistics.pstdev(heldout_aucs):.6f}`")
        lines.append(f"- std_heldout_ranking_score: `{statistics.pstdev(heldout_ranking_scores):.6f}`")
    for benchmark_id in benchmark_ids:
        pair_accs = [float(row.get("benchmarks", {}).get(benchmark_id, {}).get("pair_acc", 0.0)) for row in rows]
        aucs = [float(row.get("benchmarks", {}).get(benchmark_id, {}).get("auc", 0.0)) for row in rows]
        lines.append(f"- mean_{benchmark_id}_pair_acc: `{statistics.mean(pair_accs):.6f}`")
        lines.append(f"- mean_{benchmark_id}_auc: `{statistics.mean(aucs):.6f}`")
        if len(rows) > 1:
            lines.append(f"- std_{benchmark_id}_pair_acc: `{statistics.pstdev(pair_accs):.6f}`")
            lines.append(f"- std_{benchmark_id}_auc: `{statistics.pstdev(aucs):.6f}`")
lines.append("")
summary_path.write_text("\n".join(lines), encoding="utf-8")
PY
}

run_curriculum() {
  local curriculum_id="$1"
  local curriculum_title="$2"
  local stage_sequence="$3"
  local stage1_group="$4"
  local stage2_group="$5"
  local stage3_group="${6:-}"

  # A curriculum is a staged experiment:
  # 1. run stage 1 to get a warm-start checkpoint,
  # 2. feed that checkpoint into stage 2,
  # 3. optionally continue into stage 3.
  #
  # curriculum 是一个分阶段实验：
  # 1. 先跑 stage 1 拿到 warm-start checkpoint，
  # 2. 把它喂给 stage 2，
  # 3. 必要时再继续到 stage 3。
  local curr_prefix="${RUN_PREFIX}_${curriculum_id,,}"
  local curr_log_dir="assets/artifacts/phase_e_logs/${curr_prefix}"
  local curr_suite_log="${curr_log_dir}/suite.log"
  local curr_rows="${curr_log_dir}/seed_results.jsonl"
  local curr_summary="${curr_log_dir}/final_summary.md"
  mkdir -p "$curr_log_dir"
  : > "$curr_suite_log"
  : > "$curr_rows"

  local stage1_prefix="${curr_prefix}_${stage1_group,,}_warmstart"
  CURRENT_STAGE="${curriculum_id}_stage1"
  log_line "Curriculum ${curriculum_id}: stage1 ${stage1_group}" | tee -a "$SUITE_LOG_FILE" -a "$curr_suite_log"
  run_phase_e_subsuite "$stage1_group" "$stage1_prefix"
  local stage1_log_dir="assets/artifacts/phase_e_logs/${stage1_prefix}"

  # If the top-level wrapper does not override seeds, curricula default to the
  # canonical three-seed set.
  # 如果外层 wrapper 没覆盖 seed，curriculum 默认使用标准三 seed。
  local seeds_text="${SUITE_SEEDS_OVERRIDE:-42 43 44}"
  # shellcheck disable=SC2206
  local curriculum_seeds=(${seeds_text})

  for seed in "${curriculum_seeds[@]}"; do
    local warm_ckpt
    warm_ckpt="$(lookup_best_checkpoint "$stage1_log_dir" "$seed")"
    CURRENT_STAGE="${curriculum_id}_stage2_s${seed}"
    local stage2_prefix="${curr_prefix}_${stage2_group,,}_s${seed}"
    log_line "Curriculum ${curriculum_id}: seed=${seed} stage2 ${stage2_group} init=${warm_ckpt}" | tee -a "$SUITE_LOG_FILE" -a "$curr_suite_log"
    run_phase_e_subsuite \
      "$stage2_group" \
      "$stage2_prefix" \
      "SEEDS_OVERRIDE=${seed}" \
      "INIT_VALUE_HEAD_PATH=${warm_ckpt}"
    local stage2_log_dir="assets/artifacts/phase_e_logs/${stage2_prefix}"

    if [[ -n "$stage3_group" ]]; then
      local stage2_ckpt
      stage2_ckpt="$(lookup_best_checkpoint "$stage2_log_dir" "$seed")"
      CURRENT_STAGE="${curriculum_id}_stage3_s${seed}"
      local stage3_prefix="${curr_prefix}_${stage3_group,,}_s${seed}"
      log_line "Curriculum ${curriculum_id}: seed=${seed} stage3 ${stage3_group} init=${stage2_ckpt}" | tee -a "$SUITE_LOG_FILE" -a "$curr_suite_log"
      run_phase_e_subsuite \
        "$stage3_group" \
        "$stage3_prefix" \
        "SEEDS_OVERRIDE=${seed}" \
        "INIT_VALUE_HEAD_PATH=${stage2_ckpt}"
      cat "assets/artifacts/phase_e_logs/${stage3_prefix}/seed_results.jsonl" >> "$curr_rows"
    else
      cat "${stage2_log_dir}/seed_results.jsonl" >> "$curr_rows"
    fi
  done

  render_phase_e_like_summary \
    "$curr_rows" \
    "$curr_summary" \
    "$curriculum_id" \
    "$curriculum_title" \
    "$curr_prefix" \
    "$curr_suite_log" \
    "Curriculum run derived from the Phase E multi-source math plan." \
    "Judge whether staged warm-start beats or stabilizes the matching direct mixture." \
    "A good curriculum should preserve held-out strength while improving benchmark-facing behavior." \
    "$stage_sequence"
  append_summary_result "$curriculum_id" "E" "$curr_summary" "$CURRICULUM_RESULTS_JSONL"
}

render_final_summary() {
  # Merge direct-group results and curriculum results into one human-readable
  # top-level report.
  # 把 direct group 和 curriculum 两类结果合并成一个顶层、适合人工浏览的报告。
  python - "$DIRECT_RESULTS_JSONL" "$CURRICULUM_RESULTS_JSONL" "$SUMMARY_FILE" "$ACTIVE_PHASE_E_MM_GROUP" "$GROUP_TITLE" "$RUN_PREFIX" "$SUITE_LOG_FILE" "$GROUP_INTENTION" "$GROUP_OBSERVE" "$GROUP_EXPECT" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

direct_path = Path(sys.argv[1])
curr_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
group_id = sys.argv[4]
group_title = sys.argv[5]
run_prefix = sys.argv[6]
suite_log_file = sys.argv[7]
group_intention = sys.argv[8]
group_observe = sys.argv[9]
group_expect = sys.argv[10]

def load_rows(path: Path):
    rows = []
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows

direct_rows = load_rows(direct_path)
curr_rows = load_rows(curr_path)

def fmt(value):
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"

lines = [
    "# Phase E Multi-Source Math Suite Summary",
    "",
    f"- generated_at: {datetime.now(timezone.utc).isoformat()}",
    f"- group_id: {group_id}",
    f"- group_title: {group_title}",
    f"- run_prefix: {run_prefix}",
    f"- status: {'ok' if (direct_rows or curr_rows) else 'empty'}",
    f"- suite_log_file: {suite_log_file}",
    f"- group_intention: {group_intention}",
    f"- observe: {group_observe}",
    f"- expect: {group_expect}",
    "",
]

if direct_rows:
    lines.extend(
        [
            "## Direct Stage A-D Groups",
            "",
            "| stage | group_id | mean_pair_acc | mean_auc | mean_rank | pb_gsm_auc | pb_math_auc | prmbench_auc | std_pair_acc | std_auc |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in direct_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["stage_label"]),
                    str(row["result_id"]),
                    fmt(row["mean_heldout_pair_acc"]),
                    fmt(row["mean_heldout_auc"]),
                    fmt(row["mean_heldout_ranking_score"]),
                    fmt(row["mean_processbench_gsm8k_auc"]),
                    fmt(row["mean_processbench_math_auc"]),
                    fmt(row["mean_prmbench_preview_auc"]),
                    fmt(row["std_heldout_pair_acc"]),
                    fmt(row["std_heldout_auc"]),
                ]
            )
            + " |"
        )
        lines.append(f"Path: `{row['summary_path']}`")
    lines.append("")

if curr_rows:
    lines.extend(
        [
            "## Stage E Curricula",
            "",
            "| stage | curriculum_id | mean_pair_acc | mean_auc | mean_rank | pb_gsm_auc | pb_math_auc | prmbench_auc | std_pair_acc | std_auc |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in curr_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["stage_label"]),
                    str(row["result_id"]),
                    fmt(row["mean_heldout_pair_acc"]),
                    fmt(row["mean_heldout_auc"]),
                    fmt(row["mean_heldout_ranking_score"]),
                    fmt(row["mean_processbench_gsm8k_auc"]),
                    fmt(row["mean_processbench_math_auc"]),
                    fmt(row["mean_prmbench_preview_auc"]),
                    fmt(row["std_heldout_pair_acc"]),
                    fmt(row["std_heldout_auc"]),
                ]
            )
            + " |"
        )
        lines.append(f"Path: `{row['summary_path']}`")
    lines.append("")

summary_path.write_text("\n".join(lines), encoding="utf-8")
PY
}

resolve_group
mkdir -p "$LOG_ROOT"
: > "$SUITE_LOG_FILE"
: > "$DIRECT_RESULTS_JSONL"
: > "$CURRICULUM_RESULTS_JSONL"

{
  log_line "Phase E Multi-Source Math Suite"
  log_line "group_id=${ACTIVE_PHASE_E_MM_GROUP}"
  log_line "group_title=${GROUP_TITLE}"
  log_line "group_intention=${GROUP_INTENTION}"
  log_line "group_observe=${GROUP_OBSERVE}"
  log_line "group_expect=${GROUP_EXPECT}"
  log_line "run_prefix=${RUN_PREFIX}"
  log_line "direct_groups=${DIRECT_GROUPS[*]:-<none>}"
  log_line "curricula=${CURRICULA[*]:-<none>}"
} | tee -a "$SUITE_LOG_FILE"

for group_id in "${DIRECT_GROUPS[@]}"; do
  CURRENT_STAGE="$group_id"
  # Direct groups are run exactly as standalone Phase E groups, then summarized
  # into the multi-source program view.
  # direct group 的执行方式与单独 Phase E group 相同，只是在外层再做一次程序级汇总。
  sub_prefix="${RUN_PREFIX}_${group_id,,}"
  log_line "Launching direct sub-suite ${group_id} with RUN_PREFIX=${sub_prefix}" | tee -a "$SUITE_LOG_FILE"
  run_phase_e_subsuite "$group_id" "$sub_prefix"
  sub_summary="assets/artifacts/phase_e_logs/${sub_prefix}/final_summary.md"
  if [[ ! -f "$sub_summary" ]]; then
    echo "ERROR: Missing direct sub-suite summary: $sub_summary" >&2
    exit 1
  fi
  append_summary_result "$group_id" "$(stage_label_for_group "$group_id")" "$sub_summary" "$DIRECT_RESULTS_JSONL"
done

for curriculum_id in "${CURRICULA[@]}"; do
  # Each curriculum id expands to a concrete stage sequence from the research plan.
  # 每个 curriculum id 都会展开成研究计划里定义好的具体阶段序列。
  case "$curriculum_id" in
    CUR1_STAGEE_MS_TO_MSRPRM)
      run_curriculum \
        "$curriculum_id" \
        "CUR1 Stage E MS -> MS+R-PRM" \
        "E20_STAGEA_MS_ANCHOR_SEED3 -> E24_STAGEB_MS_RPRM_MIX_SEED3" \
        E20_STAGEA_MS_ANCHOR_SEED3 \
        E24_STAGEB_MS_RPRM_MIX_SEED3
      ;;
    CUR2_STAGEE_MS_TO_TRIMIX)
      run_curriculum \
        "$curriculum_id" \
        "CUR2 Stage E MS -> Tri-Mix" \
        "E20_STAGEA_MS_ANCHOR_SEED3 -> E27_STAGEC_MS_RPRM_PRMBENCH_MIX_SEED3" \
        E20_STAGEA_MS_ANCHOR_SEED3 \
        E27_STAGEC_MS_RPRM_PRMBENCH_MIX_SEED3
      ;;
    CUR3_STAGEE_MS_TO_MSRPRM_TO_TRIMIX)
      run_curriculum \
        "$curriculum_id" \
        "CUR3 Stage E MS -> MS+R-PRM -> Tri-Mix" \
        "E20_STAGEA_MS_ANCHOR_SEED3 -> E24_STAGEB_MS_RPRM_MIX_SEED3 -> E27_STAGEC_MS_RPRM_PRMBENCH_MIX_SEED3" \
        E20_STAGEA_MS_ANCHOR_SEED3 \
        E24_STAGEB_MS_RPRM_MIX_SEED3 \
        E27_STAGEC_MS_RPRM_PRMBENCH_MIX_SEED3
      ;;
    *)
      echo "ERROR: Unknown curriculum id: $curriculum_id" >&2
      exit 1
      ;;
  esac
done

render_final_summary
log_line "Summary file   : ${SUMMARY_FILE}" | tee -a "$SUITE_LOG_FILE"
log_line "Group complete" | tee -a "$SUITE_LOG_FILE"
