#!/usr/bin/env bash
# Phase E ProcessBench-oriented hybrid curation + architecture sweep suite.
#
# 这个脚本把当前最关键的问题直接固化成可复现实验：
# 1. 不再简单做 source concat，而是显式构造 benchmark-oriented hybrid artifact；
# 2. 以 `PRMBench` local pairs 作为锚点，按小比例补 terminal / grid；
# 3. 比较 `mlp` 和 `gated_mlp` 两种头，判断“数据问题”还是“结构问题”更主导。
#
# This wrapper packages the current ProcessBench diagnosis into one reproducible flow:
# 1. build an explicit benchmark-oriented hybrid artifact instead of naive source concat,
# 2. keep `PRMBench` local pairs as the anchor while adding bounded terminal / grid repairs,
# 3. compare `mlp` vs `gated_mlp` so we can separate data-contract gains from architecture gains.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACTIVE_PHASE_E_PB_HYBRID_GROUP="${ACTIVE_PHASE_E_PB_HYBRID_GROUP:-PH1_PRM_LOCAL_TA15_ARCH_SWEEP_SMOKE}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_processbench_hybrid}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-assets/artifacts/phase_e_logs/${RUN_PREFIX}}"
SUITE_LOG_FILE="${OUTPUT_ROOT}/suite.log"
FINAL_SUMMARY_FILE="${OUTPUT_ROOT}/final_summary.md"
CASE_RESULTS_JSONL="${OUTPUT_ROOT}/case_results.jsonl"
COMPARE_ROOT="${COMPARE_ROOT:-assets/artifacts/phase_e_transfer_compare}"

# 默认使用当前仓库里最强的 `PRMBench` local anchor 及其现成 benchmark eval。
# These defaults keep the suite runnable without manual path filling.
DEFAULT_E46_RUN="assets/artifacts/phase_e_runs/phase_e_prmbench_acc90_full_0310_1914_e46_prmbench_acc90_mlp_joint_seed3_e46_prmbench_acc90_mlp_joint_seed3_s42_value_20260310T113722Z"
DEFAULT_E46_GSM_EVAL="assets/artifacts/phase_e_eval/phase_e_rltops_0311_1124_prm_e46_processbench_gsm8k_20260311T032704Z"
DEFAULT_E46_MATH_EVAL="assets/artifacts/phase_e_eval/phase_e_rltops_0311_1124_prm_e46_processbench_math_20260311T032713Z"
DEFAULT_E82_RUN="assets/artifacts/phase_e_runs/phase_e_pb_transfer_mix_0311_1722_e82_ms_prmbench_transfer_mix_seed42_e82_ms_prmbench_transfer_mix_seed42_s42_value_20260310T171945Z"
DEFAULT_E82_GSM_EVAL="assets/artifacts/phase_e_eval/phase_e_pb_transfer_mix_0311_1722_e82_ms_prmbench_transfer_mix_seed42_e82_ms_prmbench_transfer_mix_seed42_s42_processbench_gsm8k_20260310T174541Z"
DEFAULT_E82_MATH_EVAL="assets/artifacts/phase_e_eval/phase_e_pb_transfer_mix_0311_1722_e82_ms_prmbench_transfer_mix_seed42_e82_ms_prmbench_transfer_mix_seed42_s42_processbench_math_20260310T174554Z"

CURRENT_STAGE="bootstrap"
GROUP_TITLE=""
GROUP_INTENTION=""
GROUP_OBSERVE=""
GROUP_EXPECT=""
GROUP_ARTIFACT_SPECS=()
GROUP_HEADS=()
GROUP_COMPARE_BASELINES=()
ARTIFACT_DIR_RESULT=""
VALUE_RUN_DIR_RESULT=""
BENCH_EVAL_DIR_RESULT=""

TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-96}"
BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-128}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_GPU_MEMORY_GIB="${MAX_GPU_MEMORY_GIB:-40}"
MAX_CPU_MEMORY_GIB="${MAX_CPU_MEMORY_GIB:-96}"
FEATURE_CACHE_MODE="${FEATURE_CACHE_MODE:-read_write}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-assets/artifacts/phase_e_feature_cache}"
PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-confidence}"
SOURCE_BALANCE="${SOURCE_BALANCE:-uniform}"
PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
OBJECTIVE_MODE="${OBJECTIVE_MODE:-joint}"
RANKING_TARGET_SPACE="${RANKING_TARGET_SPACE:-logit}"
RANKING_MARGIN="${RANKING_MARGIN:-0.02}"
LAMBDA_RANKING="${LAMBDA_RANKING:-1.0}"
LAMBDA_BCE="${LAMBDA_BCE:-1.0}"
HEAD_DROPOUT_PROB="${HEAD_DROPOUT_PROB:-0.05}"
HEAD_INIT_STD="${HEAD_INIT_STD:-0.02}"
HEAD_MLP_HIDDEN_SIZE="${HEAD_MLP_HIDDEN_SIZE:-1024}"
HEAD_ACTIVATION="${HEAD_ACTIVATION:-gelu}"
MODEL_PATH="${MODEL_PATH:-assets/models/Qwen2.5-7B-Instruct}"
INIT_VALUE_HEAD_PATH="${INIT_VALUE_HEAD_PATH:-${DEFAULT_E46_RUN}/best_value_head.pt}"

PRMBENCH_LOCAL_ARTIFACT="${PRMBENCH_LOCAL_ARTIFACT:-assets/artifacts/phase_e_pairs/phase_e_prmbench_acc90_full_0310_1914_e46_prmbench_acc90_mlp_joint_seed3_e46_prmbench_acc90_mlp_joint_seed3_sharedsplit_s42_pairs__8886075f9c6e}"
PRMBENCH_TERMINAL_ARTIFACT="${PRMBENCH_TERMINAL_ARTIFACT:-assets/artifacts/phase_e_pairs/phase_e_prmbench_terminal_anchor_full_0311__192ca71fd301}"
MS_GRID_ARTIFACT="${MS_GRID_ARTIFACT:-assets/artifacts/phase_e_pairs/phase_e_ms_grid_cap40k_diag_0311__4f87d4f4cea6}"

log_line() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1"
}

on_exit() {
  local exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    return
  fi
  mkdir -p "$OUTPUT_ROOT"
  {
    echo "# Phase E ProcessBench Hybrid Suite Summary"
    echo
    echo "- group_id: ${ACTIVE_PHASE_E_PB_HYBRID_GROUP}"
    echo "- group_title: ${GROUP_TITLE}"
    echo "- run_prefix: ${RUN_PREFIX}"
    echo "- status: failed"
    echo "- failed_stage: ${CURRENT_STAGE}"
    echo "- suite_log_file: ${SUITE_LOG_FILE}"
  } > "$FINAL_SUMMARY_FILE"
}
trap on_exit EXIT

resolve_group() {
  case "$ACTIVE_PHASE_E_PB_HYBRID_GROUP" in
    PH1_PRM_LOCAL_TA15_ARCH_SWEEP_SMOKE)
      GROUP_TITLE="PH1 PRMBench Local + Terminal15 Architecture Sweep Smoke"
      GROUP_INTENTION="Use PRMBench local error-step pairs as the anchor and add a bounded terminal-completion auxiliary so ProcessBench all-correct slices are no longer unseen."
      GROUP_OBSERVE="Compare whether 'gated_mlp' can preserve local discrimination while recovering more terminal behavior than a standard 'mlp' on the same curated artifact."
      GROUP_EXPECT="If the main blocker is still data-contract mismatch, both runs should beat pure E46 on terminal metrics; if structure also matters, 'gated_mlp' should keep more AUC / first-edge than 'mlp'."
      GROUP_ARTIFACT_SPECS=(
        "prm_local=${PRMBENCH_LOCAL_ARTIFACT}:3072:384:1.00"
        "prm_terminal=${PRMBENCH_TERMINAL_ARTIFACT}:512:64:0.35"
      )
      GROUP_HEADS=("mlp" "gated_mlp")
      GROUP_COMPARE_BASELINES=(
        "e46=${DEFAULT_E46_RUN}::${DEFAULT_E46_GSM_EVAL}::${DEFAULT_E46_MATH_EVAL}"
      )
      ;;
    PH2_PRM_LOCAL_TA10_MSGRID10_ARCH_SWEEP_SMOKE)
      GROUP_TITLE="PH2 PRMBench Local + Terminal10 + MS-Grid10 Architecture Sweep Smoke"
      GROUP_INTENTION="Keep PRMBench local pairs as the benchmark-aligned anchor, add a lighter terminal auxiliary, and inject a small amount of Math-Shepherd grid support for broader good-vs-bad prefix coverage."
      GROUP_OBSERVE="This asks whether a tri-mix can improve ProcessBench pair ranking without collapsing the completion-side gains seen in terminal-anchor pilots."
      GROUP_EXPECT="If the current problem is truly 'local + terminal + broader-prefix' under-coverage, this group should dominate E46 and E82 on at least one benchmark while staying competitive on the others."
      GROUP_ARTIFACT_SPECS=(
        "prm_local=${PRMBENCH_LOCAL_ARTIFACT}:3072:384:1.00"
        "prm_terminal=${PRMBENCH_TERMINAL_ARTIFACT}:384:48:0.25"
        "ms_grid=${MS_GRID_ARTIFACT}:768:96:0.20"
      )
      GROUP_HEADS=("mlp" "gated_mlp")
      GROUP_COMPARE_BASELINES=(
        "e46=${DEFAULT_E46_RUN}::${DEFAULT_E46_GSM_EVAL}::${DEFAULT_E46_MATH_EVAL}"
        "e82=${DEFAULT_E82_RUN}::${DEFAULT_E82_GSM_EVAL}::${DEFAULT_E82_MATH_EVAL}"
      )
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_E_PB_HYBRID_GROUP=$ACTIVE_PHASE_E_PB_HYBRID_GROUP" >&2
      exit 1
      ;;
  esac
}

latest_dir_by_prefix() {
  local prefix="$1"
  ls -1dt "${prefix}"__* 2>/dev/null | head -n1
}

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    exit 1
  fi
}

prepare_hybrid_artifact() {
  local run_name="$1"
  shift
  CURRENT_STAGE="prepare_pairs"
  log_line "PREPARE: ${run_name}" | tee -a "$SUITE_LOG_FILE" >&2
  "$PYTHON_BIN" -u scripts/phase_e_mix_pair_artifacts.py \
    --run-name "$run_name" \
    "$@" | tee -a "$SUITE_LOG_FILE" >&2
  ARTIFACT_DIR_RESULT="$(latest_dir_by_prefix "assets/artifacts/phase_e_pairs/${run_name}")"
  if [[ -z "$ARTIFACT_DIR_RESULT" ]]; then
    echo "ERROR: failed to resolve mixed pair artifact for run_name=${run_name}" >&2
    exit 1
  fi
}

train_case() {
  local artifact_dir="$1"
  local case_tag="$2"
  local head_arch="$3"
  CURRENT_STAGE="train_${case_tag}"
  local run_name="${RUN_PREFIX}_${case_tag}"
  local init_args=()
  if [[ -n "${INIT_VALUE_HEAD_PATH:-}" && -f "${INIT_VALUE_HEAD_PATH:-}" ]]; then
    init_args+=(--init-value-head-path "$INIT_VALUE_HEAD_PATH")
  fi
  log_line "TRAIN: ${run_name} head=${head_arch}" | tee -a "$SUITE_LOG_FILE" >&2
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" -u scripts/phase_e_train_value.py \
    --train-pairs-jsonl "${artifact_dir}/train_pairs.jsonl" \
    --eval-pairs-jsonl "${artifact_dir}/validation_pairs.jsonl" \
    --model-path "$MODEL_PATH" \
    --run-name "$run_name" \
    --objective-mode "$OBJECTIVE_MODE" \
    --learning-rate "$LEARNING_RATE" \
    --num-train-epochs "$TRAIN_EPOCHS" \
    --per-device-train-batch-size "$TRAIN_BATCH_SIZE" \
    --per-device-eval-batch-size "$EVAL_BATCH_SIZE" \
    --max-length "$MAX_LENGTH" \
    --lambda-ranking "$LAMBDA_RANKING" \
    --lambda-bce "$LAMBDA_BCE" \
    --ranking-margin "$RANKING_MARGIN" \
    --ranking-target-space "$RANKING_TARGET_SPACE" \
    --pair-weight-mode "$PAIR_WEIGHT_MODE" \
    --source-balance "$SOURCE_BALANCE" \
    --permutation-mode "$PERMUTATION_MODE" \
    --checkpoint-selection-metric pair_acc \
    --seed 42 \
    --dtype bfloat16 \
    --device-map auto \
    --max-gpu-memory-gib "$MAX_GPU_MEMORY_GIB" \
    --max-cpu-memory-gib "$MAX_CPU_MEMORY_GIB" \
    --feature-cache-root "$FEATURE_CACHE_ROOT" \
    --feature-cache-mode "$FEATURE_CACHE_MODE" \
    --head-architecture "$head_arch" \
    --head-dropout-prob "$HEAD_DROPOUT_PROB" \
    --head-init-std "$HEAD_INIT_STD" \
    --head-mlp-hidden-size "$HEAD_MLP_HIDDEN_SIZE" \
    --head-activation "$HEAD_ACTIVATION" \
    --require-cuda \
    "${init_args[@]}" | tee -a "$SUITE_LOG_FILE" >&2
  VALUE_RUN_DIR_RESULT="$(latest_dir_by_prefix "assets/artifacts/phase_e_runs/${run_name}")"
  if [[ -z "$VALUE_RUN_DIR_RESULT" ]]; then
    echo "ERROR: failed to resolve value run dir for run_name=${run_name}" >&2
    exit 1
  fi
}

eval_case_benchmark() {
  local value_run_dir="$1"
  local case_tag="$2"
  local benchmark_id="$3"
  CURRENT_STAGE="eval_${case_tag}_${benchmark_id}"
  local run_name="${RUN_PREFIX}_${case_tag}_${benchmark_id}"
  log_line "EVAL: ${run_name}" | tee -a "$SUITE_LOG_FILE" >&2
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" -u scripts/phase_e_eval_benchmark.py \
    --value-run-dir "$value_run_dir" \
    --benchmark-id "$benchmark_id" \
    --run-name "$run_name" \
    --checkpoint-name best \
    --max-samples "$BENCH_MAX_SAMPLES" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --max-gpu-memory-gib "$MAX_GPU_MEMORY_GIB" \
    --max-cpu-memory-gib "$MAX_CPU_MEMORY_GIB" \
    --feature-cache-root "$FEATURE_CACHE_ROOT" \
    --feature-cache-mode "$FEATURE_CACHE_MODE" \
    --require-cuda | tee -a "$SUITE_LOG_FILE" >&2
  BENCH_EVAL_DIR_RESULT="$(latest_dir_by_prefix "assets/artifacts/phase_e_eval/${run_name}")"
  if [[ -z "$BENCH_EVAL_DIR_RESULT" ]]; then
    echo "ERROR: failed to resolve benchmark eval dir for run_name=${run_name}" >&2
    exit 1
  fi
}

append_case_result() {
  local case_tag="$1"
  local artifact_dir="$2"
  local value_run_dir="$3"
  local gsm_eval_dir="$4"
  local math_eval_dir="$5"
  "$PYTHON_BIN" - "$case_tag" "$artifact_dir" "$value_run_dir" "$gsm_eval_dir" "$math_eval_dir" "$CASE_RESULTS_JSONL" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

case_tag = sys.argv[1]
artifact_dir = Path(sys.argv[2])
value_run_dir = Path(sys.argv[3])
gsm_eval_dir = Path(sys.argv[4])
math_eval_dir = Path(sys.argv[5])
out_path = Path(sys.argv[6])

train_summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
value_eval = json.loads((value_run_dir / "eval_metrics.json").read_text(encoding="utf-8"))
gsm_metrics = json.loads((gsm_eval_dir / "metrics.json").read_text(encoding="utf-8"))
math_metrics = json.loads((math_eval_dir / "metrics.json").read_text(encoding="utf-8"))

row = {
    "case_tag": case_tag,
    "artifact_dir": str(artifact_dir),
    "value_run_dir": str(value_run_dir),
    "train_summary": {
        "num_train_rows": int(train_summary.get("num_train_rows", 0)),
        "num_validation_rows": int(train_summary.get("num_validation_rows", 0)),
        "source_rows": train_summary.get("source_rows", []),
    },
    "heldout": {
        "pair_acc": float(value_eval.get("pair_accuracy", 0.0)),
        "auc": float(value_eval.get("auc", 0.0)),
        "ranking_score": float(value_eval.get("ranking_score", 0.0)),
    },
    "processbench_gsm8k": {
        "pair_acc": float(gsm_metrics.get("pair_accuracy_good_vs_bad", 0.0)),
        "auc": float(gsm_metrics.get("pair_auc_good_vs_bad", 0.0)),
        "first_edge": float(gsm_metrics.get("first_error_edge_accuracy", 0.0)),
        "all_correct_last": float(gsm_metrics.get("mean_all_correct_last_score", 0.0)),
    },
    "processbench_math": {
        "pair_acc": float(math_metrics.get("pair_accuracy_good_vs_bad", 0.0)),
        "auc": float(math_metrics.get("pair_auc_good_vs_bad", 0.0)),
        "first_edge": float(math_metrics.get("first_error_edge_accuracy", 0.0)),
        "all_correct_last": float(math_metrics.get("mean_all_correct_last_score", 0.0)),
    },
}
with out_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

compare_cases() {
  local benchmark_name="$1"
  shift
  CURRENT_STAGE="compare_${benchmark_name}"
  local run_name="${RUN_PREFIX}_${benchmark_name}_compare"
  log_line "COMPARE: ${benchmark_name}" | tee -a "$SUITE_LOG_FILE"
  "$PYTHON_BIN" -u scripts/phase_e_compare_processbench_transfer.py \
    --run-name "$run_name" \
    "$@" | tee -a "$SUITE_LOG_FILE"
}

parse_baseline_triplet() {
  local raw="$1"
  local name_part="${raw%%=*}"
  local rest="${raw#*=}"
  local run_dir gsm_eval_dir math_eval_dir
  if [[ "$raw" == "$rest" || "$rest" != *"::"* ]]; then
    echo "ERROR: invalid baseline spec, expected name=run_dir::gsm_eval_dir::math_eval_dir, got: $raw" >&2
    exit 1
  fi
  run_dir="${rest%%::*}"
  rest="${rest#*::}"
  if [[ "$rest" != *"::"* ]]; then
    echo "ERROR: invalid baseline spec, missing second :: separator: $raw" >&2
    exit 1
  fi
  gsm_eval_dir="${rest%%::*}"
  math_eval_dir="${rest#*::}"
  if [[ -z "$name_part" || -z "$run_dir" || -z "$gsm_eval_dir" || -z "$math_eval_dir" ]]; then
    echo "ERROR: invalid baseline spec with empty field: $raw" >&2
    exit 1
  fi
  printf '%s\n%s\n%s\n%s\n' "$name_part" "$run_dir" "$gsm_eval_dir" "$math_eval_dir"
}

write_final_summary() {
  CURRENT_STAGE="write_summary"
  "$PYTHON_BIN" - "$ACTIVE_PHASE_E_PB_HYBRID_GROUP" "$GROUP_TITLE" "$RUN_PREFIX" "$GROUP_INTENTION" "$GROUP_OBSERVE" "$GROUP_EXPECT" "$SUITE_LOG_FILE" "$CASE_RESULTS_JSONL" "$FINAL_SUMMARY_FILE" <<'PY'
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

group_id, group_title, run_prefix, intention, observe, expect, suite_log, rows_path, out_path = sys.argv[1:]
rows = [
    json.loads(line)
    for line in Path(rows_path).read_text(encoding="utf-8").splitlines()
    if line.strip()
]

def mean_for(section: str, key: str) -> float:
    vals = [float(row[section][key]) for row in rows]
    return float(statistics.mean(vals)) if vals else 0.0

lines = [
    "# Phase E ProcessBench Hybrid Suite Summary",
    "",
    f"- group_id: {group_id}",
    f"- group_title: {group_title}",
    f"- run_prefix: {run_prefix}",
    f"- status: ok",
    f"- suite_log_file: {suite_log}",
    f"- intention: {intention}",
    f"- observe: {observe}",
    f"- expect: {expect}",
    "",
    "| case | held_pair | held_auc | pb_gsm_pair | pb_gsm_auc | pb_gsm_first_edge | pb_gsm_all_correct | pb_math_pair | pb_math_auc | pb_math_first_edge | pb_math_all_correct |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['case_tag']} | "
        f"{row['heldout']['pair_acc']:.4f} | {row['heldout']['auc']:.4f} | "
        f"{row['processbench_gsm8k']['pair_acc']:.4f} | {row['processbench_gsm8k']['auc']:.4f} | "
        f"{row['processbench_gsm8k']['first_edge']:.4f} | {row['processbench_gsm8k']['all_correct_last']:.4f} | "
        f"{row['processbench_math']['pair_acc']:.4f} | {row['processbench_math']['auc']:.4f} | "
        f"{row['processbench_math']['first_edge']:.4f} | {row['processbench_math']['all_correct_last']:.4f} |"
    )
lines.extend(
    [
        "",
        "## Aggregate Means",
        "",
        f"- mean_held_pair_acc: `{mean_for('heldout', 'pair_acc'):.6f}`",
        f"- mean_held_auc: `{mean_for('heldout', 'auc'):.6f}`",
        f"- mean_pb_gsm_auc: `{mean_for('processbench_gsm8k', 'auc'):.6f}`",
        f"- mean_pb_math_auc: `{mean_for('processbench_math', 'auc'):.6f}`",
    ]
)
Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

main() {
  mkdir -p "$OUTPUT_ROOT"
  : > "$SUITE_LOG_FILE"
  : > "$CASE_RESULTS_JSONL"
  resolve_group

  require_path "$PRMBENCH_LOCAL_ARTIFACT" "PRMBench local artifact"
  require_path "$PRMBENCH_TERMINAL_ARTIFACT" "PRMBench terminal artifact"
  require_path "$MS_GRID_ARTIFACT" "Math-Shepherd grid artifact"
  require_path "$MODEL_PATH" "model path"

  local mix_args=()
  for spec in "${GROUP_ARTIFACT_SPECS[@]}"; do
    mix_args+=(--input "$spec")
  done

  local artifact_run_name="${RUN_PREFIX}_pairs"
  local artifact_dir
  prepare_hybrid_artifact "$artifact_run_name" "${mix_args[@]}"
  artifact_dir="$ARTIFACT_DIR_RESULT"
  log_line "artifact_dir=${artifact_dir}" | tee -a "$SUITE_LOG_FILE"

  local case_tags=()
  local case_run_dirs=()
  local case_gsm_eval_dirs=()
  local case_math_eval_dirs=()

  for head_arch in "${GROUP_HEADS[@]}"; do
    local case_tag="${ACTIVE_PHASE_E_PB_HYBRID_GROUP,,}_${head_arch}"
    local value_run_dir
    train_case "$artifact_dir" "$case_tag" "$head_arch"
    value_run_dir="$VALUE_RUN_DIR_RESULT"
    local gsm_eval_dir
    eval_case_benchmark "$value_run_dir" "$case_tag" processbench_gsm8k
    gsm_eval_dir="$BENCH_EVAL_DIR_RESULT"
    local math_eval_dir
    eval_case_benchmark "$value_run_dir" "$case_tag" processbench_math
    math_eval_dir="$BENCH_EVAL_DIR_RESULT"
    append_case_result "$case_tag" "$artifact_dir" "$value_run_dir" "$gsm_eval_dir" "$math_eval_dir"
    case_tags+=("$case_tag")
    case_run_dirs+=("$value_run_dir")
    case_gsm_eval_dirs+=("$gsm_eval_dir")
    case_math_eval_dirs+=("$math_eval_dir")
  done

  local compare_args_gsm=()
  local compare_args_math=()
  local baseline
  for baseline in "${GROUP_COMPARE_BASELINES[@]}"; do
    local parsed_baseline
    mapfile -t parsed_baseline < <(parse_baseline_triplet "$baseline")
    local base_name="${parsed_baseline[0]}"
    local base_run="${parsed_baseline[1]}"
    local gsm_eval="${parsed_baseline[2]}"
    local math_eval="${parsed_baseline[3]}"
    if [[ -d "$base_run" && -d "$gsm_eval" && -d "$math_eval" ]]; then
      compare_args_gsm+=(--case "${base_name}=${base_run}::${gsm_eval}")
      compare_args_math+=(--case "${base_name}=${base_run}::${math_eval}")
    fi
  done
  local idx
  for idx in "${!case_tags[@]}"; do
    compare_args_gsm+=(--case "${case_tags[$idx]}=${case_run_dirs[$idx]}::${case_gsm_eval_dirs[$idx]}")
    compare_args_math+=(--case "${case_tags[$idx]}=${case_run_dirs[$idx]}::${case_math_eval_dirs[$idx]}")
  done
  compare_cases processbench_gsm8k "${compare_args_gsm[@]}"
  compare_cases processbench_math "${compare_args_math[@]}"

  write_final_summary
  log_line "Final summary written: ${FINAL_SUMMARY_FILE}" | tee -a "$SUITE_LOG_FILE"
}

main "$@"
