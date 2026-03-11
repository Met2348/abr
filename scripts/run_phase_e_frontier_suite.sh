#!/usr/bin/env bash
# Changelog (prepend-only, newest first):
# 2026-03-11: Initial version. 2026-frontier Phase E suite for judge-filter,
#             dual-head routing, and minimal LoRA experiments on one shared
#             strong benchmark-facing artifact (PBR10).
#
# Phase E Frontier Suite
# ======================
# English
# -------
# This wrapper turns three "latest-paradigm" directions into runnable cases:
# 1. judge-filtered training data (critique / consensus style),
# 2. dual-head routing (separate local vs terminal objectives),
# 3. minimal LoRA backbone adaptation (test frozen-head ceiling).
#
# 中文
# ----
# 这个 wrapper 把三条“最新范式”方向固化成可直接运行的实验：
# 1. judge 过滤训练数据（对应 critique / consensus 思路），
# 2. dual-head 路由（把 local 与 terminal 目标拆开），
# 3. 最小 LoRA 骨干解冻（测试 frozen-head ceiling）。
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACTIVE_PHASE_E_FRONTIER_GROUP="${ACTIVE_PHASE_E_FRONTIER_GROUP:-F1_JUDGE_FILTER_PBR10}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_frontier}"
MODEL_PATH="${MODEL_PATH:-assets/models/Qwen2.5-7B-Instruct}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-assets/models/Qwen2.5-Math-7B-Instruct}"
SOURCE_PAIR_ARTIFACT="${SOURCE_PAIR_ARTIFACT:-assets/artifacts/phase_e_pairs/phase_e_pbr10_prm7b_dpo8k_s42__6184f7e62f65}"
VALUE_OUTPUT_ROOT="${VALUE_OUTPUT_ROOT:-assets/artifacts/phase_e_runs}"
PAIR_OUTPUT_ROOT="${PAIR_OUTPUT_ROOT:-assets/artifacts/phase_e_pairs}"
SAMEFAMILY_OUTPUT_ROOT="${SAMEFAMILY_OUTPUT_ROOT:-assets/artifacts/phase_e_samefamily_eval}"
BENCH_OUTPUT_ROOT="${BENCH_OUTPUT_ROOT:-assets/artifacts/phase_e_eval}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-assets/artifacts/phase_e_feature_cache}"

RECIPE_RISK_POLICY="${RECIPE_RISK_POLICY:-error}"
BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-256}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-96}"
SAMEFAMILY_BATCH_SIZE="${SAMEFAMILY_BATCH_SIZE:-96}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-4}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
HEAD_MLP_HIDDEN_SIZE="${HEAD_MLP_HIDDEN_SIZE:-1024}"
HEAD_DROPOUT_PROB="${HEAD_DROPOUT_PROB:-0.05}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-16}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-96}"
JUDGE_MIN_CONFIDENCE="${JUDGE_MIN_CONFIDENCE:-0.40}"
DUAL_TERMINAL_BCE_LAMBDA="${DUAL_TERMINAL_BCE_LAMBDA:-0.5}"
LORA_EPOCHS="${LORA_EPOCHS:-2}"
LORA_TRAIN_BATCH_SIZE="${LORA_TRAIN_BATCH_SIZE:-1}"
LORA_EVAL_BATCH_SIZE="${LORA_EVAL_BATCH_SIZE:-8}"
LORA_GRAD_ACCUM="${LORA_GRAD_ACCUM:-16}"
LORA_LEARNING_RATE="${LORA_LEARNING_RATE:-2e-5}"
LORA_MAX_LENGTH="${LORA_MAX_LENGTH:-768}"
LORA_RANK="${LORA_RANK:-4}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TOP_K_LAYERS="${LORA_TOP_K_LAYERS:-4}"

REFERENCE_PBR10_VALUE_RUN_DIR="${REFERENCE_PBR10_VALUE_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_pbr10_prm7b_dpo8k_s42_value_20260311T110527Z}"
REFERENCE_PBR10_GSM_DIR="${REFERENCE_PBR10_GSM_DIR:-assets/artifacts/phase_e_eval/phase_e_pbr10_prm7b_dpo8k_s42_gsm_fixed05_20260311T113011Z}"
REFERENCE_PBR10_MATH_DIR="${REFERENCE_PBR10_MATH_DIR:-assets/artifacts/phase_e_eval/phase_e_pbr10_prm7b_dpo8k_s42_math_fixed05_20260311T113012Z}"

LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"
RESULTS_JSONL="${LOG_ROOT}/results.jsonl"
CURRENT_STAGE="bootstrap"

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
    echo "# Phase E Frontier Suite Summary"
    echo
    echo "- group_id: ${ACTIVE_PHASE_E_FRONTIER_GROUP}"
    echo "- run_prefix: ${RUN_PREFIX}"
    echo "- status: failed"
    echo "- failed_stage: ${CURRENT_STAGE}"
    echo "- suite_log_file: ${SUITE_LOG_FILE}"
  } > "$SUMMARY_FILE"
}
trap on_exit EXIT

latest_timestamped_dir() {
  local prefix="$1"
  "$PYTHON_BIN" - "$prefix" <<'PY'
from pathlib import Path
import sys
prefix = sys.argv[1]
matches = sorted(Path(prefix).parent.glob(Path(prefix).name + "_*"), key=lambda p: p.stat().st_mtime)
if not matches:
    raise SystemExit(f"No artifact directory matches prefix: {prefix}")
print(matches[-1])
PY
}

latest_run_dir() {
  local prefix="$1"
  "$PYTHON_BIN" - "$prefix" <<'PY'
from pathlib import Path
import sys
prefix = sys.argv[1]
matches = sorted(Path(prefix).parent.glob(Path(prefix).name + "__*"), key=lambda p: p.stat().st_mtime)
if not matches:
    raise SystemExit(f"No artifact directory matches prefix: {prefix}")
print(matches[-1])
PY
}

run_cmd() {
  CURRENT_STAGE="$1"
  shift
  log_line "RUN: $*" | tee -a "$SUITE_LOG_FILE" >&2
  "$@" | tee -a "$SUITE_LOG_FILE" >&2
}

run_samefamily_eval() {
  local case_id="$1"
  local value_run_dir="$2"
  local eval_pairs_jsonl="$3"
  local run_name="${RUN_PREFIX}_${case_id}_samefamily"
  run_cmd "samefamily_${case_id}" \
    "$PYTHON_BIN" -u scripts/phase_e_eval_samefamily_trust.py \
    --value-run-dir "$value_run_dir" \
    --eval-pairs-jsonl "$eval_pairs_jsonl" \
    --run-name "$run_name" \
    --output-root "$SAMEFAMILY_OUTPUT_ROOT" \
    --checkpoint-name best \
    --batch-size "$SAMEFAMILY_BATCH_SIZE" \
    --feature-cache-root "$FEATURE_CACHE_ROOT" \
    --feature-cache-mode read_write \
    --require-cuda
  latest_timestamped_dir "${SAMEFAMILY_OUTPUT_ROOT}/${run_name}"
}

run_benchmark_eval() {
  local case_id="$1"
  local benchmark_id="$2"
  local value_run_dir="$3"
  local run_name="${RUN_PREFIX}_${case_id}_${benchmark_id}"
  run_cmd "bench_${case_id}_${benchmark_id}" \
    "$PYTHON_BIN" -u scripts/phase_e_eval_benchmark.py \
    --value-run-dir "$value_run_dir" \
    --benchmark-id "$benchmark_id" \
    --run-name "$run_name" \
    --output-root "$BENCH_OUTPUT_ROOT" \
    --max-samples "$BENCH_MAX_SAMPLES" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --feature-cache-root "$FEATURE_CACHE_ROOT" \
    --feature-cache-mode read_write \
    --processbench-f1-threshold 0.5 \
    --require-cuda
  latest_timestamped_dir "${BENCH_OUTPUT_ROOT}/${run_name}"
}

collect_result() {
  local case_id="$1"
  local variant="$2"
  local data_mode="$3"
  local value_run_dir="$4"
  local samefamily_dir="$5"
  local gsm_dir="$6"
  local math_dir="$7"
  "$PYTHON_BIN" - \
    "$case_id" "$variant" "$data_mode" "$value_run_dir" "$samefamily_dir" "$gsm_dir" "$math_dir" \
    "$RESULTS_JSONL" <<'PY'
from pathlib import Path
import json, sys
case_id, variant, data_mode, value_run_dir, samefamily_dir, gsm_dir, math_dir, out_path = sys.argv[1:]
value_summary = json.loads((Path(value_run_dir) / "summary.json").read_text(encoding="utf-8"))
samefamily_summary = json.loads((Path(samefamily_dir) / "summary.json").read_text(encoding="utf-8"))
gsm_summary = json.loads((Path(gsm_dir) / "summary.json").read_text(encoding="utf-8"))
math_summary = json.loads((Path(math_dir) / "summary.json").read_text(encoding="utf-8"))
row = {
    "case_id": case_id,
    "variant": variant,
    "data_mode": data_mode,
    "value_run_dir": value_run_dir,
    "samefamily_dir": samefamily_dir,
    "gsm_dir": gsm_dir,
    "math_dir": math_dir,
    "heldout_pair_acc": float(value_summary["eval_pairs"]["pair_accuracy"]),
    "heldout_auc": float(value_summary["eval_pairs"]["auc"]),
    "samefamily_top1": float(samefamily_summary["pool_metrics"]["top1_accuracy"]),
    "samefamily_local_first_bad": float(samefamily_summary["local_metrics"]["first_bad_accuracy"]),
    "samefamily_rej40_gain": float((samefamily_summary.get("rejection_curve") or {}).get("coverage_0.40", {}).get("gain_over_full", 0.0)),
    "gsm_auc": float(gsm_summary["metrics"]["pair_auc_good_vs_bad"]),
    "gsm_first_edge": float(gsm_summary["metrics"]["first_error_edge_accuracy"]),
    "math_auc": float(math_summary["metrics"]["pair_auc_good_vs_bad"]),
    "math_first_edge": float(math_summary["metrics"]["first_error_edge_accuracy"]),
}
with Path(out_path).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps(row, ensure_ascii=False))
PY
}

mkdir -p "$LOG_ROOT"
: > "$SUITE_LOG_FILE"
: > "$RESULTS_JSONL"

if [[ ! -d "$SOURCE_PAIR_ARTIFACT" ]]; then
  echo "ERROR: SOURCE_PAIR_ARTIFACT not found: $SOURCE_PAIR_ARTIFACT" >&2
  exit 1
fi

case "$ACTIVE_PHASE_E_FRONTIER_GROUP" in
  F1_JUDGE_FILTER_PBR10)
    log_line "Phase E frontier case: judge-filtered frozen MLP on PBR10" | tee -a "$SUITE_LOG_FILE"
    JUDGE_RUN_NAME="${RUN_PREFIX}_judgefilter"
    run_cmd judge_filter \
      "$PYTHON_BIN" -u scripts/phase_e_judge_filter_pairs.py \
      --train-pairs-jsonl "${SOURCE_PAIR_ARTIFACT}/train_pairs.jsonl" \
      --eval-pairs-jsonl "${SOURCE_PAIR_ARTIFACT}/validation_pairs.jsonl" \
      --model-path "$JUDGE_MODEL_PATH" \
      --run-name "$JUDGE_RUN_NAME" \
      --output-root "$PAIR_OUTPUT_ROOT" \
      --batch-size "$JUDGE_BATCH_SIZE" \
      --max-new-tokens "$JUDGE_MAX_NEW_TOKENS" \
      --dtype bfloat16 \
      --device-map auto \
      --min-confidence "$JUDGE_MIN_CONFIDENCE" \
      --require-cuda
    FILTERED_PAIR_DIR="$(latest_run_dir "${PAIR_OUTPUT_ROOT}/${JUDGE_RUN_NAME}")"

    VALUE_RUN_NAME="${RUN_PREFIX}_judge_mlp"
    run_cmd train_judge_mlp \
      "$PYTHON_BIN" -u scripts/phase_e_train_value.py \
      --train-pairs-jsonl "${FILTERED_PAIR_DIR}/train_pairs.jsonl" \
      --eval-pairs-jsonl "${FILTERED_PAIR_DIR}/validation_pairs.jsonl" \
      --model-path "$MODEL_PATH" \
      --run-name "$VALUE_RUN_NAME" \
      --output-root "$VALUE_OUTPUT_ROOT" \
      --objective-mode joint \
      --learning-rate "$LEARNING_RATE" \
      --num-train-epochs "$TRAIN_EPOCHS" \
      --per-device-train-batch-size "$TRAIN_BATCH_SIZE" \
      --per-device-eval-batch-size "$EVAL_BATCH_SIZE" \
      --max-length "$MAX_LENGTH" \
      --lambda-ranking 1.0 \
      --lambda-bce 1.0 \
      --ranking-margin 0.02 \
      --ranking-target-space score \
      --pair-weight-mode none \
      --source-balance none \
      --permutation-mode stable_hash \
      --checkpoint-selection-metric pair_acc \
      --recipe-risk-policy "$RECIPE_RISK_POLICY" \
      --head-architecture mlp \
      --head-mlp-hidden-size "$HEAD_MLP_HIDDEN_SIZE" \
      --head-dropout-prob "$HEAD_DROPOUT_PROB" \
      --head-init-std 0.02 \
      --head-activation gelu \
      --anti-saturation-weight 5e-4 \
      --anti-saturation-logit-threshold 3.5 \
      --feature-cache-root "$FEATURE_CACHE_ROOT" \
      --feature-cache-mode read_write \
      --require-cuda
    VALUE_RUN_DIR="$(latest_timestamped_dir "${VALUE_OUTPUT_ROOT}/${VALUE_RUN_NAME}")"
    SAMEFAMILY_DIR="$(run_samefamily_eval "judge_mlp" "$VALUE_RUN_DIR" "${FILTERED_PAIR_DIR}/validation_pairs.jsonl")"
    GSM_DIR="$(run_benchmark_eval "judge_mlp" "processbench_gsm8k" "$VALUE_RUN_DIR")"
    MATH_DIR="$(run_benchmark_eval "judge_mlp" "processbench_math" "$VALUE_RUN_DIR")"
    collect_result "judge_mlp" "frozen_mlp" "judge_filtered" "$VALUE_RUN_DIR" "$SAMEFAMILY_DIR" "$GSM_DIR" "$MATH_DIR" | tee -a "$SUITE_LOG_FILE"
    ;;
  F2_DUAL_HEAD_PBR10)
    log_line "Phase E frontier case: dual-head routing on PBR10" | tee -a "$SUITE_LOG_FILE"
    VALUE_RUN_NAME="${RUN_PREFIX}_dual_head"
    run_cmd train_dual_head \
      "$PYTHON_BIN" -u scripts/phase_e_train_value.py \
      --train-pairs-jsonl "${SOURCE_PAIR_ARTIFACT}/train_pairs.jsonl" \
      --eval-pairs-jsonl "${SOURCE_PAIR_ARTIFACT}/validation_pairs.jsonl" \
      --model-path "$MODEL_PATH" \
      --run-name "$VALUE_RUN_NAME" \
      --output-root "$VALUE_OUTPUT_ROOT" \
      --objective-mode joint \
      --learning-rate "$LEARNING_RATE" \
      --num-train-epochs "$TRAIN_EPOCHS" \
      --per-device-train-batch-size "$TRAIN_BATCH_SIZE" \
      --per-device-eval-batch-size "$EVAL_BATCH_SIZE" \
      --max-length "$MAX_LENGTH" \
      --lambda-ranking 1.0 \
      --lambda-bce 1.0 \
      --terminal-bce-lambda "$DUAL_TERMINAL_BCE_LAMBDA" \
      --ranking-margin 0.02 \
      --ranking-target-space score \
      --pair-weight-mode none \
      --source-balance none \
      --permutation-mode stable_hash \
      --checkpoint-selection-metric pair_acc \
      --recipe-risk-policy "$RECIPE_RISK_POLICY" \
      --head-architecture dual_head \
      --head-mlp-hidden-size "$HEAD_MLP_HIDDEN_SIZE" \
      --head-dropout-prob "$HEAD_DROPOUT_PROB" \
      --head-init-std 0.02 \
      --head-activation gelu \
      --head-inference-alpha 0.5 \
      --anti-saturation-weight 5e-4 \
      --anti-saturation-logit-threshold 3.5 \
      --feature-cache-root "$FEATURE_CACHE_ROOT" \
      --feature-cache-mode read_write \
      --require-cuda
    VALUE_RUN_DIR="$(latest_timestamped_dir "${VALUE_OUTPUT_ROOT}/${VALUE_RUN_NAME}")"
    SAMEFAMILY_DIR="$(run_samefamily_eval "dual_head" "$VALUE_RUN_DIR" "${SOURCE_PAIR_ARTIFACT}/validation_pairs.jsonl")"
    GSM_DIR="$(run_benchmark_eval "dual_head" "processbench_gsm8k" "$VALUE_RUN_DIR")"
    MATH_DIR="$(run_benchmark_eval "dual_head" "processbench_math" "$VALUE_RUN_DIR")"
    collect_result "dual_head" "dual_head" "raw_pbr10" "$VALUE_RUN_DIR" "$SAMEFAMILY_DIR" "$GSM_DIR" "$MATH_DIR" | tee -a "$SUITE_LOG_FILE"
    ;;
  F3_LORA_PBR10)
    log_line "Phase E frontier case: minimal LoRA on PBR10" | tee -a "$SUITE_LOG_FILE"
    VALUE_RUN_NAME="${RUN_PREFIX}_lora_mlp"
    run_cmd train_lora_mlp \
      "$PYTHON_BIN" -u scripts/phase_e_train_value_lora.py \
      --train-pairs-jsonl "${SOURCE_PAIR_ARTIFACT}/train_pairs.jsonl" \
      --eval-pairs-jsonl "${SOURCE_PAIR_ARTIFACT}/validation_pairs.jsonl" \
      --model-path "$MODEL_PATH" \
      --run-name "$VALUE_RUN_NAME" \
      --output-root "$VALUE_OUTPUT_ROOT" \
      --objective-mode joint \
      --learning-rate "$LORA_LEARNING_RATE" \
      --num-train-epochs "$LORA_EPOCHS" \
      --per-device-train-batch-size "$LORA_TRAIN_BATCH_SIZE" \
      --per-device-eval-batch-size "$LORA_EVAL_BATCH_SIZE" \
      --gradient-accumulation-steps "$LORA_GRAD_ACCUM" \
      --max-length "$LORA_MAX_LENGTH" \
      --lambda-ranking 1.0 \
      --lambda-bce 1.0 \
      --ranking-margin 0.02 \
      --ranking-target-space score \
      --pair-weight-mode none \
      --source-balance none \
      --permutation-mode stable_hash \
      --checkpoint-selection-metric pair_acc \
      --recipe-risk-policy "$RECIPE_RISK_POLICY" \
      --head-architecture mlp \
      --head-mlp-hidden-size "$HEAD_MLP_HIDDEN_SIZE" \
      --head-dropout-prob "$HEAD_DROPOUT_PROB" \
      --head-init-std 0.02 \
      --head-activation gelu \
      --anti-saturation-weight 5e-4 \
      --anti-saturation-logit-threshold 3.5 \
      --lora-rank "$LORA_RANK" \
      --lora-alpha "$LORA_ALPHA" \
      --lora-dropout "$LORA_DROPOUT" \
      --lora-top-k-layers "$LORA_TOP_K_LAYERS" \
      --gradient-checkpointing \
      --require-cuda
    VALUE_RUN_DIR="$(latest_timestamped_dir "${VALUE_OUTPUT_ROOT}/${VALUE_RUN_NAME}")"
    SAMEFAMILY_DIR="$(run_samefamily_eval "lora_mlp" "$VALUE_RUN_DIR" "${SOURCE_PAIR_ARTIFACT}/validation_pairs.jsonl")"
    GSM_DIR="$(run_benchmark_eval "lora_mlp" "processbench_gsm8k" "$VALUE_RUN_DIR")"
    MATH_DIR="$(run_benchmark_eval "lora_mlp" "processbench_math" "$VALUE_RUN_DIR")"
    collect_result "lora_mlp" "lora_mlp" "raw_pbr10" "$VALUE_RUN_DIR" "$SAMEFAMILY_DIR" "$GSM_DIR" "$MATH_DIR" | tee -a "$SUITE_LOG_FILE"
    ;;
  *)
    echo "ERROR: Unknown ACTIVE_PHASE_E_FRONTIER_GROUP=$ACTIVE_PHASE_E_FRONTIER_GROUP" >&2
    exit 1
    ;;
esac

"$PYTHON_BIN" - "$RESULTS_JSONL" "$SUMMARY_FILE" <<'PY'
from pathlib import Path
import json, sys
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
lines = [
    "# Phase E Frontier Suite Summary",
    "",
]
if not rows:
    lines.append("- status: no_results")
else:
    lines.extend([
        "| case_id | variant | data_mode | heldout_pair_acc | heldout_auc | sf_top1 | sf_local | sf_rej40_gain | gsm_auc | gsm_first_edge | math_auc | math_first_edge |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            "| {case_id} | {variant} | {data_mode} | {heldout_pair_acc:.4f} | {heldout_auc:.4f} | {samefamily_top1:.4f} | {samefamily_local_first_bad:.4f} | {samefamily_rej40_gain:.4f} | {gsm_auc:.4f} | {gsm_first_edge:.4f} | {math_auc:.4f} | {math_first_edge:.4f} |".format(**row)
        )
Path(sys.argv[2]).write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

log_line "Frontier suite summary written: $SUMMARY_FILE" | tee -a "$SUITE_LOG_FILE"
