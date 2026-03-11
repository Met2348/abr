#!/usr/bin/env bash
# Changelog (prepend-only, newest first):
# 2026-03-11: Initial creation. Same NDS1-NDS4 data profiles as run_phase_e_newdataset_suite.sh
#             but using the PROVEN best hyperparameters from ms_e43 experiment:
#             --ranking-target-space score
#             --pair-weight-mode none
#             --checkpoint-selection-metric pair_acc
#             --num-train-epochs 10
#             These changes fix the training failure seen with logit+confidence_semantic config.
#
# NDS Best-HParams Suite (NDSBH) — 使用最优超参重测新数据集
#
# English
# -------
# Same 4 data profiles as NDS suite, but with proven best hyperparameters:
#   - score target space (not logit) → avoids gradient collapse at initialization
#   - no pair weighting (not confidence_semantic) → consistent gradient signal
#   - pair_acc checkpoint selection (not ranking_score) → select best generalizing model
#   - 10 epochs (not 5) → allows convergence
#
# Experiment groups (same as NDS):
#   NDSBH1: rlhflow_align_v1 — RLHFlow-Deepseek LLM-judge labels
#   NDSBH2: math_step_dpo_v1 — Math-Step-DPO fork-point pairs + Math-Shepherd anchor
#   NDSBH3: ms_align_v1 (baseline) — same scale as NDSBH1/2
#   NDSBH4: ms_rlhflow_mixed_v1 — Mixed sources
#   NDSBH_MS_ACC90: ms_acc90 (best known baseline, ~14K pairs, for reference)
#
# 中文
# ----
# 使用最优超参（score空间, 无权重, pair_acc选择, 10epoch）重测NDS1-4各数据集，
# 将数据质量差异与超参差异解耦，确认新数据集相对于Math-Shepherd基线的真实增益。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACTIVE_PHASE_E_NDSBH_GROUP="${ACTIVE_PHASE_E_NDSBH_GROUP:-NDSBH1_RLHFLOW_SMOKE}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_ndsbh}"

MODEL_PATH="${MODEL_PATH:-assets/models/Qwen2.5-7B-Instruct}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-assets/artifacts/phase_e_feature_cache}"
PAIR_OUTPUT_ROOT="${PAIR_OUTPUT_ROOT:-assets/artifacts/phase_e_pairs}"
VALUE_OUTPUT_ROOT="${VALUE_OUTPUT_ROOT:-assets/artifacts/phase_e_runs}"
BENCH_OUTPUT_ROOT="${BENCH_OUTPUT_ROOT:-assets/artifacts/phase_e_eval}"
LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"

TARGET_TOTAL_PAIRS="${TARGET_TOTAL_PAIRS:-4096}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
HEAD_MLP_HIDDEN_SIZE="${HEAD_MLP_HIDDEN_SIZE:-1024}"
BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-128}"

# GPU device
CUDA_DEVICE="${CUDA_DEVICE:-2}"

CURRENT_STAGE="bootstrap"

GROUP_TITLE=""
GROUP_INTENTION=""
GROUP_CASES=()

log_line() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1"
}

on_exit() {
  local exit_code=$?
  mkdir -p "$LOG_ROOT"
  if [[ $exit_code -ne 0 ]]; then
    {
      echo "# Phase E NDSBH Suite Summary"
      echo "- group_id: ${ACTIVE_PHASE_E_NDSBH_GROUP}"
      echo "- status: FAILED at stage: ${CURRENT_STAGE}"
    } > "$SUMMARY_FILE"
  fi
}
trap on_exit EXIT

resolve_group() {
  case "$ACTIVE_PHASE_E_NDSBH_GROUP" in
    NDSBH1_RLHFLOW_SMOKE)
      GROUP_TITLE="NDSBH1 RLHFlow-Deepseek (Best HParams)"
      GROUP_INTENTION="Test RLHFlow-Deepseek at 4096 pairs with proven score-space hyperparameters. Direct comparison to NDS1 result reveals hparam vs data quality contribution."
      GROUP_CASES=("ndsbh1_rlhflow_align_mlp|rlhflow_align_v1")
      ;;
    NDSBH2_MATH_STEP_DPO_SMOKE)
      GROUP_TITLE="NDSBH2 Math-Step-DPO (Best HParams)"
      GROUP_INTENTION="Test Math-Step-DPO at 4096 pairs with proven hyperparameters."
      GROUP_CASES=("ndsbh2_math_step_dpo_mlp|math_step_dpo_v1")
      ;;
    NDSBH3_MS_BASELINE_SMOKE)
      GROUP_TITLE="NDSBH3 Math-Shepherd ms_align_v1 (Best HParams)"
      GROUP_INTENTION="Reproduce ms_align_v1 baseline at 4096 pairs with proven hyperparameters. Shows whether data mix vs scale vs hparams explains gap vs ms_acc90."
      GROUP_CASES=("ndsbh3_ms_align_baseline_mlp|ms_align_v1")
      ;;
    NDSBH4_MIXED_SMOKE)
      GROUP_TITLE="NDSBH4 Mixed Sources (Best HParams)"
      GROUP_INTENTION="Test mixed ms_rlhflow_mixed_v1 at 4096 pairs with proven hyperparameters."
      GROUP_CASES=("ndsbh4_ms_rlhflow_mixed_mlp|ms_rlhflow_mixed_v1")
      ;;
    NDSBH_ALL_SMOKE)
      GROUP_TITLE="NDSBH All Groups (Best HParams)"
      GROUP_INTENTION="Run all 4 NDSBH variants with proven hyperparameters for direct comparison. Also tests ms_acc90 at same scale for controlled ablation."
      GROUP_CASES=(
        "ndsbh1_rlhflow_align_mlp|rlhflow_align_v1"
        "ndsbh2_math_step_dpo_mlp|math_step_dpo_v1"
        "ndsbh3_ms_align_baseline_mlp|ms_align_v1"
        "ndsbh4_ms_rlhflow_mixed_mlp|ms_rlhflow_mixed_v1"
      )
      ;;
    NDSBH_MS_ACC90)
      GROUP_TITLE="NDSBH MS-ACC90 Baseline (Same Scale)"
      GROUP_INTENTION="Run ms_acc90 at same 4096-pair scale as NDSBH1-4 to isolate data quality effect from scale effect vs full 14K ms_e43 experiment."
      GROUP_CASES=("ndsbh0_ms_acc90_mlp|ms_acc90")
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_E_NDSBH_GROUP=$ACTIVE_PHASE_E_NDSBH_GROUP" >&2
      echo "  Valid: NDSBH1_RLHFLOW_SMOKE, NDSBH2_MATH_STEP_DPO_SMOKE, NDSBH3_MS_BASELINE_SMOKE, NDSBH4_MIXED_SMOKE, NDSBH_ALL_SMOKE, NDSBH_MS_ACC90" >&2
      exit 1
      ;;
  esac
}

run_curate() {
  local case_id="$1"
  local profile="$2"
  local run_name="${RUN_PREFIX}_${case_id}_${profile}_pairs"
  local cmd=(
    "$PYTHON_BIN" -u scripts/phase_e_curate_processbench_transfer_pairs.py
    --profile "$profile"
    --run-name "$run_name"
    --output-root "$PAIR_OUTPUT_ROOT"
    --seed 42
    --validation-ratio 0.1
    --split-granularity source_sample
    --target-total-pairs "$TARGET_TOTAL_PAIRS"
    --min-pair-confidence 0.55
  )
  # ms_acc90 profile uses the older preparation script, handle separately
  if [[ "$profile" == "ms_acc90" ]]; then
    local ms_cmd=(
      "$PYTHON_BIN" -u scripts/phase_e_prepare_pairs.py
      --source math_shepherd
      --math-shepherd-path assets/external_datasets/peiyi_math_shepherd/math-shepherd.jsonl
      --run-name "$run_name"
      --output-root "$PAIR_OUTPUT_ROOT"
      --seed 42
      --validation-ratio 0.1
      --split-granularity source_sample
      --max-local-pairs "$TARGET_TOTAL_PAIRS"
      --min-pair-confidence 0.74
      --step-label-pair-mode first_bad_edge_strict
    )
    CURRENT_STAGE="curate_${case_id}"
    log_line "RUN (ms_acc90): ${ms_cmd[*]}" | tee -a "$SUITE_LOG_FILE" >&2
    "${ms_cmd[@]}" | tee -a "$SUITE_LOG_FILE" >&2
  else
    CURRENT_STAGE="curate_${case_id}"
    log_line "RUN: ${cmd[*]}" | tee -a "$SUITE_LOG_FILE" >&2
    "${cmd[@]}" | tee -a "$SUITE_LOG_FILE" >&2
  fi
  # Return the artifact directory
  "$PYTHON_BIN" - "$PAIR_OUTPUT_ROOT" "$run_name" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
prefix = sys.argv[2]
matches = sorted(root.glob(f"{prefix}__*"))
if not matches:
    raise SystemExit(f"No artifact directory matches prefix: {prefix}")
print(matches[-1])
PY
}

run_train() {
  local case_id="$1"
  local train_jsonl="$2"
  local eval_jsonl="$3"
  local run_name="${RUN_PREFIX}_${case_id}_value"
  local cmd=(
    "$PYTHON_BIN" -u scripts/phase_e_train_value.py
    --train-pairs-jsonl "$train_jsonl"
    --eval-pairs-jsonl "$eval_jsonl"
    --model-path "$MODEL_PATH"
    --run-name "$run_name"
    --output-root "$VALUE_OUTPUT_ROOT"
    --objective-mode joint
    --learning-rate "$LEARNING_RATE"
    --num-train-epochs "$TRAIN_EPOCHS"
    --per-device-train-batch-size "$TRAIN_BATCH_SIZE"
    --per-device-eval-batch-size "$EVAL_BATCH_SIZE"
    --max-length "$MAX_LENGTH"
    --lambda-ranking 1.0
    --lambda-bce 1.0
    --ranking-margin 0.02
    --ranking-target-space score
    --pair-weight-mode none
    --source-balance none
    --permutation-mode stable_hash
    --checkpoint-selection-metric pair_acc
    --seed 42
    --feature-cache-root "$FEATURE_CACHE_ROOT"
    --feature-cache-mode read_write
    --feature-cache-lock-timeout-sec 600
    --head-architecture mlp
    --head-mlp-hidden-size "$HEAD_MLP_HIDDEN_SIZE"
    --head-dropout-prob 0.05
    --head-init-std 0.02
    --head-activation gelu
    --anti-saturation-weight 5e-4
    --anti-saturation-logit-threshold 3.5
    --nonfinite-feature-policy drop
    --require-cuda
  )
  CURRENT_STAGE="train_${case_id}"
  log_line "RUN (GPU ${CUDA_DEVICE}): ${cmd[*]}" | tee -a "$SUITE_LOG_FILE" >&2
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${cmd[@]}" | tee -a "$SUITE_LOG_FILE" >&2
  # Return latest run dir
  "$PYTHON_BIN" - "$VALUE_OUTPUT_ROOT" "$run_name" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
prefix = sys.argv[2]
matches = sorted(root.glob(f"{prefix}_*"))
if not matches:
    raise SystemExit(f"No value run dir matches prefix: {prefix}")
print(matches[-1])
PY
}

run_bench() {
  local case_id="$1"
  local value_run_dir="$2"
  local benchmark_id="$3"
  local run_name="${RUN_PREFIX}_${case_id}_${benchmark_id}"
  local cmd=(
    "$PYTHON_BIN" -u scripts/phase_e_eval_benchmark.py
    --value-run-dir "$value_run_dir"
    --benchmark-id "$benchmark_id"
    --run-name "$run_name"
    --output-root "$BENCH_OUTPUT_ROOT"
    --max-samples "$BENCH_MAX_SAMPLES"
    --batch-size "$EVAL_BATCH_SIZE"
    --require-cuda
  )
  CURRENT_STAGE="bench_${case_id}_${benchmark_id}"
  log_line "RUN (GPU ${CUDA_DEVICE}): ${cmd[*]}" | tee -a "$SUITE_LOG_FILE" >&2
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${cmd[@]}" | tee -a "$SUITE_LOG_FILE" >&2
  "$PYTHON_BIN" - "$BENCH_OUTPUT_ROOT" "$run_name" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
prefix = sys.argv[2]
matches = sorted(root.glob(f"{prefix}_*"))
if not matches:
    raise SystemExit(f"No bench dir matches prefix: {prefix}")
print(matches[-1])
PY
}

# === Main ===

mkdir -p "$LOG_ROOT"
resolve_group

log_line "Suite start: ${ACTIVE_PHASE_E_NDSBH_GROUP} — ${GROUP_TITLE}" | tee -a "$SUITE_LOG_FILE" >&2
log_line "Intention: ${GROUP_INTENTION}" | tee -a "$SUITE_LOG_FILE" >&2
log_line "Using CUDA_DEVICE=${CUDA_DEVICE}" | tee -a "$SUITE_LOG_FILE" >&2
log_line "HParams: score-space, no-pair-weight, pair_acc, ${TRAIN_EPOCHS} epochs, batch=${TRAIN_BATCH_SIZE}" | tee -a "$SUITE_LOG_FILE" >&2

declare -A CASE_BENCH_GSM=()
declare -A CASE_BENCH_MATH=()

for case_spec in "${GROUP_CASES[@]}"; do
  IFS='|' read -r case_id profile <<< "$case_spec"
  log_line "=== Case: ${case_id} (profile: ${profile}) ===" | tee -a "$SUITE_LOG_FILE" >&2

  # 1. Curate pairs
  CURRENT_STAGE="curate_${case_id}"
  pair_dir=$(run_curate "$case_id" "$profile")
  train_jsonl="${pair_dir}/train_pairs.jsonl"
  eval_jsonl="${pair_dir}/validation_pairs.jsonl"

  # 2. Train value head
  CURRENT_STAGE="train_${case_id}"
  value_dir=$(run_train "$case_id" "$train_jsonl" "$eval_jsonl")

  # 3. Eval on ProcessBench
  CURRENT_STAGE="bench_${case_id}_gsm8k"
  bench_gsm_dir=$(run_bench "$case_id" "$value_dir" "processbench_gsm8k")
  CASE_BENCH_GSM[$case_id]="$bench_gsm_dir"

  CURRENT_STAGE="bench_${case_id}_math"
  bench_math_dir=$(run_bench "$case_id" "$value_dir" "processbench_math")
  CASE_BENCH_MATH[$case_id]="$bench_math_dir"

  log_line "Case ${case_id} done." | tee -a "$SUITE_LOG_FILE" >&2
done

# === Summary ===

{
  echo "# Phase E NDSBH Suite Summary"
  echo ""
  echo "- group_id: ${ACTIVE_PHASE_E_NDSBH_GROUP}"
  echo "- group_title: ${GROUP_TITLE}"
  echo "- run_prefix: ${RUN_PREFIX}"
  echo "- status: completed"
  echo "- suite_log_file: ${SUITE_LOG_FILE}"
  echo "- cuda_device: ${CUDA_DEVICE}"
  echo "- hparams: score-space | no-pair-weight | pair_acc | ${TRAIN_EPOCHS} epochs | batch=${TRAIN_BATCH_SIZE}"
  echo ""
  echo "## Results"
  echo ""
  echo "| case_id | pb_gsm_pair_acc | pb_gsm_auc | pb_gsm_first_edge | pb_math_pair_acc | pb_math_auc | pb_math_first_edge |"
  echo "|---|---|---|---|---|---|---|"
} > "$SUMMARY_FILE"

for case_spec in "${GROUP_CASES[@]}"; do
  IFS='|' read -r case_id profile <<< "$case_spec"
  gsm_dir="${CASE_BENCH_GSM[$case_id]:-}"
  math_dir="${CASE_BENCH_MATH[$case_id]:-}"
  "$PYTHON_BIN" - "$case_id" "$gsm_dir" "$math_dir" "$SUMMARY_FILE" <<'PY'
from pathlib import Path
import json, sys
case_id, gsm_dir, math_dir, summary_file = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
def load(d, key, default=0.0):
    fp = d / "metrics.json"
    if not fp.exists():
        return default
    return json.loads(fp.read_text()).get(key, default)
row = (
    f"| {case_id} "
    f"| {load(gsm_dir,'pair_accuracy_good_vs_bad'):.4f} "
    f"| {load(gsm_dir,'pair_auc_good_vs_bad'):.4f} "
    f"| {load(gsm_dir,'first_error_edge_accuracy'):.4f} "
    f"| {load(math_dir,'pair_accuracy_good_vs_bad'):.4f} "
    f"| {load(math_dir,'pair_auc_good_vs_bad'):.4f} "
    f"| {load(math_dir,'first_error_edge_accuracy'):.4f} |"
)
with open(summary_file, "a") as f:
    f.write(row + "\n")
print(row)
PY
done

log_line "Suite completed: ${SUMMARY_FILE}" | tee -a "$SUITE_LOG_FILE" >&2
cat "$SUMMARY_FILE" >&2

CURRENT_STAGE="done"
