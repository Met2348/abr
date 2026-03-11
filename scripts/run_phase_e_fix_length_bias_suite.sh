#!/usr/bin/env bash
# Changelog (prepend-only, newest first):
# 2026-03-11: Initial creation. Length-bias fix experiment suite.
#             Root cause confirmed: 20.1% of ms_align_v1 pairs are fanout/grid types
#             with rej-cho=+194/+203 → teach "shorter=better" → inverted on ProcessBench.
#             4 fix variants tested:
#               FIX_A: ms_strict_only_v1 — MS without fanout/grid (ablation)
#               FIX_B: ms_dpo_calibrated_v1 — DPO 50% + MS strict 40% (calibrated mix)
#               FIX_C: dpo_scale_v1 — pure DPO at 8192 pairs (scale test)
#               FIX_D: rlh_strict_only_v1 — RLH without fanout/grid (quality test)
#
# Length-Bias Fix Suite — 长度偏差修复实验
#
# English
# -------
# NDSBH diagnosis showed:
#   NDS3 (ms_align_v1): 20.1% fanout/grid pairs with rej-cho=+194/+203 → teach "shorter=better"
#   ProcessBench: bad_prefix is LONGER than good_prefix → model inverts (MATH AUC=0.470)
#   NDS1 (rlhflow_align_v1): 43% fanout/grid pairs → even worse (MATH AUC=0.552)
#   NDS2 (math_step_dpo_v1): 0% length-biased pairs → correct transfer (MATH AUC=0.712 SOTA)
#
# Fix strategy:
#   FIX_A: Remove ALL fanout/grid from MS → tests if strict-only MS works at small scale
#   FIX_B: 50% DPO anchor + 40% MS strict → DPO calibrates against length bias
#   FIX_C: Pure DPO at 8192 pairs → tests scale effect on frozen-head ceiling
#   FIX_D: RLH strict-only → LLM-judge quality without length contamination
#
# 中文
# ----
# 长度偏差修复实验：诊断确认了 NDS3 失败的根本原因是 20.1% 的 fanout/grid pair
# 教模型学到"更短=更好"的长度捷径，而 ProcessBench 中 bad_prefix 比 good_prefix 长，
# 导致评分倒置。本套件测试 4 种修复策略，目标是恢复 MS/RLH 在小规模下的正向迁移。
#
# All 4 fix cases run on GPU 3 (81GB free) with proven best hparams (score-space, pair_acc).
# Expected total time: ~3-4 hours (4 × curate + train + eval).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACTIVE_PHASE_E_FLB_GROUP="${ACTIVE_PHASE_E_FLB_GROUP:-FLB_ALL_SMOKE}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_flb_0311}"

MODEL_PATH="${MODEL_PATH:-assets/models/Qwen2.5-7B-Instruct}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-assets/artifacts/phase_e_feature_cache}"
PAIR_OUTPUT_ROOT="${PAIR_OUTPUT_ROOT:-assets/artifacts/phase_e_pairs}"
VALUE_OUTPUT_ROOT="${VALUE_OUTPUT_ROOT:-assets/artifacts/phase_e_runs}"
BENCH_OUTPUT_ROOT="${BENCH_OUTPUT_ROOT:-assets/artifacts/phase_e_eval}"
LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"

# Default hyperparameters — proven best config from NDSBH suite
TARGET_TOTAL_PAIRS="${TARGET_TOTAL_PAIRS:-4096}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
HEAD_MLP_HIDDEN_SIZE="${HEAD_MLP_HIDDEN_SIZE:-1024}"
BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-256}"
RECIPE_RISK_POLICY="${RECIPE_RISK_POLICY:-error}"

# GPU device — GPU 3 is free (81GB)
CUDA_DEVICE="${CUDA_DEVICE:-3}"

CURRENT_STAGE="bootstrap"

GROUP_TITLE=""
GROUP_INTENTION=""
# case_spec format: case_id|profile|pairs_count
# pairs_count overrides TARGET_TOTAL_PAIRS for this case (useful for FIX_C 8K scale)
GROUP_CASES=()

log_line() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1"
}

on_exit() {
  local exit_code=$?
  mkdir -p "$LOG_ROOT"
  if [[ $exit_code -ne 0 ]]; then
    {
      echo "# Phase E FLB Suite Summary"
      echo "- group_id: ${ACTIVE_PHASE_E_FLB_GROUP}"
      echo "- status: FAILED at stage: ${CURRENT_STAGE}"
    } > "$SUMMARY_FILE"
  fi
}
trap on_exit EXIT

resolve_group() {
  case "$ACTIVE_PHASE_E_FLB_GROUP" in
    FLB_A_MS_STRICT)
      GROUP_TITLE="FIX A — MS Strict-Only (No Length-Bias Pairs)"
      GROUP_INTENTION="Ablation: MS without any fanout/grid pair types. If NDS3 failure was ENTIRELY due to length-biased pairs, strict-only should recover transfer even at small scale."
      GROUP_CASES=("fix_a_ms_strict_only|ms_strict_only_v1|4096")
      ;;
    FLB_B_MS_DPO_CALIBRATED)
      GROUP_TITLE="FIX B — MS + DPO Calibrated (Length-Bias Corrected)"
      GROUP_INTENTION="DPO sibling_branch pairs (rej-cho≈0) anchor the model to content quality, preventing MS strict from inheriting length shortcuts. Tests if calibrated mix outperforms pure DPO (NDS2)."
      GROUP_CASES=("fix_b_ms_dpo_calibrated|ms_dpo_calibrated_v1|4096")
      ;;
    FLB_C_DPO_SCALE)
      GROUP_TITLE="FIX C — Pure DPO Scale (8192 pairs)"
      GROUP_INTENTION="Scale NDS2's winning pure-DPO profile from 3705 to 8192 pairs. Tests if DPO data scale can push frozen-head MATH AUC beyond the current 0.712 ceiling."
      GROUP_CASES=("fix_c_dpo_scale_8k|dpo_scale_v1|8192")
      ;;
    FLB_D_RLH_STRICT)
      GROUP_TITLE="FIX D — RLHFlow Strict-Only (No Length-Bias Pairs)"
      GROUP_INTENTION="Remove ALL fanout/grid from RLHFlow (NDS1 had 43% length-biased pairs → MATH AUC=0.552). Tests if high-quality LLM-judge labels, freed from length contamination, can match NDS2."
      GROUP_CASES=("fix_d_rlh_strict_only|rlh_strict_only_v1|4096")
      ;;
    FLB_ALL_SMOKE)
      GROUP_TITLE="FLB All Fix Cases (Length-Bias Ablation)"
      GROUP_INTENTION="Run all 4 length-bias fix variants: strict-only MS, DPO-calibrated MS, pure-DPO at scale, strict-only RLH. Comprehensive test of all length-bias elimination strategies."
      GROUP_CASES=(
        "fix_a_ms_strict_only|ms_strict_only_v1|4096"
        "fix_b_ms_dpo_calibrated|ms_dpo_calibrated_v1|4096"
        "fix_c_dpo_scale_8k|dpo_scale_v1|8192"
        "fix_d_rlh_strict_only|rlh_strict_only_v1|4096"
      )
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_E_FLB_GROUP=$ACTIVE_PHASE_E_FLB_GROUP" >&2
      echo "  Valid: FLB_A_MS_STRICT, FLB_B_MS_DPO_CALIBRATED, FLB_C_DPO_SCALE, FLB_D_RLH_STRICT, FLB_ALL_SMOKE" >&2
      exit 1
      ;;
  esac
}

run_curate() {
  local case_id="$1"
  local profile="$2"
  local case_pairs="$3"
  local run_name="${RUN_PREFIX}_${case_id}_${profile}_pairs"
  local cmd=(
    "$PYTHON_BIN" -u scripts/phase_e_curate_processbench_transfer_pairs.py
    --profile "$profile"
    --run-name "$run_name"
    --output-root "$PAIR_OUTPUT_ROOT"
    --seed 42
    --validation-ratio 0.1
    --split-granularity source_sample
    --target-total-pairs "$case_pairs"
    --min-pair-confidence 0.55
  )
  CURRENT_STAGE="curate_${case_id}"
  log_line "RUN: ${cmd[*]}" | tee -a "$SUITE_LOG_FILE" >&2
  "${cmd[@]}" | tee -a "$SUITE_LOG_FILE" >&2
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
    --recipe-risk-policy "$RECIPE_RISK_POLICY"
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

log_line "Suite start: ${ACTIVE_PHASE_E_FLB_GROUP} — ${GROUP_TITLE}" | tee -a "$SUITE_LOG_FILE" >&2
log_line "Intention: ${GROUP_INTENTION}" | tee -a "$SUITE_LOG_FILE" >&2
log_line "Using CUDA_DEVICE=${CUDA_DEVICE}" | tee -a "$SUITE_LOG_FILE" >&2
log_line "HParams: score-space | no-pair-weight | pair_acc | ${TRAIN_EPOCHS} epochs | batch=${TRAIN_BATCH_SIZE}" | tee -a "$SUITE_LOG_FILE" >&2
log_line "Fix strategy: eliminate fanout/grid length-biased pairs (rej-cho=+194/+203)" | tee -a "$SUITE_LOG_FILE" >&2

declare -A CASE_BENCH_GSM=()
declare -A CASE_BENCH_MATH=()

for case_spec in "${GROUP_CASES[@]}"; do
  IFS='|' read -r case_id profile case_pairs <<< "$case_spec"
  # Default pairs to TARGET_TOTAL_PAIRS if not specified
  case_pairs="${case_pairs:-$TARGET_TOTAL_PAIRS}"
  log_line "=== Case: ${case_id} (profile: ${profile}, pairs: ${case_pairs}) ===" | tee -a "$SUITE_LOG_FILE" >&2

  # 1. Curate pairs
  CURRENT_STAGE="curate_${case_id}"
  pair_dir=$(run_curate "$case_id" "$profile" "$case_pairs")
  train_jsonl="${pair_dir}/train_pairs.jsonl"
  eval_jsonl="${pair_dir}/validation_pairs.jsonl"

  # 2. Train value head
  CURRENT_STAGE="train_${case_id}"
  value_dir=$(run_train "$case_id" "$train_jsonl" "$eval_jsonl")

  # 3. Eval on ProcessBench (GSM8K + MATH)
  CURRENT_STAGE="bench_${case_id}_gsm8k"
  bench_gsm_dir=$(run_bench "$case_id" "$value_dir" "processbench_gsm8k")
  CASE_BENCH_GSM[$case_id]="$bench_gsm_dir"

  CURRENT_STAGE="bench_${case_id}_math"
  bench_math_dir=$(run_bench "$case_id" "$value_dir" "processbench_math")
  CASE_BENCH_MATH[$case_id]="$bench_math_dir"

  log_line "Case ${case_id} done." | tee -a "$SUITE_LOG_FILE" >&2

  # Print intermediate result for this case
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" - "$case_id" "$bench_gsm_dir" "$bench_math_dir" "$SUITE_LOG_FILE" <<'PY'
from pathlib import Path
import json, sys
case_id, gsm_dir, math_dir, log_file = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
def load(d, key, default=0.0):
    fp = d / "metrics.json"
    if not fp.exists(): return default
    return json.loads(fp.read_text()).get(key, default)
line = (
    f"=== {case_id} === "
    f"GSM pair_acc={load(gsm_dir,'pair_accuracy_good_vs_bad'):.4f} "
    f"auc={load(gsm_dir,'pair_auc_good_vs_bad'):.4f} "
    f"first_edge={load(gsm_dir,'first_error_edge_accuracy'):.4f} | "
    f"MATH pair_acc={load(math_dir,'pair_accuracy_good_vs_bad'):.4f} "
    f"auc={load(math_dir,'pair_auc_good_vs_bad'):.4f} "
    f"first_edge={load(math_dir,'first_error_edge_accuracy'):.4f}"
)
print(line)
with open(log_file, "a") as f:
    f.write(line + "\n")
PY
done

# === Summary ===

{
  echo "# Phase E FLB (Length-Bias Fix) Suite Summary"
  echo ""
  echo "- group_id: ${ACTIVE_PHASE_E_FLB_GROUP}"
  echo "- group_title: ${GROUP_TITLE}"
  echo "- run_prefix: ${RUN_PREFIX}"
  echo "- status: completed"
  echo "- suite_log_file: ${SUITE_LOG_FILE}"
  echo "- cuda_device: ${CUDA_DEVICE}"
  echo "- hparams: score-space | no-pair-weight | pair_acc | ${TRAIN_EPOCHS} epochs | batch=${TRAIN_BATCH_SIZE}"
  echo ""
  echo "## Fix Strategy"
  echo ""
  echo "Root cause: fanout/grid pair types (rej-cho=+194/+203) teach 'shorter=better'."
  echo "ProcessBench bad_prefix is LONGER → inverted scores when length bias learned."
  echo ""
  echo "## Results"
  echo ""
  echo "| case_id | pairs | pb_gsm_pair_acc | pb_gsm_auc | pb_gsm_first_edge | pb_math_pair_acc | pb_math_auc | pb_math_first_edge |"
  echo "|---|---|---|---|---|---|---|---|"
} > "$SUMMARY_FILE"

for case_spec in "${GROUP_CASES[@]}"; do
  IFS='|' read -r case_id profile case_pairs <<< "$case_spec"
  case_pairs="${case_pairs:-$TARGET_TOTAL_PAIRS}"
  gsm_dir="${CASE_BENCH_GSM[$case_id]:-}"
  math_dir="${CASE_BENCH_MATH[$case_id]:-}"
  "$PYTHON_BIN" - "$case_id" "$case_pairs" "$gsm_dir" "$math_dir" "$SUMMARY_FILE" <<'PY'
from pathlib import Path
import json, sys
case_id, case_pairs, gsm_dir, math_dir, summary_file = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5]
def load(d, key, default=0.0):
    fp = d / "metrics.json"
    if not fp.exists(): return default
    return json.loads(fp.read_text()).get(key, default)
row = (
    f"| {case_id} "
    f"| {case_pairs} "
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
