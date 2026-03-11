#!/usr/bin/env bash
# Changelog (prepend-only, newest first):
# 2026-03-11: Initial creation. New dataset series NDS1-NDS4 testing RLHFlow-Deepseek,
#             Math-Step-DPO-10K, and mixed sources vs Math-Shepherd baseline on ProcessBench.
#
# Phase E New Dataset Series (NDS) — 评测新数据集能否突破当前 frozen-backbone 天花板
#
# English
# -------
# This suite tests whether higher-quality step-label datasets (LLM-judge annotated vs MC-estimated)
# and explicit fork-point pairs (Math-Step-DPO) improve ProcessBench transfer over Math-Shepherd.
#
# Experiment groups:
#   NDS1: rlhflow_align_v1 — RLHFlow-Deepseek LLM-judge labels only
#   NDS2: math_step_dpo_v1 — Math-Step-DPO fork-point pairs + Math-Shepherd anchor
#   NDS3: ms_align_v1 (baseline) — same scale as NDS1/2 for fair comparison
#   NDS4: ms_rlhflow_mixed_v1 — Mixed Math-Shepherd + RLHFlow + Math-Step-DPO
#   NDS5: ms_strict_only_v1 — remove MS fanout/grid length-bias pairs
#   NDS6: rlh_strict_only_v1 — remove RLHFlow fanout/grid length-bias pairs
#   NDS7: ms_dpo_calibrated_v1 — use DPO sibling pairs as length-bias debiaser
#
# 中文
# ----
# 测试 LLM-judge 标注质量和显式分叉点 pair 能否在 ProcessBench 迁移上超越 Math-Shepherd MC 基线。
# 所有实验使用相同规模 (4096 pairs) 和相同 head 架构 (mlp)，只变数据源。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACTIVE_PHASE_E_NDS_GROUP="${ACTIVE_PHASE_E_NDS_GROUP:-NDS1_RLHFLOW_SMOKE}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_nds}"

MODEL_PATH="${MODEL_PATH:-assets/models/Qwen2.5-7B-Instruct}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-assets/artifacts/phase_e_feature_cache}"
PAIR_OUTPUT_ROOT="${PAIR_OUTPUT_ROOT:-assets/artifacts/phase_e_pairs}"
VALUE_OUTPUT_ROOT="${VALUE_OUTPUT_ROOT:-assets/artifacts/phase_e_runs}"
BENCH_OUTPUT_ROOT="${BENCH_OUTPUT_ROOT:-assets/artifacts/phase_e_eval}"
LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"

TARGET_TOTAL_PAIRS="${TARGET_TOTAL_PAIRS:-4096}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-96}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
HEAD_MLP_HIDDEN_SIZE="${HEAD_MLP_HIDDEN_SIZE:-1024}"
BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-128}"
RECIPE_RISK_POLICY="${RECIPE_RISK_POLICY:-error}"

# GPU 设备：默认用 GPU 3（当前 GPU 1/2 已被其他实验占用）
# GPU device: default GPU 3 (GPUs 1/2 are currently occupied)
CUDA_DEVICE="${CUDA_DEVICE:-3}"

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
      echo "# Phase E NDS Suite Summary"
      echo "- group_id: ${ACTIVE_PHASE_E_NDS_GROUP}"
      echo "- status: FAILED at stage: ${CURRENT_STAGE}"
    } > "$SUMMARY_FILE"
  fi
}
trap on_exit EXIT

resolve_group() {
  case "$ACTIVE_PHASE_E_NDS_GROUP" in
    NDS1_RLHFLOW_SMOKE)
      GROUP_TITLE="NDS1 RLHFlow-Deepseek LLM-Judge Smoke"
      GROUP_INTENTION="Test whether LLM-judge annotated step labels (RLHFlow-Deepseek, 252K rows) improve ProcessBench transfer vs MC-estimated Math-Shepherd baseline at 4096 pairs scale."
      GROUP_CASES=("nds1_rlhflow_align_mlp|rlhflow_align_v1|score|none|pair_acc")
      ;;
    NDS2_MATH_STEP_DPO_SMOKE)
      GROUP_TITLE="NDS2 Math-Step-DPO Fork-Point Pairs Smoke"
      GROUP_INTENTION="Test whether explicit fork-point sibling_branch pairs from Math-Step-DPO-10K improve first_error_edge_accuracy and ProcessBench AUC vs strict first_bad_edge baseline."
      # 中文: Math-Step-DPO profile 会混合 sibling-branch + terminal anchor。
      # 对这类 mixed semantics，recipe_safety 已明确拒绝
      # `logit + confidence_semantic + ranking_score` 组合，所以这里显式切到更保守的
      # `score + none + pair_acc`。
      # The recipe safety guard already rejects
      # `logit + confidence_semantic + ranking_score` on this mixed-semantics profile,
      # so this group uses the conservative safe recipe instead.
      GROUP_CASES=("nds2_math_step_dpo_mlp|math_step_dpo_v1|score|none|pair_acc")
      ;;
    NDS3_MS_BASELINE_SMOKE)
      GROUP_TITLE="NDS3 Math-Shepherd Baseline Smoke (same scale as NDS1/2)"
      GROUP_INTENTION="Reproduce ms_align_v1 baseline at 4096 pairs with same hyperparams as NDS1/2 for fair comparison."
      GROUP_CASES=("nds3_ms_baseline_mlp|ms_align_v1|score|none|pair_acc")
      ;;
    NDS4_MIXED_SMOKE)
      GROUP_TITLE="NDS4 Mixed Multi-Source Smoke"
      GROUP_INTENTION="Test mixed Math-Shepherd + RLHFlow-Deepseek + Math-Step-DPO combination at 4096 pairs."
      GROUP_CASES=("nds4_ms_rlhflow_mixed_mlp|ms_rlhflow_mixed_v1|score|none|pair_acc")
      ;;
    NDS5_MS_STRICT_ONLY_SMOKE)
      GROUP_TITLE="NDS5 Math-Shepherd Strict-Only Smoke"
      GROUP_INTENTION="Test whether removing all Math-Shepherd fanout/grid pairs fixes the known shorter-is-better length shortcut and improves ProcessBench transfer."
      # 中文: `ms_strict_only_v1` 保留 strict + terminal 两种 mixed semantics。
      # 为避免再踩已知 mixed-terminal 反模式，这里直接使用安全配方：
      # `score + none + pair_acc`。
      # English: strict-only still mixes local + terminal semantics, so we use
      # the repository-safe recipe directly.
      GROUP_CASES=("nds5_ms_strict_only_mlp|ms_strict_only_v1|score|none|pair_acc")
      ;;
    NDS6_RLHFLOW_STRICT_ONLY_SMOKE)
      GROUP_TITLE="NDS6 RLHFlow Strict-Only Smoke"
      GROUP_INTENTION="Test whether RLHFlow-Deepseek becomes competitive once its fanout/grid length-bias pairs are removed, isolating the value of LLM-judge step labels."
      GROUP_CASES=("nds6_rlhflow_strict_only_mlp|rlh_strict_only_v1|score|none|pair_acc")
      ;;
    NDS7_MS_DPO_CALIBRATED_SMOKE)
      GROUP_TITLE="NDS7 MS + DPO Calibrated Smoke"
      GROUP_INTENTION="Test whether Math-Step-DPO sibling_branch pairs can debias Math-Shepherd length shortcuts while preserving first-error supervision."
      GROUP_CASES=("nds7_ms_dpo_calibrated_mlp|ms_dpo_calibrated_v1|score|none|pair_acc")
      ;;
    NDS_ALL_SMOKE)
      GROUP_TITLE="NDS All Groups Smoke (NDS1..NDS7)"
      GROUP_INTENTION="Run all new-dataset and length-bias-fix smoke variants in one suite pass for direct comparison."
      GROUP_CASES=(
        "nds1_rlhflow_align_mlp|rlhflow_align_v1|score|none|pair_acc"
        "nds2_math_step_dpo_mlp|math_step_dpo_v1|score|none|pair_acc"
        "nds3_ms_baseline_mlp|ms_align_v1|score|none|pair_acc"
        "nds4_ms_rlhflow_mixed_mlp|ms_rlhflow_mixed_v1|score|none|pair_acc"
        "nds5_ms_strict_only_mlp|ms_strict_only_v1|score|none|pair_acc"
        "nds6_rlhflow_strict_only_mlp|rlh_strict_only_v1|score|none|pair_acc"
        "nds7_ms_dpo_calibrated_mlp|ms_dpo_calibrated_v1|score|none|pair_acc"
      )
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_E_NDS_GROUP=$ACTIVE_PHASE_E_NDS_GROUP" >&2
      echo "  Valid: NDS1_RLHFLOW_SMOKE, NDS2_MATH_STEP_DPO_SMOKE, NDS3_MS_BASELINE_SMOKE, NDS4_MIXED_SMOKE, NDS5_MS_STRICT_ONLY_SMOKE, NDS6_RLHFLOW_STRICT_ONLY_SMOKE, NDS7_MS_DPO_CALIBRATED_SMOKE, NDS_ALL_SMOKE" >&2
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
  local ranking_target_space="${4:-logit}"
  local pair_weight_mode="${5:-confidence_semantic}"
  local checkpoint_selection_metric="${6:-ranking_score}"
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
    --ranking-target-space "$ranking_target_space"
    --pair-weight-mode "$pair_weight_mode"
    --source-balance none
    --permutation-mode stable_hash
    --checkpoint-selection-metric "$checkpoint_selection_metric"
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
  local train_status=${PIPESTATUS[0]}
  if [[ $train_status -ne 0 ]]; then
    return "$train_status"
  fi
  # Return latest run dir
  "$PYTHON_BIN" - "$VALUE_OUTPUT_ROOT" "$run_name" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
prefix = sys.argv[2]
matches = sorted(root.glob(f"{prefix}_*"))
if not matches:
    raise SystemExit(f"No value run dir matches prefix: {prefix}")
latest = matches[-1]
manifest = latest / "manifest.json"
if not manifest.exists():
    raise SystemExit(f"Value run completed without manifest: {manifest}")
print(latest)
PY
}

run_bench() {
  local case_id="$1"
  local value_run_dir="$2"
  local benchmark_id="$3"    # processbench_gsm8k or processbench_math
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
  local bench_status=${PIPESTATUS[0]}
  if [[ $bench_status -ne 0 ]]; then
    return "$bench_status"
  fi
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

print_metrics() {
  local bench_dir="$1"
  local label="$2"
  "$PYTHON_BIN" - "$bench_dir" "$label" <<'PY'
from pathlib import Path
import json, sys
d = Path(sys.argv[1]) / "metrics.json"
label = sys.argv[2]
if not d.exists():
    print(f"  {label}: metrics.json NOT FOUND")
    sys.exit(0)
m = json.loads(d.read_text())
print(f"  {label}: pair_acc={m.get('pair_accuracy_good_vs_bad',0):.4f} auc={m.get('pair_auc_good_vs_bad',0):.4f} first_edge={m.get('first_error_edge_accuracy',0):.4f}")
PY
}

# === Main ===

mkdir -p "$LOG_ROOT"
resolve_group

log_line "Suite start: ${ACTIVE_PHASE_E_NDS_GROUP} — ${GROUP_TITLE}" | tee -a "$SUITE_LOG_FILE" >&2
log_line "Intention: ${GROUP_INTENTION}" | tee -a "$SUITE_LOG_FILE" >&2
log_line "Using CUDA_DEVICE=${CUDA_DEVICE}" | tee -a "$SUITE_LOG_FILE" >&2

declare -A CASE_BENCH_GSM=()
declare -A CASE_BENCH_MATH=()

for case_spec in "${GROUP_CASES[@]}"; do
  IFS='|' read -r case_id profile ranking_target_space pair_weight_mode checkpoint_selection_metric <<< "$case_spec"
  log_line "=== Case: ${case_id} (profile: ${profile}) ===" | tee -a "$SUITE_LOG_FILE" >&2

  # 1. Curate pairs / 构造 pair artifact
  CURRENT_STAGE="curate_${case_id}"
  pair_dir=$(run_curate "$case_id" "$profile")
  train_jsonl="${pair_dir}/train_pairs.jsonl"
  eval_jsonl="${pair_dir}/validation_pairs.jsonl"

  # 2. Train value head / 训练 value head
  CURRENT_STAGE="train_${case_id}"
  value_dir=$(run_train "$case_id" "$train_jsonl" "$eval_jsonl" "$ranking_target_space" "$pair_weight_mode" "$checkpoint_selection_metric")

  # 3. Eval on ProcessBench / 在 ProcessBench 上评测
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
  echo "# Phase E NDS Suite Summary"
  echo ""
  echo "- group_id: ${ACTIVE_PHASE_E_NDS_GROUP}"
  echo "- group_title: ${GROUP_TITLE}"
  echo "- run_prefix: ${RUN_PREFIX}"
  echo "- status: completed"
  echo "- suite_log_file: ${SUITE_LOG_FILE}"
  echo "- cuda_device: ${CUDA_DEVICE}"
  echo ""
  echo "## Results"
  echo ""
  echo "| case_id | pb_gsm_pair_acc | pb_gsm_auc | pb_gsm_first_edge | pb_math_pair_acc | pb_math_auc | pb_math_first_edge |"
  echo "|---|---|---|---|---|---|---|"
} > "$SUMMARY_FILE"

for case_spec in "${GROUP_CASES[@]}"; do
  IFS='|' read -r case_id profile _ranking_target_space _pair_weight_mode _checkpoint_selection_metric <<< "$case_spec"
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
