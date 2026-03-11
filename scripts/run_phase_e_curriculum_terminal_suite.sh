#!/usr/bin/env bash
# Phase E Curriculum Terminal Training Suite
#
# English
# -------
# Two-phase curriculum that addresses the ProcessBench transfer gap caused by
# a mismatch between training supervision (local first-bad-edge pairs) and
# benchmark evaluation (40-48 % all-correct trajectories with no negative step).
#
# Core insight (from BiRM literature, 2024):
#   Terminal anchor pairs need a SEPARATE BCE loss, not the same contrastive
#   ranking loss as local pairs.  With ranking_only, a terminal pair only
#   pushes "full trajectory > truncated prefix" — but does NOT teach the model
#   to produce an absolute score near 1 for a complete correct solution.
#   Adding lambda_terminal_bce > 0 applies BCE(chosen→1, rejected→0) gated by
#   the pair's terminal route weight, which is exactly the missing signal.
#
# Curriculum design:
#   Phase 1: Local-only training
#     - Math-Shepherd first_bad_edge_strict pairs, 0 % terminal anchors
#     - Objective: ranking_only, 4 epochs, lr=5e-5
#     - Goal: learn a strong local step discriminator as the warm-start
#
#   Phase 2: Terminal fine-tuning
#     - Same MS data + 10 % terminal anchors (terminal_anchor_ratio=0.10)
#     - Objective: ranking_only + lambda_terminal_bce=0.25 (BiRM-style)
#     - Warm-start: best_value_head.pt from Phase 1
#     - 2 epochs, lr=1e-5 (lower LR to preserve local discrimination)
#     - Goal: teach the model that complete solutions score ≈ 1
#
# Baselines included:
#   baseline_local: Phase 1 result (no terminal, no terminal BCE)
#   baseline_ratio10: single-stage training with 10 % terminal, ranking_only
#                     (same as terminal ratio sweep ratio=0.10 run)
#   curriculum_bce025: this two-phase run with lambda_terminal_bce=0.25
#
# All three are evaluated on processbench_math and processbench_gsm8k.
# A comparison table is generated at the end.
#
# 中文
# ----
# 两阶段 curriculum 训练方案，解决 ProcessBench 迁移失败的根本原因：
#
# 核心诊断：
#   ProcessBench 有 40-48% 的 all-correct 轨迹，这些样本的正确评估需要模型
#   知道"完整正确解答的得分应该接近 1"——这是一个绝对得分要求，而不是相对排序。
#   单纯的 ranking loss 只会推"完整轨迹 > 截断前缀"，不能给出接近 1 的绝对得分。
#   BiRM 中用 BCE 作为额外监督正好填补这个空缺。
#
# 两阶段设计：
#   Phase 1：只用 local first_bad_edge 对，建立强大的步骤判别基础
#   Phase 2：在 Phase 1 checkpoint 基础上，加入 10% terminal anchor + BCE，
#             用较小 lr 微调，不破坏 Phase 1 已学到的 local 判别能力
#
# 基准对比：
#   baseline_local       = Phase 1 单独的结果
#   baseline_ratio10     = 单阶段 10% terminal (ranking_only，无 BCE)
#   curriculum_bce025    = 本实验的两阶段结果

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ─── Operator-facing configuration ────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_curriculum_terminal}"
MODEL_PATH="${MODEL_PATH:-assets/models/Qwen2.5-7B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
PAIR_OUTPUT_ROOT="${PAIR_OUTPUT_ROOT:-assets/artifacts/phase_e_pairs}"
VALUE_OUTPUT_ROOT="${VALUE_OUTPUT_ROOT:-assets/artifacts/phase_e_runs}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-assets/artifacts/phase_e_eval}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-assets/artifacts/phase_e_feature_cache}"
FEATURE_CACHE_MODE="${FEATURE_CACHE_MODE:-read_write}"
FEATURE_CACHE_LOCK_TIMEOUT_SEC="${FEATURE_CACHE_LOCK_TIMEOUT_SEC:-600}"

# Phase 1 training hyperparameters
# Defaults taken from the best-performing ProcessBench transfer experiment
# (phase_e_all_acc90_ms_e43, pair_acc=0.73, AUC=0.63 on processbench_math):
#   objective=joint, lambda_bce=1.0, lr=3e-5, 10 epochs,
#   pair_weight_mode=none, anti_saturation_weight=5e-4, head_dropout=0.05
PHASE1_EPOCHS="${PHASE1_EPOCHS:-10}"
PHASE1_LR="${PHASE1_LR:-3e-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
HEAD_ARCHITECTURE="${HEAD_ARCHITECTURE:-mlp}"
HEAD_DROPOUT_PROB="${HEAD_DROPOUT_PROB:-0.05}"
ANTI_SATURATION_WEIGHT="${ANTI_SATURATION_WEIGHT:-5e-4}"
ANTI_SATURATION_LOGIT_THRESHOLD="${ANTI_SATURATION_LOGIT_THRESHOLD:-3.5}"
OBJECTIVE_MODE="${OBJECTIVE_MODE:-joint}"
LAMBDA_RANKING="${LAMBDA_RANKING:-1.0}"
LAMBDA_BCE="${LAMBDA_BCE:-1.0}"
RANKING_MARGIN="${RANKING_MARGIN:-0.02}"
RANKING_TARGET_SPACE="${RANKING_TARGET_SPACE:-score}"
PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-none}"
SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-pair_acc}"
RECIPE_RISK_POLICY="${RECIPE_RISK_POLICY:-error}"
MAX_GPU_MEMORY_GIB="${MAX_GPU_MEMORY_GIB:-}"
MAX_CPU_MEMORY_GIB="${MAX_CPU_MEMORY_GIB:-}"
SEED="${SEED:-42}"

# Phase 2 fine-tuning hyperparameters
PHASE2_EPOCHS="${PHASE2_EPOCHS:-2}"
PHASE2_LR="${PHASE2_LR:-1e-5}"
PHASE2_TERMINAL_ANCHOR_RATIO="${PHASE2_TERMINAL_ANCHOR_RATIO:-0.10}"
PHASE2_TERMINAL_BCE_LAMBDA="${PHASE2_TERMINAL_BCE_LAMBDA:-0.25}"

# Math-Shepherd pair construction
# High confidence threshold (0.7) significantly improves ProcessBench transfer.
MS_MAX_LOCAL_PAIRS="${MS_MAX_LOCAL_PAIRS:-16000}"
MS_MIN_PAIR_CONFIDENCE="${MS_MIN_PAIR_CONFIDENCE:-0.7}"
MS_STEP_LABEL_PAIR_MODE="${MS_STEP_LABEL_PAIR_MODE:-first_bad_edge_strict}"
MS_INPUT_JSONL="${MS_INPUT_JSONL:-assets/external_datasets/peiyi_math_shepherd/math-shepherd.jsonl}"

# Benchmarks
BENCHMARK_IDS=(processbench_math processbench_gsm8k)
# ──────────────────────────────────────────────────────────────────────────────

# Logging
LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_TABLE_MD="${LOG_ROOT}/curriculum_summary.md"
SUMMARY_TABLE_JSON="${LOG_ROOT}/curriculum_summary.json"
mkdir -p "$LOG_ROOT"

log_line() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1"
}

log_line "=== Phase E Curriculum Terminal Training Suite ===" | tee -a "$SUITE_LOG_FILE"
log_line "model        : $MODEL_PATH" | tee -a "$SUITE_LOG_FILE"
log_line "head         : $HEAD_ARCHITECTURE" | tee -a "$SUITE_LOG_FILE"
log_line "seed         : $SEED" | tee -a "$SUITE_LOG_FILE"
log_line "phase1_epochs: $PHASE1_EPOCHS  lr=$PHASE1_LR" | tee -a "$SUITE_LOG_FILE"
log_line "phase2_epochs: $PHASE2_EPOCHS  lr=$PHASE2_LR  ta_ratio=$PHASE2_TERMINAL_ANCHOR_RATIO  bce_lambda=$PHASE2_TERMINAL_BCE_LAMBDA" | tee -a "$SUITE_LOG_FILE"

declare -a ALL_EVAL_DIRS

# Helper: build common train/eval arg arrays
_common_train_flags() {
  local -n _arr=$1
  _arr+=(
    --model-path "$MODEL_PATH"
    --head-architecture "$HEAD_ARCHITECTURE"
    --head-dropout-prob "$HEAD_DROPOUT_PROB"
    --objective-mode "$OBJECTIVE_MODE"
    --lambda-ranking "$LAMBDA_RANKING"
    --lambda-bce "$LAMBDA_BCE"
    --ranking-margin "$RANKING_MARGIN"
    --ranking-target-space "$RANKING_TARGET_SPACE"
    --anti-saturation-weight "$ANTI_SATURATION_WEIGHT"
    --anti-saturation-logit-threshold "$ANTI_SATURATION_LOGIT_THRESHOLD"
    --pair-weight-mode "$PAIR_WEIGHT_MODE"
    --source-balance "$SOURCE_BALANCE"
    --permutation-mode stable_hash
    --checkpoint-selection-metric "$CHECKPOINT_SELECTION_METRIC"
    --recipe-risk-policy "$RECIPE_RISK_POLICY"
    --seed "$SEED"
    --dtype bfloat16
    --device-map auto
    --feature-cache-root "$FEATURE_CACHE_ROOT"
    --feature-cache-mode "$FEATURE_CACHE_MODE"
    --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC"
    --require-cuda
  )
  if [[ -n "$ADAPTER_PATH" ]]; then
    _arr+=(--adapter-path "$ADAPTER_PATH")
  fi
  if [[ -n "$MAX_GPU_MEMORY_GIB" ]]; then
    _arr+=(--max-gpu-memory-gib "$MAX_GPU_MEMORY_GIB")
  fi
  if [[ -n "$MAX_CPU_MEMORY_GIB" ]]; then
    _arr+=(--max-cpu-memory-gib "$MAX_CPU_MEMORY_GIB")
  fi
}

_eval_run() {
  # Usage: _eval_run <value_run_dir> <bench_id> <run_name_suffix>
  local VALUE_DIR="$1"
  local BENCH_ID="$2"
  local EVAL_RUN_NAME="$3"
  local EVAL_CMD=(
    "$PYTHON_BIN" -u scripts/phase_e_eval_benchmark.py
    --value-run-dir "$VALUE_DIR"
    --benchmark-id "$BENCH_ID"
    --run-name "$EVAL_RUN_NAME"
    --output-root "$EVAL_OUTPUT_ROOT"
    --checkpoint-name best
    --batch-size "$EVAL_BATCH_SIZE"
    --dtype bfloat16
    --device-map auto
    --feature-cache-root "$FEATURE_CACHE_ROOT"
    --feature-cache-mode "$FEATURE_CACHE_MODE"
    --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC"
    --require-cuda
  )
  if [[ -n "$MAX_GPU_MEMORY_GIB" ]]; then
    EVAL_CMD+=(--max-gpu-memory-gib "$MAX_GPU_MEMORY_GIB")
  fi
  if [[ -n "$MAX_CPU_MEMORY_GIB" ]]; then
    EVAL_CMD+=(--max-cpu-memory-gib "$MAX_CPU_MEMORY_GIB")
  fi
  log_line "EVAL [${BENCH_ID}]: ${EVAL_CMD[*]}" | tee -a "$SUITE_LOG_FILE"
  "${EVAL_CMD[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"
  local EVAL_DIR
  EVAL_DIR="$(compgen -G "${EVAL_OUTPUT_ROOT}/${EVAL_RUN_NAME}_*" 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "$EVAL_DIR" && -d "$EVAL_DIR" ]]; then
    ALL_EVAL_DIRS+=("$EVAL_DIR")
    log_line "eval_dir: $EVAL_DIR" | tee -a "$SUITE_LOG_FILE"
  fi
}

# ─── Step 1: Prepare Phase 1 pairs (local only, no terminal anchors) ──────────
log_line "──────────────────────────────────────────────────────────────────" | tee -a "$SUITE_LOG_FILE"
log_line "STEP 1: Prepare Phase 1 local-only pairs" | tee -a "$SUITE_LOG_FILE"
log_line "──────────────────────────────────────────────────────────────────" | tee -a "$SUITE_LOG_FILE"

PHASE1_PAIRS_RUN_NAME="${RUN_PREFIX}_p1_pairs"
PHASE1_PREP_CMD=(
  "$PYTHON_BIN" -u scripts/phase_e_prepare_mathshepherd_terminal_anchor_pairs.py
  --input-jsonl "$MS_INPUT_JSONL"
  --output-root "$PAIR_OUTPUT_ROOT"
  --run-name "$PHASE1_PAIRS_RUN_NAME"
  --seed "$SEED"
  --max-local-pairs "$MS_MAX_LOCAL_PAIRS"
  --min-pair-confidence "$MS_MIN_PAIR_CONFIDENCE"
  --step-label-pair-mode "$MS_STEP_LABEL_PAIR_MODE"
  --terminal-anchor-ratio 0.0
  --split-granularity source_sample
  --validation-ratio 0.1
)
log_line "PREP: ${PHASE1_PREP_CMD[*]}" | tee -a "$SUITE_LOG_FILE"
"${PHASE1_PREP_CMD[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

PHASE1_PAIR_DIR="$(compgen -G "${PAIR_OUTPUT_ROOT}/${PHASE1_PAIRS_RUN_NAME}__*" 2>/dev/null | sort | tail -n 1 || true)"
if [[ -z "$PHASE1_PAIR_DIR" || ! -f "$PHASE1_PAIR_DIR/train_pairs.jsonl" ]]; then
  log_line "ERROR: Phase 1 pair artifact not found" | tee -a "$SUITE_LOG_FILE"
  exit 1
fi
log_line "phase1_pair_dir: $PHASE1_PAIR_DIR" | tee -a "$SUITE_LOG_FILE"

# ─── Step 2: Train Phase 1 (local discrimination warm-start) ─────────────────
log_line "──────────────────────────────────────────────────────────────────" | tee -a "$SUITE_LOG_FILE"
log_line "STEP 2: Phase 1 training (local-only, ${PHASE1_EPOCHS} epochs, lr=${PHASE1_LR})" | tee -a "$SUITE_LOG_FILE"
log_line "──────────────────────────────────────────────────────────────────" | tee -a "$SUITE_LOG_FILE"

PHASE1_TRAIN_RUN_NAME="${RUN_PREFIX}_p1_train_s${SEED}"
PHASE1_TRAIN_CMD=(
  "$PYTHON_BIN" -u scripts/phase_e_train_value.py
  --train-pairs-jsonl "$PHASE1_PAIR_DIR/train_pairs.jsonl"
  --eval-pairs-jsonl  "$PHASE1_PAIR_DIR/validation_pairs.jsonl"
  --run-name "$PHASE1_TRAIN_RUN_NAME"
  --output-root "$VALUE_OUTPUT_ROOT"
  --per-device-train-batch-size "$TRAIN_BATCH_SIZE"
  --per-device-eval-batch-size "$EVAL_BATCH_SIZE"
  --num-train-epochs "$PHASE1_EPOCHS"
  --learning-rate "$PHASE1_LR"
  --terminal-bce-lambda 0.0
)
_common_train_flags PHASE1_TRAIN_CMD
log_line "TRAIN P1: ${PHASE1_TRAIN_CMD[*]}" | tee -a "$SUITE_LOG_FILE"
"${PHASE1_TRAIN_CMD[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

PHASE1_VALUE_DIR="$(compgen -G "${VALUE_OUTPUT_ROOT}/${PHASE1_TRAIN_RUN_NAME}_*" 2>/dev/null | sort | tail -n 1 || true)"
if [[ -z "$PHASE1_VALUE_DIR" || ! -d "$PHASE1_VALUE_DIR" ]]; then
  log_line "ERROR: Phase 1 value run dir not found" | tee -a "$SUITE_LOG_FILE"
  exit 1
fi
PHASE1_BEST_CKPT="$PHASE1_VALUE_DIR/best_value_head.pt"
if [[ ! -f "$PHASE1_BEST_CKPT" ]]; then
  log_line "ERROR: Phase 1 best checkpoint not found at $PHASE1_BEST_CKPT" | tee -a "$SUITE_LOG_FILE"
  exit 1
fi
log_line "phase1_value_dir  : $PHASE1_VALUE_DIR" | tee -a "$SUITE_LOG_FILE"
log_line "phase1_best_ckpt  : $PHASE1_BEST_CKPT" | tee -a "$SUITE_LOG_FILE"

# ─── Step 3: Evaluate Phase 1 baseline ────────────────────────────────────────
log_line "──────────────────────────────────────────────────────────────────" | tee -a "$SUITE_LOG_FILE"
log_line "STEP 3: Evaluate Phase 1 baseline (no terminal)" | tee -a "$SUITE_LOG_FILE"
log_line "──────────────────────────────────────────────────────────────────" | tee -a "$SUITE_LOG_FILE"

for BENCH_ID in "${BENCHMARK_IDS[@]}"; do
  _eval_run "$PHASE1_VALUE_DIR" "$BENCH_ID" "${RUN_PREFIX}_baseline_local_${BENCH_ID}"
done

# ─── Step 4: Prepare Phase 2 pairs (local + 10 % terminal anchors) ────────────
log_line "──────────────────────────────────────────────────────────────────" | tee -a "$SUITE_LOG_FILE"
log_line "STEP 4: Prepare Phase 2 pairs (ta_ratio=${PHASE2_TERMINAL_ANCHOR_RATIO})" | tee -a "$SUITE_LOG_FILE"
log_line "──────────────────────────────────────────────────────────────────" | tee -a "$SUITE_LOG_FILE"

PHASE2_PAIRS_RUN_NAME="${RUN_PREFIX}_p2_pairs_ta$(printf '%03d' "$(echo "$PHASE2_TERMINAL_ANCHOR_RATIO * 100" | bc | cut -d. -f1)")"
PHASE2_PREP_CMD=(
  "$PYTHON_BIN" -u scripts/phase_e_prepare_mathshepherd_terminal_anchor_pairs.py
  --input-jsonl "$MS_INPUT_JSONL"
  --output-root "$PAIR_OUTPUT_ROOT"
  --run-name "$PHASE2_PAIRS_RUN_NAME"
  --seed "$SEED"
  --max-local-pairs "$MS_MAX_LOCAL_PAIRS"
  --min-pair-confidence "$MS_MIN_PAIR_CONFIDENCE"
  --step-label-pair-mode "$MS_STEP_LABEL_PAIR_MODE"
  --terminal-anchor-ratio "$PHASE2_TERMINAL_ANCHOR_RATIO"
  --split-granularity source_sample
  --validation-ratio 0.1
)
log_line "PREP P2: ${PHASE2_PREP_CMD[*]}" | tee -a "$SUITE_LOG_FILE"
"${PHASE2_PREP_CMD[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

PHASE2_PAIR_DIR="$(compgen -G "${PAIR_OUTPUT_ROOT}/${PHASE2_PAIRS_RUN_NAME}__*" 2>/dev/null | sort | tail -n 1 || true)"
if [[ -z "$PHASE2_PAIR_DIR" || ! -f "$PHASE2_PAIR_DIR/train_pairs.jsonl" ]]; then
  log_line "ERROR: Phase 2 pair artifact not found" | tee -a "$SUITE_LOG_FILE"
  exit 1
fi
log_line "phase2_pair_dir: $PHASE2_PAIR_DIR" | tee -a "$SUITE_LOG_FILE"

# ─── Step 5: Single-stage baseline (ratio=0.10, no BCE, no warm-start) ────────
# This is the same as terminal ratio sweep ratio=0.10 but run fresh here for
# a controlled baseline on the same hardware/batch.
log_line "──────────────────────────────────────────────────────────────────" | tee -a "$SUITE_LOG_FILE"
log_line "STEP 5: Single-stage baseline (ratio=0.10, ranking_only, no curriculum)" | tee -a "$SUITE_LOG_FILE"
log_line "──────────────────────────────────────────────────────────────────" | tee -a "$SUITE_LOG_FILE"

BASELINE_RATIO10_RUN_NAME="${RUN_PREFIX}_baseline_ratio10_s${SEED}"
BASELINE_RATIO10_CMD=(
  "$PYTHON_BIN" -u scripts/phase_e_train_value.py
  --train-pairs-jsonl "$PHASE2_PAIR_DIR/train_pairs.jsonl"
  --eval-pairs-jsonl  "$PHASE2_PAIR_DIR/validation_pairs.jsonl"
  --run-name "$BASELINE_RATIO10_RUN_NAME"
  --output-root "$VALUE_OUTPUT_ROOT"
  --per-device-train-batch-size "$TRAIN_BATCH_SIZE"
  --per-device-eval-batch-size "$EVAL_BATCH_SIZE"
  --num-train-epochs "$PHASE1_EPOCHS"
  --learning-rate "$PHASE1_LR"
  --terminal-bce-lambda 0.0
)
_common_train_flags BASELINE_RATIO10_CMD
log_line "TRAIN baseline_ratio10: ${BASELINE_RATIO10_CMD[*]}" | tee -a "$SUITE_LOG_FILE"
"${BASELINE_RATIO10_CMD[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

BASELINE_RATIO10_DIR="$(compgen -G "${VALUE_OUTPUT_ROOT}/${BASELINE_RATIO10_RUN_NAME}_*" 2>/dev/null | sort | tail -n 1 || true)"
if [[ -n "$BASELINE_RATIO10_DIR" && -d "$BASELINE_RATIO10_DIR" ]]; then
  for BENCH_ID in "${BENCHMARK_IDS[@]}"; do
    _eval_run "$BASELINE_RATIO10_DIR" "$BENCH_ID" "${RUN_PREFIX}_baseline_ratio10_${BENCH_ID}"
  done
else
  log_line "WARNING: baseline_ratio10 value dir not found, skipping eval" | tee -a "$SUITE_LOG_FILE"
fi

# ─── Step 6: Phase 2 curriculum training (warm-start + terminal BCE) ──────────
log_line "──────────────────────────────────────────────────────────────────" | tee -a "$SUITE_LOG_FILE"
log_line "STEP 6: Phase 2 curriculum training (warm-start, ta_ratio=${PHASE2_TERMINAL_ANCHOR_RATIO}, bce_lambda=${PHASE2_TERMINAL_BCE_LAMBDA})" | tee -a "$SUITE_LOG_FILE"
log_line "──────────────────────────────────────────────────────────────────" | tee -a "$SUITE_LOG_FILE"

PHASE2_TRAIN_RUN_NAME="${RUN_PREFIX}_p2_curriculum_bce$(printf '%03d' "$(echo "$PHASE2_TERMINAL_BCE_LAMBDA * 100" | bc | cut -d. -f1)")_s${SEED}"
PHASE2_TRAIN_CMD=(
  "$PYTHON_BIN" -u scripts/phase_e_train_value.py
  --train-pairs-jsonl "$PHASE2_PAIR_DIR/train_pairs.jsonl"
  --eval-pairs-jsonl  "$PHASE2_PAIR_DIR/validation_pairs.jsonl"
  --run-name "$PHASE2_TRAIN_RUN_NAME"
  --output-root "$VALUE_OUTPUT_ROOT"
  --per-device-train-batch-size "$TRAIN_BATCH_SIZE"
  --per-device-eval-batch-size "$EVAL_BATCH_SIZE"
  --num-train-epochs "$PHASE2_EPOCHS"
  --learning-rate "$PHASE2_LR"
  --terminal-bce-lambda "$PHASE2_TERMINAL_BCE_LAMBDA"
  --init-value-head-path "$PHASE1_BEST_CKPT"
)
_common_train_flags PHASE2_TRAIN_CMD
log_line "TRAIN P2 curriculum: ${PHASE2_TRAIN_CMD[*]}" | tee -a "$SUITE_LOG_FILE"
"${PHASE2_TRAIN_CMD[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

PHASE2_VALUE_DIR="$(compgen -G "${VALUE_OUTPUT_ROOT}/${PHASE2_TRAIN_RUN_NAME}_*" 2>/dev/null | sort | tail -n 1 || true)"
if [[ -n "$PHASE2_VALUE_DIR" && -d "$PHASE2_VALUE_DIR" ]]; then
  for BENCH_ID in "${BENCHMARK_IDS[@]}"; do
    _eval_run "$PHASE2_VALUE_DIR" "$BENCH_ID" "${RUN_PREFIX}_curriculum_bce025_${BENCH_ID}"
  done
else
  log_line "ERROR: Phase 2 curriculum value dir not found" | tee -a "$SUITE_LOG_FILE"
  exit 1
fi

# ─── Step 7: Summary table ─────────────────────────────────────────────────────
log_line "=== All phases complete. Generating summary table... ===" | tee -a "$SUITE_LOG_FILE"

if [[ ${#ALL_EVAL_DIRS[@]} -gt 0 ]]; then
  SUMMARIZE_CMD=(
    "$PYTHON_BIN" -u scripts/phase_e_summarize_transfer_results.py
    --eval-dirs "${ALL_EVAL_DIRS[@]}"
    --output-path "$SUMMARY_TABLE_MD"
    --json-output "$SUMMARY_TABLE_JSON"
    --sort-by pair_auc_good_vs_bad
  )
  log_line "SUMMARIZE: ${SUMMARIZE_CMD[*]}" | tee -a "$SUITE_LOG_FILE"
  "${SUMMARIZE_CMD[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"
  log_line "Summary table: $SUMMARY_TABLE_MD" | tee -a "$SUITE_LOG_FILE"
else
  log_line "WARNING: No eval dirs collected — skipping summary." | tee -a "$SUITE_LOG_FILE"
fi

log_line "=== Phase E Curriculum Terminal Training Suite DONE ===" | tee -a "$SUITE_LOG_FILE"
log_line "suite_log    : $SUITE_LOG_FILE" | tee -a "$SUITE_LOG_FILE"
log_line "summary_md   : $SUMMARY_TABLE_MD" | tee -a "$SUITE_LOG_FILE"
log_line "summary_json : $SUMMARY_TABLE_JSON" | tee -a "$SUITE_LOG_FILE"
