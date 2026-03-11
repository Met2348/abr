#!/usr/bin/env bash
# Phase E Terminal-Anchor Ratio Sweep
#
# English
# -------
# This suite answers the question:
#   "What terminal-anchor ratio gives the best tradeoff between
#    terminal_top1 improvement and overall pair_acc / AUC preservation?"
#
# Background: training exclusively on local first-bad-edge pairs leaves
# ProcessBench's all-correct slice (~40-48 % of examples) completely
# unsupervised, causing terminal_top1 ≈ 0.05.  A 50/50 terminal:local mix
# improved terminal_top1 but destroyed pair_acc (0.48 → 0.43).
#
# This sweep tests 4 ratios: 0.0 (baseline), 0.05, 0.10, 0.20.
# Each ratio runs one seed (42) with an MLP head on Math-Shepherd pairs,
# then evaluates on ProcessBench Math + GSM8K.
#
# After all runs, it calls phase_e_summarize_transfer_results.py to
# print a comparison table of: pair_acc | AUC | first_edge_acc | allcorr_last_score.
#
# 中文
# ----
# 这个 suite 回答：
#   "多小的 terminal-anchor 比例能在改善 terminal_top1 的同时
#    不破坏整体 pair_acc / AUC？"
#
# 背景：只用 local first-bad-edge 对，ProcessBench 约 40-48% 的
# all-correct 样本完全没有监督，导致 terminal_top1 ≈ 0.05。
# 50/50 混合虽然改善了 terminal_top1，但把 pair_acc 从 0.48 打到 0.43。
#
# 这次扫描 4 个比例：0.0（基线）、0.05、0.10、0.20。
# 每个比例 seed=42，MLP head，Math-Shepherd 数据，
# 在 ProcessBench Math + GSM8K 上评测。
#
# 全部跑完后调用 phase_e_summarize_transfer_results.py 生成对比表格。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ─── Operator-facing configuration ────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_ta_ratio_sweep}"
MODEL_PATH="${MODEL_PATH:-assets/models/Qwen2.5-7B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
PAIR_OUTPUT_ROOT="${PAIR_OUTPUT_ROOT:-assets/artifacts/phase_e_pairs}"
VALUE_OUTPUT_ROOT="${VALUE_OUTPUT_ROOT:-assets/artifacts/phase_e_runs}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-assets/artifacts/phase_e_eval}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-assets/artifacts/phase_e_feature_cache}"
FEATURE_CACHE_MODE="${FEATURE_CACHE_MODE:-read_write}"
FEATURE_CACHE_LOCK_TIMEOUT_SEC="${FEATURE_CACHE_LOCK_TIMEOUT_SEC:-600}"

# Training hyperparameters
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-192}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-192}"
HEAD_ARCHITECTURE="${HEAD_ARCHITECTURE:-mlp}"
OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-confidence}"
SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
MAX_GPU_MEMORY_GIB="${MAX_GPU_MEMORY_GIB:-}"
MAX_CPU_MEMORY_GIB="${MAX_CPU_MEMORY_GIB:-}"
SEED=42

# Benchmark targets evaluated for each ratio
BENCHMARK_IDS=("processbench_math" "processbench_gsm8k")

# Terminal-anchor ratios to sweep.
# 0.0  = no terminal anchors (baseline, same as default Math-Shepherd recipe)
# 0.05 = 5 % of local pairs → small terminal signal
# 0.10 = 10 % of local pairs → moderate terminal signal
# 0.20 = 20 % of local pairs → stronger terminal signal (below the catastrophic 50 %)
TERMINAL_ANCHOR_RATIOS=(0.0 0.05 0.10 0.20)

# Math-Shepherd pair construction defaults (same as baseline experiments)
MS_MAX_LOCAL_PAIRS="${MS_MAX_LOCAL_PAIRS:-20000}"
MS_MIN_PAIR_CONFIDENCE="${MS_MIN_PAIR_CONFIDENCE:-0.55}"
MS_STEP_LABEL_PAIR_MODE="${MS_STEP_LABEL_PAIR_MODE:-first_bad_edge_strict}"
MS_INPUT_JSONL="${MS_INPUT_JSONL:-assets/external_datasets/peiyi_math_shepherd/math-shepherd.jsonl}"
# ──────────────────────────────────────────────────────────────────────────────

# Logging
LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_TABLE_MD="${LOG_ROOT}/transfer_summary.md"
SUMMARY_TABLE_JSON="${LOG_ROOT}/transfer_summary.json"
mkdir -p "$LOG_ROOT"

log_line() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1"
}

log_line "=== Phase E Terminal-Anchor Ratio Sweep ===" | tee -a "$SUITE_LOG_FILE"
log_line "ratios : ${TERMINAL_ANCHOR_RATIOS[*]}" | tee -a "$SUITE_LOG_FILE"
log_line "model  : $MODEL_PATH" | tee -a "$SUITE_LOG_FILE"
log_line "head   : $HEAD_ARCHITECTURE" | tee -a "$SUITE_LOG_FILE"
log_line "seed   : $SEED" | tee -a "$SUITE_LOG_FILE"

# Track all eval directories produced by this sweep for final summary
declare -a ALL_EVAL_DIRS

# ─── Per-ratio loop ────────────────────────────────────────────────────────────
for RATIO in "${TERMINAL_ANCHOR_RATIOS[@]}"; do

  # Convert ratio to a safe tag for directory names (e.g. 0.05 → r005)
  RATIO_TAG="$(printf '%s' "$RATIO" | tr -d '.' | sed 's/^0//')"
  # Handle ratio=0.0 specially to produce "r000"
  if [[ -z "$RATIO_TAG" || "$RATIO_TAG" == "0" ]]; then
    RATIO_TAG="000"
  fi
  # Pad to 3 digits for consistent lexicographic sort
  RATIO_TAG="r$(printf '%03d' "${RATIO_TAG#0}")" 2>/dev/null || RATIO_TAG="r$(printf '%s' "$RATIO_TAG" | sed 's/[^0-9]//g')"

  log_line "------------------------------------------------------------" | tee -a "$SUITE_LOG_FILE"
  log_line "RATIO=${RATIO}  TAG=${RATIO_TAG}" | tee -a "$SUITE_LOG_FILE"
  log_line "------------------------------------------------------------" | tee -a "$SUITE_LOG_FILE"

  # ── 1. Prepare pairs ─────────────────────────────────────────────────────────
  PREP_RUN_NAME="phase_e_ta_sweep_${RATIO_TAG}_pairs"
  PAIR_PREP_CMD=(
    "$PYTHON_BIN" -u scripts/phase_e_prepare_mathshepherd_terminal_anchor_pairs.py
    --input-jsonl "$MS_INPUT_JSONL"
    --output-root "$PAIR_OUTPUT_ROOT"
    --run-name "$PREP_RUN_NAME"
    --seed "$SEED"
    --max-local-pairs "$MS_MAX_LOCAL_PAIRS"
    --min-pair-confidence "$MS_MIN_PAIR_CONFIDENCE"
    --step-label-pair-mode "$MS_STEP_LABEL_PAIR_MODE"
    --terminal-anchor-ratio "$RATIO"
    --split-granularity source_sample
    --validation-ratio 0.1
  )
  log_line "PREP: ${PAIR_PREP_CMD[*]}" | tee -a "$SUITE_LOG_FILE"
  "${PAIR_PREP_CMD[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

  # Locate the artifact directory (deterministic fingerprint naming: {run_name}__{hash})
  PAIR_DIR="$(compgen -G "${PAIR_OUTPUT_ROOT}/${PREP_RUN_NAME}__*" 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -z "$PAIR_DIR" || ! -f "$PAIR_DIR/train_pairs.jsonl" ]]; then
    log_line "ERROR: Could not find pair artifact for ratio=${RATIO}" | tee -a "$SUITE_LOG_FILE"
    exit 1
  fi
  log_line "pair_dir: $PAIR_DIR" | tee -a "$SUITE_LOG_FILE"

  # ── 2. Train value head ───────────────────────────────────────────────────────
  TRAIN_RUN_NAME="phase_e_ta_sweep_${RATIO_TAG}_train_s${SEED}"
  TRAIN_CMD=(
    "$PYTHON_BIN" -u scripts/phase_e_train_value.py
    --train-pairs-jsonl "$PAIR_DIR/train_pairs.jsonl"
    --eval-pairs-jsonl  "$PAIR_DIR/validation_pairs.jsonl"
    --model-path "$MODEL_PATH"
    --run-name "$TRAIN_RUN_NAME"
    --output-root "$VALUE_OUTPUT_ROOT"
    --head-architecture "$HEAD_ARCHITECTURE"
    --objective-mode "$OBJECTIVE_MODE"
    --learning-rate "$LEARNING_RATE"
    --num-train-epochs "$TRAIN_EPOCHS"
    --per-device-train-batch-size "$TRAIN_BATCH_SIZE"
    --per-device-eval-batch-size "$EVAL_BATCH_SIZE"
    --pair-weight-mode "$PAIR_WEIGHT_MODE"
    --source-balance "$SOURCE_BALANCE"
    --permutation-mode stable_hash
    --checkpoint-selection-metric ranking_score
    --seed "$SEED"
    --dtype bfloat16
    --device-map auto
    --feature-cache-root "$FEATURE_CACHE_ROOT"
    --feature-cache-mode "$FEATURE_CACHE_MODE"
    --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC"
    --require-cuda
  )
  if [[ -n "$ADAPTER_PATH" ]]; then
    TRAIN_CMD+=(--adapter-path "$ADAPTER_PATH")
  fi
  if [[ -n "$MAX_GPU_MEMORY_GIB" ]]; then
    TRAIN_CMD+=(--max-gpu-memory-gib "$MAX_GPU_MEMORY_GIB")
  fi
  if [[ -n "$MAX_CPU_MEMORY_GIB" ]]; then
    TRAIN_CMD+=(--max-cpu-memory-gib "$MAX_CPU_MEMORY_GIB")
  fi
  log_line "TRAIN: ${TRAIN_CMD[*]}" | tee -a "$SUITE_LOG_FILE"
  "${TRAIN_CMD[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

  # Locate the value run directory (latest match for this run name)
  VALUE_RUN_DIR="$(compgen -G "${VALUE_OUTPUT_ROOT}/${TRAIN_RUN_NAME}_*" 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -z "$VALUE_RUN_DIR" || ! -d "$VALUE_RUN_DIR" ]]; then
    log_line "ERROR: Could not find value run dir for ratio=${RATIO}" | tee -a "$SUITE_LOG_FILE"
    exit 1
  fi
  log_line "value_run_dir: $VALUE_RUN_DIR" | tee -a "$SUITE_LOG_FILE"

  # ── 3. Benchmark evaluation ───────────────────────────────────────────────────
  for BENCH_ID in "${BENCHMARK_IDS[@]}"; do
    EVAL_RUN_NAME="phase_e_ta_sweep_${RATIO_TAG}_s${SEED}_${BENCH_ID}"
    EVAL_CMD=(
      "$PYTHON_BIN" -u scripts/phase_e_eval_benchmark.py
      --value-run-dir "$VALUE_RUN_DIR"
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

    # Locate the eval directory and record it
    EVAL_DIR="$(compgen -G "${EVAL_OUTPUT_ROOT}/${EVAL_RUN_NAME}_*" 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -n "$EVAL_DIR" && -d "$EVAL_DIR" ]]; then
      ALL_EVAL_DIRS+=("$EVAL_DIR")
      log_line "eval_dir: $EVAL_DIR" | tee -a "$SUITE_LOG_FILE"
    fi
  done

done
# ─── End per-ratio loop ────────────────────────────────────────────────────────

log_line "=== All ratios complete. Generating summary table... ===" | tee -a "$SUITE_LOG_FILE"

# ─── Summary table ─────────────────────────────────────────────────────────────
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

log_line "=== Phase E Terminal-Anchor Ratio Sweep DONE ===" | tee -a "$SUITE_LOG_FILE"
log_line "suite_log    : $SUITE_LOG_FILE" | tee -a "$SUITE_LOG_FILE"
log_line "summary_md   : $SUMMARY_TABLE_MD" | tee -a "$SUITE_LOG_FILE"
log_line "summary_json : $SUMMARY_TABLE_JSON" | tee -a "$SUITE_LOG_FILE"
