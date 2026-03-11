#!/usr/bin/env bash
# Phase E Multi-Source Mix Suite: Math-Shepherd + PRMBench
#
# English
# -------
# This suite tests whether combining two complementary data sources
# improves ProcessBench transfer over single-source training.
#
# Motivation:
#   - Math-Shepherd: large scale (200K+), MC-labeled, good for local step
#     discrimination, but MC labels have ~10-15% noise, and the dataset
#     only contains error steps within the solution, not "all-correct" examples.
#   - PRMBench: human-curated error-step pairs, higher label quality, diverse
#     math domains. Adding it should improve generalization to cleaner signal.
#
# Mix design:
#   - MS: 60% of training pairs (12000 local + 1200 terminal anchors @ ta=0.10)
#   - PRMBench: 40% of training pairs (capped at 8000 local + 800 terminal)
#   - Combined via phase_e_mix_pair_artifacts.py with artifact_mix_source_label
#     so per-source eval metrics remain visible.
#
# Training:
#   - Two-phase curriculum (same as curriculum_terminal_suite):
#     Phase 1: local-only warm-start on mixed pairs (4 epochs, lr=5e-5)
#     Phase 2: +terminal BCE fine-tuning (2 epochs, lr=1e-5, lambda_terminal_bce=0.25)
#
# Baselines:
#   ms_only_curriculum:      curriculum on MS-only (same as curriculum suite output)
#   ms_prmbench_direct:      single-stage 4-epoch training on the mix (no curriculum)
#   ms_prmbench_curriculum:  two-phase curriculum on the mix (this experiment)
#
# Evaluated on: processbench_math, processbench_gsm8k
#
# 中文
# ----
# 测试两个互补数据源组合后的 ProcessBench 迁移效果。
#
# 动机：
#   - Math-Shepherd：规模大但标签有噪音，只有带错误步骤的轨迹。
#   - PRMBench：人工标注，标签质量高，数学领域多样。两者互补。
#
# Mix 比例：MS 60% + PRMBench 40%。
# 训练：同 curriculum_terminal_suite 的两阶段方案。
# 基准：MS 单源 curriculum vs. 混合单阶段 vs. 混合 curriculum。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ─── Operator-facing configuration ────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_ms_prmbench_mix}"
MODEL_PATH="${MODEL_PATH:-assets/models/Qwen2.5-7B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
PAIR_OUTPUT_ROOT="${PAIR_OUTPUT_ROOT:-assets/artifacts/phase_e_pairs}"
VALUE_OUTPUT_ROOT="${VALUE_OUTPUT_ROOT:-assets/artifacts/phase_e_runs}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-assets/artifacts/phase_e_eval}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-assets/artifacts/phase_e_feature_cache}"
FEATURE_CACHE_MODE="${FEATURE_CACHE_MODE:-read_write}"
FEATURE_CACHE_LOCK_TIMEOUT_SEC="${FEATURE_CACHE_LOCK_TIMEOUT_SEC:-600}"

# Training hyperparameters
# Defaults from best-performing ProcessBench transfer run (pair_acc=0.73):
#   joint objective, lr=3e-5, 10 epochs, pair_weight_mode=none, anti_sat=5e-4
PHASE1_EPOCHS="${PHASE1_EPOCHS:-10}"
PHASE1_LR="${PHASE1_LR:-3e-5}"
PHASE2_EPOCHS="${PHASE2_EPOCHS:-2}"
PHASE2_LR="${PHASE2_LR:-1e-5}"
PHASE2_TERMINAL_BCE_LAMBDA="${PHASE2_TERMINAL_BCE_LAMBDA:-0.25}"
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
MAX_GPU_MEMORY_GIB="${MAX_GPU_MEMORY_GIB:-}"
MAX_CPU_MEMORY_GIB="${MAX_CPU_MEMORY_GIB:-}"
SEED="${SEED:-42}"

# Data source hyperparameters
# Higher confidence threshold (0.7) significantly improves ProcessBench transfer.
MS_INPUT_JSONL="${MS_INPUT_JSONL:-assets/external_datasets/peiyi_math_shepherd/math-shepherd.jsonl}"
MS_MAX_LOCAL_PAIRS="${MS_MAX_LOCAL_PAIRS:-12000}"
MS_TERMINAL_ANCHOR_RATIO="${MS_TERMINAL_ANCHOR_RATIO:-0.10}"
MS_MIN_PAIR_CONFIDENCE="${MS_MIN_PAIR_CONFIDENCE:-0.7}"
MS_STEP_LABEL_PAIR_MODE="${MS_STEP_LABEL_PAIR_MODE:-first_bad_edge_strict}"

PRMBENCH_INPUT_JSONL="${PRMBENCH_INPUT_JSONL:-assets/external_datasets/hitsmy_prmbench_preview/prmbench_preview.jsonl}"
PRMBENCH_MAX_LOCAL_PAIRS="${PRMBENCH_MAX_LOCAL_PAIRS:-8000}"
PRMBENCH_TERMINAL_ANCHOR_RATIO="${PRMBENCH_TERMINAL_ANCHOR_RATIO:-0.10}"
PRMBENCH_MIN_PAIR_CONFIDENCE="${PRMBENCH_MIN_PAIR_CONFIDENCE:-0.7}"

# Mix caps: how many train pairs to draw from each source
MS_MIX_TRAIN_CAP="${MS_MIX_TRAIN_CAP:-12000}"
MS_MIX_VAL_CAP="${MS_MIX_VAL_CAP:-1200}"
PRMBENCH_MIX_TRAIN_CAP="${PRMBENCH_MIX_TRAIN_CAP:-8000}"
PRMBENCH_MIX_VAL_CAP="${PRMBENCH_MIX_VAL_CAP:-800}"

BENCHMARK_IDS=(processbench_math processbench_gsm8k)
# ──────────────────────────────────────────────────────────────────────────────

LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_TABLE_MD="${LOG_ROOT}/ms_prmbench_mix_summary.md"
SUMMARY_TABLE_JSON="${LOG_ROOT}/ms_prmbench_mix_summary.json"
mkdir -p "$LOG_ROOT"

log_line() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1"
}

log_line "=== Phase E MS+PRMBench Mix Suite ===" | tee -a "$SUITE_LOG_FILE"
log_line "model       : $MODEL_PATH" | tee -a "$SUITE_LOG_FILE"
log_line "seed        : $SEED" | tee -a "$SUITE_LOG_FILE"
log_line "MS ta_ratio : $MS_TERMINAL_ANCHOR_RATIO" | tee -a "$SUITE_LOG_FILE"
log_line "PB ta_ratio : $PRMBENCH_TERMINAL_ANCHOR_RATIO" | tee -a "$SUITE_LOG_FILE"
log_line "bce_lambda  : $PHASE2_TERMINAL_BCE_LAMBDA" | tee -a "$SUITE_LOG_FILE"

declare -a ALL_EVAL_DIRS

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
    --seed "$SEED"
    --dtype bfloat16
    --device-map auto
    --feature-cache-root "$FEATURE_CACHE_ROOT"
    --feature-cache-mode "$FEATURE_CACHE_MODE"
    --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC"
    --require-cuda
  )
  if [[ -n "$ADAPTER_PATH" ]]; then _arr+=(--adapter-path "$ADAPTER_PATH"); fi
  if [[ -n "$MAX_GPU_MEMORY_GIB" ]]; then _arr+=(--max-gpu-memory-gib "$MAX_GPU_MEMORY_GIB"); fi
  if [[ -n "$MAX_CPU_MEMORY_GIB" ]]; then _arr+=(--max-cpu-memory-gib "$MAX_CPU_MEMORY_GIB"); fi
}

_eval_run() {
  local VALUE_DIR="$1" BENCH_ID="$2" EVAL_RUN_NAME="$3"
  local EVAL_CMD=(
    "$PYTHON_BIN" -u scripts/phase_e_eval_benchmark.py
    --value-run-dir "$VALUE_DIR"
    --benchmark-id "$BENCH_ID"
    --run-name "$EVAL_RUN_NAME"
    --output-root "$EVAL_OUTPUT_ROOT"
    --checkpoint-name best
    --batch-size "$EVAL_BATCH_SIZE"
    --dtype bfloat16 --device-map auto
    --feature-cache-root "$FEATURE_CACHE_ROOT"
    --feature-cache-mode "$FEATURE_CACHE_MODE"
    --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC"
    --require-cuda
  )
  if [[ -n "$MAX_GPU_MEMORY_GIB" ]]; then EVAL_CMD+=(--max-gpu-memory-gib "$MAX_GPU_MEMORY_GIB"); fi
  if [[ -n "$MAX_CPU_MEMORY_GIB" ]]; then EVAL_CMD+=(--max-cpu-memory-gib "$MAX_CPU_MEMORY_GIB"); fi
  log_line "EVAL [${BENCH_ID}]: ${EVAL_CMD[*]}" | tee -a "$SUITE_LOG_FILE"
  "${EVAL_CMD[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"
  local EVAL_DIR
  EVAL_DIR="$(compgen -G "${EVAL_OUTPUT_ROOT}/${EVAL_RUN_NAME}_*" 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "$EVAL_DIR" && -d "$EVAL_DIR" ]]; then
    ALL_EVAL_DIRS+=("$EVAL_DIR")
    log_line "eval_dir: $EVAL_DIR" | tee -a "$SUITE_LOG_FILE"
  fi
}

# ─── Step 1: Prepare Math-Shepherd pairs ──────────────────────────────────────
log_line "─── STEP 1: Prepare Math-Shepherd pairs (ta=${MS_TERMINAL_ANCHOR_RATIO}) ───" | tee -a "$SUITE_LOG_FILE"

MS_PAIRS_RUN_NAME="${RUN_PREFIX}_ms_pairs"
MS_PREP_CMD=(
  "$PYTHON_BIN" -u scripts/phase_e_prepare_mathshepherd_terminal_anchor_pairs.py
  --input-jsonl "$MS_INPUT_JSONL"
  --output-root "$PAIR_OUTPUT_ROOT"
  --run-name "$MS_PAIRS_RUN_NAME"
  --seed "$SEED"
  --max-local-pairs "$MS_MAX_LOCAL_PAIRS"
  --min-pair-confidence "$MS_MIN_PAIR_CONFIDENCE"
  --step-label-pair-mode "$MS_STEP_LABEL_PAIR_MODE"
  --terminal-anchor-ratio "$MS_TERMINAL_ANCHOR_RATIO"
  --split-granularity source_sample
  --validation-ratio 0.1
)
log_line "PREP MS: ${MS_PREP_CMD[*]}" | tee -a "$SUITE_LOG_FILE"
"${MS_PREP_CMD[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

MS_PAIR_DIR="$(compgen -G "${PAIR_OUTPUT_ROOT}/${MS_PAIRS_RUN_NAME}__*" 2>/dev/null | sort | tail -n 1 || true)"
if [[ -z "$MS_PAIR_DIR" || ! -f "$MS_PAIR_DIR/train_pairs.jsonl" ]]; then
  log_line "ERROR: MS pair artifact not found" | tee -a "$SUITE_LOG_FILE"; exit 1
fi
log_line "ms_pair_dir: $MS_PAIR_DIR" | tee -a "$SUITE_LOG_FILE"

# ─── Step 2: Prepare PRMBench pairs ───────────────────────────────────────────
log_line "─── STEP 2: Prepare PRMBench pairs (ta=${PRMBENCH_TERMINAL_ANCHOR_RATIO}) ───" | tee -a "$SUITE_LOG_FILE"

PRMBENCH_PAIRS_RUN_NAME="${RUN_PREFIX}_prmbench_pairs"
PRMBENCH_PREP_CMD=(
  "$PYTHON_BIN" -u scripts/phase_e_prepare_prmbench_terminal_anchor_pairs.py
  --input-jsonl "$PRMBENCH_INPUT_JSONL"
  --output-root "$PAIR_OUTPUT_ROOT"
  --run-name "$PRMBENCH_PAIRS_RUN_NAME"
  --seed "$SEED"
  --max-local-pairs "$PRMBENCH_MAX_LOCAL_PAIRS"
  --min-pair-confidence "$PRMBENCH_MIN_PAIR_CONFIDENCE"
  --terminal-anchor-ratio "$PRMBENCH_TERMINAL_ANCHOR_RATIO"
  --split-granularity source_sample
  --validation-ratio 0.1
)
log_line "PREP PRMBench: ${PRMBENCH_PREP_CMD[*]}" | tee -a "$SUITE_LOG_FILE"
"${PRMBENCH_PREP_CMD[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

PRMBENCH_PAIR_DIR="$(compgen -G "${PAIR_OUTPUT_ROOT}/${PRMBENCH_PAIRS_RUN_NAME}__*" 2>/dev/null | sort | tail -n 1 || true)"
if [[ -z "$PRMBENCH_PAIR_DIR" || ! -f "$PRMBENCH_PAIR_DIR/train_pairs.jsonl" ]]; then
  log_line "ERROR: PRMBench pair artifact not found" | tee -a "$SUITE_LOG_FILE"; exit 1
fi
log_line "prmbench_pair_dir: $PRMBENCH_PAIR_DIR" | tee -a "$SUITE_LOG_FILE"

# ─── Step 3: Mix the two artifacts ────────────────────────────────────────────
log_line "─── STEP 3: Mix MS (${MS_MIX_TRAIN_CAP}) + PRMBench (${PRMBENCH_MIX_TRAIN_CAP}) ───" | tee -a "$SUITE_LOG_FILE"

MIX_RUN_NAME="${RUN_PREFIX}_mix_pairs_s${SEED}"
MIX_CMD=(
  "$PYTHON_BIN" -u scripts/phase_e_mix_pair_artifacts.py
  --input "math_shepherd=${MS_PAIR_DIR}:${MS_MIX_TRAIN_CAP}:${MS_MIX_VAL_CAP}"
  --input "prmbench=${PRMBENCH_PAIR_DIR}:${PRMBENCH_MIX_TRAIN_CAP}:${PRMBENCH_MIX_VAL_CAP}"
  --run-name "$MIX_RUN_NAME"
  --output-root "$PAIR_OUTPUT_ROOT"
  --seed "$SEED"
)
log_line "MIX: ${MIX_CMD[*]}" | tee -a "$SUITE_LOG_FILE"
"${MIX_CMD[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

MIX_PAIR_DIR="$(compgen -G "${PAIR_OUTPUT_ROOT}/${MIX_RUN_NAME}__*" 2>/dev/null | sort | tail -n 1 || true)"
if [[ -z "$MIX_PAIR_DIR" || ! -f "$MIX_PAIR_DIR/train_pairs.jsonl" ]]; then
  log_line "ERROR: Mix pair artifact not found" | tee -a "$SUITE_LOG_FILE"; exit 1
fi
log_line "mix_pair_dir: $MIX_PAIR_DIR" | tee -a "$SUITE_LOG_FILE"

# ─── Step 4: Baseline A — single-stage direct mix training (ranking_only, 4 epochs) ───
log_line "─── STEP 4: Baseline A — direct single-stage mix training ───" | tee -a "$SUITE_LOG_FILE"

DIRECT_MIX_RUN_NAME="${RUN_PREFIX}_direct_mix_s${SEED}"
DIRECT_MIX_CMD=(
  "$PYTHON_BIN" -u scripts/phase_e_train_value.py
  --train-pairs-jsonl "$MIX_PAIR_DIR/train_pairs.jsonl"
  --eval-pairs-jsonl  "$MIX_PAIR_DIR/validation_pairs.jsonl"
  --run-name "$DIRECT_MIX_RUN_NAME"
  --output-root "$VALUE_OUTPUT_ROOT"
  --per-device-train-batch-size "$TRAIN_BATCH_SIZE"
  --per-device-eval-batch-size "$EVAL_BATCH_SIZE"
  --num-train-epochs "$PHASE1_EPOCHS"
  --learning-rate "$PHASE1_LR"
  --terminal-bce-lambda 0.0
)
_common_train_flags DIRECT_MIX_CMD
log_line "TRAIN direct_mix: ${DIRECT_MIX_CMD[*]}" | tee -a "$SUITE_LOG_FILE"
"${DIRECT_MIX_CMD[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

DIRECT_MIX_DIR="$(compgen -G "${VALUE_OUTPUT_ROOT}/${DIRECT_MIX_RUN_NAME}_*" 2>/dev/null | sort | tail -n 1 || true)"
if [[ -n "$DIRECT_MIX_DIR" && -d "$DIRECT_MIX_DIR" ]]; then
  for BENCH_ID in "${BENCHMARK_IDS[@]}"; do
    _eval_run "$DIRECT_MIX_DIR" "$BENCH_ID" "${RUN_PREFIX}_direct_mix_${BENCH_ID}"
  done
  DIRECT_MIX_BEST_CKPT="$DIRECT_MIX_DIR/best_value_head.pt"
else
  log_line "ERROR: direct_mix value dir not found" | tee -a "$SUITE_LOG_FILE"; exit 1
fi

# ─── Step 5: Baseline B — two-phase curriculum on mix WITHOUT terminal BCE ────
# (Tests whether warm-start alone helps, without the extra BCE signal)
log_line "─── STEP 5: Baseline B — curriculum mix (no terminal BCE, lr=${PHASE2_LR}) ───" | tee -a "$SUITE_LOG_FILE"

CURRICULUM_NOBCE_RUN_NAME="${RUN_PREFIX}_curriculum_nobce_s${SEED}"
CURRICULUM_NOBCE_CMD=(
  "$PYTHON_BIN" -u scripts/phase_e_train_value.py
  --train-pairs-jsonl "$MIX_PAIR_DIR/train_pairs.jsonl"
  --eval-pairs-jsonl  "$MIX_PAIR_DIR/validation_pairs.jsonl"
  --run-name "$CURRICULUM_NOBCE_RUN_NAME"
  --output-root "$VALUE_OUTPUT_ROOT"
  --per-device-train-batch-size "$TRAIN_BATCH_SIZE"
  --per-device-eval-batch-size "$EVAL_BATCH_SIZE"
  --num-train-epochs "$PHASE2_EPOCHS"
  --learning-rate "$PHASE2_LR"
  --terminal-bce-lambda 0.0
  --init-value-head-path "$DIRECT_MIX_BEST_CKPT"
)
_common_train_flags CURRICULUM_NOBCE_CMD
log_line "TRAIN curriculum_nobce: ${CURRICULUM_NOBCE_CMD[*]}" | tee -a "$SUITE_LOG_FILE"
"${CURRICULUM_NOBCE_CMD[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

CURRICULUM_NOBCE_DIR="$(compgen -G "${VALUE_OUTPUT_ROOT}/${CURRICULUM_NOBCE_RUN_NAME}_*" 2>/dev/null | sort | tail -n 1 || true)"
if [[ -n "$CURRICULUM_NOBCE_DIR" && -d "$CURRICULUM_NOBCE_DIR" ]]; then
  for BENCH_ID in "${BENCHMARK_IDS[@]}"; do
    _eval_run "$CURRICULUM_NOBCE_DIR" "$BENCH_ID" "${RUN_PREFIX}_curriculum_nobce_${BENCH_ID}"
  done
else
  log_line "WARNING: curriculum_nobce dir not found, skipping eval" | tee -a "$SUITE_LOG_FILE"
fi

# ─── Step 6: Main experiment — two-phase curriculum on mix WITH terminal BCE ──
log_line "─── STEP 6: Main — curriculum mix + terminal BCE (lambda=${PHASE2_TERMINAL_BCE_LAMBDA}) ───" | tee -a "$SUITE_LOG_FILE"

CURRICULUM_BCE_RUN_NAME="${RUN_PREFIX}_curriculum_bce$(printf '%03d' "$(echo "$PHASE2_TERMINAL_BCE_LAMBDA * 100" | bc | cut -d. -f1)")_s${SEED}"
CURRICULUM_BCE_CMD=(
  "$PYTHON_BIN" -u scripts/phase_e_train_value.py
  --train-pairs-jsonl "$MIX_PAIR_DIR/train_pairs.jsonl"
  --eval-pairs-jsonl  "$MIX_PAIR_DIR/validation_pairs.jsonl"
  --run-name "$CURRICULUM_BCE_RUN_NAME"
  --output-root "$VALUE_OUTPUT_ROOT"
  --per-device-train-batch-size "$TRAIN_BATCH_SIZE"
  --per-device-eval-batch-size "$EVAL_BATCH_SIZE"
  --num-train-epochs "$PHASE2_EPOCHS"
  --learning-rate "$PHASE2_LR"
  --terminal-bce-lambda "$PHASE2_TERMINAL_BCE_LAMBDA"
  --init-value-head-path "$DIRECT_MIX_BEST_CKPT"
)
_common_train_flags CURRICULUM_BCE_CMD
log_line "TRAIN curriculum_bce: ${CURRICULUM_BCE_CMD[*]}" | tee -a "$SUITE_LOG_FILE"
"${CURRICULUM_BCE_CMD[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

CURRICULUM_BCE_DIR="$(compgen -G "${VALUE_OUTPUT_ROOT}/${CURRICULUM_BCE_RUN_NAME}_*" 2>/dev/null | sort | tail -n 1 || true)"
if [[ -n "$CURRICULUM_BCE_DIR" && -d "$CURRICULUM_BCE_DIR" ]]; then
  for BENCH_ID in "${BENCHMARK_IDS[@]}"; do
    _eval_run "$CURRICULUM_BCE_DIR" "$BENCH_ID" "${RUN_PREFIX}_curriculum_bce_${BENCH_ID}"
  done
else
  log_line "ERROR: curriculum_bce dir not found" | tee -a "$SUITE_LOG_FILE"; exit 1
fi

# ─── Step 7: Summary ──────────────────────────────────────────────────────────
log_line "=== All experiments complete. Generating summary table... ===" | tee -a "$SUITE_LOG_FILE"

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
  log_line "WARNING: No eval dirs collected." | tee -a "$SUITE_LOG_FILE"
fi

log_line "=== Phase E MS+PRMBench Mix Suite DONE ===" | tee -a "$SUITE_LOG_FILE"
log_line "suite_log    : $SUITE_LOG_FILE" | tee -a "$SUITE_LOG_FILE"
log_line "summary_md   : $SUMMARY_TABLE_MD" | tee -a "$SUITE_LOG_FILE"
log_line "summary_json : $SUMMARY_TABLE_JSON" | tee -a "$SUITE_LOG_FILE"
