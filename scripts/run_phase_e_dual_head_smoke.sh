#!/usr/bin/env bash
# Changelog (prepend-only, newest first):
# 2026-03-11: Initial version. Dual-head routing smoke experiment.
#
# Dual-Head Routing Smoke Suite
# ==============================
# English
# -------
# Validates that `dual_head` architecture with pair_semantics routing
# (local_proj for first_bad/fanout/grid pairs, terminal_proj for terminal_anchor
# pairs) outperforms or matches mlp baseline on ProcessBench after the routing
# logic was confirmed implemented in training.py.
#
# Prerequisites:
#   - ms_align_v1 pair artifact already curated (reused from PBR2a)
#   - Feature cache already populated by PBR2a run (or will be auto-built)
#   - This script runs on GPU0 (CUDA_VISIBLE_DEVICES=0)
#
# Comparison baseline:
#   - mlp + ms_align_v1 (PBR2a, running in parallel on GPU1)
#
# 中文
# ----
# 验证 dual_head 架构加 pair_semantics 路由（local_proj 接 first_bad/fanout/grid，
# terminal_proj 接 terminal_anchor）是否优于 mlp 基线。
# 前置条件：ms_align_v1 pair artifact 已经由 PBR2a 准备好，直接复用。
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-assets/models/Qwen2.5-7B-Instruct}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-assets/artifacts/phase_e_feature_cache}"
VALUE_OUTPUT_ROOT="${VALUE_OUTPUT_ROOT:-assets/artifacts/phase_e_runs}"
BENCH_OUTPUT_ROOT="${BENCH_OUTPUT_ROOT:-assets/artifacts/phase_e_eval}"
LOG_ROOT="assets/artifacts/phase_e_logs/phase_e_dual_head_smoke"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-96}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-4}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
HEAD_MLP_HIDDEN_SIZE="${HEAD_MLP_HIDDEN_SIZE:-1024}"
HEAD_DROPOUT_PROB="${HEAD_DROPOUT_PROB:-0.05}"
BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-96}"

# Reuse the existing ms_align_v1 artifact from PBR2a.
# If the PBR2a version not found, fall back to the PBR research version.
MSALIGN_PAIRS_DIR="${MSALIGN_PAIRS_DIR:-}"
if [[ -z "$MSALIGN_PAIRS_DIR" ]]; then
  for candidate in \
    "assets/artifacts/phase_e_pairs/phase_e_processbench_research_pbr2a_ms_align_mlp_s42_ms_align_v1_pairs__2a63ed682f78" \
    "assets/artifacts/phase_e_pairs/phase_e_pbr2_0311_pbr2a_ms_align_mlp_s42_ms_align_v1_pairs__2a63ed682f78"; do
    if [[ -d "$candidate" ]]; then
      MSALIGN_PAIRS_DIR="$candidate"
      break
    fi
  done
fi

if [[ -z "$MSALIGN_PAIRS_DIR" ]]; then
  echo "ERROR: ms_align_v1 pair artifact not found. Run PBR2a first." >&2
  exit 1
fi

mkdir -p "$LOG_ROOT"

log_line() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1"
}

log_to_suite() {
  log_line "$1" | tee -a "$SUITE_LOG_FILE"
}

run_train() {
  local run_name="$1"
  local head_arch="$2"
  local seed="${3:-42}"
  log_to_suite "TRAIN: $run_name (head=$head_arch seed=$seed)"
  log_to_suite "RUN: $PYTHON_BIN -u scripts/phase_e_train_value.py \
    --train-pairs-jsonl $MSALIGN_PAIRS_DIR/train_pairs.jsonl \
    --eval-pairs-jsonl $MSALIGN_PAIRS_DIR/validation_pairs.jsonl \
    --model-path $MODEL_PATH \
    --run-name $run_name \
    --output-root $VALUE_OUTPUT_ROOT \
    --objective-mode joint \
    --learning-rate $LEARNING_RATE \
    --num-train-epochs $TRAIN_EPOCHS \
    --per-device-train-batch-size $TRAIN_BATCH_SIZE \
    --per-device-eval-batch-size $EVAL_BATCH_SIZE \
    --max-length $MAX_LENGTH \
    --lambda-ranking 1.0 \
    --lambda-bce 1.0 \
    --ranking-margin 0.02 \
    --ranking-target-space logit \
    --pair-weight-mode confidence_semantic \
    --source-balance none \
    --permutation-mode stable_hash \
    --checkpoint-selection-metric ranking_score \
    --seed $seed \
    --feature-cache-root $FEATURE_CACHE_ROOT \
    --feature-cache-mode read_write \
    --feature-cache-lock-timeout-sec 600 \
    --head-architecture $head_arch \
    --head-mlp-hidden-size $HEAD_MLP_HIDDEN_SIZE \
    --head-dropout-prob $HEAD_DROPOUT_PROB \
    --head-init-std 0.02 \
    --head-activation gelu \
    --anti-saturation-weight 5e-4 \
    --anti-saturation-logit-threshold 3.5 \
    --require-cuda"
  $PYTHON_BIN -u scripts/phase_e_train_value.py \
    --train-pairs-jsonl "$MSALIGN_PAIRS_DIR/train_pairs.jsonl" \
    --eval-pairs-jsonl "$MSALIGN_PAIRS_DIR/validation_pairs.jsonl" \
    --model-path "$MODEL_PATH" \
    --run-name "$run_name" \
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
    --ranking-target-space logit \
    --pair-weight-mode confidence_semantic \
    --source-balance none \
    --permutation-mode stable_hash \
    --checkpoint-selection-metric ranking_score \
    --seed "$seed" \
    --feature-cache-root "$FEATURE_CACHE_ROOT" \
    --feature-cache-mode read_write \
    --feature-cache-lock-timeout-sec 600 \
    --head-architecture "$head_arch" \
    --head-mlp-hidden-size "$HEAD_MLP_HIDDEN_SIZE" \
    --head-dropout-prob "$HEAD_DROPOUT_PROB" \
    --head-init-std 0.02 \
    --head-activation gelu \
    --anti-saturation-weight 5e-4 \
    --anti-saturation-logit-threshold 3.5 \
    --require-cuda \
    2>&1 | tee -a "$LOG_ROOT/${run_name}.log"
}

latest_run_dir() {
  local prefix="$1"
  $PYTHON_BIN - "$prefix" <<'PY'
from pathlib import Path
import sys
prefix = sys.argv[1]
parent = Path(prefix).parent
stem = Path(prefix).name
matches = sorted(parent.glob(f"{stem}*"), key=lambda p: p.stat().st_mtime, reverse=True)
if not matches:
    print("", end="")
else:
    print(matches[0], end="")
PY
}

run_eval() {
  local value_run_dir="$1"
  local benchmark_id="$2"
  if [[ -z "$value_run_dir" ]]; then
    log_to_suite "EVAL SKIP: no run dir for benchmark=$benchmark_id"
    return 0
  fi
  local eval_run_name
  eval_run_name="$(basename "$value_run_dir")_${benchmark_id}"
  log_to_suite "EVAL: $eval_run_name benchmark=$benchmark_id"
  $PYTHON_BIN -u scripts/phase_e_eval_benchmark.py \
    --value-run-dir "$value_run_dir" \
    --benchmark-id "$benchmark_id" \
    --output-root "$BENCH_OUTPUT_ROOT" \
    --model-path "$MODEL_PATH" \
    --max-samples "$BENCH_MAX_SAMPLES" \
    --feature-cache-root "$FEATURE_CACHE_ROOT" \
    --feature-cache-mode read_write \
    2>&1 | tee -a "$LOG_ROOT/${eval_run_name}.log"
}

# -------------------------------------------------------------------------
# Main experiment: 3 architectures on same ms_align_v1 artifact
# 1. mlp  (for fresh comparison under same script settings)
# 2. gated_mlp
# 3. dual_head (the new routing-enabled architecture)
# -------------------------------------------------------------------------
log_to_suite "==================================================================="
log_to_suite "Dual-Head Routing Smoke Suite"
log_to_suite "Pairs dir: $MSALIGN_PAIRS_DIR"
log_to_suite "==================================================================="

# Case A: gated_mlp (already in PBR1/PBR2 but run fresh here for direct comparison)
RUN_PREFIX_DUALSMK="phase_e_dualhead_smoke_0311"

log_to_suite "--- Case gated_mlp ---"
run_train "${RUN_PREFIX_DUALSMK}_gated_mlp_s42" "gated_mlp" "42"
GATED_DIR=$(latest_run_dir "${VALUE_OUTPUT_ROOT}/${RUN_PREFIX_DUALSMK}_gated_mlp_s42")
run_eval "$GATED_DIR" "processbench_math"
run_eval "$GATED_DIR" "processbench_gsm8k"

log_to_suite "--- Case dual_head ---"
run_train "${RUN_PREFIX_DUALSMK}_dual_head_s42" "dual_head" "42"
DUAL_DIR=$(latest_run_dir "${VALUE_OUTPUT_ROOT}/${RUN_PREFIX_DUALSMK}_dual_head_s42")
run_eval "$DUAL_DIR" "processbench_math"
run_eval "$DUAL_DIR" "processbench_gsm8k"

log_to_suite "==================================================================="
log_to_suite "Dual-head smoke done. Check: $BENCH_OUTPUT_ROOT"
log_to_suite "==================================================================="
