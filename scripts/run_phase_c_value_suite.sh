#!/usr/bin/env bash
# Phase C value-head lifecycle suite.
#
# Why this file exists:
# - C2 requires a multi-step pipeline, not a single command:
#   1) build C1 train artifacts,
#   2) build C1 eval artifacts,
#   3) train value head,
#   4) run standalone faithfulness eval.
# - This wrapper keeps those steps reproducible and reportable in one run.
#
# What this file does:
# 1. Resolve one named Phase C group (`ACTIVE_PHASE_C_GROUP`).
# 2. Launch C1 artifact preparation for train/eval splits.
# 3. Launch C2 value-head training.
# 4. Launch standalone C2 evaluation on the eval C1 artifacts.
# 5. Write one suite summary with the key C2 metrics.
#
# Interaction with other files:
# - `scripts/phase_b_prepare_value_data.py` for C1 artifacts
# - `scripts/phase_b_train_value.py` for C2 training
# - `scripts/phase_b_eval_faithfulness.py` for C2 standalone eval
#
# Example:
#   ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_SMOKE \
#   RUN_PREFIX=phase_c_strategyqa_smoke \
#   CUDA_VISIBLE_DEVICES=1 \
#   bash scripts/run_phase_c_value_suite.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACTIVE_PHASE_C_GROUP="${ACTIVE_PHASE_C_GROUP:-C2_STRATEGYQA_SMOKE}"
RUN_PREFIX="${RUN_PREFIX:-phase_c_value}"
ENABLE_PERSISTED_LOGS="${ENABLE_PERSISTED_LOGS:-1}"
CURRENT_STAGE="init"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S %z"
}

log_line() {
  local msg="$1"
  echo "[$(timestamp)] $msg"
}

append_extra_args() {
  # Append shell-split extra CLI args into one array variable by name.
  local array_name="$1"
  local extra_text="$2"
  if [[ -z "$extra_text" ]]; then
    return 0
  fi
  # shellcheck disable=SC2206
  local extra_arr=($extra_text)
  # shellcheck disable=SC2178,SC2034
  local -n target_ref="$array_name"
  target_ref+=("${extra_arr[@]}")
}

latest_phase_c_data_run_dir_for_name() {
  # Resolve latest C1 artifact run directory for one dataset/run-name prefix.
  local dataset="$1"
  local run_name="$2"
  local latest=""
  latest="$(compgen -G "assets/artifacts/phase_c_data/${dataset}/${run_name}__*" | sort | tail -n 1 || true)"
  if [[ -z "$latest" ]]; then
    echo "ERROR: No Phase C data run directory found for dataset=$dataset run_name=$run_name" >&2
    return 1
  fi
  printf '%s\n' "$latest"
}

latest_phase_c_train_run_dir_for_name() {
  # Resolve latest C2 training run directory by run-name prefix.
  local run_name="$1"
  local latest=""
  latest="$(compgen -G "assets/artifacts/phase_c_runs/${run_name}_*" | sort | tail -n 1 || true)"
  if [[ -z "$latest" ]]; then
    echo "ERROR: No Phase C training run directory found for run_name=$run_name" >&2
    return 1
  fi
  printf '%s\n' "$latest"
}

latest_phase_c_eval_run_dir_for_name() {
  # Resolve latest C2 standalone-eval run directory by run-name prefix.
  local run_name="$1"
  local latest=""
  latest="$(compgen -G "assets/artifacts/phase_c_eval/${run_name}_*" | sort | tail -n 1 || true)"
  if [[ -z "$latest" ]]; then
    echo "ERROR: No Phase C eval run directory found for run_name=$run_name" >&2
    return 1
  fi
  printf '%s\n' "$latest"
}

json_value() {
  # Read one top-level or dotted key from a JSON file.
  #
  # Example:
  #   json_value metrics.json calibration.brier_score
  local path="$1"
  local dotted_key="$2"
  "$PYTHON_BIN" - "$path" "$dotted_key" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
value = payload
for part in key.split("."):
    if isinstance(value, dict):
        value = value.get(part)
    else:
        value = None
        break
if value is None:
    print("")
elif isinstance(value, (dict, list)):
    print(json.dumps(value, ensure_ascii=False))
else:
    print(value)
PY
}

write_failure_summary() {
  # Persist a partial summary on failures so failed stage is explicit.
  local exit_code="$1"
  [[ -z "${SUMMARY_FILE:-}" ]] && return 0
  mkdir -p "$(dirname "$SUMMARY_FILE")"
  cat > "$SUMMARY_FILE" <<EOF
# Phase C Value Suite Summary

- generated_at: $(date --iso-8601=seconds)
- group_id: ${ACTIVE_PHASE_C_GROUP:-N/A}
- group_title: ${GROUP_TITLE:-N/A}
- run_prefix: ${RUN_PREFIX:-N/A}
- run_name: ${RUN_NAME:-N/A}
- status: failed
- exit_code: ${exit_code}
- failed_stage: ${CURRENT_STAGE:-unknown}
- suite_log_file: ${SUITE_LOG_FILE:-N/A}
EOF
}

on_exit() {
  # On non-zero exit, persist a failure summary.
  local exit_code="$1"
  if [[ "$exit_code" -ne 0 ]]; then
    if [[ -n "${SUITE_LOG_FILE:-}" ]]; then
      {
        log_line "Failure stage  : ${CURRENT_STAGE:-unknown}"
        log_line "Exit code      : ${exit_code}"
      } | tee -a "$SUITE_LOG_FILE" >/dev/null
    fi
    write_failure_summary "$exit_code"
  fi
}

trap 'on_exit $?' EXIT

resolve_group() {
  # Map one group id to concrete C1/C2 settings.
  case "$ACTIVE_PHASE_C_GROUP" in
    C2_STRATEGYQA_SMOKE)
      GROUP_TITLE="Phase C2 StrategyQA Smoke Lifecycle"
      GROUP_INTENTION="Validate end-to-end C1+C2 lifecycle with a small but non-trivial StrategyQA subset."
      GROUP_OBSERVE="Check run completion, metrics persistence, and non-trivial calibration/corruption signals."
      GROUP_EXPECT="No crash, valid artifacts, and meaningful C2 metrics."
      GROUP_DATASET="strategyqa"
      TRAIN_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl"
      EVAL_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"
      EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
      ROLLOUT_COUNT="${ROLLOUT_COUNT:-4}"
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
      ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-96}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-256}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-256}"
      C2_LR="${C2_LR:-1e-3}"
      C2_EPOCHS="${C2_EPOCHS:-5}"
      C2_USE_CONTRASTIVE="${C2_USE_CONTRASTIVE:-1}"
      C2_TRAIN_EXTRA_ARGS_DEFAULT="${C2_TRAIN_EXTRA_ARGS_DEFAULT:-}"
      C2_EVAL_EXTRA_ARGS_DEFAULT="${C2_EVAL_EXTRA_ARGS_DEFAULT:-}"
      ;;
    C2_STRATEGYQA_FULL)
      GROUP_TITLE="Phase C2 StrategyQA Full Lifecycle"
      GROUP_INTENTION="Run full StrategyQA C1+C2 lifecycle for report-ready value-head metrics."
      GROUP_OBSERVE="Check calibration quality and corruption sensitivity on held-out validation C1 artifacts."
      GROUP_EXPECT="Stable full-scale run with reproducible C2 output artifacts."
      GROUP_DATASET="strategyqa"
      TRAIN_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl"
      EVAL_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-}"
      EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-}"
      ROLLOUT_COUNT="${ROLLOUT_COUNT:-4}"
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
      ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-96}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-256}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-256}"
      C2_LR="${C2_LR:-1e-3}"
      C2_EPOCHS="${C2_EPOCHS:-5}"
      C2_USE_CONTRASTIVE="${C2_USE_CONTRASTIVE:-1}"
      C2_TRAIN_EXTRA_ARGS_DEFAULT="${C2_TRAIN_EXTRA_ARGS_DEFAULT:-}"
      C2_EVAL_EXTRA_ARGS_DEFAULT="${C2_EVAL_EXTRA_ARGS_DEFAULT:-}"
      ;;
    C2_STRATEGYQA_TRICK1_BCE)
      GROUP_TITLE="Phase C2 StrategyQA Trick-1 (BCE Calibration)"
      GROUP_INTENTION="Evaluate BCE-based calibration objective against legacy MSE-only objective."
      GROUP_OBSERVE="Check Brier/Pearson shifts while keeping corruption branch disabled for isolation."
      GROUP_EXPECT="If BCE helps our target distribution, raw calibration metrics should improve."
      GROUP_DATASET="strategyqa"
      TRAIN_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl"
      EVAL_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"
      EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
      ROLLOUT_COUNT="${ROLLOUT_COUNT:-8}"
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
      ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-96}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-256}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-256}"
      C2_LR="${C2_LR:-1e-4}"
      C2_EPOCHS="${C2_EPOCHS:-8}"
      C2_USE_CONTRASTIVE="${C2_USE_CONTRASTIVE:-0}"
      C2_TRAIN_EXTRA_ARGS_DEFAULT="${C2_TRAIN_EXTRA_ARGS_DEFAULT:---calibration-loss bce --calibration-bce-pos-weight 1.0}"
      C2_EVAL_EXTRA_ARGS_DEFAULT="${C2_EVAL_EXTRA_ARGS_DEFAULT:---posthoc-calibration none}"
      ;;
    C2_STRATEGYQA_TRICK2_POSTHOC_TEMP)
      GROUP_TITLE="Phase C2 StrategyQA Trick-2 (Post-hoc Temperature)"
      GROUP_INTENTION="Test whether post-hoc temperature scaling improves calibration without retraining architecture."
      GROUP_OBSERVE="Track raw vs post-hoc Brier/Pearson and save reusable calibrator payload."
      GROUP_EXPECT="Post-hoc brier should improve over raw on the same checkpoint if miscalibration dominates."
      GROUP_DATASET="strategyqa"
      TRAIN_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl"
      EVAL_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"
      EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
      ROLLOUT_COUNT="${ROLLOUT_COUNT:-8}"
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
      ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-96}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-256}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-256}"
      C2_LR="${C2_LR:-1e-4}"
      C2_EPOCHS="${C2_EPOCHS:-8}"
      C2_USE_CONTRASTIVE="${C2_USE_CONTRASTIVE:-0}"
      C2_TRAIN_EXTRA_ARGS_DEFAULT="${C2_TRAIN_EXTRA_ARGS_DEFAULT:---calibration-loss mse --posthoc-calibration temperature --checkpoint-selection-metric posthoc_brier --posthoc-temperature-lr 0.05 --posthoc-temperature-max-iters 200}"
      C2_EVAL_EXTRA_ARGS_DEFAULT="${C2_EVAL_EXTRA_ARGS_DEFAULT:---posthoc-calibration from_run}"
      ;;
    C2_STRATEGYQA_TRICK3_ADAPTIVE_BALANCE)
      GROUP_TITLE="Phase C2 StrategyQA Trick-3 (Adaptive Loss Balancing)"
      GROUP_INTENTION="Test uncertainty-based adaptive weighting for calibration + contrastive objectives."
      GROUP_OBSERVE="Check if adaptive weighting preserves calibration while recovering corruption sensitivity."
      GROUP_EXPECT="Compared with fixed lambda, adaptive weighting should reduce manual tuning sensitivity."
      GROUP_DATASET="strategyqa"
      TRAIN_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl"
      EVAL_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"
      EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
      ROLLOUT_COUNT="${ROLLOUT_COUNT:-8}"
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
      ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-96}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-256}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-256}"
      C2_LR="${C2_LR:-1e-4}"
      C2_EPOCHS="${C2_EPOCHS:-8}"
      C2_USE_CONTRASTIVE="${C2_USE_CONTRASTIVE:-1}"
      C2_TRAIN_EXTRA_ARGS_DEFAULT="${C2_TRAIN_EXTRA_ARGS_DEFAULT:---calibration-loss bce_mse --calibration-bce-weight 1.0 --calibration-mse-weight 1.0 --lambda-contrastive 0.05 --contrastive-margin 0.02 --adaptive-loss-balancing uncertainty --adaptive-loss-init-log-variance 0.0}"
      C2_EVAL_EXTRA_ARGS_DEFAULT="${C2_EVAL_EXTRA_ARGS_DEFAULT:---posthoc-calibration none}"
      ;;
    C2_STRATEGYQA_TRICK4_ISOTONIC)
      GROUP_TITLE="Phase C2 StrategyQA Trick-4 (Isotonic Post-hoc)"
      GROUP_INTENTION="Test isotonic post-hoc calibration as an alternative to temperature scaling."
      GROUP_OBSERVE="Compare raw vs isotonic post-hoc Brier/Pearson on the same checkpoints."
      GROUP_EXPECT="If score monotonicity is useful but logit scaling is insufficient, isotonic should improve post-hoc Brier."
      GROUP_DATASET="strategyqa"
      TRAIN_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl"
      EVAL_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"
      EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
      ROLLOUT_COUNT="${ROLLOUT_COUNT:-8}"
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
      ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-96}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-256}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-256}"
      C2_LR="${C2_LR:-1e-4}"
      C2_EPOCHS="${C2_EPOCHS:-8}"
      C2_USE_CONTRASTIVE="${C2_USE_CONTRASTIVE:-0}"
      C2_TRAIN_EXTRA_ARGS_DEFAULT="${C2_TRAIN_EXTRA_ARGS_DEFAULT:---calibration-loss bce --calibration-bce-pos-weight 1.0 --posthoc-calibration isotonic --checkpoint-selection-metric posthoc_brier --posthoc-isotonic-min-points 32}"
      C2_EVAL_EXTRA_ARGS_DEFAULT="${C2_EVAL_EXTRA_ARGS_DEFAULT:---posthoc-calibration from_run}"
      ;;
    C2_STRATEGYQA_TRICK5_WEIGHTED_CAL)
      GROUP_TITLE="Phase C2 StrategyQA Trick-5 (Confidence-Weighted Calibration)"
      GROUP_INTENTION="Downweight noisy uncertain prefixes during calibration fitting using rollout-derived confidence/parseability."
      GROUP_OBSERVE="Track raw Brier/Pearson changes and whether calibration beats unweighted baselines."
      GROUP_EXPECT="Weighted calibration should reduce noisy-target overfitting and improve Brier."
      GROUP_DATASET="strategyqa"
      TRAIN_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl"
      EVAL_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"
      EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
      ROLLOUT_COUNT="${ROLLOUT_COUNT:-8}"
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
      ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-96}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-256}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-256}"
      C2_LR="${C2_LR:-1e-4}"
      C2_EPOCHS="${C2_EPOCHS:-8}"
      C2_USE_CONTRASTIVE="${C2_USE_CONTRASTIVE:-0}"
      C2_TRAIN_EXTRA_ARGS_DEFAULT="${C2_TRAIN_EXTRA_ARGS_DEFAULT:---calibration-loss bce_mse --calibration-bce-weight 1.0 --calibration-mse-weight 0.5 --calibration-sample-weighting confidence_parseable --calibration-weight-floor 0.1 --calibration-weight-gamma 1.0 --posthoc-calibration temperature --checkpoint-selection-metric posthoc_brier}"
      C2_EVAL_EXTRA_ARGS_DEFAULT="${C2_EVAL_EXTRA_ARGS_DEFAULT:---posthoc-calibration from_run}"
      ;;
    C2_STRATEGYQA_TRICK6_PAIR_FILTER)
      GROUP_TITLE="Phase C2 StrategyQA Trick-6 (Contrastive Pair Filtering)"
      GROUP_INTENTION="Reduce contrastive-noise by keeping only clean prefixes with sufficient confidence and parseability."
      GROUP_OBSERVE="Check if corruption pair metrics improve without collapsing calibration."
      GROUP_EXPECT="Pair filtering should stabilize contrastive signal and improve pair accuracy/AUC."
      GROUP_DATASET="strategyqa"
      TRAIN_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl"
      EVAL_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"
      EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
      ROLLOUT_COUNT="${ROLLOUT_COUNT:-8}"
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
      ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-96}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-256}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-256}"
      C2_LR="${C2_LR:-1e-4}"
      C2_EPOCHS="${C2_EPOCHS:-8}"
      C2_USE_CONTRASTIVE="${C2_USE_CONTRASTIVE:-1}"
      C2_TRAIN_EXTRA_ARGS_DEFAULT="${C2_TRAIN_EXTRA_ARGS_DEFAULT:---calibration-loss bce --calibration-bce-pos-weight 1.0 --calibration-sample-weighting confidence --calibration-weight-floor 0.1 --lambda-contrastive 0.05 --contrastive-margin 0.02 --contrastive-pair-filter confidence_parseable --contrastive-confidence-threshold 0.2 --contrastive-parseable-threshold 0.75 --posthoc-calibration temperature --checkpoint-selection-metric posthoc_brier}"
      C2_EVAL_EXTRA_ARGS_DEFAULT="${C2_EVAL_EXTRA_ARGS_DEFAULT:---posthoc-calibration from_run}"
      ;;
    C2_STRATEGYQA_TRICK7_COMBINED)
      GROUP_TITLE="Phase C2 StrategyQA Trick-7 (Combined C2 Retry)"
      GROUP_INTENTION="Combine weighted calibration, isotonic post-hoc fallback, and filtered contrastive to maximize usable signal."
      GROUP_OBSERVE="Check whether combined settings improve both calibration and corruption discrimination together."
      GROUP_EXPECT="If signal quality is the bottleneck, this combined run should dominate earlier smoke variants."
      GROUP_DATASET="strategyqa"
      TRAIN_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl"
      EVAL_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"
      EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
      ROLLOUT_COUNT="${ROLLOUT_COUNT:-8}"
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
      ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-96}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-256}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-256}"
      C2_LR="${C2_LR:-1e-4}"
      C2_EPOCHS="${C2_EPOCHS:-10}"
      C2_USE_CONTRASTIVE="${C2_USE_CONTRASTIVE:-1}"
      C2_TRAIN_EXTRA_ARGS_DEFAULT="${C2_TRAIN_EXTRA_ARGS_DEFAULT:---calibration-loss bce_mse --calibration-bce-weight 1.0 --calibration-mse-weight 0.5 --calibration-sample-weighting confidence_parseable --calibration-weight-floor 0.1 --calibration-weight-gamma 1.0 --lambda-contrastive 0.05 --contrastive-margin 0.02 --contrastive-pair-filter confidence_parseable --contrastive-confidence-threshold 0.2 --contrastive-parseable-threshold 0.75 --adaptive-loss-balancing uncertainty --adaptive-loss-init-log-variance 0.0 --posthoc-calibration isotonic --checkpoint-selection-metric posthoc_brier --posthoc-isotonic-min-points 32}"
      C2_EVAL_EXTRA_ARGS_DEFAULT="${C2_EVAL_EXTRA_ARGS_DEFAULT:---posthoc-calibration from_run}"
      ;;
    C2_STRATEGYQA_TRICK8_LABEL_SMOOTH)
      GROUP_TITLE="Phase C2 StrategyQA Trick-8 (Calibration Target Smoothing)"
      GROUP_INTENTION="Reduce rollout-label noise by smoothing success-rate targets before calibration loss."
      GROUP_OBSERVE="Check whether Brier/Pearson improve without harming corruption metrics."
      GROUP_EXPECT="If noisy soft targets are a bottleneck, smoothing should improve calibration stability."
      GROUP_DATASET="strategyqa"
      TRAIN_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl"
      EVAL_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"
      EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
      ROLLOUT_COUNT="${ROLLOUT_COUNT:-8}"
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
      ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-96}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-256}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-256}"
      C2_LR="${C2_LR:-1e-4}"
      C2_EPOCHS="${C2_EPOCHS:-8}"
      C2_USE_CONTRASTIVE="${C2_USE_CONTRASTIVE:-0}"
      C2_TRAIN_EXTRA_ARGS_DEFAULT="${C2_TRAIN_EXTRA_ARGS_DEFAULT:---calibration-loss bce_mse --calibration-bce-weight 1.0 --calibration-mse-weight 0.5 --calibration-target-smoothing 0.05 --posthoc-calibration temperature --checkpoint-selection-metric posthoc_brier}"
      C2_EVAL_EXTRA_ARGS_DEFAULT="${C2_EVAL_EXTRA_ARGS_DEFAULT:---posthoc-calibration from_run}"
      ;;
    C2_STRATEGYQA_TRICK9_HARD_NEG_MINING)
      GROUP_TITLE="Phase C2 StrategyQA Trick-9 (Hard-Negative Pair Mining)"
      GROUP_INTENTION="Focus contrastive updates on uncertain pairs by filtering to a narrow score-gap band."
      GROUP_OBSERVE="Check pair accuracy/AUC gains while tracking calibration regressions."
      GROUP_EXPECT="If pair noise dominates, hard-negative mining should improve corruption ordering."
      GROUP_DATASET="strategyqa"
      TRAIN_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl"
      EVAL_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"
      EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
      ROLLOUT_COUNT="${ROLLOUT_COUNT:-8}"
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
      ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-96}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-256}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-256}"
      C2_LR="${C2_LR:-1e-4}"
      C2_EPOCHS="${C2_EPOCHS:-10}"
      C2_USE_CONTRASTIVE="${C2_USE_CONTRASTIVE:-1}"
      C2_TRAIN_EXTRA_ARGS_DEFAULT="${C2_TRAIN_EXTRA_ARGS_DEFAULT:---calibration-loss bce --calibration-bce-pos-weight 1.0 --calibration-target-smoothing 0.02 --lambda-contrastive 0.1 --contrastive-margin 0.02 --contrastive-pair-filter confidence_parseable --contrastive-confidence-threshold 0.2 --contrastive-parseable-threshold 0.75 --contrastive-score-gap-min 0.0 --contrastive-score-gap-max 0.2 --posthoc-calibration temperature --checkpoint-selection-metric posthoc_brier}"
      C2_EVAL_EXTRA_ARGS_DEFAULT="${C2_EVAL_EXTRA_ARGS_DEFAULT:---posthoc-calibration from_run}"
      ;;
    C2_STRATEGYQA_TRICK10_K16_COMBINED)
      GROUP_TITLE="Phase C2 StrategyQA Trick-10 (K16 + Noise-Control Combined)"
      GROUP_INTENTION="Increase rollout signal quality (K=16) and combine smoothing + weighted calibration + filtered contrastive."
      GROUP_OBSERVE="Check whether stronger labels plus stronger filtering move both calibration and corruption metrics together."
      GROUP_EXPECT="If label-noise is the dominant issue, this should beat prior K8 variants."
      GROUP_DATASET="strategyqa"
      TRAIN_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl"
      EVAL_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"
      EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
      ROLLOUT_COUNT="${ROLLOUT_COUNT:-16}"
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
      ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-96}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-256}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-256}"
      C2_LR="${C2_LR:-1e-4}"
      C2_EPOCHS="${C2_EPOCHS:-10}"
      C2_USE_CONTRASTIVE="${C2_USE_CONTRASTIVE:-1}"
      C2_TRAIN_EXTRA_ARGS_DEFAULT="${C2_TRAIN_EXTRA_ARGS_DEFAULT:---calibration-loss bce_mse --calibration-bce-weight 1.0 --calibration-mse-weight 0.5 --calibration-target-smoothing 0.03 --calibration-sample-weighting confidence_parseable --calibration-weight-floor 0.1 --lambda-contrastive 0.08 --contrastive-margin 0.02 --contrastive-pair-filter confidence_parseable --contrastive-confidence-threshold 0.2 --contrastive-parseable-threshold 0.75 --contrastive-score-gap-min 0.0 --contrastive-score-gap-max 0.25 --adaptive-loss-balancing uncertainty --adaptive-loss-init-log-variance 0.0 --posthoc-calibration isotonic --checkpoint-selection-metric posthoc_brier --posthoc-isotonic-min-points 32}"
      C2_EVAL_EXTRA_ARGS_DEFAULT="${C2_EVAL_EXTRA_ARGS_DEFAULT:---posthoc-calibration from_run}"
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_C_GROUP=$ACTIVE_PHASE_C_GROUP"
      echo "Supported groups: C2_STRATEGYQA_SMOKE, C2_STRATEGYQA_FULL, C2_STRATEGYQA_TRICK1_BCE, C2_STRATEGYQA_TRICK2_POSTHOC_TEMP, C2_STRATEGYQA_TRICK3_ADAPTIVE_BALANCE, C2_STRATEGYQA_TRICK4_ISOTONIC, C2_STRATEGYQA_TRICK5_WEIGHTED_CAL, C2_STRATEGYQA_TRICK6_PAIR_FILTER, C2_STRATEGYQA_TRICK7_COMBINED, C2_STRATEGYQA_TRICK8_LABEL_SMOOTH, C2_STRATEGYQA_TRICK9_HARD_NEG_MINING, C2_STRATEGYQA_TRICK10_K16_COMBINED"
      exit 1
      ;;
  esac
}

run_c1_prepare() {
  # Build one C1 artifact directory (train or eval split).
  local split_label="$1"
  local input_jsonl="$2"
  local run_name="$3"
  local max_samples="$4"

  local cmd=(
    "$PYTHON_BIN" -u scripts/phase_b_prepare_value_data.py
    --input-jsonl "$input_jsonl"
    --run-name "$run_name"
    --build-corruptions
    --build-rollouts
    --model-path "assets/models/Qwen2.5-7B-Instruct"
    --batch-size "$ROLLOUT_BATCH_SIZE"
    --rollout-count "$ROLLOUT_COUNT"
    --max-new-tokens "$ROLLOUT_MAX_NEW_TOKENS"
    --temperature 0.7
    --top-p 0.95
    --dtype bfloat16
    --device-map auto
    --require-cuda
  )
  if [[ -n "$max_samples" ]]; then
    cmd+=(--max-samples "$max_samples")
  fi
  append_extra_args cmd "${PHASE_C_PREP_EXTRA_ARGS:-}"

  CURRENT_STAGE="c1_prepare_${split_label}"
  {
    log_line "C1 prepare     : ${split_label}"
    log_line "C1 input        : ${input_jsonl}"
    log_line "C1 run_name     : ${run_name}"
    log_line "C1 max_samples  : ${max_samples:-<none>}"
    log_line "C1 command      : ${cmd[*]}"
  } | tee -a "$SUITE_LOG_FILE"
  "${cmd[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"
}

resolve_group

LOG_ROOT="assets/artifacts/phase_c_logs/$RUN_PREFIX"
mkdir -p "$LOG_ROOT"
SUITE_LOG_FILE="$LOG_ROOT/suite.log"
SUMMARY_FILE="$LOG_ROOT/final_summary.md"

RUN_NAME="${RUN_PREFIX}_${ACTIVE_PHASE_C_GROUP,,}"
C1_TRAIN_RUN_NAME="${RUN_NAME}_c1_train"
C1_EVAL_RUN_NAME="${RUN_NAME}_c1_eval"
C2_TRAIN_RUN_NAME="${RUN_NAME}_c2"
C2_STANDALONE_EVAL_RUN_NAME="${RUN_NAME}_c2_eval"

{
  log_line "Repo root      : $REPO_ROOT"
  log_line "Python         : $PYTHON_BIN"
  log_line "Group          : $ACTIVE_PHASE_C_GROUP"
  log_line "Group title    : $GROUP_TITLE"
  log_line "Run prefix     : $RUN_PREFIX"
  log_line "Run name       : $RUN_NAME"
  log_line "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
  log_line "Dataset        : $GROUP_DATASET"
  log_line "Train input    : $TRAIN_INPUT_JSONL"
  log_line "Eval input     : $EVAL_INPUT_JSONL"
  log_line "Train max samp : ${TRAIN_MAX_SAMPLES:-<none>}"
  log_line "Eval max samp  : ${EVAL_MAX_SAMPLES:-<none>}"
  log_line "Rollout batch  : $ROLLOUT_BATCH_SIZE"
  log_line "Rollout count  : $ROLLOUT_COUNT"
  log_line "Rollout tokens : $ROLLOUT_MAX_NEW_TOKENS"
  log_line "C2 train batch : $C2_TRAIN_BATCH_SIZE"
  log_line "C2 eval batch  : $C2_EVAL_BATCH_SIZE"
  log_line "C2 LR          : $C2_LR"
  log_line "C2 epochs      : $C2_EPOCHS"
  log_line "C2 use ctr     : $C2_USE_CONTRASTIVE"
  log_line "C2 default train extra: ${C2_TRAIN_EXTRA_ARGS_DEFAULT:-<none>}"
  log_line "C2 default eval extra : ${C2_EVAL_EXTRA_ARGS_DEFAULT:-<none>}"
  log_line "User train extra args : ${PHASE_C_TRAIN_EXTRA_ARGS:-<none>}"
  log_line "User eval extra args  : ${PHASE_C_EVAL_EXTRA_ARGS:-<none>}"
  log_line "Intention      : $GROUP_INTENTION"
  log_line "Observe        : $GROUP_OBSERVE"
  log_line "Expectation    : $GROUP_EXPECT"
  log_line "Group run start"
} | tee "$SUITE_LOG_FILE"

run_c1_prepare "train" "$TRAIN_INPUT_JSONL" "$C1_TRAIN_RUN_NAME" "${TRAIN_MAX_SAMPLES:-}"
run_c1_prepare "eval" "$EVAL_INPUT_JSONL" "$C1_EVAL_RUN_NAME" "${EVAL_MAX_SAMPLES:-}"

CURRENT_STAGE="resolve_c1_dirs"
C1_TRAIN_DIR="$(latest_phase_c_data_run_dir_for_name "$GROUP_DATASET" "$C1_TRAIN_RUN_NAME")"
C1_EVAL_DIR="$(latest_phase_c_data_run_dir_for_name "$GROUP_DATASET" "$C1_EVAL_RUN_NAME")"
{
  log_line "Resolved C1 train dir: $C1_TRAIN_DIR"
  log_line "Resolved C1 eval dir : $C1_EVAL_DIR"
} | tee -a "$SUITE_LOG_FILE"

CURRENT_STAGE="c2_train"
c2_train_cmd=(
  "$PYTHON_BIN" -u scripts/phase_b_train_value.py
  --train-dir "$C1_TRAIN_DIR"
  --eval-dir "$C1_EVAL_DIR"
  --run-name "$C2_TRAIN_RUN_NAME"
  --require-cuda
  --dtype bfloat16
  --device-map auto
  --per-device-train-batch-size "$C2_TRAIN_BATCH_SIZE"
  --per-device-eval-batch-size "$C2_EVAL_BATCH_SIZE"
  --learning-rate "$C2_LR"
  --num-train-epochs "$C2_EPOCHS"
)
if [[ "$C2_USE_CONTRASTIVE" == "1" ]]; then
  c2_train_cmd+=(--use-contrastive-loss --lambda-contrastive 1.0 --contrastive-margin 0.1)
else
  c2_train_cmd+=(--no-use-contrastive-loss)
fi
append_extra_args c2_train_cmd "${C2_TRAIN_EXTRA_ARGS_DEFAULT:-}"
append_extra_args c2_train_cmd "${PHASE_C_TRAIN_EXTRA_ARGS:-}"
{
  log_line "C2 train command: ${c2_train_cmd[*]}"
} | tee -a "$SUITE_LOG_FILE"
"${c2_train_cmd[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

CURRENT_STAGE="resolve_c2_train_dir"
C2_TRAIN_DIR="$(latest_phase_c_train_run_dir_for_name "$C2_TRAIN_RUN_NAME")"
{
  log_line "Resolved C2 train dir: $C2_TRAIN_DIR"
} | tee -a "$SUITE_LOG_FILE"

CURRENT_STAGE="c2_eval"
c2_eval_cmd=(
  "$PYTHON_BIN" -u scripts/phase_b_eval_faithfulness.py
  --value-run-dir "$C2_TRAIN_DIR"
  --eval-dir "$C1_EVAL_DIR"
  --checkpoint-name best
  --run-name "$C2_STANDALONE_EVAL_RUN_NAME"
)
append_extra_args c2_eval_cmd "${C2_EVAL_EXTRA_ARGS_DEFAULT:-}"
append_extra_args c2_eval_cmd "${PHASE_C_EVAL_EXTRA_ARGS:-}"
{
  log_line "C2 eval command : ${c2_eval_cmd[*]}"
} | tee -a "$SUITE_LOG_FILE"
"${c2_eval_cmd[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

CURRENT_STAGE="resolve_c2_eval_dir"
C2_EVAL_DIR="$(latest_phase_c_eval_run_dir_for_name "$C2_STANDALONE_EVAL_RUN_NAME")"
C2_EVAL_METRICS="$C2_EVAL_DIR/metrics.json"
C2_BRIER="$(json_value "$C2_EVAL_METRICS" "calibration.brier_score")"
C2_PEARSON="$(json_value "$C2_EVAL_METRICS" "calibration.pearson")"
C2_ECE="$(json_value "$C2_EVAL_METRICS" "calibration.ece")"
C2_POST_BRIER="$(json_value "$C2_EVAL_METRICS" "calibration_posthoc.brier_score")"
C2_POST_PEARSON="$(json_value "$C2_EVAL_METRICS" "calibration_posthoc.pearson")"
C2_PAIR_ACC="$(json_value "$C2_EVAL_METRICS" "corruption.pair_accuracy")"
C2_AUC="$(json_value "$C2_EVAL_METRICS" "corruption.auc_clean_vs_corrupt")"

CURRENT_STAGE="final_summary"
cat > "$SUMMARY_FILE" <<EOF
# Phase C Value Suite Summary

- generated_at: $(date --iso-8601=seconds)
- group_id: $ACTIVE_PHASE_C_GROUP
- group_title: $GROUP_TITLE
- run_prefix: $RUN_PREFIX
- run_name: $RUN_NAME
- dataset: $GROUP_DATASET
- train_input_jsonl: $TRAIN_INPUT_JSONL
- eval_input_jsonl: $EVAL_INPUT_JSONL
- c1_train_dir: $C1_TRAIN_DIR
- c1_eval_dir: $C1_EVAL_DIR
- c2_train_dir: $C2_TRAIN_DIR
- c2_eval_dir: $C2_EVAL_DIR
- suite_log_file: $SUITE_LOG_FILE

## C2 Key Metrics

- brier_score: ${C2_BRIER:-N/A}
- pearson: ${C2_PEARSON:-N/A}
- ece: ${C2_ECE:-N/A}
- posthoc_brier_score: ${C2_POST_BRIER:-N/A}
- posthoc_pearson: ${C2_POST_PEARSON:-N/A}
- corruption_pair_accuracy: ${C2_PAIR_ACC:-N/A}
- corruption_auc_clean_vs_corrupt: ${C2_AUC:-N/A}
EOF

{
  log_line "Summary file   : $SUMMARY_FILE"
  log_line "C2 eval metrics: $C2_EVAL_METRICS"
  log_line "Group run complete"
} | tee -a "$SUITE_LOG_FILE"
