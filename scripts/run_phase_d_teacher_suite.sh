#!/usr/bin/env bash
# Phase D external-PRM-supported value-learning suite.
#
# Why this file exists:
# - Phase D requires a chained workflow that is tedious to run manually:
#   1) C1 train/eval preparation with teacher-score fusion (D2),
#   2) C2 training under multiple target-source settings (D3),
#   3) standalone eval per C2 run,
#   4) one consolidated summary table for quick comparison.
#
# What this file does:
# 1. Resolve one named Phase D group (`ACTIVE_PHASE_D_GROUP`).
# 2. Build D2 C1 train/eval artifacts once.
# 3. Run a bundled D3 target set defined by each group
#    (legacy D4: `q_mean_smoothed/q_teacher/q_fused`; D5/D6: `q_mean_smoothed`).
# 4. Collect key metrics into one final Markdown report.
#
# Interaction with other files:
# - `scripts/phase_b_prepare_value_data.py` (D2 fusion fields)
# - `scripts/phase_b_train_value.py` (D3 target-source switch)
# - `scripts/phase_b_eval_faithfulness.py` (standalone C2 eval)
#
# Example:
#   ACTIVE_PHASE_D_GROUP=D4_STRATEGYQA_SMOKE_3WAY_HQ \
#   RUN_PREFIX=phase_d_smoke_bundle \
#   CUDA_VISIBLE_DEVICES=1 \
#   bash scripts/run_phase_d_teacher_suite.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACTIVE_PHASE_D_GROUP="${ACTIVE_PHASE_D_GROUP:-D4_STRATEGYQA_SMOKE_3WAY_HQ}"
RUN_PREFIX="${RUN_PREFIX:-phase_d_teacher_bundle}"
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

json_value() {
  # Read one dotted key from a JSON file.
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

latest_phase_c_data_run_dir_for_name() {
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
  local run_name="$1"
  local latest=""
  latest="$(compgen -G "assets/artifacts/phase_c_eval/${run_name}_*" | sort | tail -n 1 || true)"
  if [[ -z "$latest" ]]; then
    echo "ERROR: No Phase C eval run directory found for run_name=$run_name" >&2
    return 1
  fi
  printf '%s\n' "$latest"
}

write_failure_summary() {
  local exit_code="$1"
  [[ -z "${SUMMARY_FILE:-}" ]] && return 0
  mkdir -p "$(dirname "$SUMMARY_FILE")"
  cat > "$SUMMARY_FILE" <<EOF
# Phase D Teacher Suite Summary

- generated_at: $(date --iso-8601=seconds)
- group_id: ${ACTIVE_PHASE_D_GROUP:-N/A}
- group_title: ${GROUP_TITLE:-N/A}
- run_prefix: ${RUN_PREFIX:-N/A}
- status: failed
- exit_code: ${exit_code}
- failed_stage: ${CURRENT_STAGE:-unknown}
- suite_log_file: ${SUITE_LOG_FILE:-N/A}
EOF
}

on_exit() {
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
  # 中文：这里集中定义 Phase D 成套实验配方，便于团队复用和对比。
  case "$ACTIVE_PHASE_D_GROUP" in
    D4_STRATEGYQA_SMOKE_3WAY)
      GROUP_TITLE="Phase D StrategyQA Smoke 3-Way Target Ablation"
      GROUP_INTENTION="Bundle D2 fusion prep and D3 target-source ablations for fast, comparable diagnostics."
      GROUP_OBSERVE="Compare mc vs teacher vs fused under the same C1/C2 recipe and budgets."
      GROUP_EXPECT="If teacher/fusion helps, ranking/calibration should improve beyond MC baseline."
      GROUP_DATASET="strategyqa"
      TRAIN_INPUT_JSONL="${TRAIN_INPUT_JSONL:-assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl}"
      EVAL_INPUT_JSONL="${EVAL_INPUT_JSONL:-assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl}"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"
      EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-192}"
      ROLLOUT_COUNT="${ROLLOUT_COUNT:-16}"
      ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-128}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-192}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-192}"
      C2_LR="${C2_LR:-1e-4}"
      C2_EPOCHS="${C2_EPOCHS:-10}"
      # Teacher score inputs: choose already-computed high-coverage sidecars.
      TEACHER_TRAIN_SCORES="${TEACHER_TRAIN_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_full_mcorr4_c2_strategyqa_quality_first_full_c1_train__b7a8789f1974/teacher_prefix_scores.jsonl}"
      TEACHER_EVAL_SCORES="${TEACHER_EVAL_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_c2_strategyqa_quality_first_c1_eval__f608255f810d/teacher_prefix_scores.jsonl}"
      TEACHER_FUSE_MODE="${TEACHER_FUSE_MODE:-confidence}"
      TEACHER_FUSION_LAMBDA="${TEACHER_FUSION_LAMBDA:-0.5}"
      TEACHER_CONF_CI_REF="${TEACHER_CONF_CI_REF:-0.30}"
      TEACHER_DISAGREE_THRESH="${TEACHER_DISAGREE_THRESH:-0.25}"
      TEACHER_MIN_COVERAGE="${TEACHER_MIN_COVERAGE:-0.98}"
      C1_PREP_EXTRA_ARGS_DEFAULT="${C1_PREP_EXTRA_ARGS_DEFAULT:---corruption-selection-policy cqr_balanced --max-corruptions-per-prefix 4 --min-non-step-drop-per-prefix 1 --max-step-drop-per-prefix 1 --enable-negation-flip --enable-comparator-flip --enable-condition-reversal --enable-entity-substitution --build-pair-quality --pair-rollout-count 16 --target-alpha 1.0 --target-beta 1.0 --target-ci-z 1.96 --target-weight-floor 0.1 --target-weight-gamma 1.0 --pair-delta-q-min 0.15 --pair-z-min 0.5 --rollout-two-stage --rollout-stage1-count 8 --rollout-stage2-count 24 --rollout-uncertain-band 0.2 --rollout-uncertain-ci-width 0.3}"
      C2_TRAIN_EXTRA_ARGS_DEFAULT="${C2_TRAIN_EXTRA_ARGS_DEFAULT:---use-contrastive-loss --calibration-loss bce_mse --calibration-bce-weight 1.0 --calibration-mse-weight 0.5 --calibration-sample-weighting q_weight_parseable --calibration-weight-floor 0.1 --calibration-weight-gamma 1.0 --lambda-contrastive 0.08 --contrastive-margin 0.02 --contrastive-pair-filter confidence_parseable_label --contrastive-confidence-threshold 0.2 --contrastive-parseable-threshold 0.75 --contrastive-label-delta-q-min 0.15 --contrastive-label-z-min 0.5 --contrastive-label-pair-weight-min 0.25 --contrastive-require-pair-pass-gate --contrastive-use-pair-weights --contrastive-stratified-sampling --contrastive-stratify-step-bucket-size 2 --contrastive-stratify-include-no-corruption --posthoc-calibration temperature --checkpoint-selection-metric posthoc_brier}"
      C2_EVAL_EXTRA_ARGS_DEFAULT="${C2_EVAL_EXTRA_ARGS_DEFAULT:---target-source from_run --target-source-missing-policy from_run --posthoc-calibration from_run}"
      D3_TARGETS="${D3_TARGETS:-q_mean_smoothed q_teacher q_fused}"
      ;;
    D4_STRATEGYQA_SMOKE_3WAY_HQ)
      GROUP_TITLE="Phase D StrategyQA Smoke 3-Way (HQ Pair + Two-Stage C2)"
      GROUP_INTENTION="Use quality-first pair consensus and two-stage C2 to test whether teacher-supported labels become truly learnable."
      GROUP_OBSERVE="Compare mc vs teacher vs fused under stronger pair-quality controls and top-k contrastive candidates."
      GROUP_EXPECT="If supervision is the blocker, HQ pair controls should lift corruption-ordering metrics beyond the legacy smoke baseline."
      GROUP_DATASET="strategyqa"
      TRAIN_INPUT_JSONL="${TRAIN_INPUT_JSONL:-assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl}"
      EVAL_INPUT_JSONL="${EVAL_INPUT_JSONL:-assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl}"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"
      EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-192}"
      ROLLOUT_COUNT="${ROLLOUT_COUNT:-16}"
      ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-128}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-192}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-192}"
      C2_LR="${C2_LR:-1e-4}"
      C2_EPOCHS="${C2_EPOCHS:-10}"
      TEACHER_TRAIN_SCORES="${TEACHER_TRAIN_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_full_mcorr4_c2_strategyqa_quality_first_full_c1_train__b7a8789f1974/teacher_prefix_scores.jsonl}"
      TEACHER_EVAL_SCORES="${TEACHER_EVAL_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_c2_strategyqa_quality_first_c1_eval__f608255f810d/teacher_prefix_scores.jsonl}"
      TEACHER_TRAIN_CORR_SCORES="${TEACHER_TRAIN_CORR_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_full_mcorr4_c2_strategyqa_quality_first_full_c1_train__b7a8789f1974/teacher_corruption_scores.jsonl}"
      TEACHER_EVAL_CORR_SCORES="${TEACHER_EVAL_CORR_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_c2_strategyqa_quality_first_c1_eval__f608255f810d/teacher_corruption_scores.jsonl}"
      TEACHER_FUSE_MODE="${TEACHER_FUSE_MODE:-confidence}"
      TEACHER_FUSION_LAMBDA="${TEACHER_FUSION_LAMBDA:-0.5}"
      TEACHER_CONF_CI_REF="${TEACHER_CONF_CI_REF:-0.30}"
      TEACHER_DISAGREE_THRESH="${TEACHER_DISAGREE_THRESH:-0.25}"
      TEACHER_MIN_COVERAGE="${TEACHER_MIN_COVERAGE:-0.98}"
      C1_PREP_EXTRA_ARGS_DEFAULT="${C1_PREP_EXTRA_ARGS_DEFAULT:---corruption-selection-policy cqr_balanced --max-corruptions-per-prefix 4 --min-non-step-drop-per-prefix 1 --max-step-drop-per-prefix 1 --enable-negation-flip --enable-comparator-flip --enable-condition-reversal --enable-entity-substitution --build-pair-quality --pair-rollout-count 16 --target-alpha 1.0 --target-beta 1.0 --target-ci-z 1.96 --target-weight-floor 0.1 --target-weight-gamma 1.0 --pair-delta-q-min 0.15 --pair-z-min 0.5 --rollout-two-stage --rollout-stage1-count 8 --rollout-stage2-count 24 --rollout-uncertain-band 0.2 --rollout-uncertain-ci-width 0.3 --pair-consensus-enable --pair-consensus-teacher-delta-q-min 0.03 --pair-consensus-weight-boost 1.25 --pair-consensus-fail-weight-scale 0.20 --no-pair-consensus-require-teacher-coverage}"
      C2_TRAIN_EXTRA_ARGS_DEFAULT="${C2_TRAIN_EXTRA_ARGS_DEFAULT:---train-mode two_stage --two-stage-ranking-ratio 0.45 --contrastive-max-corruptions-per-prefix 4 --use-contrastive-loss --calibration-loss bce_mse --calibration-bce-weight 1.0 --calibration-mse-weight 0.5 --calibration-sample-weighting q_weight_parseable --calibration-weight-floor 0.1 --calibration-weight-gamma 1.0 --lambda-contrastive 0.08 --contrastive-margin 0.02 --contrastive-pair-filter confidence_parseable_label --contrastive-confidence-threshold 0.2 --contrastive-parseable-threshold 0.75 --contrastive-label-delta-q-min 0.15 --contrastive-label-z-min 0.5 --contrastive-label-pair-weight-min 0.25 --contrastive-require-pair-pass-gate --contrastive-use-pair-weights --contrastive-stratified-sampling --contrastive-stratify-step-bucket-size 2 --contrastive-stratify-include-no-corruption --posthoc-calibration temperature --checkpoint-selection-metric posthoc_brier}"
      C2_EVAL_EXTRA_ARGS_DEFAULT="${C2_EVAL_EXTRA_ARGS_DEFAULT:---target-source from_run --target-source-missing-policy from_run --posthoc-calibration from_run}"
      D3_TARGETS="${D3_TARGETS:-q_mean_smoothed q_teacher q_fused}"
      ;;
    D4_STRATEGYQA_FULL_3WAY_HQ)
      GROUP_TITLE="Phase D StrategyQA Full 3-Way (HQ Pair + Two-Stage C2)"
      GROUP_INTENTION="Promotion-grade full-corpus rerun with quality-first pair controls and teacher-informed labels."
      GROUP_OBSERVE="Check whether smoke trends hold under full data and stable pair diagnostics."
      GROUP_EXPECT="If improvements are real, full run should keep or improve pair metrics while preserving calibration competitiveness."
      GROUP_DATASET="strategyqa"
      TRAIN_INPUT_JSONL="${TRAIN_INPUT_JSONL:-assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl}"
      EVAL_INPUT_JSONL="${EVAL_INPUT_JSONL:-assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl}"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-}"
      EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-}"
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-192}"
      ROLLOUT_COUNT="${ROLLOUT_COUNT:-16}"
      ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-128}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-192}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-192}"
      C2_LR="${C2_LR:-1e-4}"
      C2_EPOCHS="${C2_EPOCHS:-12}"
      TEACHER_TRAIN_SCORES="${TEACHER_TRAIN_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_full_mcorr4_c2_strategyqa_quality_first_full_c1_train__b7a8789f1974/teacher_prefix_scores.jsonl}"
      TEACHER_EVAL_SCORES="${TEACHER_EVAL_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_c2_strategyqa_quality_first_c1_eval__f608255f810d/teacher_prefix_scores.jsonl}"
      TEACHER_TRAIN_CORR_SCORES="${TEACHER_TRAIN_CORR_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_full_mcorr4_c2_strategyqa_quality_first_full_c1_train__b7a8789f1974/teacher_corruption_scores.jsonl}"
      TEACHER_EVAL_CORR_SCORES="${TEACHER_EVAL_CORR_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_c2_strategyqa_quality_first_c1_eval__f608255f810d/teacher_corruption_scores.jsonl}"
      TEACHER_FUSE_MODE="${TEACHER_FUSE_MODE:-confidence}"
      TEACHER_FUSION_LAMBDA="${TEACHER_FUSION_LAMBDA:-0.5}"
      TEACHER_CONF_CI_REF="${TEACHER_CONF_CI_REF:-0.30}"
      TEACHER_DISAGREE_THRESH="${TEACHER_DISAGREE_THRESH:-0.25}"
      TEACHER_MIN_COVERAGE="${TEACHER_MIN_COVERAGE:-0.98}"
      C1_PREP_EXTRA_ARGS_DEFAULT="${C1_PREP_EXTRA_ARGS_DEFAULT:---corruption-selection-policy cqr_balanced --max-corruptions-per-prefix 4 --min-non-step-drop-per-prefix 1 --max-step-drop-per-prefix 1 --enable-negation-flip --enable-comparator-flip --enable-condition-reversal --enable-entity-substitution --build-pair-quality --pair-rollout-count 16 --target-alpha 1.0 --target-beta 1.0 --target-ci-z 1.96 --target-weight-floor 0.1 --target-weight-gamma 1.0 --pair-delta-q-min 0.20 --pair-z-min 0.6 --rollout-two-stage --rollout-stage1-count 8 --rollout-stage2-count 24 --rollout-uncertain-band 0.2 --rollout-uncertain-ci-width 0.3 --pair-consensus-enable --pair-consensus-teacher-delta-q-min 0.05 --pair-consensus-weight-boost 1.25 --pair-consensus-fail-weight-scale 0.20 --no-pair-consensus-require-teacher-coverage}"
      C2_TRAIN_EXTRA_ARGS_DEFAULT="${C2_TRAIN_EXTRA_ARGS_DEFAULT:---train-mode two_stage --two-stage-ranking-ratio 0.45 --contrastive-max-corruptions-per-prefix 4 --use-contrastive-loss --calibration-loss bce_mse --calibration-bce-weight 1.0 --calibration-mse-weight 0.5 --calibration-sample-weighting q_weight_parseable --calibration-weight-floor 0.1 --calibration-weight-gamma 1.0 --lambda-contrastive 0.08 --contrastive-margin 0.02 --contrastive-pair-filter confidence_parseable_label --contrastive-confidence-threshold 0.2 --contrastive-parseable-threshold 0.75 --contrastive-label-delta-q-min 0.20 --contrastive-label-z-min 0.6 --contrastive-label-pair-weight-min 0.25 --contrastive-require-pair-pass-gate --contrastive-use-pair-weights --contrastive-stratified-sampling --contrastive-stratify-step-bucket-size 2 --contrastive-stratify-include-no-corruption --posthoc-calibration temperature --checkpoint-selection-metric posthoc_brier}"
      C2_EVAL_EXTRA_ARGS_DEFAULT="${C2_EVAL_EXTRA_ARGS_DEFAULT:---target-source from_run --target-source-missing-policy from_run --posthoc-calibration from_run}"
      D3_TARGETS="${D3_TARGETS:-q_mean_smoothed q_teacher q_fused}"
      ;;
    D5_STRATEGYQA_SMOKE_MC_CTRL)
      GROUP_TITLE="Phase D5 StrategyQA Smoke (MC Target, No PRM Pair Gate)"
      GROUP_INTENTION="Method-correction control: keep MC as the only training target and disable teacher-based pair consensus."
      GROUP_OBSERVE="This is the control arm for assessing whether PRM should be used as ranking/filter signal instead of direct target supervision."
      GROUP_EXPECT="Compared with D5 pair-gated run, this run isolates the effect of PRM pair-quality gating."
      GROUP_DATASET="strategyqa"
      TRAIN_INPUT_JSONL="${TRAIN_INPUT_JSONL:-assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl}"
      EVAL_INPUT_JSONL="${EVAL_INPUT_JSONL:-assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl}"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"
      EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-192}"
      ROLLOUT_COUNT="${ROLLOUT_COUNT:-16}"
      ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-128}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-192}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-192}"
      C2_LR="${C2_LR:-1e-4}"
      C2_EPOCHS="${C2_EPOCHS:-10}"
      TEACHER_TRAIN_SCORES="${TEACHER_TRAIN_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_full_mcorr4_c2_strategyqa_quality_first_full_c1_train__b7a8789f1974/teacher_prefix_scores.jsonl}"
      TEACHER_EVAL_SCORES="${TEACHER_EVAL_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_c2_strategyqa_quality_first_c1_eval__f608255f810d/teacher_prefix_scores.jsonl}"
      TEACHER_FUSE_MODE="${TEACHER_FUSE_MODE:-none}"
      TEACHER_FUSION_LAMBDA="${TEACHER_FUSION_LAMBDA:-0.5}"
      TEACHER_CONF_CI_REF="${TEACHER_CONF_CI_REF:-0.30}"
      TEACHER_DISAGREE_THRESH="${TEACHER_DISAGREE_THRESH:-0.25}"
      TEACHER_MIN_COVERAGE="${TEACHER_MIN_COVERAGE:-0.55}"
      C1_PREP_EXTRA_ARGS_DEFAULT="${C1_PREP_EXTRA_ARGS_DEFAULT:---corruption-selection-policy cqr_balanced --max-corruptions-per-prefix 4 --min-non-step-drop-per-prefix 1 --max-step-drop-per-prefix 1 --enable-negation-flip --enable-comparator-flip --enable-condition-reversal --enable-entity-substitution --build-pair-quality --pair-rollout-count 16 --target-alpha 1.0 --target-beta 1.0 --target-ci-z 1.96 --target-weight-floor 0.1 --target-weight-gamma 1.0 --pair-delta-q-min 0.15 --pair-z-min 0.5 --rollout-two-stage --rollout-stage1-count 8 --rollout-stage2-count 24 --rollout-uncertain-band 0.2 --rollout-uncertain-ci-width 0.3}"
      C2_TRAIN_EXTRA_ARGS_DEFAULT="${C2_TRAIN_EXTRA_ARGS_DEFAULT:---train-mode two_stage --two-stage-ranking-ratio 0.45 --contrastive-max-corruptions-per-prefix 4 --use-contrastive-loss --calibration-loss bce_mse --calibration-bce-weight 1.0 --calibration-mse-weight 0.5 --calibration-sample-weighting q_weight_parseable --calibration-weight-floor 0.1 --calibration-weight-gamma 1.0 --lambda-contrastive 0.08 --contrastive-margin 0.02 --contrastive-pair-filter label_quality --contrastive-label-delta-q-min 0.15 --contrastive-label-z-min 0.5 --contrastive-label-pair-weight-min 0.25 --contrastive-require-pair-pass-gate --contrastive-use-pair-weights --contrastive-stratified-sampling --contrastive-stratify-step-bucket-size 2 --contrastive-stratify-include-no-corruption --posthoc-calibration temperature --checkpoint-selection-metric posthoc_brier}"
      C2_EVAL_EXTRA_ARGS_DEFAULT="${C2_EVAL_EXTRA_ARGS_DEFAULT:---target-source from_run --target-source-missing-policy from_run --posthoc-calibration from_run}"
      D3_TARGETS="${D3_TARGETS:-q_mean_smoothed}"
      ;;
    D5_STRATEGYQA_SMOKE_MC_PRM_PAIR_GATE)
      GROUP_TITLE="Phase D5 StrategyQA Smoke (MC Target + PRM Pair Gate)"
      GROUP_INTENTION="Method-correction main arm: keep MC as training target, use PRM only for pair-quality consensus/filtering."
      GROUP_OBSERVE="Compare against D5 MC control to test whether PRM helps as ranking supervision, not as direct q target."
      GROUP_EXPECT="If correction is valid, pair metrics should improve without requiring teacher/fused target regression."
      GROUP_DATASET="strategyqa"
      TRAIN_INPUT_JSONL="${TRAIN_INPUT_JSONL:-assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl}"
      EVAL_INPUT_JSONL="${EVAL_INPUT_JSONL:-assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl}"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"
      EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-192}"
      ROLLOUT_COUNT="${ROLLOUT_COUNT:-16}"
      ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-128}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-192}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-192}"
      C2_LR="${C2_LR:-1e-4}"
      C2_EPOCHS="${C2_EPOCHS:-10}"
      TEACHER_TRAIN_SCORES="${TEACHER_TRAIN_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_full_mcorr4_c2_strategyqa_quality_first_full_c1_train__b7a8789f1974/teacher_prefix_scores.jsonl}"
      TEACHER_EVAL_SCORES="${TEACHER_EVAL_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_c2_strategyqa_quality_first_c1_eval__f608255f810d/teacher_prefix_scores.jsonl}"
      TEACHER_TRAIN_CORR_SCORES="${TEACHER_TRAIN_CORR_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_full_mcorr4_c2_strategyqa_quality_first_full_c1_train__b7a8789f1974/teacher_corruption_scores.jsonl}"
      TEACHER_EVAL_CORR_SCORES="${TEACHER_EVAL_CORR_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_c2_strategyqa_quality_first_c1_eval__f608255f810d/teacher_corruption_scores.jsonl}"
      TEACHER_FUSE_MODE="${TEACHER_FUSE_MODE:-none}"
      TEACHER_FUSION_LAMBDA="${TEACHER_FUSION_LAMBDA:-0.5}"
      TEACHER_CONF_CI_REF="${TEACHER_CONF_CI_REF:-0.30}"
      TEACHER_DISAGREE_THRESH="${TEACHER_DISAGREE_THRESH:-0.25}"
      TEACHER_MIN_COVERAGE="${TEACHER_MIN_COVERAGE:-0.55}"
      C1_PREP_EXTRA_ARGS_DEFAULT="${C1_PREP_EXTRA_ARGS_DEFAULT:---corruption-selection-policy cqr_balanced --max-corruptions-per-prefix 4 --min-non-step-drop-per-prefix 1 --max-step-drop-per-prefix 1 --enable-negation-flip --enable-comparator-flip --enable-condition-reversal --enable-entity-substitution --build-pair-quality --pair-rollout-count 16 --target-alpha 1.0 --target-beta 1.0 --target-ci-z 1.96 --target-weight-floor 0.1 --target-weight-gamma 1.0 --pair-delta-q-min 0.15 --pair-z-min 0.5 --rollout-two-stage --rollout-stage1-count 8 --rollout-stage2-count 24 --rollout-uncertain-band 0.2 --rollout-uncertain-ci-width 0.3 --pair-consensus-enable --pair-consensus-teacher-delta-q-min 0.03 --pair-consensus-weight-boost 1.25 --pair-consensus-fail-weight-scale 0.20 --no-pair-consensus-require-teacher-coverage}"
      C2_TRAIN_EXTRA_ARGS_DEFAULT="${C2_TRAIN_EXTRA_ARGS_DEFAULT:---train-mode two_stage --two-stage-ranking-ratio 0.45 --contrastive-max-corruptions-per-prefix 4 --use-contrastive-loss --calibration-loss bce_mse --calibration-bce-weight 1.0 --calibration-mse-weight 0.5 --calibration-sample-weighting q_weight_parseable --calibration-weight-floor 0.1 --calibration-weight-gamma 1.0 --lambda-contrastive 0.08 --contrastive-margin 0.02 --contrastive-pair-filter label_quality --contrastive-label-delta-q-min 0.15 --contrastive-label-z-min 0.5 --contrastive-label-pair-weight-min 0.25 --contrastive-require-pair-pass-gate --contrastive-use-pair-weights --contrastive-stratified-sampling --contrastive-stratify-step-bucket-size 2 --contrastive-stratify-include-no-corruption --posthoc-calibration temperature --checkpoint-selection-metric posthoc_brier}"
      C2_EVAL_EXTRA_ARGS_DEFAULT="${C2_EVAL_EXTRA_ARGS_DEFAULT:---target-source from_run --target-source-missing-policy from_run --posthoc-calibration from_run}"
      D3_TARGETS="${D3_TARGETS:-q_mean_smoothed}"
      ;;
    D5_STRATEGYQA_FULL_MC_CTRL)
      GROUP_TITLE="Phase D5 StrategyQA Full (MC Target, No PRM Pair Gate)"
      GROUP_INTENTION="Full-data control for method correction with MC-only supervision."
      GROUP_OBSERVE="Provides full-data reference before enabling PRM pair gate."
      GROUP_EXPECT="Used as paired control against D5 full pair-gated run."
      GROUP_DATASET="strategyqa"
      TRAIN_INPUT_JSONL="${TRAIN_INPUT_JSONL:-assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl}"
      EVAL_INPUT_JSONL="${EVAL_INPUT_JSONL:-assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl}"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-}"
      EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-}"
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-192}"
      ROLLOUT_COUNT="${ROLLOUT_COUNT:-16}"
      ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-128}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-192}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-192}"
      C2_LR="${C2_LR:-1e-4}"
      C2_EPOCHS="${C2_EPOCHS:-12}"
      TEACHER_TRAIN_SCORES="${TEACHER_TRAIN_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_full_mcorr4_c2_strategyqa_quality_first_full_c1_train__b7a8789f1974/teacher_prefix_scores.jsonl}"
      TEACHER_EVAL_SCORES="${TEACHER_EVAL_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_c2_strategyqa_quality_first_c1_eval__f608255f810d/teacher_prefix_scores.jsonl}"
      TEACHER_FUSE_MODE="${TEACHER_FUSE_MODE:-none}"
      TEACHER_FUSION_LAMBDA="${TEACHER_FUSION_LAMBDA:-0.5}"
      TEACHER_CONF_CI_REF="${TEACHER_CONF_CI_REF:-0.30}"
      TEACHER_DISAGREE_THRESH="${TEACHER_DISAGREE_THRESH:-0.25}"
      TEACHER_MIN_COVERAGE="${TEACHER_MIN_COVERAGE:-0.55}"
      C1_PREP_EXTRA_ARGS_DEFAULT="${C1_PREP_EXTRA_ARGS_DEFAULT:---corruption-selection-policy cqr_balanced --max-corruptions-per-prefix 4 --min-non-step-drop-per-prefix 1 --max-step-drop-per-prefix 1 --enable-negation-flip --enable-comparator-flip --enable-condition-reversal --enable-entity-substitution --build-pair-quality --pair-rollout-count 16 --target-alpha 1.0 --target-beta 1.0 --target-ci-z 1.96 --target-weight-floor 0.1 --target-weight-gamma 1.0 --pair-delta-q-min 0.20 --pair-z-min 0.6 --rollout-two-stage --rollout-stage1-count 8 --rollout-stage2-count 24 --rollout-uncertain-band 0.2 --rollout-uncertain-ci-width 0.3}"
      C2_TRAIN_EXTRA_ARGS_DEFAULT="${C2_TRAIN_EXTRA_ARGS_DEFAULT:---train-mode two_stage --two-stage-ranking-ratio 0.45 --contrastive-max-corruptions-per-prefix 4 --use-contrastive-loss --calibration-loss bce_mse --calibration-bce-weight 1.0 --calibration-mse-weight 0.5 --calibration-sample-weighting q_weight_parseable --calibration-weight-floor 0.1 --calibration-weight-gamma 1.0 --lambda-contrastive 0.08 --contrastive-margin 0.02 --contrastive-pair-filter label_quality --contrastive-label-delta-q-min 0.20 --contrastive-label-z-min 0.6 --contrastive-label-pair-weight-min 0.25 --contrastive-require-pair-pass-gate --contrastive-use-pair-weights --contrastive-stratified-sampling --contrastive-stratify-step-bucket-size 2 --contrastive-stratify-include-no-corruption --posthoc-calibration temperature --checkpoint-selection-metric posthoc_brier}"
      C2_EVAL_EXTRA_ARGS_DEFAULT="${C2_EVAL_EXTRA_ARGS_DEFAULT:---target-source from_run --target-source-missing-policy from_run --posthoc-calibration from_run}"
      D3_TARGETS="${D3_TARGETS:-q_mean_smoothed}"
      ;;
    D5_STRATEGYQA_FULL_MC_PRM_PAIR_GATE)
      GROUP_TITLE="Phase D5 StrategyQA Full (MC Target + PRM Pair Gate)"
      GROUP_INTENTION="Full-data method-correction main arm using PRM only as pair gate (not direct q supervision)."
      GROUP_OBSERVE="Primary promotion candidate after smoke confirms pair-gate gain."
      GROUP_EXPECT="Should outperform D5 full control on corruption ordering while preserving calibration competitiveness."
      GROUP_DATASET="strategyqa"
      TRAIN_INPUT_JSONL="${TRAIN_INPUT_JSONL:-assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl}"
      EVAL_INPUT_JSONL="${EVAL_INPUT_JSONL:-assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl}"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-}"
      EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-}"
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-192}"
      ROLLOUT_COUNT="${ROLLOUT_COUNT:-16}"
      ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-128}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-192}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-192}"
      C2_LR="${C2_LR:-1e-4}"
      C2_EPOCHS="${C2_EPOCHS:-12}"
      TEACHER_TRAIN_SCORES="${TEACHER_TRAIN_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_full_mcorr4_c2_strategyqa_quality_first_full_c1_train__b7a8789f1974/teacher_prefix_scores.jsonl}"
      TEACHER_EVAL_SCORES="${TEACHER_EVAL_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_c2_strategyqa_quality_first_c1_eval__f608255f810d/teacher_prefix_scores.jsonl}"
      TEACHER_TRAIN_CORR_SCORES="${TEACHER_TRAIN_CORR_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_full_mcorr4_c2_strategyqa_quality_first_full_c1_train__b7a8789f1974/teacher_corruption_scores.jsonl}"
      TEACHER_EVAL_CORR_SCORES="${TEACHER_EVAL_CORR_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_c2_strategyqa_quality_first_c1_eval__f608255f810d/teacher_corruption_scores.jsonl}"
      TEACHER_FUSE_MODE="${TEACHER_FUSE_MODE:-none}"
      TEACHER_FUSION_LAMBDA="${TEACHER_FUSION_LAMBDA:-0.5}"
      TEACHER_CONF_CI_REF="${TEACHER_CONF_CI_REF:-0.30}"
      TEACHER_DISAGREE_THRESH="${TEACHER_DISAGREE_THRESH:-0.25}"
      TEACHER_MIN_COVERAGE="${TEACHER_MIN_COVERAGE:-0.55}"
      C1_PREP_EXTRA_ARGS_DEFAULT="${C1_PREP_EXTRA_ARGS_DEFAULT:---corruption-selection-policy cqr_balanced --max-corruptions-per-prefix 4 --min-non-step-drop-per-prefix 1 --max-step-drop-per-prefix 1 --enable-negation-flip --enable-comparator-flip --enable-condition-reversal --enable-entity-substitution --build-pair-quality --pair-rollout-count 16 --target-alpha 1.0 --target-beta 1.0 --target-ci-z 1.96 --target-weight-floor 0.1 --target-weight-gamma 1.0 --pair-delta-q-min 0.20 --pair-z-min 0.6 --rollout-two-stage --rollout-stage1-count 8 --rollout-stage2-count 24 --rollout-uncertain-band 0.2 --rollout-uncertain-ci-width 0.3 --pair-consensus-enable --pair-consensus-teacher-delta-q-min 0.05 --pair-consensus-weight-boost 1.25 --pair-consensus-fail-weight-scale 0.20 --no-pair-consensus-require-teacher-coverage}"
      C2_TRAIN_EXTRA_ARGS_DEFAULT="${C2_TRAIN_EXTRA_ARGS_DEFAULT:---train-mode two_stage --two-stage-ranking-ratio 0.45 --contrastive-max-corruptions-per-prefix 4 --use-contrastive-loss --calibration-loss bce_mse --calibration-bce-weight 1.0 --calibration-mse-weight 0.5 --calibration-sample-weighting q_weight_parseable --calibration-weight-floor 0.1 --calibration-weight-gamma 1.0 --lambda-contrastive 0.08 --contrastive-margin 0.02 --contrastive-pair-filter label_quality --contrastive-label-delta-q-min 0.20 --contrastive-label-z-min 0.6 --contrastive-label-pair-weight-min 0.25 --contrastive-require-pair-pass-gate --contrastive-use-pair-weights --contrastive-stratified-sampling --contrastive-stratify-step-bucket-size 2 --contrastive-stratify-include-no-corruption --posthoc-calibration temperature --checkpoint-selection-metric posthoc_brier}"
      C2_EVAL_EXTRA_ARGS_DEFAULT="${C2_EVAL_EXTRA_ARGS_DEFAULT:---target-source from_run --target-source-missing-policy from_run --posthoc-calibration from_run}"
      D3_TARGETS="${D3_TARGETS:-q_mean_smoothed}"
      ;;
    D6_STRATEGYQA_SMOKE_RANKING_CTRL)
      GROUP_TITLE="Phase D6 StrategyQA Smoke (Ranking-First MC Control)"
      GROUP_INTENTION="Directly address objective-mismatch risk: use MC targets, ranking-only optimization, and ranking-based checkpoint selection."
      GROUP_OBSERVE="Reference arm for D6 pair-discrimination gains without PRM pair gate."
      GROUP_EXPECT="Compared with D5 MC control, this should improve corr_pair_acc/corr_auc consistency by aligning training and selection to ranking objectives."
      GROUP_DATASET="strategyqa"
      TRAIN_INPUT_JSONL="${TRAIN_INPUT_JSONL:-assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl}"
      EVAL_INPUT_JSONL="${EVAL_INPUT_JSONL:-assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl}"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"
      EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-192}"
      ROLLOUT_COUNT="${ROLLOUT_COUNT:-16}"
      ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-128}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-192}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-192}"
      C2_LR="${C2_LR:-1e-4}"
      C2_EPOCHS="${C2_EPOCHS:-8}"
      # D6 MC control must remain teacher-free in C1 fusion path.
      TEACHER_TRAIN_SCORES="${TEACHER_TRAIN_SCORES:-}"
      TEACHER_EVAL_SCORES="${TEACHER_EVAL_SCORES:-}"
      TEACHER_TRAIN_CORR_SCORES="${TEACHER_TRAIN_CORR_SCORES:-}"
      TEACHER_EVAL_CORR_SCORES="${TEACHER_EVAL_CORR_SCORES:-}"
      TEACHER_FUSE_MODE="${TEACHER_FUSE_MODE:-none}"
      TEACHER_FUSION_LAMBDA="${TEACHER_FUSION_LAMBDA:-0.5}"
      TEACHER_CONF_CI_REF="${TEACHER_CONF_CI_REF:-0.30}"
      TEACHER_DISAGREE_THRESH="${TEACHER_DISAGREE_THRESH:-0.25}"
      TEACHER_MIN_COVERAGE="${TEACHER_MIN_COVERAGE:-0.0}"
      C1_PREP_EXTRA_ARGS_DEFAULT="${C1_PREP_EXTRA_ARGS_DEFAULT:---corruption-selection-policy cqr_balanced --max-corruptions-per-prefix 4 --min-non-step-drop-per-prefix 1 --max-step-drop-per-prefix 1 --enable-negation-flip --enable-comparator-flip --enable-condition-reversal --enable-entity-substitution --build-pair-quality --pair-rollout-count 16 --target-alpha 1.0 --target-beta 1.0 --target-ci-z 1.96 --target-weight-floor 0.1 --target-weight-gamma 1.0 --pair-delta-q-min 0.20 --pair-z-min 0.6 --rollout-two-stage --rollout-stage1-count 8 --rollout-stage2-count 24 --rollout-uncertain-band 0.2 --rollout-uncertain-ci-width 0.3}"
      C2_TRAIN_EXTRA_ARGS_DEFAULT="${C2_TRAIN_EXTRA_ARGS_DEFAULT:---train-mode ranking_only --use-contrastive-loss --calibration-loss bce_mse --calibration-bce-weight 1.0 --calibration-mse-weight 0.5 --calibration-sample-weighting q_weight_parseable --calibration-weight-floor 0.1 --calibration-weight-gamma 1.0 --lambda-contrastive 0.10 --contrastive-margin 0.02 --contrastive-pair-filter label_quality --contrastive-label-delta-q-min 0.20 --contrastive-label-z-min 0.6 --contrastive-label-pair-weight-min 0.30 --contrastive-require-pair-pass-gate --contrastive-use-pair-weights --contrastive-stratified-sampling --contrastive-stratify-step-bucket-size 2 --contrastive-stratify-include-no-corruption --checkpoint-selection-metric corr_auc --posthoc-calibration none}"
      C2_EVAL_EXTRA_ARGS_DEFAULT="${C2_EVAL_EXTRA_ARGS_DEFAULT:---target-source from_run --target-source-missing-policy from_run --posthoc-calibration none}"
      D3_TARGETS="${D3_TARGETS:-q_mean_smoothed}"
      ;;
    D6_STRATEGYQA_SMOKE_RANKING_PRM_GATE)
      GROUP_TITLE="Phase D6 StrategyQA Smoke (Ranking-First + PRM Pair Gate)"
      GROUP_INTENTION="Ranking-first main arm: keep MC target, use PRM only to strengthen pair-quality gating in C1."
      GROUP_OBSERVE="Direct A/B against D6 ranking control to measure PRM gate benefit under the same ranking objective."
      GROUP_EXPECT="corr_pair_acc/corr_auc should improve over D6 control if teacher-gated pairs add real ranking signal."
      GROUP_DATASET="strategyqa"
      TRAIN_INPUT_JSONL="${TRAIN_INPUT_JSONL:-assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl}"
      EVAL_INPUT_JSONL="${EVAL_INPUT_JSONL:-assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl}"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"
      EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-192}"
      ROLLOUT_COUNT="${ROLLOUT_COUNT:-16}"
      ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-128}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-192}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-192}"
      C2_LR="${C2_LR:-1e-4}"
      C2_EPOCHS="${C2_EPOCHS:-8}"
      TEACHER_TRAIN_SCORES="${TEACHER_TRAIN_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_full_mcorr4_c2_strategyqa_quality_first_full_c1_train__b7a8789f1974/teacher_prefix_scores.jsonl}"
      TEACHER_EVAL_SCORES="${TEACHER_EVAL_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_c2_strategyqa_quality_first_c1_eval__f608255f810d/teacher_prefix_scores.jsonl}"
      TEACHER_TRAIN_CORR_SCORES="${TEACHER_TRAIN_CORR_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_full_mcorr4_c2_strategyqa_quality_first_full_c1_train__b7a8789f1974/teacher_corruption_scores.jsonl}"
      TEACHER_EVAL_CORR_SCORES="${TEACHER_EVAL_CORR_SCORES:-assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_c2_strategyqa_quality_first_c1_eval__f608255f810d/teacher_corruption_scores.jsonl}"
      TEACHER_FUSE_MODE="${TEACHER_FUSE_MODE:-none}"
      TEACHER_FUSION_LAMBDA="${TEACHER_FUSION_LAMBDA:-0.5}"
      TEACHER_CONF_CI_REF="${TEACHER_CONF_CI_REF:-0.30}"
      TEACHER_DISAGREE_THRESH="${TEACHER_DISAGREE_THRESH:-0.25}"
      TEACHER_MIN_COVERAGE="${TEACHER_MIN_COVERAGE:-0.70}"
      C1_PREP_EXTRA_ARGS_DEFAULT="${C1_PREP_EXTRA_ARGS_DEFAULT:---corruption-selection-policy cqr_balanced --max-corruptions-per-prefix 4 --min-non-step-drop-per-prefix 1 --max-step-drop-per-prefix 1 --enable-negation-flip --enable-comparator-flip --enable-condition-reversal --enable-entity-substitution --build-pair-quality --pair-rollout-count 16 --target-alpha 1.0 --target-beta 1.0 --target-ci-z 1.96 --target-weight-floor 0.1 --target-weight-gamma 1.0 --pair-delta-q-min 0.20 --pair-z-min 0.6 --rollout-two-stage --rollout-stage1-count 8 --rollout-stage2-count 24 --rollout-uncertain-band 0.2 --rollout-uncertain-ci-width 0.3 --pair-consensus-enable --pair-consensus-teacher-delta-q-min 0.05 --pair-consensus-weight-boost 1.25 --pair-consensus-fail-weight-scale 0.20 --pair-consensus-require-teacher-coverage}"
      C2_TRAIN_EXTRA_ARGS_DEFAULT="${C2_TRAIN_EXTRA_ARGS_DEFAULT:---train-mode ranking_only --use-contrastive-loss --calibration-loss bce_mse --calibration-bce-weight 1.0 --calibration-mse-weight 0.5 --calibration-sample-weighting q_weight_parseable --calibration-weight-floor 0.1 --calibration-weight-gamma 1.0 --lambda-contrastive 0.10 --contrastive-margin 0.02 --contrastive-pair-filter label_quality --contrastive-label-delta-q-min 0.20 --contrastive-label-z-min 0.6 --contrastive-label-pair-weight-min 0.30 --contrastive-require-pair-pass-gate --contrastive-use-pair-weights --contrastive-stratified-sampling --contrastive-stratify-step-bucket-size 2 --contrastive-stratify-include-no-corruption --checkpoint-selection-metric corr_auc --posthoc-calibration none}"
      C2_EVAL_EXTRA_ARGS_DEFAULT="${C2_EVAL_EXTRA_ARGS_DEFAULT:---target-source from_run --target-source-missing-policy from_run --posthoc-calibration none}"
      D3_TARGETS="${D3_TARGETS:-q_mean_smoothed}"
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_D_GROUP=$ACTIVE_PHASE_D_GROUP"
      echo "Supported groups: D4_STRATEGYQA_SMOKE_3WAY, D4_STRATEGYQA_SMOKE_3WAY_HQ, D4_STRATEGYQA_FULL_3WAY_HQ, D5_STRATEGYQA_SMOKE_MC_CTRL, D5_STRATEGYQA_SMOKE_MC_PRM_PAIR_GATE, D5_STRATEGYQA_FULL_MC_CTRL, D5_STRATEGYQA_FULL_MC_PRM_PAIR_GATE, D6_STRATEGYQA_SMOKE_RANKING_CTRL, D6_STRATEGYQA_SMOKE_RANKING_PRM_GATE"
      exit 1
      ;;
  esac
}

run_c1_prepare() {
  local split_label="$1"
  local input_jsonl="$2"
  local run_name="$3"
  local max_samples="$4"
  local teacher_scores="$5"
  local teacher_corr_scores="$6"

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
  # D6 safety: teacher arguments are only passed when teacher files are
  # explicitly configured. This keeps MC-only control arms truly teacher-free.
  if [[ -n "$teacher_scores" ]]; then
    cmd+=(
      --teacher-prefix-scores-jsonl "$teacher_scores"
      --teacher-fuse-mode "$TEACHER_FUSE_MODE"
      --teacher-fusion-lambda "$TEACHER_FUSION_LAMBDA"
      --teacher-confidence-ci-ref "$TEACHER_CONF_CI_REF"
      --teacher-disagree-threshold "$TEACHER_DISAGREE_THRESH"
      --teacher-min-coverage "$TEACHER_MIN_COVERAGE"
    )
  fi
  if [[ -n "$max_samples" ]]; then
    cmd+=(--max-samples "$max_samples")
  fi
  if [[ -n "$teacher_corr_scores" ]]; then
    cmd+=(--teacher-corruption-scores-jsonl "$teacher_corr_scores")
  fi
  append_extra_args cmd "${C1_PREP_EXTRA_ARGS_DEFAULT:-}"
  append_extra_args cmd "${PHASE_D_PREP_EXTRA_ARGS:-}"

  CURRENT_STAGE="d2_prepare_${split_label}"
  {
    log_line "D2 prepare      : ${split_label}"
    log_line "D2 input         : ${input_jsonl}"
    log_line "D2 run_name      : ${run_name}"
    log_line "D2 teacher score : ${teacher_scores}"
    log_line "D2 teacher corr  : ${teacher_corr_scores:-<none>}"
    log_line "D2 command       : ${cmd[*]}"
  } | tee -a "$SUITE_LOG_FILE"
  "${cmd[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"
}

run_c2_target() {
  local target_source="$1"
  local label="$2"
  local c2_train_run_name="${RUN_NAME}_${label}_c2"
  local c2_eval_run_name="${RUN_NAME}_${label}_c2_eval"

  local train_cmd=(
    "$PYTHON_BIN" -u scripts/phase_b_train_value.py
    --train-dir "$C1_TRAIN_DIR"
    --eval-dir "$C1_EVAL_DIR"
    --run-name "$c2_train_run_name"
    --target-source "$target_source"
    --target-source-missing-policy fail
    --require-cuda
    --dtype bfloat16
    --device-map auto
    --per-device-train-batch-size "$C2_TRAIN_BATCH_SIZE"
    --per-device-eval-batch-size "$C2_EVAL_BATCH_SIZE"
    --learning-rate "$C2_LR"
    --num-train-epochs "$C2_EPOCHS"
  )
  append_extra_args train_cmd "${C2_TRAIN_EXTRA_ARGS_DEFAULT:-}"
  append_extra_args train_cmd "${PHASE_D_TRAIN_EXTRA_ARGS:-}"

  CURRENT_STAGE="d3_train_${label}"
  {
    log_line "D3 train label   : ${label}"
    log_line "D3 target_source : ${target_source}"
    log_line "D3 train command : ${train_cmd[*]}"
  } | tee -a "$SUITE_LOG_FILE"
  "${train_cmd[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

  CURRENT_STAGE="d3_resolve_train_${label}"
  local c2_train_dir
  c2_train_dir="$(latest_phase_c_train_run_dir_for_name "$c2_train_run_name")"
  log_line "Resolved C2 dir  : ${c2_train_dir}" | tee -a "$SUITE_LOG_FILE"

  local eval_cmd=(
    "$PYTHON_BIN" -u scripts/phase_b_eval_faithfulness.py
    --value-run-dir "$c2_train_dir"
    --eval-dir "$C1_EVAL_DIR"
    --checkpoint-name best
    --run-name "$c2_eval_run_name"
  )
  append_extra_args eval_cmd "${C2_EVAL_EXTRA_ARGS_DEFAULT:-}"
  append_extra_args eval_cmd "${PHASE_D_EVAL_EXTRA_ARGS:-}"

  CURRENT_STAGE="d3_eval_${label}"
  {
    log_line "D3 eval command  : ${eval_cmd[*]}"
  } | tee -a "$SUITE_LOG_FILE"
  "${eval_cmd[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

  CURRENT_STAGE="d3_resolve_eval_${label}"
  local c2_eval_dir c2_eval_metrics
  c2_eval_dir="$(latest_phase_c_eval_run_dir_for_name "$c2_eval_run_name")"
  c2_eval_metrics="$c2_eval_dir/metrics.json"
  log_line "Resolved eval dir: ${c2_eval_dir}" | tee -a "$SUITE_LOG_FILE"

  local brier pearson post_brier pair_acc auc cov dis
  brier="$(json_value "$c2_eval_metrics" "calibration.brier_score")"
  pearson="$(json_value "$c2_eval_metrics" "calibration.pearson")"
  post_brier="$(json_value "$c2_eval_metrics" "calibration_posthoc.brier_score")"
  pair_acc="$(json_value "$c2_eval_metrics" "corruption.pair_accuracy")"
  auc="$(json_value "$c2_eval_metrics" "corruption.auc_clean_vs_corrupt")"
  cov="$(json_value "$c2_eval_metrics" "target_source_stats.coverage_ratio")"
  dis="$(json_value "$c2_eval_metrics" "target_source_stats.teacher_disagree_ratio")"

  RESULTS_ROWS+=("${label}|${target_source}|${brier:-N/A}|${pearson:-N/A}|${post_brier:-N/A}|${pair_acc:-N/A}|${auc:-N/A}|${cov:-N/A}|${dis:-N/A}|${c2_train_dir}|${c2_eval_dir}")
}

resolve_group

LOG_ROOT="assets/artifacts/phase_d_logs/$RUN_PREFIX"
mkdir -p "$LOG_ROOT"
SUITE_LOG_FILE="$LOG_ROOT/suite.log"
SUMMARY_FILE="$LOG_ROOT/final_summary.md"
RUN_NAME="${RUN_PREFIX}_${ACTIVE_PHASE_D_GROUP,,}"
C1_TRAIN_RUN_NAME="${RUN_NAME}_d2_c1_train"
C1_EVAL_RUN_NAME="${RUN_NAME}_d2_c1_eval"

{
  log_line "Repo root      : $REPO_ROOT"
  log_line "Python         : $PYTHON_BIN"
  log_line "Group          : $ACTIVE_PHASE_D_GROUP"
  log_line "Group title    : $GROUP_TITLE"
  log_line "Run prefix     : $RUN_PREFIX"
  log_line "Run name       : $RUN_NAME"
  log_line "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
  log_line "Dataset        : $GROUP_DATASET"
  log_line "Train input    : $TRAIN_INPUT_JSONL"
  log_line "Eval input     : $EVAL_INPUT_JSONL"
  log_line "Teacher train  : $TEACHER_TRAIN_SCORES"
  log_line "Teacher trainc : ${TEACHER_TRAIN_CORR_SCORES:-<none>}"
  log_line "Teacher eval   : $TEACHER_EVAL_SCORES"
  log_line "Teacher evalc  : ${TEACHER_EVAL_CORR_SCORES:-<none>}"
  log_line "Fuse mode      : $TEACHER_FUSE_MODE"
  log_line "D3 targets     : $D3_TARGETS"
  log_line "Intention      : $GROUP_INTENTION"
  log_line "Observe        : $GROUP_OBSERVE"
  log_line "Expectation    : $GROUP_EXPECT"
  log_line "Group run start"
} | tee "$SUITE_LOG_FILE"

# D6 safety gate:
# - teacher files are optional for MC-only control groups.
# - when configured, they must be complete and consistent.
if [[ -n "${TEACHER_TRAIN_SCORES:-}" && ! -f "$TEACHER_TRAIN_SCORES" ]]; then
  echo "ERROR: Missing teacher score file: $TEACHER_TRAIN_SCORES" >&2
  exit 1
fi
if [[ -n "${TEACHER_EVAL_SCORES:-}" && ! -f "$TEACHER_EVAL_SCORES" ]]; then
  echo "ERROR: Missing teacher score file: $TEACHER_EVAL_SCORES" >&2
  exit 1
fi
if [[ -n "${TEACHER_TRAIN_CORR_SCORES:-}" && ! -f "$TEACHER_TRAIN_CORR_SCORES" ]]; then
  echo "ERROR: Missing teacher corruption score file: $TEACHER_TRAIN_CORR_SCORES" >&2
  exit 1
fi
if [[ -n "${TEACHER_EVAL_CORR_SCORES:-}" && ! -f "$TEACHER_EVAL_CORR_SCORES" ]]; then
  echo "ERROR: Missing teacher corruption score file: $TEACHER_EVAL_CORR_SCORES" >&2
  exit 1
fi
if [[ -z "${TEACHER_TRAIN_SCORES:-}" && -n "${TEACHER_TRAIN_CORR_SCORES:-}" ]]; then
  echo "ERROR: TEACHER_TRAIN_CORR_SCORES is set but TEACHER_TRAIN_SCORES is empty." >&2
  exit 1
fi
if [[ -z "${TEACHER_EVAL_SCORES:-}" && -n "${TEACHER_EVAL_CORR_SCORES:-}" ]]; then
  echo "ERROR: TEACHER_EVAL_CORR_SCORES is set but TEACHER_EVAL_SCORES is empty." >&2
  exit 1
fi
if [[ -z "${TEACHER_TRAIN_SCORES:-}" || -z "${TEACHER_EVAL_SCORES:-}" ]]; then
  if [[ "${TEACHER_FUSE_MODE:-none}" != "none" ]]; then
    echo "ERROR: teacher_fuse_mode=${TEACHER_FUSE_MODE} requires train/eval teacher prefix scores." >&2
    exit 1
  fi
fi

run_c1_prepare "train" "$TRAIN_INPUT_JSONL" "$C1_TRAIN_RUN_NAME" "${TRAIN_MAX_SAMPLES:-}" "$TEACHER_TRAIN_SCORES" "${TEACHER_TRAIN_CORR_SCORES:-}"
run_c1_prepare "eval" "$EVAL_INPUT_JSONL" "$C1_EVAL_RUN_NAME" "${EVAL_MAX_SAMPLES:-}" "$TEACHER_EVAL_SCORES" "${TEACHER_EVAL_CORR_SCORES:-}"

CURRENT_STAGE="resolve_d2_dirs"
C1_TRAIN_DIR="$(latest_phase_c_data_run_dir_for_name "$GROUP_DATASET" "$C1_TRAIN_RUN_NAME")"
C1_EVAL_DIR="$(latest_phase_c_data_run_dir_for_name "$GROUP_DATASET" "$C1_EVAL_RUN_NAME")"
{
  log_line "Resolved D2 train dir: $C1_TRAIN_DIR"
  log_line "Resolved D2 eval dir : $C1_EVAL_DIR"
} | tee -a "$SUITE_LOG_FILE"

declare -a RESULTS_ROWS=()
for target_source in $D3_TARGETS; do
  case "$target_source" in
    q_mean_smoothed) label="mc" ;;
    q_teacher) label="teacher" ;;
    q_fused) label="fused" ;;
    *) label="$target_source" ;;
  esac
  run_c2_target "$target_source" "$label"
done

CURRENT_STAGE="final_summary"
{
  echo "# Phase D Teacher Suite Summary"
  echo
  echo "- generated_at: $(date --iso-8601=seconds)"
  echo "- group_id: $ACTIVE_PHASE_D_GROUP"
  echo "- group_title: $GROUP_TITLE"
  echo "- run_prefix: $RUN_PREFIX"
  echo "- run_name: $RUN_NAME"
  echo "- dataset: $GROUP_DATASET"
  echo "- train_input_jsonl: $TRAIN_INPUT_JSONL"
  echo "- eval_input_jsonl: $EVAL_INPUT_JSONL"
  echo "- d2_train_dir: $C1_TRAIN_DIR"
  echo "- d2_eval_dir: $C1_EVAL_DIR"
  echo "- suite_log_file: $SUITE_LOG_FILE"
  echo
  echo "## D2 Settings"
  echo
  echo "- teacher_train_scores: \`$TEACHER_TRAIN_SCORES\`"
  echo "- teacher_train_corruption_scores: \`${TEACHER_TRAIN_CORR_SCORES:-<none>}\`"
  echo "- teacher_eval_scores: \`$TEACHER_EVAL_SCORES\`"
  echo "- teacher_eval_corruption_scores: \`${TEACHER_EVAL_CORR_SCORES:-<none>}\`"
  echo "- teacher_fuse_mode: \`$TEACHER_FUSE_MODE\`"
  echo "- teacher_fusion_lambda: \`$TEACHER_FUSION_LAMBDA\`"
  echo "- teacher_confidence_ci_ref: \`$TEACHER_CONF_CI_REF\`"
  echo "- teacher_disagree_threshold: \`$TEACHER_DISAGREE_THRESH\`"
  echo "- teacher_min_coverage: \`$TEACHER_MIN_COVERAGE\`"
  echo
  echo "## D3 Result Table"
  echo
  echo "| label | target_source | brier | pearson | posthoc_brier | pair_acc | auc | target_cov | teacher_dis | c2_train_dir | c2_eval_dir |"
  echo "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|"
  for row in "${RESULTS_ROWS[@]}"; do
    IFS='|' read -r label target brier pear postb pair auc cov dis train_dir eval_dir <<< "$row"
    echo "| ${label} | ${target} | ${brier} | ${pear} | ${postb} | ${pair} | ${auc} | ${cov} | ${dis} | \`${train_dir}\` | \`${eval_dir}\` |"
  done
} > "$SUMMARY_FILE"

{
  log_line "Summary file   : $SUMMARY_FILE"
  log_line "Group run complete"
} | tee -a "$SUITE_LOG_FILE"
