#!/usr/bin/env bash
# Phase C P(IK) lifecycle suite.
#
# Why this file exists:
# - P(IK) requires a multi-stage run for reproducibility:
#   1) build train question-level rollout artifacts,
#   2) build eval question-level rollout artifacts,
#   3) train the question-level value head,
#   4) run standalone evaluation and summarize key metrics.
# - This wrapper keeps that flow deterministic and report-ready.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACTIVE_PHASE_C_PIK_GROUP="${ACTIVE_PHASE_C_PIK_GROUP:-PIK_STRATEGYQA_SMOKE}"
RUN_PREFIX="${RUN_PREFIX:-phase_c_pik}"
CURRENT_STAGE="init"
# 中文：这个入口专门跑问题级 P(IK) 管线；切组只改 ACTIVE_PHASE_C_PIK_GROUP。


timestamp() {
  date "+%Y-%m-%d %H:%M:%S %z"
}


log_line() {
  local msg="$1"
  echo "[$(timestamp)] $msg"
}


append_extra_args() {
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


latest_pik_data_run_dir_for_name() {
  local dataset="$1"
  local run_name="$2"
  local latest=""
  latest="$(compgen -G "assets/artifacts/phase_c_pik_data/${dataset}/${run_name}__*" | sort | tail -n 1 || true)"
  if [[ -z "$latest" ]]; then
    echo "ERROR: No P(IK) data run dir found for dataset=$dataset run_name=$run_name" >&2
    return 1
  fi
  printf '%s\n' "$latest"
}


latest_pik_train_run_dir_for_name() {
  local run_name="$1"
  local latest=""
  latest="$(compgen -G "assets/artifacts/phase_c_pik_runs/${run_name}_*" | sort | tail -n 1 || true)"
  if [[ -z "$latest" ]]; then
    echo "ERROR: No P(IK) training run dir found for run_name=$run_name" >&2
    return 1
  fi
  printf '%s\n' "$latest"
}


latest_pik_eval_run_dir_for_name() {
  local run_name="$1"
  local latest=""
  latest="$(compgen -G "assets/artifacts/phase_c_pik_eval/${run_name}_*" | sort | tail -n 1 || true)"
  if [[ -z "$latest" ]]; then
    echo "ERROR: No P(IK) eval run dir found for run_name=$run_name" >&2
    return 1
  fi
  printf '%s\n' "$latest"
}


json_value() {
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
  local exit_code="$1"
  [[ -z "${SUMMARY_FILE:-}" ]] && return 0
  mkdir -p "$(dirname "$SUMMARY_FILE")"
  cat > "$SUMMARY_FILE" <<EOF
# Phase C P(IK) Suite Summary

- generated_at: $(date --iso-8601=seconds)
- group_id: ${ACTIVE_PHASE_C_PIK_GROUP:-N/A}
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
  # 组定义同时决定数据规模与训练配置，是复现实验的主入口。
  # 中文：新增组时优先复制现有组，仅替换数据路径和训练超参，避免漏字段。
  case "$ACTIVE_PHASE_C_PIK_GROUP" in
    PIK_STRATEGYQA_SMOKE)
      GROUP_TITLE="Phase C P(IK) StrategyQA Smoke"
      GROUP_INTENTION="Quick end-to-end validation for question-level P(IK) path."
      GROUP_OBSERVE="Check whether P(IK) head learns non-trivial calibration/AUROC signal."
      GROUP_EXPECT="No crash, usable artifacts, and reportable P(IK) metrics."
      GROUP_DATASET="strategyqa"
      TRAIN_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl"
      EVAL_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"
      EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-192}"
      ROLLOUT_COUNT="${ROLLOUT_COUNT:-20}"
      ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-96}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-192}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-192}"
      C2_LR="${C2_LR:-1e-4}"
      C2_EPOCHS="${C2_EPOCHS:-8}"
      C2_TRAIN_EXTRA_ARGS_DEFAULT="${C2_TRAIN_EXTRA_ARGS_DEFAULT:---calibration-loss bce --known-threshold 0.5 --posthoc-calibration temperature --checkpoint-selection-metric posthoc_brier}"
      C2_EVAL_EXTRA_ARGS_DEFAULT="${C2_EVAL_EXTRA_ARGS_DEFAULT:---posthoc-calibration from_run}"
      ;;
    PIK_STRATEGYQA_FULL)
      GROUP_TITLE="Phase C P(IK) StrategyQA Full"
      GROUP_INTENTION="Full-scale question-level P(IK) verification before deeper Phase C branches."
      GROUP_OBSERVE="Track held-out Brier/Pearson/AUROC and compare against prior prefix-level failures."
      GROUP_EXPECT="If supervision is usable, AUROC should exceed random and Brier should beat baseline."
      GROUP_DATASET="strategyqa"
      TRAIN_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl"
      EVAL_INPUT_JSONL="assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-}"
      EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-}"
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-192}"
      ROLLOUT_COUNT="${ROLLOUT_COUNT:-32}"
      ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-96}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-192}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-192}"
      C2_LR="${C2_LR:-1e-4}"
      C2_EPOCHS="${C2_EPOCHS:-10}"
      C2_TRAIN_EXTRA_ARGS_DEFAULT="${C2_TRAIN_EXTRA_ARGS_DEFAULT:---calibration-loss bce --known-threshold 0.5 --posthoc-calibration temperature --checkpoint-selection-metric posthoc_brier}"
      C2_EVAL_EXTRA_ARGS_DEFAULT="${C2_EVAL_EXTRA_ARGS_DEFAULT:---posthoc-calibration from_run}"
      ;;
    PIK_GSM8K_SMOKE)
      GROUP_TITLE="Phase C P(IK) GSM8K Smoke"
      GROUP_INTENTION="Cross-check question-level P(IK) learnability on math-heavy data."
      GROUP_OBSERVE="Check whether GSM8K P(IK) signal is similarly weak or stronger than StrategyQA."
      GROUP_EXPECT="Provide direct evidence for whether current issues are task-specific or global."
      GROUP_DATASET="gsm8k"
      TRAIN_INPUT_JSONL="assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/train.jsonl"
      EVAL_INPUT_JSONL="assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/validation.jsonl"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-512}"
      EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-172}"
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-192}"
      ROLLOUT_COUNT="${ROLLOUT_COUNT:-20}"
      ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-192}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-192}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-192}"
      C2_LR="${C2_LR:-1e-4}"
      C2_EPOCHS="${C2_EPOCHS:-8}"
      C2_TRAIN_EXTRA_ARGS_DEFAULT="${C2_TRAIN_EXTRA_ARGS_DEFAULT:---calibration-loss bce --known-threshold 0.5 --posthoc-calibration temperature --checkpoint-selection-metric posthoc_brier}"
      C2_EVAL_EXTRA_ARGS_DEFAULT="${C2_EVAL_EXTRA_ARGS_DEFAULT:---posthoc-calibration from_run}"
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_C_PIK_GROUP=$ACTIVE_PHASE_C_PIK_GROUP"
      echo "Supported groups: PIK_STRATEGYQA_SMOKE, PIK_STRATEGYQA_FULL, PIK_GSM8K_SMOKE"
      exit 1
      ;;
  esac
}


run_c1_prepare() {
  local split_label="$1"
  local input_jsonl="$2"
  local run_name="$3"
  local max_samples="$4"

  local cmd=(
    "$PYTHON_BIN" -u scripts/phase_c_prepare_pik_data.py
    --input-jsonl "$input_jsonl"
    --run-name "$run_name"
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
  # 中文：这里是 P(IK) C1 阶段唯一的用户注入口，适合临时改 rollout 参数。
  append_extra_args cmd "${PHASE_C_PIK_PREP_EXTRA_ARGS:-}"

  CURRENT_STAGE="c1_prepare_${split_label}"
  {
    log_line "PIK C1 prepare  : ${split_label}"
    log_line "PIK C1 input     : ${input_jsonl}"
    log_line "PIK C1 run_name  : ${run_name}"
    log_line "PIK C1 max_samp  : ${max_samples:-<none>}"
    log_line "PIK C1 command   : ${cmd[*]}"
  } | tee -a "$SUITE_LOG_FILE"
  "${cmd[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"
}


resolve_group

LOG_ROOT="assets/artifacts/phase_c_pik_logs/$RUN_PREFIX"
mkdir -p "$LOG_ROOT"
SUITE_LOG_FILE="$LOG_ROOT/suite.log"
SUMMARY_FILE="$LOG_ROOT/final_summary.md"

RUN_NAME="${RUN_PREFIX}_${ACTIVE_PHASE_C_PIK_GROUP,,}"
C1_TRAIN_RUN_NAME="${RUN_NAME}_c1_train"
C1_EVAL_RUN_NAME="${RUN_NAME}_c1_eval"
C2_TRAIN_RUN_NAME="${RUN_NAME}_c2"
C2_STANDALONE_EVAL_RUN_NAME="${RUN_NAME}_c2_eval"
# 中文：目录解析依赖这套命名后缀，改名规则会影响后续自动查找。

{
  log_line "Repo root      : $REPO_ROOT"
  log_line "Python         : $PYTHON_BIN"
  log_line "Group          : $ACTIVE_PHASE_C_PIK_GROUP"
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
  log_line "C2 default train extra: ${C2_TRAIN_EXTRA_ARGS_DEFAULT:-<none>}"
  log_line "C2 default eval extra : ${C2_EVAL_EXTRA_ARGS_DEFAULT:-<none>}"
  log_line "User train extra args : ${PHASE_C_PIK_TRAIN_EXTRA_ARGS:-<none>}"
  log_line "User eval extra args  : ${PHASE_C_PIK_EVAL_EXTRA_ARGS:-<none>}"
  log_line "Intention      : $GROUP_INTENTION"
  log_line "Observe        : $GROUP_OBSERVE"
  log_line "Expectation    : $GROUP_EXPECT"
  log_line "Group run start"
} | tee "$SUITE_LOG_FILE"

run_c1_prepare "train" "$TRAIN_INPUT_JSONL" "$C1_TRAIN_RUN_NAME" "${TRAIN_MAX_SAMPLES:-}"
run_c1_prepare "eval" "$EVAL_INPUT_JSONL" "$C1_EVAL_RUN_NAME" "${EVAL_MAX_SAMPLES:-}"

CURRENT_STAGE="resolve_c1_dirs"
C1_TRAIN_DIR="$(latest_pik_data_run_dir_for_name "$GROUP_DATASET" "$C1_TRAIN_RUN_NAME")"
C1_EVAL_DIR="$(latest_pik_data_run_dir_for_name "$GROUP_DATASET" "$C1_EVAL_RUN_NAME")"
{
  log_line "Resolved PIK C1 train dir: $C1_TRAIN_DIR"
  log_line "Resolved PIK C1 eval dir : $C1_EVAL_DIR"
} | tee -a "$SUITE_LOG_FILE"

CURRENT_STAGE="c2_train"
c2_train_cmd=(
  "$PYTHON_BIN" -u scripts/phase_c_train_pik.py
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
append_extra_args c2_train_cmd "${C2_TRAIN_EXTRA_ARGS_DEFAULT:-}"
append_extra_args c2_train_cmd "${PHASE_C_PIK_TRAIN_EXTRA_ARGS:-}"
{
  log_line "PIK C2 train command: ${c2_train_cmd[*]}"
} | tee -a "$SUITE_LOG_FILE"
"${c2_train_cmd[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

CURRENT_STAGE="resolve_c2_train_dir"
C2_TRAIN_DIR="$(latest_pik_train_run_dir_for_name "$C2_TRAIN_RUN_NAME")"
{
  log_line "Resolved PIK C2 train dir: $C2_TRAIN_DIR"
} | tee -a "$SUITE_LOG_FILE"

CURRENT_STAGE="c2_eval"
c2_eval_cmd=(
  "$PYTHON_BIN" -u scripts/phase_c_eval_pik.py
  --value-run-dir "$C2_TRAIN_DIR"
  --eval-dir "$C1_EVAL_DIR"
  --checkpoint-name best
  --run-name "$C2_STANDALONE_EVAL_RUN_NAME"
)
append_extra_args c2_eval_cmd "${C2_EVAL_EXTRA_ARGS_DEFAULT:-}"
append_extra_args c2_eval_cmd "${PHASE_C_PIK_EVAL_EXTRA_ARGS:-}"
# 中文：train/eval 的 extra args 分开传，避免把训练参数误注入评估脚本。
{
  log_line "PIK C2 eval command : ${c2_eval_cmd[*]}"
} | tee -a "$SUITE_LOG_FILE"
"${c2_eval_cmd[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

CURRENT_STAGE="resolve_c2_eval_dir"
# 注意：下面指标全部来自 standalone eval，而不是训练过程中的中间日志。
C2_EVAL_DIR="$(latest_pik_eval_run_dir_for_name "$C2_STANDALONE_EVAL_RUN_NAME")"
C2_EVAL_METRICS="$C2_EVAL_DIR/metrics.json"
C2_BRIER="$(json_value "$C2_EVAL_METRICS" "calibration.brier_score")"
C2_PEARSON="$(json_value "$C2_EVAL_METRICS" "calibration.pearson")"
C2_ECE="$(json_value "$C2_EVAL_METRICS" "calibration.ece")"
C2_KNOWN_AUC="$(json_value "$C2_EVAL_METRICS" "known_auc")"
C2_POST_BRIER="$(json_value "$C2_EVAL_METRICS" "calibration_posthoc.brier_score")"
C2_POST_PEARSON="$(json_value "$C2_EVAL_METRICS" "calibration_posthoc.pearson")"
C2_KNOWN_AUC_POST="$(json_value "$C2_EVAL_METRICS" "known_auc_posthoc")"

CURRENT_STAGE="final_summary"
cat > "$SUMMARY_FILE" <<EOF
# Phase C P(IK) Suite Summary

- generated_at: $(date --iso-8601=seconds)
- group_id: $ACTIVE_PHASE_C_PIK_GROUP
- group_title: $GROUP_TITLE
- run_prefix: $RUN_PREFIX
- run_name: $RUN_NAME
- intention: $GROUP_INTENTION
- observe: $GROUP_OBSERVE
- expectation: $GROUP_EXPECT
- status: success

## Inputs

- dataset: $GROUP_DATASET
- train_input_jsonl: $TRAIN_INPUT_JSONL
- eval_input_jsonl: $EVAL_INPUT_JSONL
- train_max_samples: ${TRAIN_MAX_SAMPLES:-<none>}
- eval_max_samples: ${EVAL_MAX_SAMPLES:-<none>}
- rollout_batch_size: $ROLLOUT_BATCH_SIZE
- rollout_count: $ROLLOUT_COUNT
- rollout_max_new_tokens: $ROLLOUT_MAX_NEW_TOKENS
- c2_train_batch_size: $C2_TRAIN_BATCH_SIZE
- c2_eval_batch_size: $C2_EVAL_BATCH_SIZE
- c2_learning_rate: $C2_LR
- c2_num_train_epochs: $C2_EPOCHS

## Resolved Artifact Dirs

- c1_train_dir: $C1_TRAIN_DIR
- c1_eval_dir: $C1_EVAL_DIR
- c2_train_dir: $C2_TRAIN_DIR
- c2_eval_dir: $C2_EVAL_DIR

## P(IK) Metrics

- brier_score: ${C2_BRIER:-n/a}
- pearson: ${C2_PEARSON:-n/a}
- ece: ${C2_ECE:-n/a}
- known_auc: ${C2_KNOWN_AUC:-n/a}
- posthoc_brier: ${C2_POST_BRIER:-n/a}
- posthoc_pearson: ${C2_POST_PEARSON:-n/a}
- known_auc_posthoc: ${C2_KNOWN_AUC_POST:-n/a}

## Files

- suite_log_file: $SUITE_LOG_FILE
- metrics_file: $C2_EVAL_METRICS
EOF

{
  log_line "Summary file   : $SUMMARY_FILE"
  log_line "PIK eval metrics: $C2_EVAL_METRICS"
  log_line "Group run complete"
} | tee -a "$SUITE_LOG_FILE"
