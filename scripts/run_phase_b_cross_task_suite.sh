#!/usr/bin/env bash
# Phase B cross-task evaluation suite.
#
# Why this file exists:
# - Phase B training tells us whether an adapter helps on its own task,
# - but the BCR/ABR research goal also needs cross-task interference evidence:
#   does a StrategyQA-trained adapter hurt GSM8K, and does a GSM8K-trained adapter
#   hurt StrategyQA?
#
# What this file does:
# 1. resolve one named cross-task group,
# 2. locate the finished source Phase B run directory,
# 3. evaluate the frozen base model on the target task,
# 4. evaluate the source adapter/model on the same target task,
# 5. summarize before/after deltas in one markdown/json report.
#
# Interaction with other files:
# - `scripts/phase_b_eval.py`: executes each target-task eval.
# - `scripts/phase_b_compare_eval.py`: produces the held-out comparison summary.
# - `assets/artifacts/phase_b_runs/*`: supplies the trained adapter/model.
#
# Example:
#   ACTIVE_CROSS_TASK_GROUP=B3_XTASK_STRAT_R32_TO_GSM8K \
#   RUN_PREFIX=xtask_strat_to_gsm \
#   CUDA_VISIBLE_DEVICES=1 \
#   PHASE_B_EVAL_BATCH_SIZE=64 \
#   bash scripts/run_phase_b_cross_task_suite.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACTIVE_CROSS_TASK_GROUP="${ACTIVE_CROSS_TASK_GROUP:-B3_XTASK_STRAT_R32_TO_GSM8K}"
RUN_PREFIX="${RUN_PREFIX:-phase_b_xtask}"
PHASE_B_EVAL_BATCH_SIZE="${PHASE_B_EVAL_BATCH_SIZE:-64}"
ENABLE_PERSISTED_LOGS="${ENABLE_PERSISTED_LOGS:-1}"
CURRENT_STAGE="init"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S %z"
}

log_line() {
  local msg="$1"
  echo "[$(timestamp)] $msg"
}

write_failure_summary() {
  local exit_code="$1"
  [[ -z "${SUMMARY_FILE:-}" ]] && return 0
  mkdir -p "$(dirname "$SUMMARY_FILE")"
  cat > "$SUMMARY_FILE" <<EOF
# Phase B Cross-Task Suite Summary

- generated_at: $(date --iso-8601=seconds)
- group_id: ${ACTIVE_CROSS_TASK_GROUP:-N/A}
- group_title: ${GROUP_TITLE:-N/A}
- run_prefix: ${RUN_PREFIX:-N/A}
- run_name: ${RUN_NAME:-N/A}
- source_run_name_prefix: ${SOURCE_RUN_NAME_PREFIX:-N/A}
- source_phase_b_run_dir: ${SOURCE_PHASE_B_RUN_DIR:-N/A}
- target_dataset: ${TARGET_DATASET:-N/A}
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

latest_phase_b_run_dir_for_name() {
  local run_name="$1"
  local latest=""
  latest="$(compgen -G "assets/artifacts/phase_b_runs/${run_name}_*" | sort | tail -n 1 || true)"
  if [[ -z "$latest" ]]; then
    echo "ERROR: No Phase B run directory found for run-name: $run_name" >&2
    return 1
  fi
  printf '%s\n' "$latest"
}

latest_phase_a_metrics_for_name() {
  local run_name="$1"
  local latest=""
  latest="$(compgen -G "assets/artifacts/phase_a_runs/${run_name}_*/metrics.json" | sort | tail -n 1 || true)"
  if [[ -z "$latest" ]]; then
    echo "ERROR: No Phase A metrics.json found for run-name: $run_name" >&2
    return 1
  fi
  printf '%s\n' "$latest"
}

json_manifest_value() {
  local manifest_path="$1"
  local key="$2"
  "$PYTHON_BIN" - "$manifest_path" "$key" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
value = payload.get(key)
if value is None:
    print("")
else:
    print(value)
PY
}

resolve_group() {
  TARGET_DATASET=""
  EVAL_SPECS=""

  case "$ACTIVE_CROSS_TASK_GROUP" in
    B3_XTASK_STRAT_R32_TO_GSM8K)
      GROUP_TITLE="B3 Cross-Task: StrategyQA Rank-32 Adapter -> GSM8K"
      GROUP_INTENTION="Measure whether the current best StrategyQA adapter transfers or interferes when evaluated on GSM8K."
      GROUP_OBSERVE="Compare frozen-base vs StrategyQA-trained adapter on GSM8K held-out validation/test."
      GROUP_EXPECT="Any meaningful GSM8K drop would indicate cross-task interference from StrategyQA-focused alignment."
      SOURCE_RUN_NAME_PREFIX="strategyqa_diag_r32_b2_strategyqa_diag_lora_r32"
      TARGET_DATASET="gsm8k"
      EVAL_SPECS="$(cat <<'EOF'
validation|assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/validation.jsonl|freeform|192
test|assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/test.jsonl|freeform|192
EOF
)"
      ;;
    B3_XTASK_GSM8K_FULL_TO_STRAT)
      GROUP_TITLE="B3 Cross-Task: GSM8K Full-CoT Adapter -> StrategyQA"
      GROUP_INTENTION="Measure whether the harmful GSM8K full-CoT adapter also harms or shifts StrategyQA behavior."
      GROUP_OBSERVE="Compare frozen-base vs GSM8K full-CoT adapter on StrategyQA held-out validation/test."
      GROUP_EXPECT="A StrategyQA drop would indicate broader cross-task capability conflict, not only same-task failure."
      SOURCE_RUN_NAME_PREFIX="phase_b_gsm8k_full_b2_gsm8k_full"
      TARGET_DATASET="strategyqa"
      EVAL_SPECS="$(cat <<'EOF'
validation|assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl|freeform|96
test|assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/test.jsonl|freeform|96
EOF
)"
      ;;
    B3_XTASK_GSM8K_DIRECT_TO_STRAT)
      GROUP_TITLE="B3 Cross-Task: GSM8K Direct-Style Adapter -> StrategyQA"
      GROUP_INTENTION="Measure whether a shorter, safer GSM8K-style adapter transfers to StrategyQA better than the full CoT GSM8K adapter."
      GROUP_OBSERVE="Compare frozen-base vs GSM8K direct-style adapter on StrategyQA held-out validation/test."
      GROUP_EXPECT="If this adapter harms StrategyQA less than the full CoT adapter, the cross-task damage is tied to the long-CoT GSM8K target."
      SOURCE_RUN_NAME_PREFIX="gsm8k_diag_direct_b2_gsm8k_diag_direct_style"
      TARGET_DATASET="strategyqa"
      EVAL_SPECS="$(cat <<'EOF'
validation|assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl|freeform|96
test|assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/test.jsonl|freeform|96
EOF
)"
      ;;
    B3_XTASK_GSM8K_EQUATION_TO_STRAT)
      GROUP_TITLE="B3 Cross-Task: GSM8K Equation-Style Adapter -> StrategyQA"
      GROUP_INTENTION="Measure whether the GSM8K equation-style adapter preserves StrategyQA better than the full GSM8K CoT adapter."
      GROUP_OBSERVE="Compare frozen-base vs GSM8K equation-style adapter on StrategyQA held-out validation/test."
      GROUP_EXPECT="If this adapter interferes less, cross-task damage is again pointing to long-CoT GSM8K supervision rather than GSM8K tuning in general."
      SOURCE_RUN_NAME_PREFIX="gsm8k_diag_equation_b2_gsm8k_diag_equation_style"
      TARGET_DATASET="strategyqa"
      EVAL_SPECS="$(cat <<'EOF'
validation|assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl|freeform|96
test|assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/test.jsonl|freeform|96
EOF
)"
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_CROSS_TASK_GROUP=$ACTIVE_CROSS_TASK_GROUP"
      echo "Supported groups: B3_XTASK_STRAT_R32_TO_GSM8K, B3_XTASK_GSM8K_FULL_TO_STRAT, B3_XTASK_GSM8K_DIRECT_TO_STRAT, B3_XTASK_GSM8K_EQUATION_TO_STRAT"
      exit 1
      ;;
  esac
}

run_eval_spec() {
  local stage="$1"
  local label="$2"
  local input_jsonl="$3"
  local decode_mode="$4"
  local max_new_tokens="$5"
  local run_name="${RUN_NAME}_${stage}_${label}"

  local cmd=(
    "$PYTHON_BIN" -u scripts/phase_b_eval.py
    --input-jsonl "$input_jsonl"
    --run-name "$run_name"
    --batch-size "$PHASE_B_EVAL_BATCH_SIZE"
    --strategyqa-decode-mode "$decode_mode"
    --max-new-tokens "$max_new_tokens"
    --require-cuda
  )

  if [[ "$stage" == "pre" ]]; then
    cmd+=(--model-path "$BASE_MODEL_PATH")
  else
    cmd+=(--phase-b-run-dir "$SOURCE_PHASE_B_RUN_DIR")
  fi

  if [[ -n "${PHASE_B_EVAL_EXTRA_ARGS:-}" ]]; then
    cmd+=(--extra-args)
    # shellcheck disable=SC2206
    local eval_extra_arr=(${PHASE_B_EVAL_EXTRA_ARGS})
    cmd+=("${eval_extra_arr[@]}")
  fi

  CURRENT_STAGE="eval_${stage}_${label}"
  {
    log_line "Eval stage     : $stage"
    log_line "Eval label     : $label"
    log_line "Eval input      : $input_jsonl"
    log_line "Eval decode     : $decode_mode"
    log_line "Eval tokens     : $max_new_tokens"
    log_line "Eval batch      : $PHASE_B_EVAL_BATCH_SIZE"
    log_line "Eval command    : ${cmd[*]}"
  } | tee -a "$SUITE_LOG_FILE"

  "${cmd[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"
}

resolve_group

LOG_ROOT="assets/artifacts/phase_b_logs/$RUN_PREFIX"
mkdir -p "$LOG_ROOT"
SUITE_LOG_FILE="$LOG_ROOT/suite.log"
SUMMARY_FILE="$LOG_ROOT/final_summary.md"
GAIN_SUMMARY_JSON="$LOG_ROOT/cross_task_gain_summary.json"
GAIN_SUMMARY_MD="$LOG_ROOT/cross_task_gain_summary.md"

RUN_NAME="${RUN_PREFIX}_${ACTIVE_CROSS_TASK_GROUP,,}"
SOURCE_PHASE_B_RUN_DIR="${SOURCE_PHASE_B_RUN_DIR:-$(latest_phase_b_run_dir_for_name "$SOURCE_RUN_NAME_PREFIX")}"
BASE_MODEL_PATH="$(json_manifest_value "$SOURCE_PHASE_B_RUN_DIR/manifest.json" "model_path")"

{
  log_line "Repo root      : $REPO_ROOT"
  log_line "Python         : $PYTHON_BIN"
  log_line "Group          : $ACTIVE_CROSS_TASK_GROUP"
  log_line "Group title    : $GROUP_TITLE"
  log_line "Run prefix     : $RUN_PREFIX"
  log_line "Run name       : $RUN_NAME"
  log_line "Source run prefix: $SOURCE_RUN_NAME_PREFIX"
  log_line "Source Phase B : $SOURCE_PHASE_B_RUN_DIR"
  log_line "Base model     : $BASE_MODEL_PATH"
  log_line "Target dataset : $TARGET_DATASET"
  log_line "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
  log_line "Eval batch     : $PHASE_B_EVAL_BATCH_SIZE"
  log_line "Intention      : $GROUP_INTENTION"
  log_line "Observe        : $GROUP_OBSERVE"
  log_line "Expectation    : $GROUP_EXPECT"
  log_line "Group run start"
} | tee "$SUITE_LOG_FILE"

updated_compare_args=()
while IFS='|' read -r label input_jsonl decode_mode max_new_tokens; do
  [[ -z "$label" ]] && continue
  run_eval_spec "pre" "$label" "$input_jsonl" "$decode_mode" "$max_new_tokens"
  run_eval_spec "post" "$label" "$input_jsonl" "$decode_mode" "$max_new_tokens"
  before_metrics="$(latest_phase_a_metrics_for_name "${RUN_NAME}_pre_${label}")"
  after_metrics="$(latest_phase_a_metrics_for_name "${RUN_NAME}_post_${label}")"
  updated_compare_args+=(--compare "$label" "$before_metrics" "$after_metrics")
done <<< "$EVAL_SPECS"

CURRENT_STAGE="cross_task_analysis"
compare_cmd=(
  "$PYTHON_BIN" -u scripts/phase_b_compare_eval.py
  --dataset "$TARGET_DATASET"
  --phase-b-run-dir "$SOURCE_PHASE_B_RUN_DIR"
  --title "$GROUP_TITLE"
  --output-json "$GAIN_SUMMARY_JSON"
  --output-markdown "$GAIN_SUMMARY_MD"
  "${updated_compare_args[@]}"
)

{
  log_line "Cross-task analysis: ${compare_cmd[*]}"
} | tee -a "$SUITE_LOG_FILE"
"${compare_cmd[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

CURRENT_STAGE="final_summary"
cat > "$SUMMARY_FILE" <<EOF
# Phase B Cross-Task Suite Summary

- generated_at: $(date --iso-8601=seconds)
- group_id: $ACTIVE_CROSS_TASK_GROUP
- group_title: $GROUP_TITLE
- run_prefix: $RUN_PREFIX
- run_name: $RUN_NAME
- source_run_name_prefix: $SOURCE_RUN_NAME_PREFIX
- source_phase_b_run_dir: $SOURCE_PHASE_B_RUN_DIR
- base_model_path: $BASE_MODEL_PATH
- target_dataset: $TARGET_DATASET
- suite_log_file: $SUITE_LOG_FILE
- gain_summary_json: $GAIN_SUMMARY_JSON
- gain_summary_markdown: $GAIN_SUMMARY_MD

$(cat "$GAIN_SUMMARY_MD")
EOF

{
  log_line "Summary file   : $SUMMARY_FILE"
  log_line "Gain summary   : $GAIN_SUMMARY_MD"
  log_line "Group run complete"
} | tee -a "$SUITE_LOG_FILE"

if [[ "$ENABLE_PERSISTED_LOGS" -eq 0 ]]; then
  rm -f "$SUITE_LOG_FILE" "$SUMMARY_FILE" "$GAIN_SUMMARY_JSON" "$GAIN_SUMMARY_MD"
fi
