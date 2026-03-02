#!/usr/bin/env bash
# Phase B suite launcher with optional pre/post benchmark gain evaluation.
#
# Why this file exists:
# - provide one-click named Phase B training groups,
# - keep supervisor-reportable intentions/observations/expectations beside the run,
# - optionally measure real PEFT gain with the frozen Phase A evaluator before and
#   after training,
# - persist suite-level logs separately from per-run training/eval artifacts.
#
# What this file does:
# 1. resolve `ACTIVE_PHASE_B_GROUP` into a concrete training config and eval plan,
# 2. optionally run baseline evaluation on held-out splits before training,
# 3. launch `scripts/phase_b_train_sft.py`,
# 4. optionally run post-train evaluation on the same held-out splits,
# 5. generate a markdown/json gain report showing how much PEFT changed accuracy.
#
# Interaction with other files:
# - `scripts/phase_b_train_sft.py` performs the real training run.
# - `scripts/phase_b_eval.py` bridges Phase B artifacts back into the frozen
#   Phase A evaluator.
# - `scripts/phase_b_compare_eval.py` summarizes before/after benchmark deltas.
# - `configs/phase_b/*.json` define the concrete training defaults.
#
# Example:
#   ACTIVE_PHASE_B_GROUP=B2_STRATEGYQA_FULL RUN_PREFIX=phase_b_strategyqa_full \
#   bash scripts/run_phase_b_training_suite.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACTIVE_PHASE_B_GROUP="${ACTIVE_PHASE_B_GROUP:-B1_SMOKE}"
RUN_PREFIX="${RUN_PREFIX:-phase_b}"
ENABLE_PERSISTED_LOGS="${ENABLE_PERSISTED_LOGS:-1}"
ENABLE_AUTO_GAIN_EVAL="${ENABLE_AUTO_GAIN_EVAL:-1}"
PHASE_B_EVAL_BATCH_SIZE="${PHASE_B_EVAL_BATCH_SIZE:-}"

timestamp() {
  # Print timestamps in a format that matches other suite logs.
  date "+%Y-%m-%d %H:%M:%S %z"
}

log_line() {
  # Add a timestamp prefix to one human-readable log message.
  local msg="$1"
  echo "[$(timestamp)] $msg"
}

json_config_value() {
  # Read one top-level key from the active JSON config.
  #
  # Example:
  #   model_path="$(json_config_value model_path)"
  local key="$1"
  "$PYTHON_BIN" - "$CONFIG_JSON" "$key" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
value = payload.get(key)
if value is None:
    print("")
elif isinstance(value, (dict, list)):
    print(json.dumps(value, ensure_ascii=False))
else:
    print(value)
PY
}

latest_phase_b_run_dir_for_name() {
  # Resolve the latest Phase B run directory for one run-name prefix.
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
  # Resolve the latest Phase A metrics.json for one eval run-name prefix.
  local run_name="$1"
  local latest=""
  latest="$(compgen -G "assets/artifacts/phase_a_runs/${run_name}_*/metrics.json" | sort | tail -n 1 || true)"
  if [[ -z "$latest" ]]; then
    echo "ERROR: No Phase A metrics.json found for run-name: $run_name" >&2
    return 1
  fi
  printf '%s\n' "$latest"
}

append_extra_args() {
  # Append shell-split extra CLI args into one array variable by name.
  #
  # Example:
  #   cmd=(python script.py)
  #   PHASE_B_EXTRA_ARGS="--max-steps 20"
  #   append_extra_args cmd "$PHASE_B_EXTRA_ARGS"
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

resolve_group() {
  # Map a short group id to a stable config block and optional eval plan.
  #
  # Keep this function explicit rather than data-driven so a novice can inspect
  # all supported Phase B entrypoints in one place.
  AUTO_GAIN_EVAL_GROUP=0
  GROUP_DATASET=""
  EVAL_SPECS=""
  DEFAULT_EVAL_BATCH_SIZE=4

  case "$ACTIVE_PHASE_B_GROUP" in
    B1_SMOKE)
      GROUP_TITLE="B1 Smoke Training"
      GROUP_INTENTION="Fast end-to-end validation of training/eval/checkpoint path."
      GROUP_OBSERVE="Check run completion, loss logging, checkpoint save, and eval artifact writing."
      GROUP_EXPECT="Finish in minutes; no crash; all required files emitted."
      CONFIG_JSON="configs/phase_b/peft_smoke_strategyqa.json"
      ;;
    B1_FIRST)
      GROUP_TITLE="B1 First Candidate"
      GROUP_INTENTION="Run first full candidate config for StrategyQA PEFT baseline."
      GROUP_OBSERVE="Check stability, throughput, and post-train eval outputs."
      GROUP_EXPECT="Stable train/eval, reproducible manifest, checkpoint artifacts."
      CONFIG_JSON="configs/phase_b/peft_first_run_strategyqa.json"
      ;;
    B2_STRATEGYQA_FULL)
      GROUP_TITLE="B2 Full StrategyQA PEFT Gain"
      GROUP_INTENTION="Train PEFT on the full StrategyQA CoT-compact train split and measure held-out gain against the frozen base model."
      GROUP_OBSERVE="Check validation/test accuracy before vs after PEFT under the same freeform Phase A evaluator."
      GROUP_EXPECT="Held-out accuracy should improve or stay flat with low parse-error drift."
      CONFIG_JSON="configs/phase_b/peft_full_strategyqa_cot.json"
      GROUP_DATASET="strategyqa"
      AUTO_GAIN_EVAL_GROUP=1
      DEFAULT_EVAL_BATCH_SIZE=8
      EVAL_SPECS="$(cat <<'EOF'
validation|assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl|freeform|96
test|assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/test.jsonl|freeform|96
EOF
)"
      ;;
    B2_GSM8K_FULL)
      GROUP_TITLE="B2 Full GSM8K PEFT Gain"
      GROUP_INTENTION="Train PEFT on the full GSM8K CoT-compact train split and measure held-out gain against the frozen base model."
      GROUP_OBSERVE="Check validation/test accuracy before vs after PEFT under the same math evaluator and truncation safeguards."
      GROUP_EXPECT="Held-out accuracy should improve with parse errors remaining at zero."
      CONFIG_JSON="configs/phase_b/peft_full_gsm8k_cot.json"
      GROUP_DATASET="gsm8k"
      AUTO_GAIN_EVAL_GROUP=1
      DEFAULT_EVAL_BATCH_SIZE=8
      EVAL_SPECS="$(cat <<'EOF'
validation|assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/validation.jsonl|freeform|192
test|assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/test.jsonl|freeform|192
EOF
)"
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_B_GROUP=$ACTIVE_PHASE_B_GROUP"
      echo "Supported groups: B1_SMOKE, B1_FIRST, B2_STRATEGYQA_FULL, B2_GSM8K_FULL"
      exit 1
      ;;
  esac
}

run_eval_spec() {
  # Run one baseline or post-train eval spec through `scripts/phase_b_eval.py`.
  #
  # Arguments:
  #   $1 = pre|post
  #   $2 = split label (e.g. validation)
  #   $3 = input JSONL
  #   $4 = decode mode
  #   $5 = max_new_tokens
  local stage="$1"
  local label="$2"
  local input_jsonl="$3"
  local decode_mode="$4"
  local max_new_tokens="$5"
  local run_name="${RUN_NAME}_${stage}_${label}"
  local eval_batch_size="${PHASE_B_EVAL_BATCH_SIZE:-$DEFAULT_EVAL_BATCH_SIZE}"

  local cmd=(
    "$PYTHON_BIN" -u scripts/phase_b_eval.py
    --input-jsonl "$input_jsonl"
    --run-name "$run_name"
    --batch-size "$eval_batch_size"
    --strategyqa-decode-mode "$decode_mode"
    --max-new-tokens "$max_new_tokens"
    --require-cuda
  )

  if [[ "$stage" == "pre" ]]; then
    cmd+=(--model-path "$BASE_MODEL_PATH")
  else
    cmd+=(--phase-b-run-dir "$TRAIN_RUN_DIR")
  fi

  if [[ -n "${PHASE_B_EVAL_EXTRA_ARGS:-}" ]]; then
    cmd+=(--extra-args)
    # shellcheck disable=SC2206
    local eval_extra_arr=(${PHASE_B_EVAL_EXTRA_ARGS})
    cmd+=("${eval_extra_arr[@]}")
  fi

  {
    log_line "Eval stage     : $stage"
    log_line "Eval label     : $label"
    log_line "Eval input      : $input_jsonl"
    log_line "Eval decode     : $decode_mode"
    log_line "Eval tokens     : $max_new_tokens"
    log_line "Eval batch      : $eval_batch_size"
    log_line "Eval command    : ${cmd[*]}"
  } | tee -a "$SUITE_LOG_FILE"

  "${cmd[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"
}

resolve_group

LOG_ROOT="assets/artifacts/phase_b_logs/$RUN_PREFIX"
mkdir -p "$LOG_ROOT"
SUITE_LOG_FILE="$LOG_ROOT/suite.log"
SUMMARY_FILE="$LOG_ROOT/final_summary.md"
GAIN_SUMMARY_JSON="$LOG_ROOT/peft_gain_summary.json"
GAIN_SUMMARY_MD="$LOG_ROOT/peft_gain_summary.md"

RUN_NAME="${RUN_PREFIX}_${ACTIVE_PHASE_B_GROUP,,}"
AUTO_GAIN_EVAL=0
if [[ "$AUTO_GAIN_EVAL_GROUP" -eq 1 && "$ENABLE_AUTO_GAIN_EVAL" -eq 1 ]]; then
  AUTO_GAIN_EVAL=1
fi

BASE_MODEL_PATH="$(json_config_value model_path)"

run_cmd=(
  "$PYTHON_BIN" -u scripts/phase_b_train_sft.py
  --config-json "$CONFIG_JSON"
  --run-name "$RUN_NAME"
)
append_extra_args run_cmd "${PHASE_B_EXTRA_ARGS:-}"

{
  log_line "Repo root      : $REPO_ROOT"
  log_line "Python         : $PYTHON_BIN"
  log_line "Group          : $ACTIVE_PHASE_B_GROUP"
  log_line "Group title    : $GROUP_TITLE"
  log_line "Run prefix     : $RUN_PREFIX"
  log_line "Run name       : $RUN_NAME"
  log_line "Config JSON    : $CONFIG_JSON"
  log_line "Base model     : ${BASE_MODEL_PATH:-<missing>}"
  log_line "Intention      : $GROUP_INTENTION"
  log_line "Observe        : $GROUP_OBSERVE"
  log_line "Expectation    : $GROUP_EXPECT"
  log_line "Auto gain eval : $AUTO_GAIN_EVAL"
  log_line "Command        : ${run_cmd[*]}"
  log_line "Group run start"
} | tee "$SUITE_LOG_FILE"

COMPARE_ARGS=()

if [[ "$AUTO_GAIN_EVAL" -eq 1 ]]; then
  while IFS='|' read -r label input_jsonl decode_mode max_new_tokens; do
    [[ -z "$label" ]] && continue
    run_eval_spec "pre" "$label" "$input_jsonl" "$decode_mode" "$max_new_tokens"
    before_metrics="$(latest_phase_a_metrics_for_name "${RUN_NAME}_pre_${label}")"
    COMPARE_ARGS+=(--compare "$label" "$before_metrics" "__POST__${label}")
  done <<< "$EVAL_SPECS"
fi

"${run_cmd[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

TRAIN_RUN_DIR="$(latest_phase_b_run_dir_for_name "$RUN_NAME")"

if [[ "$AUTO_GAIN_EVAL" -eq 1 ]]; then
  updated_compare_args=()
  while IFS='|' read -r label input_jsonl decode_mode max_new_tokens; do
    [[ -z "$label" ]] && continue
    run_eval_spec "post" "$label" "$input_jsonl" "$decode_mode" "$max_new_tokens"
    before_metrics="$(latest_phase_a_metrics_for_name "${RUN_NAME}_pre_${label}")"
    after_metrics="$(latest_phase_a_metrics_for_name "${RUN_NAME}_post_${label}")"
    updated_compare_args+=(--compare "$label" "$before_metrics" "$after_metrics")
  done <<< "$EVAL_SPECS"

  compare_cmd=(
    "$PYTHON_BIN" -u scripts/phase_b_compare_eval.py
    --dataset "$GROUP_DATASET"
    --phase-b-run-dir "$TRAIN_RUN_DIR"
    --title "$GROUP_TITLE"
    --output-json "$GAIN_SUMMARY_JSON"
    --output-markdown "$GAIN_SUMMARY_MD"
    "${updated_compare_args[@]}"
  )

  {
    log_line "Gain analysis  : ${compare_cmd[*]}"
  } | tee -a "$SUITE_LOG_FILE"
  "${compare_cmd[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"
fi

cat > "$SUMMARY_FILE" <<EOF
# Phase B Training Suite Summary

- generated_at: $(date --iso-8601=seconds)
- group_id: $ACTIVE_PHASE_B_GROUP
- group_title: $GROUP_TITLE
- run_prefix: $RUN_PREFIX
- run_name: $RUN_NAME
- config_json: $CONFIG_JSON
- base_model_path: ${BASE_MODEL_PATH:-N/A}
- train_run_dir: ${TRAIN_RUN_DIR:-N/A}
- intention: $GROUP_INTENTION
- observe: $GROUP_OBSERVE
- expectation: $GROUP_EXPECT
- auto_gain_eval: $AUTO_GAIN_EVAL
- suite_log_file: $SUITE_LOG_FILE
EOF

if [[ "$AUTO_GAIN_EVAL" -eq 1 ]]; then
  {
    echo "- gain_summary_json: $GAIN_SUMMARY_JSON"
    echo "- gain_summary_markdown: $GAIN_SUMMARY_MD"
    echo ""
    echo "## Eval Plan"
    while IFS='|' read -r label input_jsonl decode_mode max_new_tokens; do
      [[ -z "$label" ]] && continue
      echo "- $label | input=\`$input_jsonl\` | decode=\`$decode_mode\` | tok=\`$max_new_tokens\`"
    done <<< "$EVAL_SPECS"
    echo ""
    cat "$GAIN_SUMMARY_MD"
  } >> "$SUMMARY_FILE"
fi

{
  log_line "Summary file   : $SUMMARY_FILE"
  if [[ "$AUTO_GAIN_EVAL" -eq 1 ]]; then
    log_line "Gain summary   : $GAIN_SUMMARY_MD"
  fi
  log_line "Group run complete"
} | tee -a "$SUITE_LOG_FILE"

if [[ "$ENABLE_PERSISTED_LOGS" -eq 0 ]]; then
  rm -f "$SUITE_LOG_FILE" "$SUMMARY_FILE" "$GAIN_SUMMARY_JSON" "$GAIN_SUMMARY_MD"
fi
