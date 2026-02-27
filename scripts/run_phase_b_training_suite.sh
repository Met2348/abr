#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACTIVE_PHASE_B_GROUP="${ACTIVE_PHASE_B_GROUP:-B1_SMOKE}"
RUN_PREFIX="${RUN_PREFIX:-phase_b}"
ENABLE_PERSISTED_LOGS="${ENABLE_PERSISTED_LOGS:-1}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S %z"
}

log_line() {
  local msg="$1"
  echo "[$(timestamp)] $msg"
}

resolve_group() {
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
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_B_GROUP=$ACTIVE_PHASE_B_GROUP"
      echo "Supported groups: B1_SMOKE, B1_FIRST"
      exit 1
      ;;
  esac
}

resolve_group

LOG_ROOT="assets/artifacts/phase_b_logs/$RUN_PREFIX"
mkdir -p "$LOG_ROOT"
SUITE_LOG_FILE="$LOG_ROOT/suite.log"
SUMMARY_FILE="$LOG_ROOT/final_summary.md"

RUN_NAME="${RUN_PREFIX}_${ACTIVE_PHASE_B_GROUP,,}"

run_cmd=(
  "$PYTHON_BIN" -u scripts/phase_b_train_sft.py
  --config-json "$CONFIG_JSON"
  --run-name "$RUN_NAME"
)

if [[ -n "${PHASE_B_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_arr=(${PHASE_B_EXTRA_ARGS})
  run_cmd+=("${extra_arr[@]}")
fi

{
  log_line "Repo root      : $REPO_ROOT"
  log_line "Python         : $PYTHON_BIN"
  log_line "Group          : $ACTIVE_PHASE_B_GROUP"
  log_line "Group title    : $GROUP_TITLE"
  log_line "Run prefix     : $RUN_PREFIX"
  log_line "Run name       : $RUN_NAME"
  log_line "Config JSON    : $CONFIG_JSON"
  log_line "Intention      : $GROUP_INTENTION"
  log_line "Observe        : $GROUP_OBSERVE"
  log_line "Expectation    : $GROUP_EXPECT"
  log_line "Command        : ${run_cmd[*]}"
  log_line "Group run start"
} | tee "$SUITE_LOG_FILE"

"${run_cmd[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

cat > "$SUMMARY_FILE" <<EOF
# Phase B Training Suite Summary

- generated_at: $(date --iso-8601=seconds)
- group_id: $ACTIVE_PHASE_B_GROUP
- group_title: $GROUP_TITLE
- run_prefix: $RUN_PREFIX
- run_name: $RUN_NAME
- config_json: $CONFIG_JSON
- intention: $GROUP_INTENTION
- observe: $GROUP_OBSERVE
- expectation: $GROUP_EXPECT
- suite_log_file: $SUITE_LOG_FILE
EOF

{
  log_line "Summary file   : $SUMMARY_FILE"
  log_line "Group run complete"
} | tee -a "$SUITE_LOG_FILE"

if [[ "$ENABLE_PERSISTED_LOGS" -eq 0 ]]; then
  rm -f "$SUITE_LOG_FILE" "$SUMMARY_FILE"
fi
