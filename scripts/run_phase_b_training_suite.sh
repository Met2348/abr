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
CURRENT_STAGE="init"
# 中文：通常只需要改 ACTIVE_PHASE_B_GROUP 和 RUN_PREFIX；其它建议先保持默认。

timestamp() {
  # Print timestamps in a format that matches other suite logs.
  date "+%Y-%m-%d %H:%M:%S %z"
}

log_line() {
  # Add a timestamp prefix to one human-readable log message.
  local msg="$1"
  echo "[$(timestamp)] $msg"
}

write_failure_summary() {
  # Persist a short failure report when the suite exits early.
  #
  # This is intentionally shell-only and lightweight so that even failures that
  # happen before training has started still leave a readable breadcrumb.
  local exit_code="$1"
  [[ -z "${SUMMARY_FILE:-}" ]] && return 0
  mkdir -p "$(dirname "$SUMMARY_FILE")"
  cat > "$SUMMARY_FILE" <<EOF
# Phase B Training Suite Summary

- generated_at: $(date --iso-8601=seconds)
- group_id: ${ACTIVE_PHASE_B_GROUP:-N/A}
- group_title: ${GROUP_TITLE:-N/A}
- run_prefix: ${RUN_PREFIX:-N/A}
- run_name: ${RUN_NAME:-N/A}
- config_json: ${CONFIG_JSON:-N/A}
- base_model_path: ${BASE_MODEL_PATH:-N/A}
- train_run_dir: ${TRAIN_RUN_DIR:-N/A}
- intention: ${GROUP_INTENTION:-N/A}
- observe: ${GROUP_OBSERVE:-N/A}
- expectation: ${GROUP_EXPECT:-N/A}
- auto_gain_eval: ${AUTO_GAIN_EVAL:-0}
- suite_log_file: ${SUITE_LOG_FILE:-N/A}
- status: failed
- exit_code: ${exit_code}
- failed_stage: ${CURRENT_STAGE:-unknown}
- note: This suite stopped before normal completion. Inspect \`${SUITE_LOG_FILE:-suite.log}\` for the last completed command.
EOF
}

on_exit() {
  # Emit an explicit failure line and a partial summary for non-zero exits.
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
  AUTO_CKPT_SWEEP_GROUP=0
  GROUP_DATASET=""
  EVAL_SPECS=""
  DEFAULT_EVAL_BATCH_SIZE=4
  # 中文：EVAL_SPECS 每行格式：
  # label|input_jsonl|decode_mode|max_new_tokens

  # 中文：新增 B 组时，优先复制一个最接近的分组并只改 CONFIG_JSON/EVAL_SPECS。
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
    B2_STRATEGYQA_DIAG_EPOCH_200)
      GROUP_TITLE="B2 StrategyQA Diagnostic: 2.0 Epoch"
      GROUP_INTENTION="Test whether the current StrategyQA PEFT gain can be pushed higher simply by training longer on the same CoT-compact data."
      GROUP_OBSERVE="Compare held-out validation/test gain against the 1.0 epoch full StrategyQA baseline."
      GROUP_EXPECT="If accuracy improves again, the 1.0 epoch run was still under-trained; if not, the baseline is near saturation."
      CONFIG_JSON="configs/phase_b/peft_diag_strategyqa_epoch200.json"
      GROUP_DATASET="strategyqa"
      AUTO_GAIN_EVAL_GROUP=1
      DEFAULT_EVAL_BATCH_SIZE=8
      EVAL_SPECS="$(cat <<'EOF'
validation|assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl|freeform|96
test|assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/test.jsonl|freeform|96
EOF
)"
      ;;
    B2_STRATEGYQA_DIAG_EPOCH_300)
      GROUP_TITLE="B2 StrategyQA Diagnostic: 3.0 Epoch"
      GROUP_INTENTION="Test whether further increasing StrategyQA exposure continues to improve held-out accuracy or starts to flatten/regress."
      GROUP_OBSERVE="Compare against both 1.0 epoch and 2.0 epoch runs under the same prompt/eval setup."
      GROUP_EXPECT="If this run stalls or drops, StrategyQA performance has likely reached its useful training horizon."
      CONFIG_JSON="configs/phase_b/peft_diag_strategyqa_epoch300.json"
      GROUP_DATASET="strategyqa"
      AUTO_GAIN_EVAL_GROUP=1
      DEFAULT_EVAL_BATCH_SIZE=8
      EVAL_SPECS="$(cat <<'EOF'
validation|assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl|freeform|96
test|assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/test.jsonl|freeform|96
EOF
)"
      ;;
    B2_STRATEGYQA_DIAG_LORA_R8)
      GROUP_TITLE="B2 StrategyQA Diagnostic: LoRA Rank 8"
      GROUP_INTENTION="Test whether the current StrategyQA gain can be preserved with a smaller adapter, indicating the baseline rank-16 adapter may be over-provisioned."
      GROUP_OBSERVE="Compare held-out gain and stability against the baseline rank-16 PEFT run."
      GROUP_EXPECT="If performance matches baseline, rank-8 is sufficient and more efficient."
      CONFIG_JSON="configs/phase_b/peft_diag_strategyqa_lora_r8.json"
      GROUP_DATASET="strategyqa"
      AUTO_GAIN_EVAL_GROUP=1
      DEFAULT_EVAL_BATCH_SIZE=8
      EVAL_SPECS="$(cat <<'EOF'
validation|assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl|freeform|96
test|assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/test.jsonl|freeform|96
EOF
)"
      ;;
    B2_STRATEGYQA_DIAG_LORA_R32)
      GROUP_TITLE="B2 StrategyQA Diagnostic: LoRA Rank 32"
      GROUP_INTENTION="Test whether StrategyQA is currently capacity-limited by the baseline rank-16 LoRA adapter."
      GROUP_OBSERVE="Compare held-out gain against the baseline rank-16 and smaller rank-8 runs."
      GROUP_EXPECT="If rank-32 improves materially, LoRA capacity is a real bottleneck on StrategyQA."
      CONFIG_JSON="configs/phase_b/peft_diag_strategyqa_lora_r32.json"
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
    B2_GSM8K_DIAG_LR_5E5)
      GROUP_TITLE="B2 GSM8K Diagnostic: LR 5e-5"
      GROUP_INTENTION="Test whether the GSM8K drop is caused by an overly aggressive learning rate while keeping data/style fixed to the CoT baseline."
      GROUP_OBSERVE="Compare held-out validation/test gain against B2_GSM8K_FULL with the same prompts and sequence length."
      GROUP_EXPECT="If accuracy recovers materially, the original 2e-4 learning rate was too large for stable math adaptation."
      CONFIG_JSON="configs/phase_b/peft_diag_gsm8k_cot_lr5e5.json"
      GROUP_DATASET="gsm8k"
      AUTO_GAIN_EVAL_GROUP=1
      DEFAULT_EVAL_BATCH_SIZE=8
      EVAL_SPECS="$(cat <<'EOF'
validation|assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/validation.jsonl|freeform|192
test|assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/test.jsonl|freeform|192
EOF
)"
      ;;
    B2_GSM8K_DIAG_LR_1E4)
      GROUP_TITLE="B2 GSM8K Diagnostic: LR 1e-4"
      GROUP_INTENTION="Test whether a moderately smaller learning rate is enough to recover GSM8K held-out accuracy under the same CoT supervision."
      GROUP_OBSERVE="Compare held-out deltas against both LR 2e-4 and LR 5e-5 runs."
      GROUP_EXPECT="If this run improves over the original and matches/exceeds LR 5e-5, the failure is mostly optimizer overshoot rather than data-target mismatch."
      CONFIG_JSON="configs/phase_b/peft_diag_gsm8k_cot_lr1e4.json"
      GROUP_DATASET="gsm8k"
      AUTO_GAIN_EVAL_GROUP=1
      DEFAULT_EVAL_BATCH_SIZE=8
      EVAL_SPECS="$(cat <<'EOF'
validation|assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/validation.jsonl|freeform|192
test|assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/test.jsonl|freeform|192
EOF
)"
      ;;
    B2_GSM8K_DIAG_EPOCH_025)
      GROUP_TITLE="B2 GSM8K Diagnostic: 0.25 Epoch"
      GROUP_INTENTION="Test whether the GSM8K regression is caused by too much exposure to CoT supervision rather than by the optimizer step size."
      GROUP_OBSERVE="Hold learning rate fixed and reduce training exposure to one quarter epoch."
      GROUP_EXPECT="If accuracy recovers, the original one-epoch run was over-training the adapter on style patterns."
      CONFIG_JSON="configs/phase_b/peft_diag_gsm8k_cot_epoch025.json"
      GROUP_DATASET="gsm8k"
      AUTO_GAIN_EVAL_GROUP=1
      DEFAULT_EVAL_BATCH_SIZE=8
      EVAL_SPECS="$(cat <<'EOF'
validation|assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/validation.jsonl|freeform|192
test|assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/test.jsonl|freeform|192
EOF
)"
      ;;
    B2_GSM8K_DIAG_EPOCH_050)
      GROUP_TITLE="B2 GSM8K Diagnostic: 0.50 Epoch"
      GROUP_INTENTION="Test whether a half-epoch exposure preserves more of the base model's math reasoning while still learning useful supervision patterns."
      GROUP_OBSERVE="Compare against 0.25 epoch and 1.0 epoch runs under the same CoT data and LR."
      GROUP_EXPECT="If the half-epoch run beats both the 1.0 epoch and 0.25 epoch runs, the GSM8K issue is primarily an exposure-tuning problem."
      CONFIG_JSON="configs/phase_b/peft_diag_gsm8k_cot_epoch050.json"
      GROUP_DATASET="gsm8k"
      AUTO_GAIN_EVAL_GROUP=1
      DEFAULT_EVAL_BATCH_SIZE=8
      EVAL_SPECS="$(cat <<'EOF'
validation|assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/validation.jsonl|freeform|192
test|assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/test.jsonl|freeform|192
EOF
)"
      ;;
    B2_GSM8K_DIAG_DIRECT_STYLE)
      GROUP_TITLE="B2 GSM8K Diagnostic: Direct-Final Style"
      GROUP_INTENTION="Test whether the GSM8K regression is caused by CoT-target imitation itself by training on a much shorter direct-final target format."
      GROUP_OBSERVE="Evaluate before/after gain on the matched direct-final benchmark protocol."
      GROUP_EXPECT="If this run avoids the CoT regression pattern, the main problem is the CoT supervision target rather than PEFT itself."
      CONFIG_JSON="configs/phase_b/peft_diag_gsm8k_direct_style.json"
      GROUP_DATASET="gsm8k"
      AUTO_GAIN_EVAL_GROUP=1
      DEFAULT_EVAL_BATCH_SIZE=8
      EVAL_SPECS="$(cat <<'EOF'
validation|assets/artifacts/phase_a_prepared/gsm8k/18ffeb7b40f2/validation.jsonl|freeform|32
test|assets/artifacts/phase_a_prepared/gsm8k/18ffeb7b40f2/test.jsonl|freeform|32
EOF
)"
      ;;
    B2_GSM8K_DIAG_EQUATION_STYLE)
      GROUP_TITLE="B2 GSM8K Diagnostic: Equation-Then-Final Style"
      GROUP_INTENTION="Test whether the equation-markup supervision style is specifically responsible for the cleaner-but-wrong GSM8K outputs."
      GROUP_OBSERVE="Evaluate before/after gain on the matched equation-style benchmark protocol."
      GROUP_EXPECT="If this run reproduces the same failure mode, the equation-heavy style is likely amplifying local arithmetic mistakes."
      CONFIG_JSON="configs/phase_b/peft_diag_gsm8k_equation_style.json"
      GROUP_DATASET="gsm8k"
      AUTO_GAIN_EVAL_GROUP=1
      DEFAULT_EVAL_BATCH_SIZE=8
      EVAL_SPECS="$(cat <<'EOF'
validation|assets/artifacts/phase_a_prepared/gsm8k/bdcce4830551/validation.jsonl|freeform|64
test|assets/artifacts/phase_a_prepared/gsm8k/bdcce4830551/test.jsonl|freeform|64
EOF
)"
      ;;
    B2_GSM8K_DIAG_CHECKPOINT_SWEEP)
      GROUP_TITLE="B2 GSM8K Diagnostic: Checkpoint Sweep"
      GROUP_INTENTION="Test whether GSM8K benchmark quality peaks before the final checkpoint under the current full CoT PEFT recipe."
      GROUP_OBSERVE="Save many checkpoints, then compare held-out accuracy for each retained checkpoint and the final adapter."
      GROUP_EXPECT="If an earlier checkpoint beats the final model, the GSM8K drop is partly late-run drift rather than purely bad supervision."
      CONFIG_JSON="configs/phase_b/peft_diag_gsm8k_checkpoint_sweep.json"
      GROUP_DATASET="gsm8k"
      AUTO_CKPT_SWEEP_GROUP=1
      DEFAULT_EVAL_BATCH_SIZE=64
      EVAL_SPECS="$(cat <<'EOF'
validation|assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/validation.jsonl|freeform|192
test|assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/test.jsonl|freeform|192
EOF
)"
      ;;
    B2_GSM8K_DIAG_SHORT_COT)
      GROUP_TITLE="B2 GSM8K Diagnostic: Short-CoT Supervision"
      GROUP_INTENTION="Test whether shortening the supervised GSM8K rationale preserves arithmetic quality better than the full CoT target."
      GROUP_OBSERVE="Train on the same CoT-prepared corpus, but keep only the last two reasoning lines plus the final answer during supervision."
      GROUP_EXPECT="If held-out accuracy recovers, long rationale imitation is a major contributor to the GSM8K regression."
      CONFIG_JSON="configs/phase_b/peft_diag_gsm8k_short_cot.json"
      GROUP_DATASET="gsm8k"
      AUTO_GAIN_EVAL_GROUP=1
      DEFAULT_EVAL_BATCH_SIZE=64
      EVAL_SPECS="$(cat <<'EOF'
validation|assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/validation.jsonl|freeform|192
test|assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/test.jsonl|freeform|192
EOF
)"
      ;;
    B2_GSM8K_DIAG_ANSWER_WEIGHTED)
      GROUP_TITLE="B2 GSM8K Diagnostic: Answer-Weighted Supervision"
      GROUP_INTENTION="Test whether GSM8K training loss is over-focused on rationale tokens by weighting the final-answer line more strongly than the reasoning lines."
      GROUP_OBSERVE="Compare held-out gain against the full CoT baseline while keeping the same data and optimizer regime."
      GROUP_EXPECT="If accuracy recovers, the main issue is likely target-loss imbalance rather than PEFT itself."
      CONFIG_JSON="configs/phase_b/peft_diag_gsm8k_answer_weighted.json"
      GROUP_DATASET="gsm8k"
      AUTO_GAIN_EVAL_GROUP=1
      DEFAULT_EVAL_BATCH_SIZE=64
      EVAL_SPECS="$(cat <<'EOF'
validation|assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/validation.jsonl|freeform|192
test|assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/test.jsonl|freeform|192
EOF
)"
      ;;
    B2_GSM8K_REPAIR_ANSWER_WEIGHTED_CKPT)
      GROUP_TITLE="B2 GSM8K Repair: Answer-Weighted + Checkpoint Sweep"
      GROUP_INTENTION="Combine the two strongest GSM8K repair signals discovered so far: answer-weighted supervision and dense checkpoint selection."
      GROUP_OBSERVE="Check whether held-out best-checkpoint accuracy can fully recover or exceed the frozen GSM8K baseline under the same long-CoT training data."
      GROUP_EXPECT="If the best retained checkpoint matches or beats the frozen base model, late-run drift plus loss imbalance explains most of the previous GSM8K drop."
      CONFIG_JSON="configs/phase_b/peft_repair_gsm8k_answer_weighted_ckpt.json"
      GROUP_DATASET="gsm8k"
      AUTO_CKPT_SWEEP_GROUP=1
      DEFAULT_EVAL_BATCH_SIZE=64
      EVAL_SPECS="$(cat <<'EOF'
validation|assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/validation.jsonl|freeform|192
test|assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/test.jsonl|freeform|192
EOF
)"
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_B_GROUP=$ACTIVE_PHASE_B_GROUP"
      echo "Supported groups: B1_SMOKE, B1_FIRST, B2_STRATEGYQA_FULL, B2_STRATEGYQA_DIAG_EPOCH_200, B2_STRATEGYQA_DIAG_EPOCH_300, B2_STRATEGYQA_DIAG_LORA_R8, B2_STRATEGYQA_DIAG_LORA_R32, B2_GSM8K_FULL, B2_GSM8K_DIAG_LR_5E5, B2_GSM8K_DIAG_LR_1E4, B2_GSM8K_DIAG_EPOCH_025, B2_GSM8K_DIAG_EPOCH_050, B2_GSM8K_DIAG_DIRECT_STYLE, B2_GSM8K_DIAG_EQUATION_STYLE, B2_GSM8K_DIAG_CHECKPOINT_SWEEP, B2_GSM8K_DIAG_SHORT_COT, B2_GSM8K_DIAG_ANSWER_WEIGHTED, B2_GSM8K_REPAIR_ANSWER_WEIGHTED_CKPT"
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
  # 中文：pre 阶段评估 frozen base，post 阶段评估训练后 adapter；便于直接算增益。

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

  CURRENT_STAGE="eval_${stage}_${label}"
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
CKPT_SWEEP_JSON="$LOG_ROOT/checkpoint_sweep_summary.json"
CKPT_SWEEP_MD="$LOG_ROOT/checkpoint_sweep_summary.md"

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
  log_line "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
  log_line "Intention      : $GROUP_INTENTION"
  log_line "Observe        : $GROUP_OBSERVE"
  log_line "Expectation    : $GROUP_EXPECT"
  log_line "Auto gain eval : $AUTO_GAIN_EVAL"
  log_line "Auto ckpt sweep: $AUTO_CKPT_SWEEP_GROUP"
  log_line "Command        : ${run_cmd[*]}"
  log_line "Group run start"
} | tee "$SUITE_LOG_FILE"

COMPARE_ARGS=()

if [[ "$AUTO_GAIN_EVAL" -eq 1 ]]; then
  # 中文：训练前先跑同一套评估，形成可对比的 before metrics。
  while IFS='|' read -r label input_jsonl decode_mode max_new_tokens; do
    [[ -z "$label" ]] && continue
    run_eval_spec "pre" "$label" "$input_jsonl" "$decode_mode" "$max_new_tokens"
    before_metrics="$(latest_phase_a_metrics_for_name "${RUN_NAME}_pre_${label}")"
    COMPARE_ARGS+=(--compare "$label" "$before_metrics" "__POST__${label}")
  done <<< "$EVAL_SPECS"
fi

CURRENT_STAGE="train_launch"
{
  log_line "Train launch   : ${run_cmd[*]}"
} | tee -a "$SUITE_LOG_FILE"
"${run_cmd[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

TRAIN_RUN_DIR="$(latest_phase_b_run_dir_for_name "$RUN_NAME")"

if [[ "$AUTO_GAIN_EVAL" -eq 1 ]]; then
  updated_compare_args=()
  # 中文：训练后复跑同一套评估；若这里改了输入或 token，上下对比会失真。
  while IFS='|' read -r label input_jsonl decode_mode max_new_tokens; do
    [[ -z "$label" ]] && continue
    run_eval_spec "post" "$label" "$input_jsonl" "$decode_mode" "$max_new_tokens"
    before_metrics="$(latest_phase_a_metrics_for_name "${RUN_NAME}_pre_${label}")"
    after_metrics="$(latest_phase_a_metrics_for_name "${RUN_NAME}_post_${label}")"
    updated_compare_args+=(--compare "$label" "$before_metrics" "$after_metrics")
  done <<< "$EVAL_SPECS"

  CURRENT_STAGE="gain_analysis"
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

if [[ "$AUTO_CKPT_SWEEP_GROUP" -eq 1 ]]; then
  CURRENT_STAGE="checkpoint_sweep"
  ckpt_sweep_cmd=(
    "$PYTHON_BIN" -u scripts/phase_b_checkpoint_sweep.py
    --phase-b-run-dir "$TRAIN_RUN_DIR"
    --dataset "$GROUP_DATASET"
    --title "$GROUP_TITLE"
    --run-name-prefix "${RUN_NAME}_sweep"
    --batch-size "${PHASE_B_EVAL_BATCH_SIZE:-$DEFAULT_EVAL_BATCH_SIZE}"
    --require-cuda
    --output-json "$CKPT_SWEEP_JSON"
    --output-markdown "$CKPT_SWEEP_MD"
  )

  if [[ -n "${PHASE_B_SWEEP_CHECKPOINTS:-}" ]]; then
    ckpt_sweep_cmd+=(--checkpoint-labels "$PHASE_B_SWEEP_CHECKPOINTS")
  fi
  while IFS='|' read -r label input_jsonl decode_mode max_new_tokens; do
    [[ -z "$label" ]] && continue
    ckpt_sweep_cmd+=(
      --eval-spec "$label" "$input_jsonl" "$decode_mode" "$max_new_tokens"
    )
  done <<< "$EVAL_SPECS"
  if [[ -n "${PHASE_B_EVAL_EXTRA_ARGS:-}" ]]; then
    ckpt_sweep_cmd+=(--extra-args)
    # shellcheck disable=SC2206
    eval_extra_arr=(${PHASE_B_EVAL_EXTRA_ARGS})
    ckpt_sweep_cmd+=("${eval_extra_arr[@]}")
  fi

  {
    log_line "Checkpoint sweep: ${ckpt_sweep_cmd[*]}"
  } | tee -a "$SUITE_LOG_FILE"
  "${ckpt_sweep_cmd[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"
fi

CURRENT_STAGE="final_summary"
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

if [[ "$AUTO_CKPT_SWEEP_GROUP" -eq 1 ]]; then
  {
    echo "- checkpoint_sweep_json: $CKPT_SWEEP_JSON"
    echo "- checkpoint_sweep_markdown: $CKPT_SWEEP_MD"
    echo ""
    cat "$CKPT_SWEEP_MD"
  } >> "$SUMMARY_FILE"
fi

{
  log_line "Summary file   : $SUMMARY_FILE"
  if [[ "$AUTO_GAIN_EVAL" -eq 1 ]]; then
    log_line "Gain summary   : $GAIN_SUMMARY_MD"
  fi
  if [[ "$AUTO_CKPT_SWEEP_GROUP" -eq 1 ]]; then
    log_line "Checkpoint sweep: $CKPT_SWEEP_MD"
  fi
  log_line "Group run complete"
} | tee -a "$SUITE_LOG_FILE"

if [[ "$ENABLE_PERSISTED_LOGS" -eq 0 ]]; then
  rm -f "$SUITE_LOG_FILE" "$SUMMARY_FILE" "$GAIN_SUMMARY_JSON" "$GAIN_SUMMARY_MD" "$CKPT_SWEEP_JSON" "$CKPT_SWEEP_MD"
fi
