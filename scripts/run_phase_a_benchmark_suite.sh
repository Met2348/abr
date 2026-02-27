#!/usr/bin/env bash

# Phase A benchmark runner with one-click "param groups".
#
# Why this exists
# ---------------
# We want reproducible experiments without repeatedly copy-pasting long commands.
# This script lets you choose a param group (A1/A2/...) and run it end-to-end.
#
# One-click usage
# ---------------
# 1) Open this file.
# 2) Change ACTIVE_PARAM_GROUP from A1 to A2 (or others).
# 3) Save.
# 4) Run: bash scripts/run_phase_a_benchmark_suite.sh
#
# You can also override from CLI:
#   ACTIVE_PARAM_GROUP=A2 bash scripts/run_phase_a_benchmark_suite.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# -----------------------------------------------------------------------------
# One-click selector.
# -----------------------------------------------------------------------------
ACTIVE_PARAM_GROUP="${ACTIVE_PARAM_GROUP:-A1}"

# Track which runtime knobs were explicitly provided by user env.
# We use this so param groups can set sensible defaults without overriding user intent.
USER_SET_BATCH_SIZE="${BATCH_SIZE+x}"
USER_SET_MAX_PROGRESS_LINES="${MAX_PROGRESS_LINES+x}"
USER_SET_OOM_BACKOFF="${OOM_BACKOFF+x}"
USER_SET_TRUNCATION_RECOVERY="${TRUNCATION_RECOVERY+x}"
USER_SET_TRUNCATION_RECOVERY_ROUNDS="${TRUNCATION_RECOVERY_ROUNDS+x}"
USER_SET_TRUNCATION_RECOVERY_EXTRA_TOKENS="${TRUNCATION_RECOVERY_EXTRA_TOKENS+x}"
USER_SET_TRUNCATION_RECOVERY_DATASETS="${TRUNCATION_RECOVERY_DATASETS+x}"
USER_SET_TRUNCATION_RECOVERY_REQUIRE_FINAL_SIGNAL="${TRUNCATION_RECOVERY_REQUIRE_FINAL_SIGNAL+x}"

# -----------------------------------------------------------------------------
# Global runtime knobs (override with env vars if needed).
# -----------------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-strategyqa}"
SOURCE_SPLIT="${SOURCE_SPLIT:-train}"
SPLIT_POLICY="${SPLIT_POLICY:-hash}"
LIMIT="${LIMIT:-2000}"
SEED="${SEED:-42}"
TEMPLATE_VERSION="${TEMPLATE_VERSION:-1.0.0}"

MODEL_PATH="${MODEL_PATH:-assets/models/Qwen2.5-7B-Instruct}"
PREP_ROOT="${PREP_ROOT:-assets/artifacts/phase_a_prepared}"
RUN_ROOT="${RUN_ROOT:-assets/artifacts/phase_a_runs}"

DTYPE="${DTYPE:-bfloat16}"
LOG_EVERY="${LOG_EVERY:-5}"
MAX_PROGRESS_LINES="${MAX_PROGRESS_LINES:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OOM_BACKOFF="${OOM_BACKOFF:-1}"
STRATEGYQA_DECODE_MODE="${STRATEGYQA_DECODE_MODE:-freeform}"
TRUNCATE_CHAT_MARKERS="${TRUNCATE_CHAT_MARKERS:-1}"
TRUNCATION_RECOVERY="${TRUNCATION_RECOVERY:-1}"
TRUNCATION_RECOVERY_ROUNDS="${TRUNCATION_RECOVERY_ROUNDS:-2}"
TRUNCATION_RECOVERY_EXTRA_TOKENS="${TRUNCATION_RECOVERY_EXTRA_TOKENS:-96}"
TRUNCATION_RECOVERY_DATASETS="${TRUNCATION_RECOVERY_DATASETS:-gsm8k,hendrycks_math}"
TRUNCATION_RECOVERY_REQUIRE_FINAL_SIGNAL="${TRUNCATION_RECOVERY_REQUIRE_FINAL_SIGNAL:-1}"

set_default_if_not_user_set() {
  local user_flag="$1"
  local var_name="$2"
  local default_value="$3"
  if [[ "${user_flag}" != "x" ]]; then
    printf -v "${var_name}" '%s' "${default_value}"
  fi
}

configure_a11_token_stress_variant() {
  local max_new_tokens="$1"
  local default_batch_size="$2"

  GROUP_TITLE="StrategyQA Whole-Corpus Token Stress t${max_new_tokens}"
  GROUP_INTENTION="Stress-test suspected token-limit effects on whole-corpus StrategyQA with larger CoT budgets."
  GROUP_OBSERVE="Track split-wise and aggregate accuracy while monitoring truncation-recovery activity and throughput."
  GROUP_EXPECTATION="Larger token budgets should reduce cap-related failures but increase runtime; gains should eventually plateau."
  DATASET="strategyqa"
  LIMIT="None"
  STRATEGYQA_DECODE_MODE="freeform"

  # Long-run defaults: verbose enough for monitoring, conservative enough for stability.
  # User env values still take precedence over these defaults.
  set_default_if_not_user_set "${USER_SET_MAX_PROGRESS_LINES}" MAX_PROGRESS_LINES "50"
  set_default_if_not_user_set "${USER_SET_BATCH_SIZE}" BATCH_SIZE "${default_batch_size}"
  set_default_if_not_user_set "${USER_SET_OOM_BACKOFF}" OOM_BACKOFF "1"
  set_default_if_not_user_set "${USER_SET_TRUNCATION_RECOVERY}" TRUNCATION_RECOVERY "1"
  set_default_if_not_user_set "${USER_SET_TRUNCATION_RECOVERY_ROUNDS}" TRUNCATION_RECOVERY_ROUNDS "2"
  set_default_if_not_user_set "${USER_SET_TRUNCATION_RECOVERY_EXTRA_TOKENS}" TRUNCATION_RECOVERY_EXTRA_TOKENS "96"
  set_default_if_not_user_set "${USER_SET_TRUNCATION_RECOVERY_DATASETS}" TRUNCATION_RECOVERY_DATASETS "gsm8k,hendrycks_math,strategyqa"
  set_default_if_not_user_set "${USER_SET_TRUNCATION_RECOVERY_REQUIRE_FINAL_SIGNAL}" TRUNCATION_RECOVERY_REQUIRE_FINAL_SIGNAL "1"

  COT_TEMPLATE_ID="qa_strategyqa_cot_compact"
  COT_TARGET_STYLE="cot_then_answer"
  GROUP_NEED_COT=1
  GROUP_ENABLE_WHOLE_AGG=1

  # Keep these stress runs compute-aware: no additional repro run by default.
  # Aggregate summary remains report-ready across train/validation/test.
  GROUP_RUN_SPECS=(
    "full_train_t${max_new_tokens}|cot_train|${RUN_PREFIX}_full_train_t${max_new_tokens}|${max_new_tokens}|no"
    "full_validation_t${max_new_tokens}|cot_validation|${RUN_PREFIX}_full_validation_t${max_new_tokens}|${max_new_tokens}|no"
    "full_test_t${max_new_tokens}|cot_test|${RUN_PREFIX}_full_test_t${max_new_tokens}|${max_new_tokens}|no"
  )
}

# Template slots used by groups.
# Groups can override these values to compare prompt styles while reusing
# the same suite execution logic.
DIRECT_TEMPLATE_ID="${DIRECT_TEMPLATE_ID:-qa_direct}"
DIRECT_TARGET_STYLE="${DIRECT_TARGET_STYLE:-answer_only}"
COT_TEMPLATE_ID="${COT_TEMPLATE_ID:-qa_cot_then_final}"
COT_TARGET_STYLE="${COT_TARGET_STYLE:-cot_then_answer}"
STRICT_TEMPLATE_ID="${STRICT_TEMPLATE_ID:-qa_binary_strict}"
STRICT_TARGET_STYLE="${STRICT_TARGET_STYLE:-answer_only}"

# Keep a copy of baseline slot values so each group starts from a clean reset.
BASE_DIRECT_TEMPLATE_ID="${DIRECT_TEMPLATE_ID}"
BASE_DIRECT_TARGET_STYLE="${DIRECT_TARGET_STYLE}"
BASE_COT_TEMPLATE_ID="${COT_TEMPLATE_ID}"
BASE_COT_TARGET_STYLE="${COT_TARGET_STYLE}"
BASE_STRICT_TEMPLATE_ID="${STRICT_TEMPLATE_ID}"
BASE_STRICT_TARGET_STYLE="${STRICT_TARGET_STYLE}"

# Sweep defaults used by some groups.
COT_SWEEP_TOKENS="${COT_SWEEP_TOKENS:-128 192 256 320 384}"
DIRECT_SWEEP_TOKENS="${DIRECT_SWEEP_TOKENS:-16 24 32 48 64}"

# For 7B, single GPU is usually simpler and often faster than multi-GPU sharding.
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

# Include group name in run prefix so outputs are easy to track.
RUN_PREFIX="${RUN_PREFIX:-${ACTIVE_PARAM_GROUP}_$(date -u +%Y%m%dT%H%M%SZ)}"

# Persist full suite stdout/stderr in one place.
LOG_ROOT="${LOG_ROOT:-assets/artifacts/phase_a_logs}"
SUITE_LOG_DIR="${SUITE_LOG_DIR:-${LOG_ROOT}/${RUN_PREFIX}}"
ENABLE_PERSISTED_LOGS="${ENABLE_PERSISTED_LOGS:-1}"
mkdir -p "${SUITE_LOG_DIR}"
SUITE_SUMMARY_FILE="${SUITE_SUMMARY_FILE:-${SUITE_LOG_DIR}/final_summary.md}"
if [[ "${ENABLE_PERSISTED_LOGS}" == "1" ]]; then
  SUITE_LOG_FILE="${SUITE_LOG_FILE:-${SUITE_LOG_DIR}/suite.log}"
  # Mirror everything to terminal + file for live monitoring and postmortem.
  exec > >(tee -a "${SUITE_LOG_FILE}") 2>&1
fi

# -----------------------------------------------------------------------------
# Internal state populated by group config.
# -----------------------------------------------------------------------------
GROUP_TITLE=""
GROUP_INTENTION=""
GROUP_OBSERVE=""
GROUP_EXPECTATION=""
GROUP_NEED_DIRECT=0
GROUP_NEED_COT=0
GROUP_NEED_STRICT=0
GROUP_ENABLE_WHOLE_AGG=0
declare -a GROUP_RUN_SPECS=()
declare -a RESULT_LINES=()
DIRECT_TRAIN_JSONL=""
DIRECT_VAL_JSONL=""
DIRECT_TEST_JSONL=""
COT_TRAIN_JSONL=""
COT_VAL_JSONL=""
COT_TEST_JSONL=""
STRICT_TRAIN_JSONL=""
STRICT_VAL_JSONL=""
STRICT_TEST_JSONL=""

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

run_cmd() {
  log "RUN: $*"
  "$@"
}

assert_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "Required file not found: ${path}"
}

assert_dir() {
  local path="$1"
  [[ -d "${path}" ]] || die "Required directory not found: ${path}"
}

warn_if_concurrent_generation() {
  if pgrep -fa "scripts/phase_a_generate_and_eval.py" >/dev/null; then
    log "WARNING: Detected other running phase_a_generate_and_eval.py processes."
    log "WARNING: This suite will continue, but concurrent runs can distort speed/comparison fairness."
  fi
}

resolve_prepared_dir() {
  local target_style="$1"
  local template_id="$2"

  "${PYTHON_BIN}" - "${PREP_ROOT}" "${DATASET}" "${SOURCE_SPLIT}" "${SPLIT_POLICY}" \
    "${LIMIT}" "${target_style}" "${template_id}" "${TEMPLATE_VERSION}" "${SEED}" <<'PY'
import json
import sys
from pathlib import Path

prep_root = Path(sys.argv[1])
dataset = sys.argv[2].lower()
source_split = sys.argv[3]
split_policy = sys.argv[4]
limit_raw = sys.argv[5]
target_style = sys.argv[6]
template_id = sys.argv[7]
template_version = sys.argv[8]
seed = int(sys.argv[9])

limit = None if limit_raw in {"None", "none", "ALL", "all", ""} else int(limit_raw)
dataset_dir = prep_root / dataset
if not dataset_dir.exists():
    raise SystemExit(f"Missing prepared root: {dataset_dir}")

matches: list[tuple[str, Path]] = []
for manifest in dataset_dir.glob("*/manifest.json"):
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    spec = payload.get("run_spec", {})
    split_cfg = spec.get("split_config", {})

    if (
        spec.get("dataset", "").lower() == dataset
        and spec.get("source_split") == source_split
        and spec.get("split_policy") == split_policy
        and spec.get("limit") == limit
        and spec.get("target_style") == target_style
        and spec.get("template_id") == template_id
        and spec.get("template_version") == template_version
        and int(split_cfg.get("seed", -1)) == seed
    ):
        created = str(payload.get("created_at_utc", ""))
        matches.append((created, manifest.parent))

if not matches:
    raise SystemExit(
        "Could not find prepared directory for "
        f"dataset={dataset}, target_style={target_style}, template_id={template_id}"
    )

matches.sort(key=lambda x: x[0])
print(matches[-1][1])
PY
}

latest_run_dir() {
  local run_name="$1"
  "${PYTHON_BIN}" - "${RUN_ROOT}" "${run_name}" <<'PY'
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
run_name = sys.argv[2]
matches = sorted(run_root.glob(f"{run_name}_*"))
if not matches:
    raise SystemExit(f"No run directories found for run_name={run_name}")
print(matches[-1])
PY
}

prepare_variant() {
  local template_id="$1"
  local target_style="$2"
  local cmd=("${PYTHON_BIN}" scripts/phase_a_prepare.py \
    --datasets "${DATASET}" \
    --source-split "${SOURCE_SPLIT}" \
    --split-policy "${SPLIT_POLICY}" \
    --template-id "${template_id}" \
    --template-version "${TEMPLATE_VERSION}" \
    --target-style "${target_style}" \
    --seed "${SEED}" \
    --resume)

  # Allow full-dataset runs by omitting --limit entirely.
  # Supported env values: LIMIT=None, LIMIT=all, or LIMIT="".
  if [[ "${LIMIT}" != "None" && "${LIMIT}" != "none" && "${LIMIT}" != "ALL" && "${LIMIT}" != "all" && -n "${LIMIT}" ]]; then
    cmd+=(--limit "${LIMIT}")
  fi

  run_cmd "${cmd[@]}"
}

run_infer_eval() {
  local input_jsonl="$1"
  local run_name="$2"
  local max_new_tokens="$3"
  local compare_mode="$4"  # yes|no

  local compare_flag=()
  if [[ "${compare_mode}" == "no" ]]; then
    compare_flag=(--no-compare-latest-same-name)
  fi

  local truncate_flag=(--truncate-chat-markers)
  if [[ "${TRUNCATE_CHAT_MARKERS}" == "0" ]]; then
    truncate_flag=(--no-truncate-chat-markers)
  fi

  local oom_backoff_flag=(--oom-backoff)
  if [[ "${OOM_BACKOFF}" == "0" ]]; then
    oom_backoff_flag=(--no-oom-backoff)
  fi

  local trunc_recovery_flag=(--truncation-recovery)
  if [[ "${TRUNCATION_RECOVERY}" == "0" ]]; then
    trunc_recovery_flag=(--no-truncation-recovery)
  fi

  local trunc_recovery_require_signal_flag=(--truncation-recovery-require-final-answer-signal)
  if [[ "${TRUNCATION_RECOVERY_REQUIRE_FINAL_SIGNAL}" == "0" ]]; then
    trunc_recovery_require_signal_flag=(--no-truncation-recovery-require-final-answer-signal)
  fi

  run_cmd "${PYTHON_BIN}" -u scripts/phase_a_generate_and_eval.py \
    --input-jsonl "${input_jsonl}" \
    --model-path "${MODEL_PATH}" \
    --run-name "${run_name}" \
    --require-cuda \
    --dtype "${DTYPE}" \
    --device-map auto \
    --no-do-sample \
    --seed "${SEED}" \
    --max-new-tokens "${max_new_tokens}" \
    --strategyqa-decode-mode "${STRATEGYQA_DECODE_MODE}" \
    "${truncate_flag[@]}" \
    --batch-size "${BATCH_SIZE}" \
    "${oom_backoff_flag[@]}" \
    "${trunc_recovery_flag[@]}" \
    --truncation-recovery-rounds "${TRUNCATION_RECOVERY_ROUNDS}" \
    --truncation-recovery-extra-tokens "${TRUNCATION_RECOVERY_EXTRA_TOKENS}" \
    --truncation-recovery-datasets "${TRUNCATION_RECOVERY_DATASETS}" \
    "${trunc_recovery_require_signal_flag[@]}" \
    --log-every "${LOG_EVERY}" \
    --max-progress-lines "${MAX_PROGRESS_LINES}" \
    "${compare_flag[@]}"
}

configure_param_group() {
  local group_id="$1"

  GROUP_RUN_SPECS=()
  GROUP_NEED_DIRECT=0
  GROUP_NEED_COT=0
  GROUP_NEED_STRICT=0
  GROUP_ENABLE_WHOLE_AGG=0
  DIRECT_TRAIN_JSONL=""
  DIRECT_VAL_JSONL=""
  DIRECT_TEST_JSONL=""
  COT_TRAIN_JSONL=""
  COT_VAL_JSONL=""
  COT_TEST_JSONL=""
  STRICT_TRAIN_JSONL=""
  STRICT_VAL_JSONL=""
  STRICT_TEST_JSONL=""
  DIRECT_TEMPLATE_ID="${BASE_DIRECT_TEMPLATE_ID}"
  DIRECT_TARGET_STYLE="${BASE_DIRECT_TARGET_STYLE}"
  COT_TEMPLATE_ID="${BASE_COT_TEMPLATE_ID}"
  COT_TARGET_STYLE="${BASE_COT_TARGET_STYLE}"
  STRICT_TEMPLATE_ID="${BASE_STRICT_TEMPLATE_ID}"
  STRICT_TARGET_STYLE="${BASE_STRICT_TARGET_STYLE}"

  case "${group_id}" in
    A1)
      # Intention:
      # Reproduce the core story we already observed:
      # direct baseline vs CoT at two token budgets.
      #
      # Observe:
      # - parse_error_rate gap between direct and CoT.
      # - CoT improvement from t128 to t256.
      #
      # Expectation:
      # - direct_t32 should be strong and stable.
      # - cot_t128 should have high parse errors.
      # - cot_t256 should improve but may still trail direct on total accuracy.
      GROUP_TITLE="Core Reproduction (Direct vs CoT)"
      GROUP_INTENTION="Rebuild baseline evidence with deterministic settings."
      GROUP_OBSERVE="Focus on parse_error_rate and total accuracy across direct_t32, cot_t128, cot_t256."
      GROUP_EXPECTATION="direct_t32 robust; cot_t128 weak compliance; cot_t256 improved but still compliance-limited."
      GROUP_NEED_DIRECT=1
      GROUP_NEED_COT=1
      GROUP_RUN_SPECS=(
        "direct_t32_r1|direct|${RUN_PREFIX}_direct_t32|32|no"
        "direct_t32_r2|direct|${RUN_PREFIX}_direct_t32|32|yes"
        "cot_t128|cot|${RUN_PREFIX}_cot_t128|128|no"
        "cot_t256|cot|${RUN_PREFIX}_cot_t256|256|no"
      )
      ;;
    A2)
      # Intention:
      # Diagnose truncation/completion limits for CoT prompts.
      #
      # Observe:
      # - Does parse_error_rate decrease as max_new_tokens increases?
      # - Where does improvement plateau?
      #
      # Expectation:
      # - t128 usually worst.
      # - larger budgets should reduce parse errors, but with diminishing returns.
      GROUP_TITLE="CoT Token Sweep"
      GROUP_INTENTION="Measure how CoT token budget affects compliance and accuracy."
      GROUP_OBSERVE="Look for monotonic or near-monotonic parse_error reductions with larger token budgets."
      GROUP_EXPECTATION="Accuracy should rise with token budget until a plateau; runtime rises sharply."
      GROUP_NEED_COT=1
      for tok in ${COT_SWEEP_TOKENS}; do
        GROUP_RUN_SPECS+=("cot_t${tok}|cot|${RUN_PREFIX}_cot_t${tok}|${tok}|no")
      done
      ;;
    A3)
      # Intention:
      # Build a fast, practical baseline frontier for direct-answer prompting.
      #
      # Observe:
      # - Which token budget gives best speed/accuracy tradeoff?
      # - Does parse error increase sharply at very small budgets?
      #
      # Expectation:
      # - tiny budgets can under-answer (higher parse errors),
      # - mid-range budgets often stabilize metrics.
      GROUP_TITLE="Direct Token Sweep"
      GROUP_INTENTION="Find practical direct baseline budget with good speed and stable accuracy."
      GROUP_OBSERVE="Compare accuracy and parse_error_rate across direct token budgets."
      GROUP_EXPECTATION="A mid-range budget (often around 24-48) should be close to best tradeoff."
      GROUP_NEED_DIRECT=1
      for tok in ${DIRECT_SWEEP_TOKENS}; do
        GROUP_RUN_SPECS+=("direct_t${tok}|direct|${RUN_PREFIX}_direct_t${tok}|${tok}|no")
      done
      ;;
    A4)
      # Intention:
      # Verify strict reproducibility by repeating the same configs.
      #
      # Observe:
      # - delta_accuracy and delta_parse_err should be 0 on second run.
      # - changed_samples should be 0.
      #
      # Expectation:
      # - deterministic settings should produce identical outputs.
      GROUP_TITLE="Determinism Check"
      GROUP_INTENTION="Validate exact rerun reproducibility for one direct and one CoT config."
      GROUP_OBSERVE="Second run should report zero metric deltas and zero changed samples."
      GROUP_EXPECTATION="No differences between run1 and run2 for both configs."
      GROUP_NEED_DIRECT=1
      GROUP_NEED_COT=1
      GROUP_RUN_SPECS=(
        "direct_t32_repro_r1|direct|${RUN_PREFIX}_direct_t32_repro|32|no"
        "direct_t32_repro_r2|direct|${RUN_PREFIX}_direct_t32_repro|32|yes"
        "cot_t256_repro_r1|cot|${RUN_PREFIX}_cot_t256_repro|256|no"
        "cot_t256_repro_r2|cot|${RUN_PREFIX}_cot_t256_repro|256|yes"
      )
      ;;
    A5)
      # Intention:
      # Directly target three observed problems in direct runs:
      # 1) parse/compliance misses,
      # 2) wasted tokens due to long answers hitting caps,
      # 3) potential default-label bias.
      #
      # Observe:
      # - parse_error_rate improvement vs direct baseline,
      # - whether tiny token budgets (4/8/16) still keep strong accuracy,
      # - determinism on strict setting (second run should match first).
      #
      # Expectation:
      # - strict template should reduce parse errors and output length,
      # - small token budgets should be sufficient for yes/no output,
      # - accuracy should stay comparable or improve vs direct_t16 baseline.
      GROUP_TITLE="Strict Yes/No Compliance Fix"
      GROUP_INTENTION="Use strict binary prompt to improve format compliance and efficiency."
      GROUP_OBSERVE="Watch parse_error_rate first; then total accuracy and reproducibility deltas."
      GROUP_EXPECTATION="Strict runs should be shorter and cleaner; determinism should hold on repeat."
      GROUP_NEED_DIRECT=1
      GROUP_NEED_STRICT=1
      GROUP_RUN_SPECS=(
        "baseline_direct_t16|direct|${RUN_PREFIX}_baseline_direct_t16|16|no"
        "strict_t4|strict|${RUN_PREFIX}_strict_t4|4|no"
        "strict_t8|strict|${RUN_PREFIX}_strict_t8|8|no"
        "strict_t16_r1|strict|${RUN_PREFIX}_strict_t16|16|no"
        "strict_t16_r2|strict|${RUN_PREFIX}_strict_t16|16|yes"
      )
      ;;
    A6)
      # Intention:
      # Remove format-related parse failures by forcing binary decision decoding
      # for StrategyQA (score `yes` vs `no` directly).
      #
      # Observe:
      # - parse_error_rate should approach 0 for StrategyQA.
      # - compare direct and CoT prompt styles under identical binary-choice decode.
      #
      # Expectation:
      # - parse errors largely disappear,
      # - accuracy differences become more attributable to model decision quality
      #   rather than output-format noise.
      GROUP_TITLE="Binary-Choice Decode Validation"
      GROUP_INTENTION="Diagnose model quality after removing free-form format failures."
      GROUP_OBSERVE="Check parse_error first (should be near zero), then compare direct vs CoT accuracy."
      GROUP_EXPECTATION="Coverage improves sharply; remaining errors mostly reflect model reasoning/calibration."
      STRATEGYQA_DECODE_MODE="binary_choice"
      GROUP_NEED_DIRECT=1
      GROUP_NEED_COT=1
      GROUP_RUN_SPECS=(
        "direct_binchoice|direct|${RUN_PREFIX}_direct_binchoice|16|no"
        "cot_binchoice|cot|${RUN_PREFIX}_cot_binchoice|16|no"
        "direct_binchoice_repro|direct|${RUN_PREFIX}_direct_binchoice|16|yes"
      )
      ;;
    A7)
      # Intention:
      # Prompt-style A/B/C test for StrategyQA.
      #
      # Observe:
      # - style-level differences in parse_error and accuracy,
      # - whether compact CoT helps or hurts under fixed decoding.
      #
      # Expectation:
      # - minimal binary style should be strongest on compliance,
      # - evidence/verdict style may improve interpretability,
      # - compact CoT may trade speed for quality.
      GROUP_TITLE="StrategyQA Prompt Style Sweep"
      GROUP_INTENTION="Compare three StrategyQA prompt styles under one reproducible setup."
      GROUP_OBSERVE="Check style ranking on accuracy, parse_error_rate, and generation speed."
      GROUP_EXPECTATION="Minimal-binary style should be cleanest; CoT style may help only if token budget is sufficient."
      DATASET="strategyqa"
      STRATEGYQA_DECODE_MODE="freeform"
      DIRECT_TEMPLATE_ID="qa_strategyqa_minimal_binary"
      DIRECT_TARGET_STYLE="answer_only"
      COT_TEMPLATE_ID="qa_strategyqa_cot_compact"
      COT_TARGET_STYLE="cot_then_answer"
      STRICT_TEMPLATE_ID="qa_strategyqa_evidence_verdict"
      STRICT_TARGET_STYLE="answer_only"
      GROUP_NEED_DIRECT=1
      GROUP_NEED_COT=1
      GROUP_NEED_STRICT=1
      GROUP_RUN_SPECS=(
        "style_minimal_t16|direct|${RUN_PREFIX}_style_minimal_t16|16|no"
        "style_cot_compact_t96|cot|${RUN_PREFIX}_style_cot_compact_t96|96|no"
        "style_evidence_verdict_t32|strict|${RUN_PREFIX}_style_evidence_verdict_t32|32|no"
      )
      ;;
    A8)
      # Intention:
      # Prompt-style A/B/C test for GSM8K.
      #
      # Observe:
      # - direct final-only vs compact CoT vs equation-first formats,
      # - extraction reliability and accuracy under each style.
      #
      # Expectation:
      # - direct-final should be fastest,
      # - compact CoT can improve quality if decode budget is large enough,
      # - equation-first can be a balanced middle point.
      GROUP_TITLE="GSM8K Prompt Style Sweep"
      GROUP_INTENTION="Compare three GSM8K prompt styles with deterministic decode settings."
      GROUP_OBSERVE="Check accuracy first, then runtime and extraction diagnostics."
      GROUP_EXPECTATION="CoT style may win on quality; direct style should win on speed."
      DATASET="gsm8k"
      STRATEGYQA_DECODE_MODE="freeform"
      DIRECT_TEMPLATE_ID="qa_gsm8k_direct_final_only"
      DIRECT_TARGET_STYLE="answer_only"
      COT_TEMPLATE_ID="qa_gsm8k_cot_compact_final"
      COT_TARGET_STYLE="cot_then_answer"
      STRICT_TEMPLATE_ID="qa_gsm8k_equation_then_final"
      STRICT_TARGET_STYLE="answer_only"
      GROUP_NEED_DIRECT=1
      GROUP_NEED_COT=1
      GROUP_NEED_STRICT=1
      GROUP_RUN_SPECS=(
        "style_direct_final_t32|direct|${RUN_PREFIX}_style_direct_final_t32|32|no"
        "style_cot_compact_t192|cot|${RUN_PREFIX}_style_cot_compact_t192|192|no"
        "style_equation_t64|strict|${RUN_PREFIX}_style_equation_t64|64|no"
      )
      ;;
    A9)
      # Intention:
      # Run the current best StrategyQA setting on the full dataset
      # (no preparation-time limit cap), with a reproducibility check.
      #
      # Observe:
      # - full-data accuracy and parse_error_rate,
      # - run2 determinism delta vs run1.
      #
      # Expectation:
      # - accuracy close to small-scale best setting,
      # - zero (or near-zero) reproducibility deltas.
      GROUP_TITLE="StrategyQA Full-Data Best Setting"
      GROUP_INTENTION="Validate full-data performance of the best StrategyQA prompt configuration."
      GROUP_OBSERVE="Track full-data accuracy, parse_error_rate, and deterministic rerun deltas."
      GROUP_EXPECTATION="CoT compact t96 should remain top quality with low parse error on full-data validation."
      DATASET="strategyqa"
      LIMIT="None"
      STRATEGYQA_DECODE_MODE="freeform"
      COT_TEMPLATE_ID="qa_strategyqa_cot_compact"
      COT_TARGET_STYLE="cot_then_answer"
      GROUP_NEED_COT=1
      GROUP_RUN_SPECS=(
        "full_best_cot_t96_r1|cot|${RUN_PREFIX}_full_best_cot_t96|96|no"
        "full_best_cot_t96_r2|cot|${RUN_PREFIX}_full_best_cot_t96|96|yes"
      )
      ;;
    A10)
      # Intention:
      # Run the current best GSM8K setting on the full dataset
      # (no preparation-time limit cap), with a reproducibility check.
      #
      # Observe:
      # - full-data accuracy,
      # - truncation recovery activity,
      # - run2 determinism delta vs run1.
      #
      # Expectation:
      # - CoT compact t192 remains strongest,
      # - truncation recovery should trigger on a small subset.
      GROUP_TITLE="GSM8K Full-Data Best Setting"
      GROUP_INTENTION="Validate full-data performance of the best GSM8K prompt configuration."
      GROUP_OBSERVE="Track full-data accuracy, truncation-recovery rows, and deterministic rerun deltas."
      GROUP_EXPECTATION="CoT compact t192 should remain near best-known GSM8K accuracy on full-data validation."
      DATASET="gsm8k"
      LIMIT="None"
      STRATEGYQA_DECODE_MODE="freeform"
      COT_TEMPLATE_ID="qa_gsm8k_cot_compact_final"
      COT_TARGET_STYLE="cot_then_answer"
      GROUP_NEED_COT=1
      GROUP_RUN_SPECS=(
        "full_best_cot_t192_r1|cot|${RUN_PREFIX}_full_best_cot_t192|192|no"
        "full_best_cot_t192_r2|cot|${RUN_PREFIX}_full_best_cot_t192|192|yes"
      )
      ;;
    A11)
      # Intention:
      # Evaluate one best StrategyQA setting over the full prepared corpus
      # (train+validation+test = all samples) for report-ready "whole picture".
      #
      # Observe:
      # - per-split accuracy/parse_error_rate,
      # - weighted aggregate metrics over all rows,
      # - deterministic rerun on the aggregate.
      #
      # Expectation:
      # - aggregate n should equal full prepared corpus count,
      # - parse errors should mainly come from cap-hit freeform outputs,
      # - rerun deltas should remain zero.
      GROUP_TITLE="StrategyQA Whole-Corpus Review (2290)"
      GROUP_INTENTION="Produce report-ready whole-corpus metrics using the current best StrategyQA prompt setting."
      GROUP_OBSERVE="Check split-wise metrics and the weighted aggregate over train+validation+test."
      GROUP_EXPECTATION="Aggregate should be stable and reproducible; truncation-safe settings should keep parse errors low."
      DATASET="strategyqa"
      LIMIT="None"
      STRATEGYQA_DECODE_MODE="freeform"
      set_default_if_not_user_set "${USER_SET_MAX_PROGRESS_LINES}" MAX_PROGRESS_LINES "50"
      set_default_if_not_user_set "${USER_SET_BATCH_SIZE}" BATCH_SIZE "16"
      set_default_if_not_user_set "${USER_SET_OOM_BACKOFF}" OOM_BACKOFF "1"
      set_default_if_not_user_set "${USER_SET_TRUNCATION_RECOVERY}" TRUNCATION_RECOVERY "1"
      set_default_if_not_user_set "${USER_SET_TRUNCATION_RECOVERY_ROUNDS}" TRUNCATION_RECOVERY_ROUNDS "2"
      set_default_if_not_user_set "${USER_SET_TRUNCATION_RECOVERY_EXTRA_TOKENS}" TRUNCATION_RECOVERY_EXTRA_TOKENS "96"
      set_default_if_not_user_set "${USER_SET_TRUNCATION_RECOVERY_DATASETS}" TRUNCATION_RECOVERY_DATASETS "gsm8k,hendrycks_math,strategyqa"
      set_default_if_not_user_set "${USER_SET_TRUNCATION_RECOVERY_REQUIRE_FINAL_SIGNAL}" TRUNCATION_RECOVERY_REQUIRE_FINAL_SIGNAL "1"
      COT_TEMPLATE_ID="qa_strategyqa_cot_compact"
      COT_TARGET_STYLE="cot_then_answer"
      GROUP_NEED_COT=1
      GROUP_ENABLE_WHOLE_AGG=1
      GROUP_RUN_SPECS=(
        "full_train_t96|cot_train|${RUN_PREFIX}_full_train_t96|96|no"
        "full_validation_t96|cot_validation|${RUN_PREFIX}_full_validation_t96|96|no"
        "full_test_t96|cot_test|${RUN_PREFIX}_full_test_t96|96|no"
        "full_train_t96_repro|cot_train|${RUN_PREFIX}_full_train_t96|96|yes"
      )
      ;;
    A11_128)
      # StrategyQA whole-corpus token-stress subgroup (t128).
      # Slightly reduced default batch vs A11 to keep long-decoding runs stable.
      configure_a11_token_stress_variant "128" "64"
      ;;
    A11_256)
      # StrategyQA whole-corpus token-stress subgroup (t256).
      # Batch default is lowered as token budget grows to reduce OOM risk.
      configure_a11_token_stress_variant "256" "32"
      ;;
    A11_384)
      # StrategyQA whole-corpus token-stress subgroup (t384).
      configure_a11_token_stress_variant "384" "24"
      ;;
    A11_512)
      # StrategyQA whole-corpus token-stress subgroup (t512).
      configure_a11_token_stress_variant "512" "16"
      ;;
    A11_1024)
      # StrategyQA whole-corpus token-stress subgroup (t1024).
      # Very long decode budget; conservative default batch for stability.
      configure_a11_token_stress_variant "1024" "8"
      ;;
    A12)
      # Intention:
      # Evaluate one best GSM8K setting over the full prepared corpus
      # (train+validation+test from source train split) for report-ready totals.
      #
      # Observe:
      # - per-split accuracy/parse_error_rate,
      # - weighted aggregate metrics over all rows,
      # - deterministic rerun check on validation split.
      #
      # Expectation:
      # - aggregate reflects the full prepared corpus,
      # - truncation recovery should trigger on a subset of long CoT outputs,
      # - validation rerun deltas should remain zero.
      GROUP_TITLE="GSM8K Whole-Corpus Review"
      GROUP_INTENTION="Produce report-ready whole-corpus metrics using the current best GSM8K prompt setting."
      GROUP_OBSERVE="Check split-wise metrics, aggregate quality, truncation-recovery activity, and reproducibility."
      GROUP_EXPECTATION="CoT compact t192 should stay strongest on quality with bounded truncation failures under safeguards."
      DATASET="gsm8k"
      LIMIT="None"
      STRATEGYQA_DECODE_MODE="freeform"
      set_default_if_not_user_set "${USER_SET_MAX_PROGRESS_LINES}" MAX_PROGRESS_LINES "50"
      set_default_if_not_user_set "${USER_SET_BATCH_SIZE}" BATCH_SIZE "16"
      set_default_if_not_user_set "${USER_SET_OOM_BACKOFF}" OOM_BACKOFF "1"
      set_default_if_not_user_set "${USER_SET_TRUNCATION_RECOVERY}" TRUNCATION_RECOVERY "1"
      set_default_if_not_user_set "${USER_SET_TRUNCATION_RECOVERY_ROUNDS}" TRUNCATION_RECOVERY_ROUNDS "2"
      set_default_if_not_user_set "${USER_SET_TRUNCATION_RECOVERY_EXTRA_TOKENS}" TRUNCATION_RECOVERY_EXTRA_TOKENS "96"
      set_default_if_not_user_set "${USER_SET_TRUNCATION_RECOVERY_DATASETS}" TRUNCATION_RECOVERY_DATASETS "gsm8k,hendrycks_math"
      set_default_if_not_user_set "${USER_SET_TRUNCATION_RECOVERY_REQUIRE_FINAL_SIGNAL}" TRUNCATION_RECOVERY_REQUIRE_FINAL_SIGNAL "1"
      COT_TEMPLATE_ID="qa_gsm8k_cot_compact_final"
      COT_TARGET_STYLE="cot_then_answer"
      GROUP_NEED_COT=1
      GROUP_ENABLE_WHOLE_AGG=1
      GROUP_RUN_SPECS=(
        "full_train_t192|cot_train|${RUN_PREFIX}_full_train_t192|192|no"
        "full_validation_t192|cot_validation|${RUN_PREFIX}_full_validation_t192|192|no"
        "full_test_t192|cot_test|${RUN_PREFIX}_full_test_t192|192|no"
        "full_validation_t192_repro|cot_validation|${RUN_PREFIX}_full_validation_t192|192|yes"
      )
      ;;
    *)
      die "Unsupported ACTIVE_PARAM_GROUP=${group_id}. Supported: A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A11_128, A11_256, A11_384, A11_512, A11_1024, A12"
      ;;
  esac
}

prepare_needed_inputs() {
  if [[ "${GROUP_NEED_DIRECT}" == "1" ]]; then
    prepare_variant "${DIRECT_TEMPLATE_ID}" "${DIRECT_TARGET_STYLE}"
    local direct_prep_dir
    direct_prep_dir="$(resolve_prepared_dir "${DIRECT_TARGET_STYLE}" "${DIRECT_TEMPLATE_ID}")"
    DIRECT_TRAIN_JSONL="${direct_prep_dir}/train.jsonl"
    DIRECT_VAL_JSONL="${direct_prep_dir}/validation.jsonl"
    DIRECT_TEST_JSONL="${direct_prep_dir}/test.jsonl"
    assert_file "${DIRECT_TRAIN_JSONL}"
    assert_file "${DIRECT_VAL_JSONL}"
    assert_file "${DIRECT_TEST_JSONL}"
    log "Prepared direct files        : train=${DIRECT_TRAIN_JSONL}, validation=${DIRECT_VAL_JSONL}, test=${DIRECT_TEST_JSONL} (${DIRECT_TEMPLATE_ID}, ${DIRECT_TARGET_STYLE})"
  fi

  if [[ "${GROUP_NEED_COT}" == "1" ]]; then
    prepare_variant "${COT_TEMPLATE_ID}" "${COT_TARGET_STYLE}"
    local cot_prep_dir
    cot_prep_dir="$(resolve_prepared_dir "${COT_TARGET_STYLE}" "${COT_TEMPLATE_ID}")"
    COT_TRAIN_JSONL="${cot_prep_dir}/train.jsonl"
    COT_VAL_JSONL="${cot_prep_dir}/validation.jsonl"
    COT_TEST_JSONL="${cot_prep_dir}/test.jsonl"
    assert_file "${COT_TRAIN_JSONL}"
    assert_file "${COT_VAL_JSONL}"
    assert_file "${COT_TEST_JSONL}"
    log "Prepared CoT files           : train=${COT_TRAIN_JSONL}, validation=${COT_VAL_JSONL}, test=${COT_TEST_JSONL} (${COT_TEMPLATE_ID}, ${COT_TARGET_STYLE})"
  fi

  if [[ "${GROUP_NEED_STRICT}" == "1" ]]; then
    prepare_variant "${STRICT_TEMPLATE_ID}" "${STRICT_TARGET_STYLE}"
    local strict_prep_dir
    strict_prep_dir="$(resolve_prepared_dir "${STRICT_TARGET_STYLE}" "${STRICT_TEMPLATE_ID}")"
    STRICT_TRAIN_JSONL="${strict_prep_dir}/train.jsonl"
    STRICT_VAL_JSONL="${strict_prep_dir}/validation.jsonl"
    STRICT_TEST_JSONL="${strict_prep_dir}/test.jsonl"
    assert_file "${STRICT_TRAIN_JSONL}"
    assert_file "${STRICT_VAL_JSONL}"
    assert_file "${STRICT_TEST_JSONL}"
    log "Prepared strict files        : train=${STRICT_TRAIN_JSONL}, validation=${STRICT_VAL_JSONL}, test=${STRICT_TEST_JSONL} (${STRICT_TEMPLATE_ID}, ${STRICT_TARGET_STYLE})"
  fi
}

execute_group_runs() {
  local spec label input_kind run_name max_new_tokens compare_mode
  local input_jsonl run_dir

  for spec in "${GROUP_RUN_SPECS[@]}"; do
    IFS='|' read -r label input_kind run_name max_new_tokens compare_mode <<< "${spec}"

    case "${input_kind}" in
      direct)
        [[ -n "${DIRECT_VAL_JSONL}" ]] || die "Direct input is not prepared."
        input_jsonl="${DIRECT_VAL_JSONL}"
        ;;
      direct_train)
        [[ -n "${DIRECT_TRAIN_JSONL}" ]] || die "Direct train input is not prepared."
        input_jsonl="${DIRECT_TRAIN_JSONL}"
        ;;
      direct_validation)
        [[ -n "${DIRECT_VAL_JSONL}" ]] || die "Direct validation input is not prepared."
        input_jsonl="${DIRECT_VAL_JSONL}"
        ;;
      direct_test)
        [[ -n "${DIRECT_TEST_JSONL}" ]] || die "Direct test input is not prepared."
        input_jsonl="${DIRECT_TEST_JSONL}"
        ;;
      cot)
        [[ -n "${COT_VAL_JSONL}" ]] || die "CoT input is not prepared."
        input_jsonl="${COT_VAL_JSONL}"
        ;;
      cot_train)
        [[ -n "${COT_TRAIN_JSONL}" ]] || die "CoT train input is not prepared."
        input_jsonl="${COT_TRAIN_JSONL}"
        ;;
      cot_validation)
        [[ -n "${COT_VAL_JSONL}" ]] || die "CoT validation input is not prepared."
        input_jsonl="${COT_VAL_JSONL}"
        ;;
      cot_test)
        [[ -n "${COT_TEST_JSONL}" ]] || die "CoT test input is not prepared."
        input_jsonl="${COT_TEST_JSONL}"
        ;;
      strict)
        [[ -n "${STRICT_VAL_JSONL}" ]] || die "Strict input is not prepared."
        input_jsonl="${STRICT_VAL_JSONL}"
        ;;
      strict_train)
        [[ -n "${STRICT_TRAIN_JSONL}" ]] || die "Strict train input is not prepared."
        input_jsonl="${STRICT_TRAIN_JSONL}"
        ;;
      strict_validation)
        [[ -n "${STRICT_VAL_JSONL}" ]] || die "Strict validation input is not prepared."
        input_jsonl="${STRICT_VAL_JSONL}"
        ;;
      strict_test)
        [[ -n "${STRICT_TEST_JSONL}" ]] || die "Strict test input is not prepared."
        input_jsonl="${STRICT_TEST_JSONL}"
        ;;
      *)
        die "Unknown input kind in run spec: ${input_kind}"
        ;;
    esac

    run_infer_eval "${input_jsonl}" "${run_name}" "${max_new_tokens}" "${compare_mode}"
    run_dir="$(latest_run_dir "${run_name}")"
    RESULT_LINES+=("${label}|${run_dir}")
  done
}

print_summary_table() {
  local tmp_results_file
  local tmp_specs_file
  tmp_results_file="$(mktemp /tmp/phasea_group_results_XXXXXX.txt)"
  tmp_specs_file="$(mktemp /tmp/phasea_group_specs_XXXXXX.txt)"
  trap 'rm -f "${tmp_results_file:-}" "${tmp_specs_file:-}"' RETURN

  printf '%s\n' "${RESULT_LINES[@]}" > "${tmp_results_file}"
  printf '%s\n' "${GROUP_RUN_SPECS[@]}" > "${tmp_specs_file}"

  PHASEA_GROUP_ID="${ACTIVE_PARAM_GROUP}" \
  PHASEA_GROUP_TITLE="${GROUP_TITLE}" \
  PHASEA_GROUP_INTENTION="${GROUP_INTENTION}" \
  PHASEA_GROUP_OBSERVE="${GROUP_OBSERVE}" \
  PHASEA_GROUP_EXPECTATION="${GROUP_EXPECTATION}" \
  PHASEA_GROUP_ENABLE_WHOLE_AGG="${GROUP_ENABLE_WHOLE_AGG}" \
  PHASEA_RUN_PREFIX="${RUN_PREFIX}" \
  PHASEA_DATASET="${DATASET}" \
  PHASEA_SOURCE_SPLIT="${SOURCE_SPLIT}" \
  PHASEA_SPLIT_POLICY="${SPLIT_POLICY}" \
  PHASEA_LIMIT="${LIMIT}" \
  PHASEA_SEED="${SEED}" \
  PHASEA_DTYPE="${DTYPE}" \
  PHASEA_LOG_EVERY="${LOG_EVERY}" \
  PHASEA_MAX_PROGRESS_LINES="${MAX_PROGRESS_LINES}" \
  PHASEA_BATCH_SIZE="${BATCH_SIZE}" \
  PHASEA_OOM_BACKOFF="${OOM_BACKOFF}" \
  PHASEA_TRUNCATION_RECOVERY="${TRUNCATION_RECOVERY}" \
  PHASEA_TRUNCATION_RECOVERY_ROUNDS="${TRUNCATION_RECOVERY_ROUNDS}" \
  PHASEA_TRUNCATION_RECOVERY_EXTRA_TOKENS="${TRUNCATION_RECOVERY_EXTRA_TOKENS}" \
  PHASEA_TRUNCATION_RECOVERY_DATASETS="${TRUNCATION_RECOVERY_DATASETS}" \
  PHASEA_TRUNCATION_RECOVERY_REQUIRE_FINAL_SIGNAL="${TRUNCATION_RECOVERY_REQUIRE_FINAL_SIGNAL}" \
  PHASEA_STRATEGYQA_DECODE_MODE="${STRATEGYQA_DECODE_MODE}" \
  PHASEA_TRUNCATE_CHAT_MARKERS="${TRUNCATE_CHAT_MARKERS}" \
  PHASEA_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  PHASEA_MODEL_PATH="${MODEL_PATH}" \
  PHASEA_DIRECT_TEMPLATE_ID="${DIRECT_TEMPLATE_ID}" \
  PHASEA_DIRECT_TARGET_STYLE="${DIRECT_TARGET_STYLE}" \
  PHASEA_COT_TEMPLATE_ID="${COT_TEMPLATE_ID}" \
  PHASEA_COT_TARGET_STYLE="${COT_TARGET_STYLE}" \
  PHASEA_STRICT_TEMPLATE_ID="${STRICT_TEMPLATE_ID}" \
  PHASEA_STRICT_TARGET_STYLE="${STRICT_TARGET_STYLE}" \
  PHASEA_DIRECT_TRAIN_JSONL="${DIRECT_TRAIN_JSONL}" \
  PHASEA_DIRECT_VAL_JSONL="${DIRECT_VAL_JSONL}" \
  PHASEA_DIRECT_TEST_JSONL="${DIRECT_TEST_JSONL}" \
  PHASEA_COT_TRAIN_JSONL="${COT_TRAIN_JSONL}" \
  PHASEA_COT_VAL_JSONL="${COT_VAL_JSONL}" \
  PHASEA_COT_TEST_JSONL="${COT_TEST_JSONL}" \
  PHASEA_STRICT_TRAIN_JSONL="${STRICT_TRAIN_JSONL}" \
  PHASEA_STRICT_VAL_JSONL="${STRICT_VAL_JSONL}" \
  PHASEA_STRICT_TEST_JSONL="${STRICT_TEST_JSONL}" \
  PHASEA_SUITE_LOG_FILE="${SUITE_LOG_FILE:-<disabled>}" \
  PHASEA_SUITE_SUMMARY_FILE="${SUITE_SUMMARY_FILE}" \
  "${PYTHON_BIN}" - "${tmp_results_file}" "${tmp_specs_file}" <<'PY'
import json
import os
import sys
from datetime import datetime
from pathlib import Path

def _bootstrap_src_path() -> None:
    repo_root = Path.cwd()
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_a.instability import (
    compute_pairwise_prediction_flip,
    index_rows_by_sample_id,
    summarize_strategyqa_instability,
)

results_path = Path(sys.argv[1])
specs_path = Path(sys.argv[2])

group_id = os.environ["PHASEA_GROUP_ID"]
group_title = os.environ["PHASEA_GROUP_TITLE"]
group_intention = os.environ["PHASEA_GROUP_INTENTION"]
group_observe = os.environ["PHASEA_GROUP_OBSERVE"]
group_expectation = os.environ["PHASEA_GROUP_EXPECTATION"]
summary_path = Path(os.environ["PHASEA_SUITE_SUMMARY_FILE"])

rows = []
instability_by_label = {}
sample_index_by_label = {}
for raw in results_path.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    label, run_dir_str = raw.split("|", 1)
    run_dir = Path(run_dir_str)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    scored_path = run_dir / "scored_predictions.jsonl"

    scored_rows = []
    parsed = []
    for line in scored_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scored_rows.append(row)
        if not bool(row.get("parse_error", False)):
            parsed.append(row)
    inst = summarize_strategyqa_instability(scored_rows)
    instability_by_label[label] = inst
    sample_index_by_label[label] = index_rows_by_sample_id(scored_rows)
    parseable_correct = sum(1 for r in parsed if bool(r.get("is_correct", False)))
    acc_parsed = (parseable_correct / len(parsed)) if parsed else 0.0

    gen_cfg = manifest.get("generation_config", {})
    gen_stats = metrics.get("generation_stats", {}) if isinstance(metrics.get("generation_stats"), dict) else {}
    comparison = metrics.get("comparison", {}) if isinstance(metrics.get("comparison"), dict) else {}
    delta_acc = comparison.get("delta_accuracy")
    delta_parse = comparison.get("delta_parse_error_rate")
    changed = comparison.get("changed_prediction_count")

    rows.append(
        {
            "label": label,
            "run_dir": str(run_dir),
            "max_new_tokens": int(gen_cfg.get("max_new_tokens", 0)),
            "n": int(metrics.get("n_total", 0)),
            "n_correct": int(metrics.get("n_correct", 0)),
            "n_parse_error": int(metrics.get("n_parse_error", 0)),
            "acc": float(metrics.get("accuracy", 0.0)),
            "parse_err": float(metrics.get("parse_error_rate", 0.0)),
            "parseable_n": len(parsed),
            "parseable_correct": int(parseable_correct),
            "acc_parseable": float(acc_parsed),
            "delta_acc": delta_acc,
            "delta_parse": delta_parse,
            "changed": changed,
            "vram_mean_gib": (
                float(gen_stats.get("vram_mean_total_reserved_gib_sampled"))
                if gen_stats.get("vram_mean_total_reserved_gib_sampled") is not None
                else None
            ),
            "vram_max_gib": (
                float(gen_stats.get("vram_max_total_reserved_gib_sampled"))
                if gen_stats.get("vram_max_total_reserved_gib_sampled") is not None
                else None
            ),
        }
    )

spec_rows = []
for raw in specs_path.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    label, input_kind, run_name, max_new_tokens, compare_mode = raw.split("|", 4)
    spec_rows.append(
        {
            "label": label,
            "input_kind": input_kind,
            "run_name": run_name,
            "max_new_tokens": max_new_tokens,
            "compare_mode": compare_mode,
        }
    )

spec_compare_mode_by_label = {s["label"]: s["compare_mode"] for s in spec_rows}

best_acc = max(rows, key=lambda r: r["acc"]) if rows else None
best_parse = min(rows, key=lambda r: r["parse_err"]) if rows else None

out_lines: list[str] = []
out_lines.append("=" * 138)
out_lines.append("FINAL EXPERIMENT SUMMARY")
out_lines.append("=" * 138)
out_lines.append(f"generated_at      : {datetime.now().astimezone().isoformat()}")
out_lines.append(f"group_id          : {group_id}")
out_lines.append(f"group_title       : {group_title}")
out_lines.append(f"run_prefix        : {os.environ['PHASEA_RUN_PREFIX']}")
out_lines.append(f"intention         : {group_intention}")
out_lines.append(f"observe           : {group_observe}")
out_lines.append(f"expectation       : {group_expectation}")
out_lines.append("-" * 138)
out_lines.append("SETTINGS")
out_lines.append(f"dataset           : {os.environ['PHASEA_DATASET']}")
out_lines.append(f"source_split      : {os.environ['PHASEA_SOURCE_SPLIT']}")
out_lines.append(f"split_policy      : {os.environ['PHASEA_SPLIT_POLICY']}")
out_lines.append(f"limit             : {os.environ['PHASEA_LIMIT']}")
out_lines.append(f"seed              : {os.environ['PHASEA_SEED']}")
out_lines.append(f"dtype             : {os.environ['PHASEA_DTYPE']}")
out_lines.append(f"log_every         : {os.environ['PHASEA_LOG_EVERY']}")
out_lines.append(f"max_progress_lines: {os.environ['PHASEA_MAX_PROGRESS_LINES']}")
out_lines.append(f"batch_size        : {os.environ['PHASEA_BATCH_SIZE']}")
out_lines.append(f"oom_backoff       : {os.environ['PHASEA_OOM_BACKOFF']}")
out_lines.append(f"trunc_recovery    : {os.environ['PHASEA_TRUNCATION_RECOVERY']}")
out_lines.append(f"trunc_recov_rounds: {os.environ['PHASEA_TRUNCATION_RECOVERY_ROUNDS']}")
out_lines.append(f"trunc_recov_extra : {os.environ['PHASEA_TRUNCATION_RECOVERY_EXTRA_TOKENS']}")
out_lines.append(f"trunc_recov_data  : {os.environ['PHASEA_TRUNCATION_RECOVERY_DATASETS']}")
out_lines.append(f"trunc_recov_reqfa : {os.environ['PHASEA_TRUNCATION_RECOVERY_REQUIRE_FINAL_SIGNAL']}")
out_lines.append(f"strategyqa_decode : {os.environ['PHASEA_STRATEGYQA_DECODE_MODE']}")
out_lines.append(f"truncate_markers  : {os.environ['PHASEA_TRUNCATE_CHAT_MARKERS']}")
out_lines.append(f"cuda_devices      : {os.environ['PHASEA_CUDA_VISIBLE_DEVICES']}")
out_lines.append(f"model_path        : {os.environ['PHASEA_MODEL_PATH']}")
out_lines.append(
    "direct_template  : "
    f"{os.environ['PHASEA_DIRECT_TEMPLATE_ID']} ({os.environ['PHASEA_DIRECT_TARGET_STYLE']})"
)
out_lines.append(
    "cot_template     : "
    f"{os.environ['PHASEA_COT_TEMPLATE_ID']} ({os.environ['PHASEA_COT_TARGET_STYLE']})"
)
out_lines.append(
    "strict_template  : "
    f"{os.environ['PHASEA_STRICT_TEMPLATE_ID']} ({os.environ['PHASEA_STRICT_TARGET_STYLE']})"
)
direct_train = os.environ.get("PHASEA_DIRECT_TRAIN_JSONL", "")
direct_val = os.environ.get("PHASEA_DIRECT_VAL_JSONL", "")
direct_test = os.environ.get("PHASEA_DIRECT_TEST_JSONL", "")
cot_train = os.environ.get("PHASEA_COT_TRAIN_JSONL", "")
cot_val = os.environ.get("PHASEA_COT_VAL_JSONL", "")
cot_test = os.environ.get("PHASEA_COT_TEST_JSONL", "")
strict_train = os.environ.get("PHASEA_STRICT_TRAIN_JSONL", "")
strict_val = os.environ.get("PHASEA_STRICT_VAL_JSONL", "")
strict_test = os.environ.get("PHASEA_STRICT_TEST_JSONL", "")

if direct_train:
    out_lines.append(f"direct_train_input: {direct_train}")
if direct_val:
    out_lines.append(f"direct_val_input  : {direct_val}")
if direct_test:
    out_lines.append(f"direct_test_input : {direct_test}")
if cot_train:
    out_lines.append(f"cot_train_input   : {cot_train}")
if cot_val:
    out_lines.append(f"cot_val_input     : {cot_val}")
if cot_test:
    out_lines.append(f"cot_test_input    : {cot_test}")
if strict_train:
    out_lines.append(f"strict_train_input: {strict_train}")
if strict_val:
    out_lines.append(f"strict_val_input  : {strict_val}")
if strict_test:
    out_lines.append(f"strict_test_input : {strict_test}")
out_lines.append(f"suite_log_file    : {os.environ.get('PHASEA_SUITE_LOG_FILE', '<disabled>')}")
out_lines.append(f"summary_file      : {summary_path}")
out_lines.append("-" * 138)
out_lines.append("PLANNED RUN SPECS")
for spec in spec_rows:
    out_lines.append(
        f"- label={spec['label']} | input={spec['input_kind']} | "
        f"tok={spec['max_new_tokens']} | compare={spec['compare_mode']} | "
        f"run_name={spec['run_name']}"
    )
out_lines.append("-" * 138)
out_lines.append("RESULT TABLE")
out_lines.append(
    f"{'label':<22} {'tok':>4} {'n':>5} {'acc':>8} {'parse_err':>10} "
    f"{'parseable_n':>11} {'acc_parseable':>13} {'vram_mean':>10} {'vram_max':>10} "
    f"{'delta_acc':>10} {'changed':>8}"
)
out_lines.append("-" * 138)
for row in rows:
    delta_acc = f"{row['delta_acc']:+.4f}" if row["delta_acc"] is not None else "n/a"
    changed = str(row["changed"]) if row["changed"] is not None else "n/a"
    vram_mean = f"{row['vram_mean_gib']:.2f}" if row["vram_mean_gib"] is not None else "n/a"
    vram_max = f"{row['vram_max_gib']:.2f}" if row["vram_max_gib"] is not None else "n/a"
    out_lines.append(
        f"{row['label']:<22} {row['max_new_tokens']:>4d} {row['n']:>5d} "
        f"{row['acc']:>8.4f} {row['parse_err']:>10.4f} "
        f"{row['parseable_n']:>11d} {row['acc_parseable']:>13.4f} "
        f"{vram_mean:>10} {vram_max:>10} "
        f"{delta_acc:>10} {changed:>8}"
    )
out_lines.append("-" * 138)
out_lines.append("INSTABILITY INDICATORS (ARTIFACT ANALYSIS)")
out_lines.append(
    f"{'label':<22} {'tagged':>8} {'multi_tag':>10} {'first_last_flip':>15} "
    f"{'tag_switch':>11} {'mean_tags':>10}"
)
out_lines.append("-" * 138)
for row in rows:
    inst = instability_by_label.get(row["label"], {})
    out_lines.append(
        f"{row['label']:<22} "
        f"{float(inst.get('with_final_tag_rate', 0.0)):>8.4f} "
        f"{float(inst.get('multi_final_tag_rate', 0.0)):>10.4f} "
        f"{float(inst.get('first_last_disagree_rate', 0.0)):>15.4f} "
        f"{float(inst.get('tag_switch_rate', 0.0)):>11.4f} "
        f"{float(inst.get('mean_final_tag_count_tagged', 0.0)):>10.2f}"
    )
out_lines.append("-" * 138)

pairwise_rows = []
for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        label_a = rows[i]["label"]
        label_b = rows[j]["label"]
        flip = compute_pairwise_prediction_flip(
            sample_index_by_label.get(label_a, {}),
            sample_index_by_label.get(label_b, {}),
        )
        if int(flip.get("n_overlap", 0)) <= 0:
            continue
        pairwise_rows.append(
            {
                "label_a": label_a,
                "label_b": label_b,
                **flip,
            }
        )

if pairwise_rows:
    out_lines.append("PAIRWISE FLIP ANALYSIS (OVERLAPPING SAMPLE_IDS)")
    out_lines.append(
        f"{'run_a':<24} {'run_b':<24} {'overlap':>8} {'pred_flip':>10} "
        f"{'corr_flip':>10} {'yes->no':>8} {'no->yes':>8}"
    )
    out_lines.append("-" * 138)
    for item in pairwise_rows:
        out_lines.append(
            f"{item['label_a']:<24} {item['label_b']:<24} "
            f"{int(item.get('n_overlap', 0)):>8d} "
            f"{float(item.get('pred_flip_rate', 0.0)):>10.4f} "
            f"{float(item.get('correctness_flip_rate', 0.0)):>10.4f} "
            f"{int(item.get('n_yes_to_no', 0)):>8d} "
            f"{int(item.get('n_no_to_yes', 0)):>8d}"
        )
    out_lines.append("-" * 138)

if os.environ.get("PHASEA_GROUP_ENABLE_WHOLE_AGG", "0") == "1" and rows:
    primary_rows = [
        r
        for r in rows
        if spec_compare_mode_by_label.get(r["label"], "no") != "yes"
    ]
    agg_n = sum(r["n"] for r in primary_rows)
    agg_correct = sum(r["n_correct"] for r in primary_rows)
    agg_parse_err_n = sum(r["n_parse_error"] for r in primary_rows)
    agg_parseable_n = sum(r["parseable_n"] for r in primary_rows)
    agg_parseable_correct = sum(r["parseable_correct"] for r in primary_rows)
    agg_acc = (agg_correct / agg_n) if agg_n else 0.0
    agg_parse_err_rate = (agg_parse_err_n / agg_n) if agg_n else 0.0
    agg_acc_parseable = (
        agg_parseable_correct / agg_parseable_n if agg_parseable_n else 0.0
    )
    out_lines.append("WHOLE-CORPUS AGGREGATE")
    out_lines.append(
        f"included_runs={len(primary_rows)} (compare=no only, reproducibility reruns excluded)"
    )
    out_lines.append(
        f"n_total={agg_n} | n_correct={agg_correct} | n_parse_error={agg_parse_err_n} | "
        f"n_parseable={agg_parseable_n}"
    )
    out_lines.append(
        f"accuracy={agg_acc:.4f} | parse_error_rate={agg_parse_err_rate:.4f} | "
        f"acc_parseable={agg_acc_parseable:.4f}"
    )
    out_lines.append("-" * 138)
if best_acc is not None:
    out_lines.append(
        "best_accuracy     : "
        f"{best_acc['label']} (acc={best_acc['acc']:.4f}, parse_err={best_acc['parse_err']:.4f})"
    )
if best_parse is not None:
    out_lines.append(
        "lowest_parse_err  : "
        f"{best_parse['label']} (parse_err={best_parse['parse_err']:.4f}, acc={best_parse['acc']:.4f})"
    )
out_lines.append("=" * 138)

text = "\n".join(out_lines) + "\n"
print(text, end="")
summary_path.write_text(text, encoding="utf-8")
PY
}

main() {
  log "Repo root      : ${REPO_ROOT}"
  log "Python         : ${PYTHON_BIN}"
  log "CUDA devices   : ${CUDA_VISIBLE_DEVICES}"
  log "Model path     : ${MODEL_PATH}"
  log "Run prefix     : ${RUN_PREFIX}"
  log "Log settings   : log_every=${LOG_EVERY}, max_progress_lines=${MAX_PROGRESS_LINES}"
  log "Batch settings : batch_size=${BATCH_SIZE}, oom_backoff=${OOM_BACKOFF}"
  log "Trunc settings : enabled=${TRUNCATION_RECOVERY}, rounds=${TRUNCATION_RECOVERY_ROUNDS}, extra_tokens=${TRUNCATION_RECOVERY_EXTRA_TOKENS}, datasets=${TRUNCATION_RECOVERY_DATASETS}, require_final_signal=${TRUNCATION_RECOVERY_REQUIRE_FINAL_SIGNAL}"
  if [[ "${ENABLE_PERSISTED_LOGS}" == "1" ]]; then
    log "Suite log file : ${SUITE_LOG_FILE}"
  else
    log "Suite log file : <disabled>"
  fi
  log "Summary file   : ${SUITE_SUMMARY_FILE}"
  assert_file "scripts/phase_a_prepare.py"
  assert_file "scripts/phase_a_generate_and_eval.py"
  assert_dir "${MODEL_PATH}"
  warn_if_concurrent_generation

  configure_param_group "${ACTIVE_PARAM_GROUP}"

  log "Param group    : ${ACTIVE_PARAM_GROUP} (${GROUP_TITLE})"
  log "Intention      : ${GROUP_INTENTION}"
  log "Observe        : ${GROUP_OBSERVE}"
  log "Expectation    : ${GROUP_EXPECTATION}"
  log "Dataset config : dataset=${DATASET}, source_split=${SOURCE_SPLIT}, split_policy=${SPLIT_POLICY}, limit=${LIMIT}"
  log "Decode mode    : strategyqa_decode_mode=${STRATEGYQA_DECODE_MODE}, truncate_chat_markers=${TRUNCATE_CHAT_MARKERS}"
  log "Template slots : direct=${DIRECT_TEMPLATE_ID}/${DIRECT_TARGET_STYLE}, cot=${COT_TEMPLATE_ID}/${COT_TARGET_STYLE}, strict=${STRICT_TEMPLATE_ID}/${STRICT_TARGET_STYLE}"

  prepare_needed_inputs
  execute_group_runs
  print_summary_table

  log "Group run complete."
}

main "$@"
