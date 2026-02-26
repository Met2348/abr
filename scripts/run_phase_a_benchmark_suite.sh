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
declare -a GROUP_RUN_SPECS=()
declare -a RESULT_LINES=()
DIRECT_VAL_JSONL=""
COT_VAL_JSONL=""
STRICT_VAL_JSONL=""

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

limit = None if limit_raw in {"None", ""} else int(limit_raw)
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
  run_cmd "${PYTHON_BIN}" scripts/phase_a_prepare.py \
    --datasets "${DATASET}" \
    --source-split "${SOURCE_SPLIT}" \
    --split-policy "${SPLIT_POLICY}" \
    --limit "${LIMIT}" \
    --template-id "${template_id}" \
    --template-version "${TEMPLATE_VERSION}" \
    --target-style "${target_style}" \
    --seed "${SEED}" \
    --resume
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
    *)
      die "Unsupported ACTIVE_PARAM_GROUP=${group_id}. Supported: A1, A2, A3, A4, A5, A6"
      ;;
  esac
}

prepare_needed_inputs() {
  if [[ "${GROUP_NEED_DIRECT}" == "1" ]]; then
    prepare_variant "qa_direct" "answer_only"
    local direct_prep_dir
    direct_prep_dir="$(resolve_prepared_dir "answer_only" "qa_direct")"
    DIRECT_VAL_JSONL="${direct_prep_dir}/validation.jsonl"
    assert_file "${DIRECT_VAL_JSONL}"
    log "Prepared direct validation file: ${DIRECT_VAL_JSONL}"
  fi

  if [[ "${GROUP_NEED_COT}" == "1" ]]; then
    prepare_variant "qa_cot_then_final" "cot_then_answer"
    local cot_prep_dir
    cot_prep_dir="$(resolve_prepared_dir "cot_then_answer" "qa_cot_then_final")"
    COT_VAL_JSONL="${cot_prep_dir}/validation.jsonl"
    assert_file "${COT_VAL_JSONL}"
    log "Prepared CoT validation file   : ${COT_VAL_JSONL}"
  fi

  if [[ "${GROUP_NEED_STRICT}" == "1" ]]; then
    prepare_variant "qa_binary_strict" "answer_only"
    local strict_prep_dir
    strict_prep_dir="$(resolve_prepared_dir "answer_only" "qa_binary_strict")"
    STRICT_VAL_JSONL="${strict_prep_dir}/validation.jsonl"
    assert_file "${STRICT_VAL_JSONL}"
    log "Prepared strict validation file: ${STRICT_VAL_JSONL}"
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
      cot)
        [[ -n "${COT_VAL_JSONL}" ]] || die "CoT input is not prepared."
        input_jsonl="${COT_VAL_JSONL}"
        ;;
      strict)
        [[ -n "${STRICT_VAL_JSONL}" ]] || die "Strict input is not prepared."
        input_jsonl="${STRICT_VAL_JSONL}"
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
  trap 'rm -f "${tmp_results_file}" "${tmp_specs_file}"' RETURN

  printf '%s\n' "${RESULT_LINES[@]}" > "${tmp_results_file}"
  printf '%s\n' "${GROUP_RUN_SPECS[@]}" > "${tmp_specs_file}"

  PHASEA_GROUP_ID="${ACTIVE_PARAM_GROUP}" \
  PHASEA_GROUP_TITLE="${GROUP_TITLE}" \
  PHASEA_GROUP_INTENTION="${GROUP_INTENTION}" \
  PHASEA_GROUP_OBSERVE="${GROUP_OBSERVE}" \
  PHASEA_GROUP_EXPECTATION="${GROUP_EXPECTATION}" \
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
  PHASEA_STRATEGYQA_DECODE_MODE="${STRATEGYQA_DECODE_MODE}" \
  PHASEA_TRUNCATE_CHAT_MARKERS="${TRUNCATE_CHAT_MARKERS}" \
  PHASEA_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  PHASEA_MODEL_PATH="${MODEL_PATH}" \
  PHASEA_DIRECT_VAL_JSONL="${DIRECT_VAL_JSONL}" \
  PHASEA_COT_VAL_JSONL="${COT_VAL_JSONL}" \
  PHASEA_STRICT_VAL_JSONL="${STRICT_VAL_JSONL}" \
  PHASEA_SUITE_LOG_FILE="${SUITE_LOG_FILE:-<disabled>}" \
  PHASEA_SUITE_SUMMARY_FILE="${SUITE_SUMMARY_FILE}" \
  "${PYTHON_BIN}" - "${tmp_results_file}" "${tmp_specs_file}" <<'PY'
import json
import os
import sys
from datetime import datetime
from pathlib import Path

results_path = Path(sys.argv[1])
specs_path = Path(sys.argv[2])

group_id = os.environ["PHASEA_GROUP_ID"]
group_title = os.environ["PHASEA_GROUP_TITLE"]
group_intention = os.environ["PHASEA_GROUP_INTENTION"]
group_observe = os.environ["PHASEA_GROUP_OBSERVE"]
group_expectation = os.environ["PHASEA_GROUP_EXPECTATION"]
summary_path = Path(os.environ["PHASEA_SUITE_SUMMARY_FILE"])

rows = []
for raw in results_path.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    label, run_dir_str = raw.split("|", 1)
    run_dir = Path(run_dir_str)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    scored_path = run_dir / "scored_predictions.jsonl"

    parsed = []
    for line in scored_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not bool(row.get("parse_error", False)):
            parsed.append(row)
    acc_parsed = (
        sum(1 for r in parsed if bool(r.get("is_correct", False))) / len(parsed)
        if parsed
        else 0.0
    )

    gen_cfg = manifest.get("generation_config", {})
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
            "acc": float(metrics.get("accuracy", 0.0)),
            "parse_err": float(metrics.get("parse_error_rate", 0.0)),
            "parseable_n": len(parsed),
            "acc_parseable": float(acc_parsed),
            "delta_acc": delta_acc,
            "delta_parse": delta_parse,
            "changed": changed,
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
out_lines.append(f"strategyqa_decode : {os.environ['PHASEA_STRATEGYQA_DECODE_MODE']}")
out_lines.append(f"truncate_markers  : {os.environ['PHASEA_TRUNCATE_CHAT_MARKERS']}")
out_lines.append(f"cuda_devices      : {os.environ['PHASEA_CUDA_VISIBLE_DEVICES']}")
out_lines.append(f"model_path        : {os.environ['PHASEA_MODEL_PATH']}")
direct_path = os.environ.get("PHASEA_DIRECT_VAL_JSONL", "")
cot_path = os.environ.get("PHASEA_COT_VAL_JSONL", "")
strict_path = os.environ.get("PHASEA_STRICT_VAL_JSONL", "")
if direct_path:
    out_lines.append(f"direct_input      : {direct_path}")
if cot_path:
    out_lines.append(f"cot_input         : {cot_path}")
if strict_path:
    out_lines.append(f"strict_input      : {strict_path}")
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
    f"{'parseable_n':>11} {'acc_parseable':>13} {'delta_acc':>10} {'changed':>8}"
)
out_lines.append("-" * 138)
for row in rows:
    delta_acc = f"{row['delta_acc']:+.4f}" if row["delta_acc"] is not None else "n/a"
    changed = str(row["changed"]) if row["changed"] is not None else "n/a"
    out_lines.append(
        f"{row['label']:<22} {row['max_new_tokens']:>4d} {row['n']:>5d} "
        f"{row['acc']:>8.4f} {row['parse_err']:>10.4f} "
        f"{row['parseable_n']:>11d} {row['acc_parseable']:>13.4f} "
        f"{delta_acc:>10} {changed:>8}"
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
  log "Decode mode    : strategyqa_decode_mode=${STRATEGYQA_DECODE_MODE}, truncate_chat_markers=${TRUNCATE_CHAT_MARKERS}"
  if [[ "${ENABLE_PERSISTED_LOGS}" == "1" ]]; then
    log "Suite log file : ${SUITE_LOG_FILE}"
  else
    log "Suite log file : <disabled>"
  fi
  log "Summary file   : ${SUITE_SUMMARY_FILE}"
  log "Dataset config : dataset=${DATASET}, source_split=${SOURCE_SPLIT}, split_policy=${SPLIT_POLICY}, limit=${LIMIT}"

  assert_file "scripts/phase_a_prepare.py"
  assert_file "scripts/phase_a_generate_and_eval.py"
  assert_dir "${MODEL_PATH}"
  warn_if_concurrent_generation

  configure_param_group "${ACTIVE_PARAM_GROUP}"

  log "Param group    : ${ACTIVE_PARAM_GROUP} (${GROUP_TITLE})"
  log "Intention      : ${GROUP_INTENTION}"
  log "Observe        : ${GROUP_OBSERVE}"
  log "Expectation    : ${GROUP_EXPECTATION}"

  prepare_needed_inputs
  execute_group_runs
  print_summary_table

  log "Group run complete."
}

main "$@"
