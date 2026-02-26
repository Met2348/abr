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

# Sweep defaults used by some groups.
COT_SWEEP_TOKENS="${COT_SWEEP_TOKENS:-128 192 256 320 384}"
DIRECT_SWEEP_TOKENS="${DIRECT_SWEEP_TOKENS:-16 24 32 48 64}"

# For 7B, single GPU is usually simpler and often faster than multi-GPU sharding.
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

# Include group name in run prefix so outputs are easy to track.
RUN_PREFIX="${RUN_PREFIX:-${ACTIVE_PARAM_GROUP}_$(date -u +%Y%m%dT%H%M%SZ)}"

# -----------------------------------------------------------------------------
# Internal state populated by group config.
# -----------------------------------------------------------------------------
GROUP_TITLE=""
GROUP_INTENTION=""
GROUP_OBSERVE=""
GROUP_EXPECTATION=""
GROUP_NEED_DIRECT=0
GROUP_NEED_COT=0
declare -a GROUP_RUN_SPECS=()
declare -a RESULT_LINES=()
DIRECT_VAL_JSONL=""
COT_VAL_JSONL=""

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

assert_no_concurrent_generation() {
  if [[ "${ALLOW_CONCURRENT:-0}" == "1" ]]; then
    return 0
  fi
  if pgrep -fa "scripts/phase_a_generate_and_eval.py" >/dev/null; then
    die "Detected another running phase_a_generate_and_eval.py process. \
Set ALLOW_CONCURRENT=1 to bypass, or wait for current run to finish."
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
    --log-every "${LOG_EVERY}" \
    "${compare_flag[@]}"
}

configure_param_group() {
  local group_id="$1"

  GROUP_RUN_SPECS=()
  GROUP_NEED_DIRECT=0
  GROUP_NEED_COT=0

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
    *)
      die "Unsupported ACTIVE_PARAM_GROUP=${group_id}. Supported: A1, A2, A3, A4"
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
  local tmp_file
  tmp_file="$(mktemp /tmp/phasea_group_summary_XXXXXX.txt)"
  trap 'rm -f "${tmp_file}"' RETURN

  printf '%s\n' "${RESULT_LINES[@]}" > "${tmp_file}"

  "${PYTHON_BIN}" - "${tmp_file}" "${ACTIVE_PARAM_GROUP}" "${GROUP_TITLE}" <<'PY'
import json
import sys
from pathlib import Path

lines_path = Path(sys.argv[1])
group_id = sys.argv[2]
group_title = sys.argv[3]

rows = []
for raw in lines_path.read_text(encoding="utf-8").splitlines():
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

print("=" * 138)
print(f"Group Summary: {group_id} - {group_title}")
print("=" * 138)
print(
    f"{'label':<22} {'tok':>4} {'n':>5} {'acc':>8} {'parse_err':>10} "
    f"{'parseable_n':>11} {'acc_parseable':>13} {'delta_acc':>10} {'changed':>8}"
)
print("-" * 138)
for row in rows:
    delta_acc = f"{row['delta_acc']:+.4f}" if row["delta_acc"] is not None else "n/a"
    changed = str(row["changed"]) if row["changed"] is not None else "n/a"
    print(
        f"{row['label']:<22} {row['max_new_tokens']:>4d} {row['n']:>5d} "
        f"{row['acc']:>8.4f} {row['parse_err']:>10.4f} "
        f"{row['parseable_n']:>11d} {row['acc_parseable']:>13.4f} "
        f"{delta_acc:>10} {changed:>8}"
    )
print("=" * 138)
PY
}

main() {
  log "Repo root      : ${REPO_ROOT}"
  log "Python         : ${PYTHON_BIN}"
  log "CUDA devices   : ${CUDA_VISIBLE_DEVICES}"
  log "Model path     : ${MODEL_PATH}"
  log "Run prefix     : ${RUN_PREFIX}"
  log "Dataset config : dataset=${DATASET}, source_split=${SOURCE_SPLIT}, split_policy=${SPLIT_POLICY}, limit=${LIMIT}"

  assert_file "scripts/phase_a_prepare.py"
  assert_file "scripts/phase_a_generate_and_eval.py"
  assert_dir "${MODEL_PATH}"
  assert_no_concurrent_generation

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

