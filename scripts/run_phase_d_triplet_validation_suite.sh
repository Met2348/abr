#!/usr/bin/env bash
# Phase D6-T triplet-validation suite (mentor-mandated ranking branch).
#
# Why this file exists:
# - D6-T requires one strict, reproducible branch to validate ranking learnability
#   on high-confidence same-question triplets.
# - Manual execution of DT1~DT6 groups is easy to misconfigure and hard to compare.
#
# What this file does:
# 1) Resolve one group via ACTIVE_PHASE_D6T_GROUP.
# 2) Build canonical external pair artifacts (Math-Shepherd / PRM800K).
# 3) Train C2 with ranking-first policy (or configured ablation mode).
# 4) Evaluate directly on external held-out pairs.
# 5) Write per-seed rows and aggregated summary markdown/json.
#
# Quick start (DT1 smoke):
#   ACTIVE_PHASE_D6T_GROUP=DT1_MATH_SHEPHERD_SMOKE \
#   RUN_PREFIX=d6t_smoke \
#   PHASE_C_TRAIN_DIR=assets/artifacts/phase_c_data/strategyqa/<train_dir> \
#   PHASE_C_EVAL_DIR=assets/artifacts/phase_c_data/strategyqa/<eval_dir> \
#   CUDA_VISIBLE_DEVICES=1 \
#   bash scripts/run_phase_d_triplet_validation_suite.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACTIVE_PHASE_D6T_GROUP="${ACTIVE_PHASE_D6T_GROUP:-DT1_MATH_SHEPHERD_SMOKE}"
RUN_PREFIX="${RUN_PREFIX:-phase_d6t_triplet_suite}"
CURRENT_STAGE="init"

GROUP_DATASET="${GROUP_DATASET:-strategyqa}"
PHASE_C_TRAIN_DIR="${PHASE_C_TRAIN_DIR:-}"
PHASE_C_EVAL_DIR="${PHASE_C_EVAL_DIR:-}"

PAIR_OUTPUT_ROOT="${PAIR_OUTPUT_ROOT:-assets/artifacts/phase_d_external_pairs}"
C2_OUTPUT_ROOT="${C2_OUTPUT_ROOT:-assets/artifacts/phase_c_runs}"
D6T_EVAL_OUTPUT_ROOT="${D6T_EVAL_OUTPUT_ROOT:-assets/artifacts/phase_d_triplet_eval}"
LOG_ROOT="${LOG_ROOT:-assets/artifacts/phase_d6t_logs/${RUN_PREFIX}}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SEED_RESULTS_JSONL="${LOG_ROOT}/seed_results.jsonl"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"

MATH_SHEPHERD_PATH="${MATH_SHEPHERD_PATH:-assets/external_datasets/peiyi_math_shepherd/math-shepherd.jsonl}"
PRM800K_PATH="${PRM800K_PATH:-assets/external_datasets/openai_prm800k}"

FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-assets/artifacts/phase_c_feature_cache}"
FEATURE_CACHE_MODE="${FEATURE_CACHE_MODE:-read_write}"
FEATURE_CACHE_LOCK_TIMEOUT_SEC="${FEATURE_CACHE_LOCK_TIMEOUT_SEC:-600}"

# Optional toggles / append-only args.
RUN_C1_STANDALONE_EVAL="${RUN_C1_STANDALONE_EVAL:-0}"
PAIR_PREP_EXTRA_ARGS="${PAIR_PREP_EXTRA_ARGS:-}"
C2_TRAIN_EXTRA_ARGS="${C2_TRAIN_EXTRA_ARGS:-}"
D6T_EXT_EVAL_EXTRA_ARGS="${D6T_EXT_EVAL_EXTRA_ARGS:-}"
C1_EVAL_EXTRA_ARGS="${C1_EVAL_EXTRA_ARGS:-}"
GROUP_PAIR_PREP_EXTRA_DEFAULT=""
GROUP_C2_TRAIN_EXTRA_DEFAULT=""

# Pass-gate thresholds (for seed aggregation).
PASS_PAIR_ACC_MIN="${PASS_PAIR_ACC_MIN:-0.65}"
PASS_AUC_MIN="${PASS_AUC_MIN:-0.65}"
PASS_STD_MAX="${PASS_STD_MAX:-0.03}"

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

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: Missing ${label}: $path" >&2
    return 1
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    echo "ERROR: Missing ${label}: $path" >&2
    return 1
  fi
}

latest_phase_c_data_guess() {
  local dataset="$1"
  local marker="$2"
  local latest=""
  latest="$(compgen -G "assets/artifacts/phase_c_data/${dataset}/*${marker}*__*" | sort | tail -n 1 || true)"
  printf '%s\n' "$latest"
}

latest_dir_for_prefix() {
  local glob_pattern="$1"
  local latest=""
  latest="$(compgen -G "$glob_pattern" | sort | tail -n 1 || true)"
  if [[ -z "$latest" ]]; then
    return 1
  fi
  printf '%s\n' "$latest"
}

write_failure_summary() {
  local exit_code="$1"
  mkdir -p "$(dirname "$SUMMARY_FILE")"
  cat > "$SUMMARY_FILE" <<EOM
# Phase D6-T Triplet Suite Summary

- generated_at: $(date --iso-8601=seconds)
- group_id: ${ACTIVE_PHASE_D6T_GROUP:-N/A}
- group_title: ${GROUP_TITLE:-N/A}
- run_prefix: ${RUN_PREFIX:-N/A}
- status: failed
- exit_code: ${exit_code}
- failed_stage: ${CURRENT_STAGE:-unknown}
- suite_log_file: ${SUITE_LOG_FILE:-N/A}
EOM
}

on_exit() {
  local exit_code="$1"
  if [[ "$exit_code" -ne 0 ]]; then
    {
      log_line "Failure stage: ${CURRENT_STAGE:-unknown}"
      log_line "Exit code: ${exit_code}"
    } >> "$SUITE_LOG_FILE"
    write_failure_summary "$exit_code"
  fi
}
trap 'on_exit $?' EXIT

resolve_group() {
  case "$ACTIVE_PHASE_D6T_GROUP" in
    DT1_MATH_SHEPHERD_SMOKE)
      GROUP_TITLE="DT1 Math-Shepherd Smoke"
      GROUP_INTENTION="Fast sanity check for triplet construction + ranking-only training."
      GROUP_OBSERVE="Verify end-to-end run and metrics above random baseline."
      GROUP_EXPECT="pair_acc/auc should be > 0.5 with stable training."
      SEEDS=(42)
      USE_MATH_SHEPHERD=1
      USE_PRM800K=0
      C2_EPOCHS="${C2_EPOCHS:-2}"
      C2_LR="${C2_LR:-1e-4}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-64}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-64}"
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-2500}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-5000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.55}"
      MIN_CHARS="${MIN_CHARS:-12}"
      MAX_LENGTH_RATIO="${MAX_LENGTH_RATIO:-3.0}"
      MAX_TOKEN_OVERLAP="${MAX_TOKEN_OVERLAP:-0.99}"
      MAX_PAIRS_PER_SAMPLE="${MAX_PAIRS_PER_SAMPLE:-2}"
      C2_TRAIN_MODE="${C2_TRAIN_MODE:-ranking_only}"
      C2_CALIBRATION_LOSS="${C2_CALIBRATION_LOSS:-mse}"
      EXTERNAL_PAIR_WEIGHT="${EXTERNAL_PAIR_WEIGHT:-1.0}"
      EXTERNAL_PAIR_SOURCE_BALANCE="${EXTERNAL_PAIR_SOURCE_BALANCE:-uniform}"
      EXTERNAL_PAIR_ONLY="${EXTERNAL_PAIR_ONLY:-1}"
      GROUP_C2_TRAIN_EXTRA_DEFAULT="${GROUP_C2_TRAIN_EXTRA_DEFAULT:---lambda-contrastive 1.0}"
      ;;
    DT2_MATH_SHEPHERD_SEED3)
      GROUP_TITLE="DT2 Math-Shepherd Seed3"
      GROUP_INTENTION="Main phase-1 gate: test ranking learnability under 3 seeds."
      GROUP_OBSERVE="Check mean and std of pair_acc/auc against pass thresholds."
      GROUP_EXPECT="pair_acc/auc cross 0.65 with std <= 0.03."
      SEEDS=(42 43 44)
      USE_MATH_SHEPHERD=1
      USE_PRM800K=0
      C2_EPOCHS="${C2_EPOCHS:-4}"
      C2_LR="${C2_LR:-1e-4}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-64}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-64}"
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-10000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-20000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.55}"
      MIN_CHARS="${MIN_CHARS:-12}"
      MAX_LENGTH_RATIO="${MAX_LENGTH_RATIO:-3.0}"
      MAX_TOKEN_OVERLAP="${MAX_TOKEN_OVERLAP:-0.99}"
      MAX_PAIRS_PER_SAMPLE="${MAX_PAIRS_PER_SAMPLE:-2}"
      C2_TRAIN_MODE="${C2_TRAIN_MODE:-ranking_only}"
      C2_CALIBRATION_LOSS="${C2_CALIBRATION_LOSS:-mse}"
      EXTERNAL_PAIR_WEIGHT="${EXTERNAL_PAIR_WEIGHT:-1.0}"
      EXTERNAL_PAIR_SOURCE_BALANCE="${EXTERNAL_PAIR_SOURCE_BALANCE:-uniform}"
      EXTERNAL_PAIR_ONLY="${EXTERNAL_PAIR_ONLY:-1}"
      GROUP_C2_TRAIN_EXTRA_DEFAULT="${GROUP_C2_TRAIN_EXTRA_DEFAULT:---lambda-contrastive 1.0}"
      ;;
    DT3_PRM800K_SMOKE)
      GROUP_TITLE="DT3 PRM800K Smoke"
      GROUP_INTENTION="Schema/learning sanity check on PRM800K-only triplets."
      GROUP_OBSERVE="Confirm no parser/schema collapse and metrics above random."
      GROUP_EXPECT="trend should align with DT1 direction."
      SEEDS=(42)
      USE_MATH_SHEPHERD=0
      USE_PRM800K=1
      C2_EPOCHS="${C2_EPOCHS:-2}"
      C2_LR="${C2_LR:-1e-4}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-64}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-64}"
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-2500}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-5000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.55}"
      MIN_CHARS="${MIN_CHARS:-12}"
      MAX_LENGTH_RATIO="${MAX_LENGTH_RATIO:-3.0}"
      MAX_TOKEN_OVERLAP="${MAX_TOKEN_OVERLAP:-0.99}"
      MAX_PAIRS_PER_SAMPLE="${MAX_PAIRS_PER_SAMPLE:-2}"
      C2_TRAIN_MODE="${C2_TRAIN_MODE:-ranking_only}"
      C2_CALIBRATION_LOSS="${C2_CALIBRATION_LOSS:-mse}"
      EXTERNAL_PAIR_WEIGHT="${EXTERNAL_PAIR_WEIGHT:-1.0}"
      EXTERNAL_PAIR_SOURCE_BALANCE="${EXTERNAL_PAIR_SOURCE_BALANCE:-uniform}"
      EXTERNAL_PAIR_ONLY="${EXTERNAL_PAIR_ONLY:-1}"
      GROUP_C2_TRAIN_EXTRA_DEFAULT="${GROUP_C2_TRAIN_EXTRA_DEFAULT:---lambda-contrastive 1.0}"
      ;;
    DT4_MIXED_MS_PRM800K_SEED3)
      GROUP_TITLE="DT4 Mixed Math-Shepherd + PRM800K Seed3"
      GROUP_INTENTION="Phase-2 mixed-source robustness gate."
      GROUP_OBSERVE="Check if ranking remains stable after source mixing."
      GROUP_EXPECT="metrics keep pass-level performance without source collapse."
      SEEDS=(42 43 44)
      USE_MATH_SHEPHERD=1
      USE_PRM800K=1
      C2_EPOCHS="${C2_EPOCHS:-4}"
      C2_LR="${C2_LR:-1e-4}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-64}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-64}"
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-10000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-24000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.55}"
      MIN_CHARS="${MIN_CHARS:-12}"
      MAX_LENGTH_RATIO="${MAX_LENGTH_RATIO:-3.0}"
      MAX_TOKEN_OVERLAP="${MAX_TOKEN_OVERLAP:-0.99}"
      MAX_PAIRS_PER_SAMPLE="${MAX_PAIRS_PER_SAMPLE:-2}"
      C2_TRAIN_MODE="${C2_TRAIN_MODE:-ranking_only}"
      C2_CALIBRATION_LOSS="${C2_CALIBRATION_LOSS:-mse}"
      EXTERNAL_PAIR_WEIGHT="${EXTERNAL_PAIR_WEIGHT:-1.0}"
      EXTERNAL_PAIR_SOURCE_BALANCE="${EXTERNAL_PAIR_SOURCE_BALANCE:-uniform}"
      EXTERNAL_PAIR_ONLY="${EXTERNAL_PAIR_ONLY:-1}"
      GROUP_C2_TRAIN_EXTRA_DEFAULT="${GROUP_C2_TRAIN_EXTRA_DEFAULT:---lambda-contrastive 1.0}"
      ;;
    DT5_ABLATION_NO_FILTER)
      GROUP_TITLE="DT5 Ablation No Filter"
      GROUP_INTENTION="Ablate pair-quality filters to measure noise impact."
      GROUP_OBSERVE="Compare against DT2/DT4 to verify filters are necessary."
      GROUP_EXPECT="ranking quality should degrade when filters are relaxed."
      SEEDS=(42)
      USE_MATH_SHEPHERD=1
      USE_PRM800K=1
      C2_EPOCHS="${C2_EPOCHS:-4}"
      C2_LR="${C2_LR:-1e-4}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-64}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-64}"
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-12000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-30000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.0}"
      MIN_CHARS="${MIN_CHARS:-4}"
      MAX_LENGTH_RATIO="${MAX_LENGTH_RATIO:-8.0}"
      MAX_TOKEN_OVERLAP="${MAX_TOKEN_OVERLAP:-1.0}"
      MAX_PAIRS_PER_SAMPLE="${MAX_PAIRS_PER_SAMPLE:-4}"
      C2_TRAIN_MODE="${C2_TRAIN_MODE:-ranking_only}"
      C2_CALIBRATION_LOSS="${C2_CALIBRATION_LOSS:-mse}"
      EXTERNAL_PAIR_WEIGHT="${EXTERNAL_PAIR_WEIGHT:-1.0}"
      EXTERNAL_PAIR_SOURCE_BALANCE="${EXTERNAL_PAIR_SOURCE_BALANCE:-uniform}"
      EXTERNAL_PAIR_ONLY="${EXTERNAL_PAIR_ONLY:-1}"
      GROUP_C2_TRAIN_EXTRA_DEFAULT="${GROUP_C2_TRAIN_EXTRA_DEFAULT:---lambda-contrastive 1.0}"
      ;;
    DT6_ABLATION_WITH_CAL_AUX)
      GROUP_TITLE="DT6 Ablation With Calibration Aux"
      GROUP_INTENTION="Test whether calibration auxiliary helps or hurts ranking."
      GROUP_OBSERVE="Compare ranking metrics against DT2/DT4 baseline."
      GROUP_EXPECT="if ranking drops, keep calibration out of D6-T core path."
      SEEDS=(42)
      USE_MATH_SHEPHERD=1
      USE_PRM800K=1
      C2_EPOCHS="${C2_EPOCHS:-4}"
      C2_LR="${C2_LR:-1e-4}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-64}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-64}"
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-10000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-24000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.55}"
      MIN_CHARS="${MIN_CHARS:-12}"
      MAX_LENGTH_RATIO="${MAX_LENGTH_RATIO:-3.0}"
      MAX_TOKEN_OVERLAP="${MAX_TOKEN_OVERLAP:-0.99}"
      MAX_PAIRS_PER_SAMPLE="${MAX_PAIRS_PER_SAMPLE:-2}"
      C2_TRAIN_MODE="${C2_TRAIN_MODE:-joint}"
      C2_CALIBRATION_LOSS="${C2_CALIBRATION_LOSS:-bce_mse}"
      EXTERNAL_PAIR_WEIGHT="${EXTERNAL_PAIR_WEIGHT:-1.0}"
      EXTERNAL_PAIR_SOURCE_BALANCE="${EXTERNAL_PAIR_SOURCE_BALANCE:-uniform}"
      EXTERNAL_PAIR_ONLY="${EXTERNAL_PAIR_ONLY:-1}"
      GROUP_C2_TRAIN_EXTRA_DEFAULT="${GROUP_C2_TRAIN_EXTRA_DEFAULT:---lambda-contrastive 1.0 --calibration-mse-weight 0.2 --calibration-bce-weight 0.2}"
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_D6T_GROUP=$ACTIVE_PHASE_D6T_GROUP" >&2
      echo "Supported groups:" >&2
      echo "  DT1_MATH_SHEPHERD_SMOKE" >&2
      echo "  DT2_MATH_SHEPHERD_SEED3" >&2
      echo "  DT3_PRM800K_SMOKE" >&2
      echo "  DT4_MIXED_MS_PRM800K_SEED3" >&2
      echo "  DT5_ABLATION_NO_FILTER" >&2
      echo "  DT6_ABLATION_WITH_CAL_AUX" >&2
      exit 1
      ;;
  esac
}

auto_resolve_phase_c_dirs() {
  if [[ -z "$PHASE_C_TRAIN_DIR" ]]; then
    PHASE_C_TRAIN_DIR="$(latest_phase_c_data_guess "$GROUP_DATASET" "train")"
  fi
  if [[ -z "$PHASE_C_EVAL_DIR" ]]; then
    PHASE_C_EVAL_DIR="$(latest_phase_c_data_guess "$GROUP_DATASET" "eval")"
  fi
  if [[ -z "$PHASE_C_TRAIN_DIR" || -z "$PHASE_C_EVAL_DIR" ]]; then
    echo "ERROR: Could not auto-resolve PHASE_C_TRAIN_DIR / PHASE_C_EVAL_DIR." >&2
    echo "Please set both variables explicitly." >&2
    return 1
  fi

  require_dir "$PHASE_C_TRAIN_DIR" "PHASE_C_TRAIN_DIR"
  require_dir "$PHASE_C_EVAL_DIR" "PHASE_C_EVAL_DIR"
  require_file "$PHASE_C_TRAIN_DIR/manifest.json" "train manifest"
  require_file "$PHASE_C_EVAL_DIR/manifest.json" "eval manifest"
  require_file "$PHASE_C_TRAIN_DIR/prefixes.jsonl" "train prefixes"
  require_file "$PHASE_C_EVAL_DIR/prefixes.jsonl" "eval prefixes"
}

append_seed_result() {
  local seed="$1"
  local pair_run_dir="$2"
  local c2_run_dir="$3"
  local ext_eval_run_dir="$4"
  local c1_eval_run_dir="$5"
  "$PYTHON_BIN" - "$seed" "$pair_run_dir" "$c2_run_dir" "$ext_eval_run_dir" "$c1_eval_run_dir" "$SEED_RESULTS_JSONL" <<'PY'
import json
import sys
from pathlib import Path

seed = int(sys.argv[1])
pair_run_dir = Path(sys.argv[2])
c2_run_dir = Path(sys.argv[3])
ext_eval_run_dir = Path(sys.argv[4])
c1_eval_run_dir_raw = sys.argv[5]
out_path = Path(sys.argv[6])

pair_summary = json.loads((pair_run_dir / "summary.json").read_text(encoding="utf-8"))
ext_metrics = json.loads((ext_eval_run_dir / "metrics.json").read_text(encoding="utf-8"))

c1_metrics = None
if c1_eval_run_dir_raw != "":
    c1_metrics_path = Path(c1_eval_run_dir_raw) / "metrics.json"
    if c1_metrics_path.exists():
        c1_metrics = json.loads(c1_metrics_path.read_text(encoding="utf-8"))

row = {
    "seed": seed,
    "pair_run_dir": str(pair_run_dir),
    "c2_run_dir": str(c2_run_dir),
    "ext_eval_run_dir": str(ext_eval_run_dir),
    "c1_eval_run_dir": (c1_eval_run_dir_raw if c1_eval_run_dir_raw else None),
    "num_train_pairs": int(pair_summary.get("num_train_rows", 0)),
    "num_val_pairs": int(pair_summary.get("num_validation_rows", 0)),
    "pair_sources": pair_summary.get("train_summary", {}).get("by_source", {}),
    "pair_mean_conf": float(pair_summary.get("train_summary", {}).get("mean_pair_confidence", 0.0)),
    "ext_pair_acc": float(ext_metrics.get("pair_accuracy", 0.0)),
    "ext_auc": float(ext_metrics.get("auc_chosen_vs_rejected", 0.0)),
    "ext_mean_margin": float(ext_metrics.get("mean_margin", 0.0)),
    "ext_median_margin": float(ext_metrics.get("median_margin", 0.0)),
    "c1_corr_pair_acc": (
        float(c1_metrics.get("corruption", {}).get("pair_accuracy", 0.0))
        if isinstance(c1_metrics, dict)
        else None
    ),
    "c1_corr_auc": (
        float(c1_metrics.get("corruption", {}).get("auc_clean_vs_corrupt", 0.0))
        if isinstance(c1_metrics, dict)
        else None
    ),
}
with out_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

render_final_summary() {
  "$PYTHON_BIN" - "$SEED_RESULTS_JSONL" "$SUMMARY_FILE" "$ACTIVE_PHASE_D6T_GROUP" "$GROUP_TITLE" \
    "$GROUP_INTENTION" "$GROUP_OBSERVE" "$GROUP_EXPECT" "$RUN_PREFIX" "$SUITE_LOG_FILE" \
    "$PASS_PAIR_ACC_MIN" "$PASS_AUC_MIN" "$PASS_STD_MAX" <<'PY'
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

rows_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
group_id = sys.argv[3]
group_title = sys.argv[4]
group_intention = sys.argv[5]
group_observe = sys.argv[6]
group_expect = sys.argv[7]
run_prefix = sys.argv[8]
suite_log = sys.argv[9]
pass_pair_acc_min = float(sys.argv[10])
pass_auc_min = float(sys.argv[11])
pass_std_max = float(sys.argv[12])

rows = []
if rows_path.exists():
    for raw in rows_path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text:
            continue
        rows.append(json.loads(text))

lines = [
    "# Phase D6-T Triplet Validation Suite Summary",
    "",
    f"- generated_at: {datetime.now(timezone.utc).isoformat()}",
    f"- group_id: {group_id}",
    f"- group_title: {group_title}",
    f"- run_prefix: {run_prefix}",
    f"- status: {'ok' if rows else 'empty'}",
    f"- suite_log_file: {suite_log}",
    f"- group_intention: {group_intention}",
    f"- observe: {group_observe}",
    f"- expect: {group_expect}",
    "",
]

if not rows:
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    raise SystemExit(0)

lines.extend(
    [
        "## Per-Seed Metrics",
        "",
        "| seed | train_pairs | val_pairs | ext_pair_acc | ext_auc | ext_mean_margin | c1_pair_acc | c1_auc |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
)
for row in rows:
    c1_acc = row.get("c1_corr_pair_acc")
    c1_auc = row.get("c1_corr_auc")
    lines.append(
        "| {seed} | {train} | {val} | {ext_acc:.4f} | {ext_auc:.4f} | {margin:.4f} | {c1_acc} | {c1_auc} |".format(
            seed=int(row.get("seed", 0)),
            train=int(row.get("num_train_pairs", 0)),
            val=int(row.get("num_val_pairs", 0)),
            ext_acc=float(row.get("ext_pair_acc", 0.0)),
            ext_auc=float(row.get("ext_auc", 0.0)),
            margin=float(row.get("ext_mean_margin", 0.0)),
            c1_acc=("{:.4f}".format(float(c1_acc)) if c1_acc is not None else "N/A"),
            c1_auc=("{:.4f}".format(float(c1_auc)) if c1_auc is not None else "N/A"),
        )
    )
lines.append("")

pair_acc_values = [float(row.get("ext_pair_acc", 0.0)) for row in rows]
auc_values = [float(row.get("ext_auc", 0.0)) for row in rows]
pair_mean = statistics.mean(pair_acc_values)
auc_mean = statistics.mean(auc_values)
pair_std = statistics.pstdev(pair_acc_values) if len(pair_acc_values) > 1 else 0.0
auc_std = statistics.pstdev(auc_values) if len(auc_values) > 1 else 0.0

pass_pair = pair_mean >= pass_pair_acc_min
pass_auc = auc_mean >= pass_auc_min
pass_std = pair_std <= pass_std_max and auc_std <= pass_std_max
overall_pass = bool(pass_pair and pass_auc and pass_std)

lines.extend(
    [
        "## Aggregated Gate",
        "",
        f"- mean_ext_pair_acc: `{pair_mean:.6f}` (threshold `{pass_pair_acc_min:.2f}`)",
        f"- mean_ext_auc: `{auc_mean:.6f}` (threshold `{pass_auc_min:.2f}`)",
        f"- std_ext_pair_acc: `{pair_std:.6f}` (threshold `{pass_std_max:.2f}`)",
        f"- std_ext_auc: `{auc_std:.6f}` (threshold `{pass_std_max:.2f}`)",
        f"- gate_pass: `{overall_pass}`",
        "",
    ]
)

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
}

run_one_seed() {
  local seed="$1"
  CURRENT_STAGE="seed_${seed}_setup"
  local group_tag
  group_tag="$(echo "$ACTIVE_PHASE_D6T_GROUP" | tr '[:upper:]' '[:lower:]')"

  local pair_run_name="${RUN_PREFIX}_${group_tag}_s${seed}_pairs"
  local c2_run_name="${RUN_PREFIX}_${group_tag}_s${seed}_c2"
  local ext_eval_name="${RUN_PREFIX}_${group_tag}_s${seed}_ext_eval"
  local c1_eval_name="${RUN_PREFIX}_${group_tag}_s${seed}_c1_eval"

  local prep_args=(
    -u scripts/phase_d_prepare_external_pairs.py
    --run-name "$pair_run_name"
    --output-root "$PAIR_OUTPUT_ROOT"
    --seed "$seed"
    --validation-ratio "${VALIDATION_RATIO:-0.1}"
    --build-step-converted
    --max-pairs-per-source "$MAX_PAIRS_PER_SOURCE"
    --max-pairs-total "$MAX_PAIRS_TOTAL"
    --min-pair-confidence "$MIN_PAIR_CONFIDENCE"
    --min-chars "$MIN_CHARS"
    --max-length-ratio "$MAX_LENGTH_RATIO"
    --max-token-overlap "$MAX_TOKEN_OVERLAP"
    --max-pairs-per-sample "$MAX_PAIRS_PER_SAMPLE"
  )
  if [[ "$USE_MATH_SHEPHERD" -eq 1 ]]; then
    prep_args+=(--math-shepherd-path "$MATH_SHEPHERD_PATH")
  fi
  if [[ "$USE_PRM800K" -eq 1 ]]; then
    prep_args+=(--prm800k-path "$PRM800K_PATH")
  fi
  append_extra_args prep_args "$GROUP_PAIR_PREP_EXTRA_DEFAULT"
  append_extra_args prep_args "$PAIR_PREP_EXTRA_ARGS"

  CURRENT_STAGE="seed_${seed}_prepare_pairs"
  {
    log_line "[seed=${seed}] Prepare external triplet pairs"
    log_line "[seed=${seed}] Command: $PYTHON_BIN ${prep_args[*]}"
  } | tee -a "$SUITE_LOG_FILE" >/dev/null
  "$PYTHON_BIN" "${prep_args[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

  local pair_run_dir=""
  pair_run_dir="$(latest_dir_for_prefix "${PAIR_OUTPUT_ROOT}/${pair_run_name}__*")"
  local pair_train_jsonl="${pair_run_dir}/train_pairs.jsonl"
  local pair_val_jsonl="${pair_run_dir}/validation_pairs.jsonl"
  require_file "$pair_train_jsonl" "train_pairs.jsonl"
  require_file "$pair_val_jsonl" "validation_pairs.jsonl"

  local train_args=(
    -u scripts/phase_b_train_value.py
    --train-dir "$PHASE_C_TRAIN_DIR"
    --eval-dir "$PHASE_C_EVAL_DIR"
    --run-name "$c2_run_name"
    --output-root "$C2_OUTPUT_ROOT"
    --target-source "${TARGET_SOURCE:-q_mean_smoothed}"
    --target-source-missing-policy "${TARGET_SOURCE_MISSING_POLICY:-fail}"
    --require-cuda
    --dtype "${C2_DTYPE:-bfloat16}"
    --device-map "${C2_DEVICE_MAP:-auto}"
    --learning-rate "$C2_LR"
    --num-train-epochs "$C2_EPOCHS"
    --per-device-train-batch-size "$C2_TRAIN_BATCH_SIZE"
    --per-device-eval-batch-size "$C2_EVAL_BATCH_SIZE"
    --seed "$seed"
    --external-pair-jsonl "$pair_train_jsonl"
    --external-pair-weight "$EXTERNAL_PAIR_WEIGHT"
    --external-pair-source-balance "$EXTERNAL_PAIR_SOURCE_BALANCE"
    --external-pair-min-confidence "$MIN_PAIR_CONFIDENCE"
    --external-pair-use-confidence-weights
    --train-mode "$C2_TRAIN_MODE"
    --calibration-loss "$C2_CALIBRATION_LOSS"
    --checkpoint-selection-metric "${CHECKPOINT_SELECTION_METRIC:-corr_auc}"
    --posthoc-calibration "${POSTHOC_CALIBRATION:-none}"
    --feature-cache-root "$FEATURE_CACHE_ROOT"
    --feature-cache-mode "$FEATURE_CACHE_MODE"
    --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC"
  )
  if [[ "$EXTERNAL_PAIR_ONLY" -eq 1 ]]; then
    train_args+=(--external-pair-only)
  fi
  append_extra_args train_args "$GROUP_C2_TRAIN_EXTRA_DEFAULT"
  append_extra_args train_args "$C2_TRAIN_EXTRA_ARGS"

  CURRENT_STAGE="seed_${seed}_train_c2"
  {
    log_line "[seed=${seed}] Train C2 triplet branch"
    log_line "[seed=${seed}] Command: $PYTHON_BIN ${train_args[*]}"
  } | tee -a "$SUITE_LOG_FILE" >/dev/null
  "$PYTHON_BIN" "${train_args[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

  local c2_run_dir=""
  c2_run_dir="$(latest_dir_for_prefix "${C2_OUTPUT_ROOT}/${c2_run_name}_*")"
  require_file "${c2_run_dir}/manifest.json" "c2 manifest"

  local ext_eval_args=(
    -u scripts/phase_d_eval_external_pairs.py
    --value-run-dir "$c2_run_dir"
    --external-pair-jsonl "$pair_val_jsonl"
    --run-name "$ext_eval_name"
    --output-root "$D6T_EVAL_OUTPUT_ROOT"
    --checkpoint-name "${EXT_EVAL_CHECKPOINT_NAME:-best}"
    --batch-size "$C2_EVAL_BATCH_SIZE"
    --require-cuda
    --feature-cache-root "$FEATURE_CACHE_ROOT"
    --feature-cache-mode "$FEATURE_CACHE_MODE"
    --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC"
  )
  append_extra_args ext_eval_args "$D6T_EXT_EVAL_EXTRA_ARGS"

  CURRENT_STAGE="seed_${seed}_eval_external"
  {
    log_line "[seed=${seed}] Evaluate external held-out pairs"
    log_line "[seed=${seed}] Command: $PYTHON_BIN ${ext_eval_args[*]}"
  } | tee -a "$SUITE_LOG_FILE" >/dev/null
  "$PYTHON_BIN" "${ext_eval_args[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

  local ext_eval_run_dir=""
  ext_eval_run_dir="$(latest_dir_for_prefix "${D6T_EVAL_OUTPUT_ROOT}/${ext_eval_name}_*")"
  require_file "${ext_eval_run_dir}/metrics.json" "external eval metrics"

  local c1_eval_run_dir=""
  if [[ "$RUN_C1_STANDALONE_EVAL" -eq 1 ]]; then
    local c1_eval_args=(
      -u scripts/phase_b_eval_faithfulness.py
      --value-run-dir "$c2_run_dir"
      --eval-dir "$PHASE_C_EVAL_DIR"
      --run-name "$c1_eval_name"
      --output-root "${C1_EVAL_OUTPUT_ROOT:-assets/artifacts/phase_c_eval}"
      --checkpoint-name best
      --target-source from_run
      --target-source-missing-policy from_run
      --posthoc-calibration none
      --feature-cache-root "$FEATURE_CACHE_ROOT"
      --feature-cache-mode "$FEATURE_CACHE_MODE"
      --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC"
    )
    append_extra_args c1_eval_args "$C1_EVAL_EXTRA_ARGS"
    CURRENT_STAGE="seed_${seed}_eval_c1"
    {
      log_line "[seed=${seed}] Optional C1 standalone eval"
      log_line "[seed=${seed}] Command: $PYTHON_BIN ${c1_eval_args[*]}"
    } | tee -a "$SUITE_LOG_FILE" >/dev/null
    "$PYTHON_BIN" "${c1_eval_args[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"
    c1_eval_run_dir="$(latest_dir_for_prefix "${C1_EVAL_OUTPUT_ROOT:-assets/artifacts/phase_c_eval}/${c1_eval_name}_*")"
    require_file "${c1_eval_run_dir}/metrics.json" "c1 standalone eval metrics"
  fi

  CURRENT_STAGE="seed_${seed}_append_result"
  append_seed_result "$seed" "$pair_run_dir" "$c2_run_dir" "$ext_eval_run_dir" "$c1_eval_run_dir"
}

main() {
  mkdir -p "$LOG_ROOT"
  : > "$SUITE_LOG_FILE"
  : > "$SEED_RESULTS_JSONL"

  resolve_group
  auto_resolve_phase_c_dirs
  if [[ "$USE_MATH_SHEPHERD" -eq 1 ]]; then
    require_file "$MATH_SHEPHERD_PATH" "Math-Shepherd JSONL"
  fi
  if [[ "$USE_PRM800K" -eq 1 ]]; then
    if [[ ! -e "$PRM800K_PATH" ]]; then
      echo "ERROR: PRM800K path required by group but not found: $PRM800K_PATH" >&2
      echo "Hint: set PRM800K_PATH to your local PRM800K root/file." >&2
      exit 1
    fi
  fi

  {
    log_line "Phase D6-T Triplet Validation Suite"
    log_line "group_id=${ACTIVE_PHASE_D6T_GROUP}"
    log_line "group_title=${GROUP_TITLE}"
    log_line "group_intention=${GROUP_INTENTION}"
    log_line "group_observe=${GROUP_OBSERVE}"
    log_line "group_expect=${GROUP_EXPECT}"
    log_line "run_prefix=${RUN_PREFIX}"
    log_line "phase_c_train_dir=${PHASE_C_TRAIN_DIR}"
    log_line "phase_c_eval_dir=${PHASE_C_EVAL_DIR}"
    log_line "seeds=${SEEDS[*]}"
    log_line "feature_cache_root=${FEATURE_CACHE_ROOT}"
    log_line "feature_cache_mode=${FEATURE_CACHE_MODE}"
  } | tee -a "$SUITE_LOG_FILE"

  local seed
  for seed in "${SEEDS[@]}"; do
    run_one_seed "$seed"
  done

  CURRENT_STAGE="final_summary"
  render_final_summary | tee "$SUMMARY_FILE"
  log_line "Final summary written: $SUMMARY_FILE" | tee -a "$SUITE_LOG_FILE"
  log_line "Suite log written: $SUITE_LOG_FILE" | tee -a "$SUITE_LOG_FILE"
}

main "$@"
