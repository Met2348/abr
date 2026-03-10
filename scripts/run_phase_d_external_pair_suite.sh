#!/usr/bin/env bash
# Phase D4 external-pair bootstrap suite.
#
# Why this file exists:
# - D4 introduces a new supervision source (external chosen/rejected pairs).
# - Running D4A/D4B/D4C manually is error-prone and hard to compare.
# - This suite keeps one reproducible pipeline for:
#   1) external pair preparation,
#   2) C2 training with external pair branch,
#   3) standalone eval,
#   4) final side-by-side summary.
#
# What this file does:
# 1. Resolve a D4 group (`ACTIVE_PHASE_D4_GROUP`).
# 2. Auto-locate or use provided C1 train/eval dirs.
# 3. Execute D4A / D4B / D4C stages (or a single stage).
# 4. Save logs and a Markdown summary in `assets/artifacts/phase_d_logs/<RUN_PREFIX>/`.
#
# Quick start (smoke):
#   ACTIVE_PHASE_D4_GROUP=D4ABC_STRATEGYQA_SMOKE \
#   RUN_PREFIX=phase_d4abc_smoke \
#   PHASE_C_TRAIN_DIR=assets/artifacts/phase_c_data/strategyqa/<train_dir> \
#   PHASE_C_EVAL_DIR=assets/artifacts/phase_c_data/strategyqa/<eval_dir> \
#   CUDA_VISIBLE_DEVICES=0 \
#   bash scripts/run_phase_d_external_pair_suite.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACTIVE_PHASE_D4_GROUP="${ACTIVE_PHASE_D4_GROUP:-D4ABC_STRATEGYQA_SMOKE}"
RUN_PREFIX="${RUN_PREFIX:-phase_d4_external_suite}"
STEP_LABEL_PAIR_MODE="${STEP_LABEL_PAIR_MODE:-first_bad_edge_strict}"
CURRENT_STAGE="init"

PAIR_OUTPUT_ROOT="${PAIR_OUTPUT_ROOT:-assets/artifacts/phase_d_external_pairs}"
C2_OUTPUT_ROOT="${C2_OUTPUT_ROOT:-assets/artifacts/phase_c_runs}"
C2_EVAL_OUTPUT_ROOT="${C2_EVAL_OUTPUT_ROOT:-assets/artifacts/phase_c_eval}"
LOG_ROOT="assets/artifacts/phase_d_logs/$RUN_PREFIX"
SUITE_LOG_FILE="$LOG_ROOT/suite.log"
SUMMARY_FILE="$LOG_ROOT/final_summary.md"
STAGE_RESULTS_JSONL="$LOG_ROOT/stage_results.jsonl"

GROUP_DATASET="${GROUP_DATASET:-strategyqa}"
PHASE_C_TRAIN_DIR="${PHASE_C_TRAIN_DIR:-}"
PHASE_C_EVAL_DIR="${PHASE_C_EVAL_DIR:-}"

# External data roots (local snapshots already downloaded in this repo).
R_PRM_ROOT="${R_PRM_ROOT:-assets/external_datasets/kevinpro_r_prm}"
PRMBENCH_PREVIEW_PATH="${PRMBENCH_PREVIEW_PATH:-assets/external_datasets/hitsmy_prmbench_preview/prmbench_preview.jsonl}"
MATH_SHEPHERD_PATH="${MATH_SHEPHERD_PATH:-assets/external_datasets/peiyi_math_shepherd/math-shepherd.jsonl}"
RLHFLOW_MISTRAL_ROOT="${RLHFLOW_MISTRAL_ROOT:-assets/external_datasets/rlhflow_mistral_prm}"
RLHFLOW_DEEPSEEK_PATH="${RLHFLOW_DEEPSEEK_PATH:-assets/external_datasets/rlhflow_deepseek_prm/deepseek_instruct_data.jsonl}"

# Optional append-only extra args.
PAIR_PREP_EXTRA_ARGS="${PAIR_PREP_EXTRA_ARGS:-}"
C2_TRAIN_EXTRA_ARGS="${C2_TRAIN_EXTRA_ARGS:-}"
C2_EVAL_EXTRA_ARGS="${C2_EVAL_EXTRA_ARGS:-}"

D4A_PAIR_PREP_EXTRA_ARGS="${D4A_PAIR_PREP_EXTRA_ARGS:-}"
D4B_PAIR_PREP_EXTRA_ARGS="${D4B_PAIR_PREP_EXTRA_ARGS:-}"
D4C_PAIR_PREP_EXTRA_ARGS="${D4C_PAIR_PREP_EXTRA_ARGS:-}"

D4A_C2_TRAIN_EXTRA_ARGS="${D4A_C2_TRAIN_EXTRA_ARGS:-}"
D4B_C2_TRAIN_EXTRA_ARGS="${D4B_C2_TRAIN_EXTRA_ARGS:-}"
D4C_C2_TRAIN_EXTRA_ARGS="${D4C_C2_TRAIN_EXTRA_ARGS:-}"
# Shared persistent feature-cache controls for C2 train/eval.
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-assets/artifacts/phase_c_feature_cache}"
FEATURE_CACHE_MODE="${FEATURE_CACHE_MODE:-read_write}"
FEATURE_CACHE_LOCK_TIMEOUT_SEC="${FEATURE_CACHE_LOCK_TIMEOUT_SEC:-600}"
# Eval post-hoc mode:
# - auto: use from_run when calibrator payload exists, otherwise fallback none.
# - from_run: always require saved calibrator payload.
# - none/temperature/isotonic: force that mode.
D4_EVAL_POSTHOC_MODE="${D4_EVAL_POSTHOC_MODE:-auto}"

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

latest_phase_c_data_guess() {
  local dataset="$1"
  local marker="$2"
  local latest=""
  latest="$(compgen -G "assets/artifacts/phase_c_data/${dataset}/*${marker}*__*" | sort | tail -n 1 || true)"
  printf '%s\n' "$latest"
}

latest_phase_c_run_for_name() {
  local run_name="$1"
  local latest=""
  latest="$(compgen -G "${C2_OUTPUT_ROOT}/${run_name}_*" | sort | tail -n 1 || true)"
  if [[ -z "$latest" ]]; then
    echo "ERROR: No C2 run directory found for run_name=$run_name" >&2
    return 1
  fi
  printf '%s\n' "$latest"
}

latest_phase_c_eval_for_name() {
  local run_name="$1"
  local latest=""
  latest="$(compgen -G "${C2_EVAL_OUTPUT_ROOT}/${run_name}_*" | sort | tail -n 1 || true)"
  if [[ -z "$latest" ]]; then
    echo "ERROR: No C2 eval directory found for run_name=$run_name" >&2
    return 1
  fi
  printf '%s\n' "$latest"
}

latest_pair_run_dir_for_name() {
  local run_name="$1"
  local latest=""
  latest="$(compgen -G "${PAIR_OUTPUT_ROOT}/${run_name}__*" | sort | tail -n 1 || true)"
  if [[ -z "$latest" ]]; then
    echo "ERROR: No external pair run directory found for run_name=$run_name" >&2
    return 1
  fi
  printf '%s\n' "$latest"
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

write_failure_summary() {
  local exit_code="$1"
  mkdir -p "$(dirname "$SUMMARY_FILE")"
  cat > "$SUMMARY_FILE" <<EOM
# Phase D4 External Pair Suite Summary

- generated_at: $(date --iso-8601=seconds)
- group_id: ${ACTIVE_PHASE_D4_GROUP:-N/A}
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
  case "$ACTIVE_PHASE_D4_GROUP" in
    D4A_STRATEGYQA_SMOKE)
      GROUP_TITLE="D4A StrategyQA Smoke (Direct Pair Warm Start)"
      GROUP_INTENTION="Use direct external pairs (R-PRM + PRMBench Preview) as ranking warm start."
      STAGE_ORDER=("D4A")
      C2_EPOCHS="${C2_EPOCHS:-3}"
      C2_LR="${C2_LR:-1e-4}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-128}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-128}"
      ;;
    D4B_STRATEGYQA_SMOKE)
      GROUP_TITLE="D4B StrategyQA Smoke (Step-Converted Expansion)"
      GROUP_INTENTION="Add step-converted external pairs (Math-Shepherd + RLHFlow) with stricter filtering."
      STAGE_ORDER=("D4B")
      C2_EPOCHS="${C2_EPOCHS:-4}"
      C2_LR="${C2_LR:-1e-4}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-128}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-128}"
      ;;
    D4C_STRATEGYQA_SMOKE)
      GROUP_TITLE="D4C StrategyQA Smoke (In-Domain Stabilization)"
      GROUP_INTENTION="Combine external pairs with in-domain C1 contrastive path using conservative external weight."
      STAGE_ORDER=("D4C")
      C2_EPOCHS="${C2_EPOCHS:-5}"
      C2_LR="${C2_LR:-1e-4}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-128}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-128}"
      ;;
    D4ABC_STRATEGYQA_SMOKE)
      GROUP_TITLE="D4A+B+C StrategyQA Smoke (Full External Bootstrap Chain)"
      GROUP_INTENTION="Run D4A -> D4B -> D4C in one reproducible chain for quick comparison."
      STAGE_ORDER=("D4A" "D4B" "D4C")
      C2_EPOCHS="${C2_EPOCHS:-4}"
      C2_LR="${C2_LR:-1e-4}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-128}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-128}"
      ;;
    D4ABC_STRATEGYQA_FULL)
      GROUP_TITLE="D4A+B+C StrategyQA Full"
      GROUP_INTENTION="Promotion-oriented full-scale rerun of D4A/B/C with larger pair pools."
      STAGE_ORDER=("D4A" "D4B" "D4C")
      C2_EPOCHS="${C2_EPOCHS:-8}"
      C2_LR="${C2_LR:-1e-4}"
      C2_TRAIN_BATCH_SIZE="${C2_TRAIN_BATCH_SIZE:-192}"
      C2_EVAL_BATCH_SIZE="${C2_EVAL_BATCH_SIZE:-192}"
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_D4_GROUP=$ACTIVE_PHASE_D4_GROUP" >&2
      echo "Supported groups:" >&2
      echo "  D4A_STRATEGYQA_SMOKE" >&2
      echo "  D4B_STRATEGYQA_SMOKE" >&2
      echo "  D4C_STRATEGYQA_SMOKE" >&2
      echo "  D4ABC_STRATEGYQA_SMOKE" >&2
      echo "  D4ABC_STRATEGYQA_FULL" >&2
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
  require_file "$PHASE_C_TRAIN_DIR/rollout_targets.jsonl" "train rollout_targets"
  require_file "$PHASE_C_EVAL_DIR/prefixes.jsonl" "eval prefixes"
  require_file "$PHASE_C_EVAL_DIR/rollout_targets.jsonl" "eval rollout_targets"
}

append_stage_result() {
  local stage_id="$1"
  local pair_dir="$2"
  local c2_run_dir="$3"
  local eval_run_dir="$4"
  "$PYTHON_BIN" - "$stage_id" "$pair_dir" "$c2_run_dir" "$eval_run_dir" "$STAGE_RESULTS_JSONL" <<'PY'
import json
import sys
from pathlib import Path

stage_id = sys.argv[1]
pair_dir = Path(sys.argv[2])
c2_run_dir = Path(sys.argv[3])
eval_run_dir = Path(sys.argv[4])
out_path = Path(sys.argv[5])

pair_summary = json.loads((pair_dir / "summary.json").read_text(encoding="utf-8"))
metrics = json.loads((eval_run_dir / "metrics.json").read_text(encoding="utf-8"))

row = {
    "stage": stage_id,
    "pair_dir": str(pair_dir),
    "c2_run_dir": str(c2_run_dir),
    "eval_run_dir": str(eval_run_dir),
    "num_train_pairs": int(pair_summary.get("num_train_rows", 0)),
    "num_val_pairs": int(pair_summary.get("num_validation_rows", 0)),
    "pair_sources": pair_summary.get("train_summary", {}).get("by_source", {}),
    "pair_mean_conf": pair_summary.get("train_summary", {}).get("mean_pair_confidence", 0.0),
    "brier": metrics.get("calibration", {}).get("brier_score", None),
    "corr_pair_acc": metrics.get("corruption", {}).get("pair_accuracy", None),
    "corr_auc": metrics.get("corruption", {}).get("auc_clean_vs_corrupt", None),
}
with out_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

render_final_summary() {
  "$PYTHON_BIN" - "$STAGE_RESULTS_JSONL" "$SUMMARY_FILE" "$ACTIVE_PHASE_D4_GROUP" "$GROUP_TITLE" "$RUN_PREFIX" "$SUITE_LOG_FILE" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

rows_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
group_id = sys.argv[3]
group_title = sys.argv[4]
run_prefix = sys.argv[5]
log_file = sys.argv[6]

rows = []
if rows_path.exists():
    for raw in rows_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rows.append(json.loads(raw))

lines = []
lines.append("# Phase D4 External Pair Suite Summary")
lines.append("")
lines.append(f"- generated_at: {datetime.now(timezone.utc).isoformat()}")
lines.append(f"- group_id: {group_id}")
lines.append(f"- group_title: {group_title}")
lines.append(f"- run_prefix: {run_prefix}")
lines.append(f"- status: {'ok' if rows else 'empty'}")
lines.append(f"- suite_log_file: {log_file}")
lines.append("")

if rows:
    lines.append("## Stage Metrics")
    lines.append("")
    lines.append("| Stage | TrainPairs | ValPairs | MeanConf | CorrPairAcc | CorrAUC | Brier |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {stage} | {train} | {val} | {conf:.4f} | {acc:.4f} | {auc:.4f} | {brier:.4f} |".format(
                stage=row.get("stage", ""),
                train=int(row.get("num_train_pairs") or 0),
                val=int(row.get("num_val_pairs") or 0),
                conf=float(row.get("pair_mean_conf") or 0.0),
                acc=float(row.get("corr_pair_acc") or 0.0),
                auc=float(row.get("corr_auc") or 0.0),
                brier=float(row.get("brier") or 0.0),
            )
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for row in rows:
        lines.append(f"### {row.get('stage')}")
        lines.append(f"- pair_dir: `{row.get('pair_dir')}`")
        lines.append(f"- c2_run_dir: `{row.get('c2_run_dir')}`")
        lines.append(f"- eval_run_dir: `{row.get('eval_run_dir')}`")
        lines.append(f"- pair_sources: `{json.dumps(row.get('pair_sources', {}), ensure_ascii=False)}`")
        lines.append("")

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
}

run_one_stage() {
  local stage_id="$1"
  CURRENT_STAGE="${stage_id}_setup"

  local pair_run_name="${RUN_PREFIX}_${stage_id,,}_pairs"
  local c2_run_name="${RUN_PREFIX}_${stage_id,,}_c2"
  local c2_eval_name="${RUN_PREFIX}_${stage_id,,}_eval"

  local prep_args=(
    -u scripts/phase_d_prepare_external_pairs.py
    --run-name "$pair_run_name"
    --output-root "$PAIR_OUTPUT_ROOT"
    --seed 42
  )

  local external_weight="0.0"
  local external_source_balance="none"
  local external_min_conf="0.0"
  local train_mode="joint"
  local two_stage_ratio="0.5"

  case "$stage_id" in
    D4A)
      # D4A: direct pair warm start only.
      prep_args+=(
        --r-prm-root "$R_PRM_ROOT"
        --r-prm-split train
        --prmbench-preview-path "$PRMBENCH_PREVIEW_PATH"
        --no-build-step-converted
        --min-pair-confidence "${D4A_MIN_PAIR_CONFIDENCE:-0.0}"
      )
      if [[ "$ACTIVE_PHASE_D4_GROUP" == *"SMOKE"* ]]; then
        prep_args+=(--max-pairs-per-source "${D4A_MAX_PAIRS_PER_SOURCE:-3000}" --max-pairs-total "${D4A_MAX_PAIRS_TOTAL:-8000}")
      fi
      external_weight="${D4A_EXTERNAL_PAIR_WEIGHT:-0.20}"
      external_source_balance="${D4A_EXTERNAL_PAIR_SOURCE_BALANCE:-uniform}"
      external_min_conf="${D4A_EXTERNAL_PAIR_MIN_CONFIDENCE:-0.0}"
      append_extra_args prep_args "$D4A_PAIR_PREP_EXTRA_ARGS"
      ;;
    D4B)
      # D4B: add converted step-label pairs with stronger filtering.
      prep_args+=(
        --r-prm-root "$R_PRM_ROOT"
        --r-prm-split train
        --prmbench-preview-path "$PRMBENCH_PREVIEW_PATH"
        --math-shepherd-path "$MATH_SHEPHERD_PATH"
        --rlhflow-mistral-root "$RLHFLOW_MISTRAL_ROOT"
        --rlhflow-deepseek-path "$RLHFLOW_DEEPSEEK_PATH"
        --build-step-converted
        --min-pair-confidence "${D4B_MIN_PAIR_CONFIDENCE:-0.55}"
        --step-label-pair-mode "${STEP_LABEL_PAIR_MODE}"
        --max-length-ratio "${D4B_MAX_LENGTH_RATIO:-3.5}"
        --max-token-overlap "${D4B_MAX_TOKEN_OVERLAP:-0.99}"
      )
      if [[ "$ACTIVE_PHASE_D4_GROUP" == *"SMOKE"* ]]; then
        prep_args+=(--max-pairs-per-source "${D4B_MAX_PAIRS_PER_SOURCE:-4000}" --max-pairs-total "${D4B_MAX_PAIRS_TOTAL:-12000}")
      fi
      external_weight="${D4B_EXTERNAL_PAIR_WEIGHT:-0.12}"
      external_source_balance="${D4B_EXTERNAL_PAIR_SOURCE_BALANCE:-uniform}"
      external_min_conf="${D4B_EXTERNAL_PAIR_MIN_CONFIDENCE:-0.55}"
      append_extra_args prep_args "$D4B_PAIR_PREP_EXTRA_ARGS"
      ;;
    D4C)
      # D4C: in-domain stabilization with conservative external branch.
      prep_args+=(
        --r-prm-root "$R_PRM_ROOT"
        --r-prm-split train
        --prmbench-preview-path "$PRMBENCH_PREVIEW_PATH"
        --math-shepherd-path "$MATH_SHEPHERD_PATH"
        --rlhflow-mistral-root "$RLHFLOW_MISTRAL_ROOT"
        --rlhflow-deepseek-path "$RLHFLOW_DEEPSEEK_PATH"
        --build-step-converted
        --min-pair-confidence "${D4C_MIN_PAIR_CONFIDENCE:-0.60}"
        --step-label-pair-mode "${STEP_LABEL_PAIR_MODE}"
        --max-length-ratio "${D4C_MAX_LENGTH_RATIO:-3.0}"
        --max-token-overlap "${D4C_MAX_TOKEN_OVERLAP:-0.985}"
      )
      if [[ "$ACTIVE_PHASE_D4_GROUP" == *"SMOKE"* ]]; then
        prep_args+=(--max-pairs-per-source "${D4C_MAX_PAIRS_PER_SOURCE:-4500}" --max-pairs-total "${D4C_MAX_PAIRS_TOTAL:-14000}")
      fi
      external_weight="${D4C_EXTERNAL_PAIR_WEIGHT:-0.08}"
      external_source_balance="${D4C_EXTERNAL_PAIR_SOURCE_BALANCE:-uniform}"
      external_min_conf="${D4C_EXTERNAL_PAIR_MIN_CONFIDENCE:-0.60}"
      train_mode="two_stage"
      two_stage_ratio="${D4C_TWO_STAGE_RANKING_RATIO:-0.45}"
      append_extra_args prep_args "$D4C_PAIR_PREP_EXTRA_ARGS"
      ;;
    *)
      echo "ERROR: Unsupported stage_id=$stage_id" >&2
      return 1
      ;;
  esac

  append_extra_args prep_args "$PAIR_PREP_EXTRA_ARGS"

  CURRENT_STAGE="${stage_id}_prepare_pairs"
  {
    log_line "[$stage_id] Preparing external pairs"
    log_line "[$stage_id] Command: $PYTHON_BIN ${prep_args[*]}"
  } | tee -a "$SUITE_LOG_FILE" >/dev/null
  "$PYTHON_BIN" "${prep_args[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

  local pair_run_dir=""
  pair_run_dir="$(latest_pair_run_dir_for_name "$pair_run_name")"
  local pair_train_jsonl="$pair_run_dir/train_pairs.jsonl"
  require_file "$pair_train_jsonl" "external pair train jsonl"

  local train_args=(
    -u scripts/phase_b_train_value.py
    --train-dir "$PHASE_C_TRAIN_DIR"
    --eval-dir "$PHASE_C_EVAL_DIR"
    --run-name "$c2_run_name"
    --output-root "$C2_OUTPUT_ROOT"
    --target-source q_fused
    --target-source-missing-policy fail
    --require-cuda
    --dtype bfloat16
    --device-map auto
    --learning-rate "$C2_LR"
    --num-train-epochs "$C2_EPOCHS"
    --per-device-train-batch-size "$C2_TRAIN_BATCH_SIZE"
    --per-device-eval-batch-size "$C2_EVAL_BATCH_SIZE"
    --external-pair-jsonl "$pair_train_jsonl"
    --external-pair-weight "$external_weight"
    --external-pair-source-balance "$external_source_balance"
    --external-pair-min-confidence "$external_min_conf"
    --external-pair-use-confidence-weights
    --train-mode "$train_mode"
    --two-stage-ranking-ratio "$two_stage_ratio"
    --feature-cache-root "$FEATURE_CACHE_ROOT"
    --feature-cache-mode "$FEATURE_CACHE_MODE"
    --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC"
  )

  case "$stage_id" in
    D4A)
      append_extra_args train_args "$D4A_C2_TRAIN_EXTRA_ARGS"
      ;;
    D4B)
      append_extra_args train_args "$D4B_C2_TRAIN_EXTRA_ARGS"
      ;;
    D4C)
      append_extra_args train_args "$D4C_C2_TRAIN_EXTRA_ARGS"
      ;;
  esac
  append_extra_args train_args "$C2_TRAIN_EXTRA_ARGS"

  CURRENT_STAGE="${stage_id}_train_c2"
  {
    log_line "[$stage_id] Training C2 with external pair branch"
    log_line "[$stage_id] Command: $PYTHON_BIN ${train_args[*]}"
  } | tee -a "$SUITE_LOG_FILE" >/dev/null
  "$PYTHON_BIN" "${train_args[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

  local c2_run_dir=""
  c2_run_dir="$(latest_phase_c_run_for_name "$c2_run_name")"

  local eval_args=(
    -u scripts/phase_b_eval_faithfulness.py
    --value-run-dir "$c2_run_dir"
    --eval-dir "$PHASE_C_EVAL_DIR"
    --run-name "$c2_eval_name"
    --output-root "$C2_EVAL_OUTPUT_ROOT"
    --checkpoint-name best
    --target-source from_run
    --target-source-missing-policy from_run
    --feature-cache-root "$FEATURE_CACHE_ROOT"
    --feature-cache-mode "$FEATURE_CACHE_MODE"
    --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC"
  )
  local resolved_eval_posthoc="$D4_EVAL_POSTHOC_MODE"
  if [[ "$resolved_eval_posthoc" == "auto" ]]; then
    if [[ -f "$c2_run_dir/best_posthoc_calibration.json" || -f "$c2_run_dir/final_posthoc_calibration.json" ]]; then
      resolved_eval_posthoc="from_run"
    else
      resolved_eval_posthoc="none"
      log_line "[$stage_id] No saved posthoc payload in $c2_run_dir, fallback eval posthoc=none" | tee -a "$SUITE_LOG_FILE" >/dev/null
    fi
  fi
  eval_args+=(--posthoc-calibration "$resolved_eval_posthoc")
  append_extra_args eval_args "$C2_EVAL_EXTRA_ARGS"

  CURRENT_STAGE="${stage_id}_eval_c2"
  {
    log_line "[$stage_id] Running standalone eval"
    log_line "[$stage_id] Command: $PYTHON_BIN ${eval_args[*]}"
  } | tee -a "$SUITE_LOG_FILE" >/dev/null
  "$PYTHON_BIN" "${eval_args[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

  local eval_run_dir=""
  eval_run_dir="$(latest_phase_c_eval_for_name "$c2_eval_name")"
  require_file "$pair_run_dir/summary.json" "external pair summary"
  require_file "$eval_run_dir/metrics.json" "standalone eval metrics"

  CURRENT_STAGE="${stage_id}_append_summary"
  append_stage_result "$stage_id" "$pair_run_dir" "$c2_run_dir" "$eval_run_dir"

  {
    log_line "[$stage_id] Completed"
    log_line "[$stage_id] pair_run_dir=$pair_run_dir"
    log_line "[$stage_id] c2_run_dir=$c2_run_dir"
    log_line "[$stage_id] eval_run_dir=$eval_run_dir"
  } | tee -a "$SUITE_LOG_FILE" >/dev/null
}

main() {
  mkdir -p "$LOG_ROOT"
  : > "$SUITE_LOG_FILE"
  : > "$STAGE_RESULTS_JSONL"

  resolve_group
  auto_resolve_phase_c_dirs

  # Basic input sanity checks for external sources.
  require_dir "$R_PRM_ROOT" "R-PRM root"
  require_file "$PRMBENCH_PREVIEW_PATH" "PRMBench preview"
  if [[ " ${STAGE_ORDER[*]} " == *" D4B "* || " ${STAGE_ORDER[*]} " == *" D4C "* ]]; then
    require_file "$MATH_SHEPHERD_PATH" "Math-Shepherd"
    require_dir "$RLHFLOW_MISTRAL_ROOT" "RLHFlow mistral root"
    require_file "$RLHFLOW_DEEPSEEK_PATH" "RLHFlow deepseek"
  fi

  {
    log_line "Phase D4 External Pair Suite"
    log_line "group_id=${ACTIVE_PHASE_D4_GROUP}"
    log_line "group_title=${GROUP_TITLE}"
    log_line "group_intention=${GROUP_INTENTION}"
    log_line "run_prefix=${RUN_PREFIX}"
    log_line "step_label_pair_mode=${STEP_LABEL_PAIR_MODE}"
    log_line "phase_c_train_dir=${PHASE_C_TRAIN_DIR}"
    log_line "phase_c_eval_dir=${PHASE_C_EVAL_DIR}"
    log_line "stage_order=${STAGE_ORDER[*]}"
    log_line "pair_output_root=${PAIR_OUTPUT_ROOT}"
    log_line "c2_output_root=${C2_OUTPUT_ROOT}"
    log_line "c2_eval_output_root=${C2_EVAL_OUTPUT_ROOT}"
    log_line "feature_cache_root=${FEATURE_CACHE_ROOT}"
    log_line "feature_cache_mode=${FEATURE_CACHE_MODE}"
    log_line "feature_cache_lock_timeout_sec=${FEATURE_CACHE_LOCK_TIMEOUT_SEC}"
  } | tee -a "$SUITE_LOG_FILE"

  for stage_id in "${STAGE_ORDER[@]}"; do
    run_one_stage "$stage_id"
  done

  CURRENT_STAGE="final_summary"
  render_final_summary | tee "$SUMMARY_FILE"
  log_line "Final summary written: $SUMMARY_FILE" | tee -a "$SUITE_LOG_FILE"
  log_line "Suite log written: $SUITE_LOG_FILE" | tee -a "$SUITE_LOG_FILE"
}

main "$@"
