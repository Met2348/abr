#!/usr/bin/env bash
# Phase E deeper R-PRM diagnostic suite.
#
# Why this file exists
# --------------------
# The first dedicated R-PRM length sweep already established:
# 1. compact `R-PRM` becomes truncation-clean at `1536`,
# 2. longer length helps,
# 3. but the source still remains far below `ACC90`.
#
# This wrapper therefore diagnoses the next layer:
# 1. raw compact-contract yield and verdict balance,
# 2. objective geometry (`score` vs `logit`),
# 3. head capacity (`linear` vs `mlp`),
# 4. auxiliary loss (`ranking_only` vs `joint`),
# 5. residual length benefit (`1536` vs `2048`).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ACTIVE_PHASE_E_RPRM_DEEP_GROUP="${ACTIVE_PHASE_E_RPRM_DEEP_GROUP:-RD2_RPRM_DEEP_DIAG_SMOKE}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_rprm_deep_diag}"
LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"
RESULTS_JSONL="${LOG_ROOT}/config_results.jsonl"
ARTIFACT_JSON="${LOG_ROOT}/artifact_paths.json"
CURRENT_STAGE="bootstrap"

GROUP_TITLE=""
GROUP_INTENTION=""
GROUP_OBSERVE=""
GROUP_EXPECT=""
PAIR_SEED=42
PAIR_SPLIT_GRANULARITY="pair_id"
PAIR_MIN_CONFIDENCE=0.75
PAIR_MAX_TOTAL=3000
PAIR_MAX_PER_SOURCE=3000
AUDIT_MAX_ROWS=3000
MODEL_PATH="${MODEL_PATH:-assets/models/Qwen2.5-7B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-assets/artifacts/phase_e_feature_cache}"
FEATURE_CACHE_MODE="${FEATURE_CACHE_MODE:-read_write}"
FEATURE_CACHE_LOCK_TIMEOUT_SEC="${FEATURE_CACHE_LOCK_TIMEOUT_SEC:-600}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
RECIPE_RISK_POLICY="${RECIPE_RISK_POLICY:-error}"
RPRM_STEP_LABEL_PAIR_MODE="${RPRM_STEP_LABEL_PAIR_MODE:-first_bad_edge_strict}"

log_line() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1"
}

on_exit() {
  local exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    return
  fi
  mkdir -p "$LOG_ROOT"
  {
    echo "# Phase E R-PRM Deep Diagnostic Summary"
    echo
    echo "- group_id: ${ACTIVE_PHASE_E_RPRM_DEEP_GROUP}"
    echo "- group_title: ${GROUP_TITLE}"
    echo "- run_prefix: ${RUN_PREFIX}"
    echo "- status: failed"
    echo "- failed_stage: ${CURRENT_STAGE}"
    echo "- suite_log_file: ${SUITE_LOG_FILE}"
  } > "$SUMMARY_FILE"
}
trap on_exit EXIT

resolve_group() {
  case "$ACTIVE_PHASE_E_RPRM_DEEP_GROUP" in
    RD2_RPRM_DEEP_DIAG_SMOKE)
      GROUP_TITLE="RD2 R-PRM Deep Diagnosis Smoke"
      GROUP_INTENTION="Diagnose the remaining R-PRM bottleneck after compact-contract truncation repair."
      GROUP_OBSERVE="Audit raw compact-contract yield, then run a controlled same-artifact config matrix that isolates head, geometry, auxiliary BCE, and residual length effects."
      GROUP_EXPECT="The suite should tell us whether current compact R-PRM is mainly limited by extraction yield, recipe geometry, head capacity, or still by longer-sequence needs."
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_E_RPRM_DEEP_GROUP=$ACTIVE_PHASE_E_RPRM_DEEP_GROUP" >&2
      exit 1
      ;;
  esac
}

length_train_batch_size() {
  local max_length="$1"
  case "$max_length" in
    1536) echo "${TRAIN_BATCH_SIZE_1536:-48}" ;;
    2048) echo "${TRAIN_BATCH_SIZE_2048:-32}" ;;
    *) echo "${TRAIN_BATCH_SIZE_DEFAULT:-32}" ;;
  esac
}

length_eval_batch_size() {
  local max_length="$1"
  case "$max_length" in
    1536) echo "${EVAL_BATCH_SIZE_1536:-64}" ;;
    2048) echo "${EVAL_BATCH_SIZE_2048:-32}" ;;
    *) echo "${EVAL_BATCH_SIZE_DEFAULT:-32}" ;;
  esac
}

run_and_capture() {
  local __resultvar="$1"
  shift
  local output
  output="$("$@" 2>&1)"
  local exit_code=$?
  printf '%s\n' "$output" | tee -a "$SUITE_LOG_FILE"
  printf -v "$__resultvar" '%s' "$output"
  return "$exit_code"
}

append_result_row() {
  local payload_json="$1"
  python - "$RESULTS_JSONL" "$payload_json" <<'PY'
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
payload = json.loads(sys.argv[2])
with out_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
PY
}

render_summary() {
  python - "$ARTIFACT_JSON" "$RESULTS_JSONL" "$SUMMARY_FILE" "$ACTIVE_PHASE_E_RPRM_DEEP_GROUP" "$GROUP_TITLE" "$RUN_PREFIX" "$SUITE_LOG_FILE" "$GROUP_INTENTION" "$GROUP_OBSERVE" "$GROUP_EXPECT" <<'PY'
import json
import sys
from pathlib import Path

artifact_path = Path(sys.argv[1])
results_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
group_id = sys.argv[4]
group_title = sys.argv[5]
run_prefix = sys.argv[6]
suite_log_file = sys.argv[7]
group_intention = sys.argv[8]
group_observe = sys.argv[9]
group_expect = sys.argv[10]

artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
rows = []
if results_path.exists():
    for raw in results_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            rows.append(json.loads(raw))

audit = json.loads(Path(artifact["audit_summary"]).read_text(encoding="utf-8"))

def fmt(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"

best_pair = max(rows, key=lambda row: float(row.get("heldout_pair_acc", 0.0))) if rows else None
best_auc = max(rows, key=lambda row: float(row.get("heldout_auc", 0.0))) if rows else None

lines = [
    "# Phase E R-PRM Deep Diagnostic Summary",
    "",
    f"- group_id: {group_id}",
    f"- group_title: {group_title}",
    f"- run_prefix: {run_prefix}",
    f"- status: ok",
    f"- suite_log_file: {suite_log_file}",
    f"- group_intention: {group_intention}",
    f"- observe: {group_observe}",
    f"- expect: {group_expect}",
    f"- pair_artifact_dir: `{artifact['pair_run_dir']}`",
    f"- audit_summary: `{artifact['audit_summary']}`",
    "",
    "## Raw Compact-Contract Audit",
    "",
    f"- raw_rows_audited: `{audit['raw_rows_audited']}`",
    f"- accepted_rows: `{audit['accepted_rows']}`",
    f"- acceptance_rate: `{audit['acceptance_rate']:.4f}`",
    f"- chosen_yes: `{int(audit['chosen_verdict_counts'].get('yes', 0))}`",
    f"- chosen_no: `{int(audit['chosen_verdict_counts'].get('no', 0))}`",
    f"- prompt_token_p95: `{int(audit['prompt_token_lengths']['p95'])}`",
    f"- first_diff_p95: `{int(audit['first_diff_token_index']['p95'])}`",
    "",
    "## Config Matrix",
    "",
    "| config_id | objective | rank_space | head | max_length | pair_acc | auc | rank_score | yes_acc | no_acc | yes_no_gap | short_q1_acc | long_q4_acc | q1_q4_gap |",
    "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        "| "
        + " | ".join(
            [
                str(row["config_id"]),
                str(row["objective_mode"]),
                str(row["ranking_target_space"]),
                str(row["head_architecture"]),
                str(int(row["max_length"])),
                fmt(row.get("heldout_pair_acc")),
                fmt(row.get("heldout_auc")),
                fmt(row.get("heldout_ranking_score")),
                fmt(row.get("yes_pair_acc")),
                fmt(row.get("no_pair_acc")),
                fmt(row.get("yes_no_gap")),
                fmt(row.get("short_q1_pair_acc")),
                fmt(row.get("long_q4_pair_acc")),
                fmt(row.get("q1_q4_gap")),
            ]
        )
    )
        + " |"
    )
    lines.append(f"Run: `{row['run_dir']}`")
lines.extend(
    [
        "",
        "## Best Configs",
        "",
        f"- best_by_pair_acc: `{best_pair['config_id'] if best_pair else 'none'}`",
        f"- best_pair_acc: `{fmt(best_pair.get('heldout_pair_acc') if best_pair else None)}`",
        f"- best_by_auc: `{best_auc['config_id'] if best_auc else 'none'}`",
        f"- best_auc: `{fmt(best_auc.get('heldout_auc') if best_auc else None)}`",
    ]
)
summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

resolve_group
mkdir -p "$LOG_ROOT"
: > "$SUITE_LOG_FILE"
: > "$RESULTS_JSONL"

{
  log_line "Phase E R-PRM Deep Diagnostic Suite"
  log_line "group_id=${ACTIVE_PHASE_E_RPRM_DEEP_GROUP}"
  log_line "group_title=${GROUP_TITLE}"
  log_line "group_intention=${GROUP_INTENTION}"
  log_line "group_observe=${GROUP_OBSERVE}"
  log_line "group_expect=${GROUP_EXPECT}"
  log_line "run_prefix=${RUN_PREFIX}"
} | tee -a "$SUITE_LOG_FILE"

CURRENT_STAGE="audit_raw_contract"
audit_cmd=(
  "$PYTHON_BIN" -u scripts/phase_e_audit_rprm_contract.py
  --r-prm-root assets/external_datasets/kevinpro_r_prm
  --split train
  --model-path "$MODEL_PATH"
  --run-name "${RUN_PREFIX}_audit"
  --output-root "${LOG_ROOT}/audit"
  --max-rows "$AUDIT_MAX_ROWS"
  --max-lengths 1024 1280 1536 2048
)
if [[ -n "$ADAPTER_PATH" ]]; then
  audit_cmd+=(--adapter-path "$ADAPTER_PATH")
fi
log_line "RUN: ${audit_cmd[*]}" | tee -a "$SUITE_LOG_FILE"
audit_output=""
run_and_capture audit_output "${audit_cmd[@]}"
AUDIT_SUMMARY="$(printf '%s\n' "$audit_output" | awk -F': ' '/^summary_json/{print $2}' | tail -n1)"
if [[ -z "$AUDIT_SUMMARY" ]]; then
  echo "ERROR: Failed to parse audit summary path" >&2
  exit 1
fi

CURRENT_STAGE="prepare_pairs"
prep_cmd=(
  "$PYTHON_BIN" -u scripts/phase_e_prepare_pairs.py
  --source-bundle r_prm_train
  --run-name "${RUN_PREFIX}_pairs"
  --output-root assets/artifacts/phase_e_pairs
  --seed "$PAIR_SEED"
  --split-granularity "$PAIR_SPLIT_GRANULARITY"
  --max-pairs-total "$PAIR_MAX_TOTAL"
  --max-pairs-per-source "$PAIR_MAX_PER_SOURCE"
  --min-pair-confidence "$PAIR_MIN_CONFIDENCE"
  --r-prm-pair-mode compact_verdict
  --step-label-pair-mode "$RPRM_STEP_LABEL_PAIR_MODE"
)
log_line "RUN: ${prep_cmd[*]}" | tee -a "$SUITE_LOG_FILE"
prep_output=""
run_and_capture prep_output "${prep_cmd[@]}"
PAIR_RUN_DIR="$(printf '%s\n' "$prep_output" | awk -F': ' '/^run_dir/{print $2}' | tail -n1)"
if [[ -z "$PAIR_RUN_DIR" ]]; then
  echo "ERROR: Failed to parse pair artifact run_dir" >&2
  exit 1
fi
TRAIN_PAIRS_JSONL="${PAIR_RUN_DIR}/train_pairs.jsonl"
VALIDATION_PAIRS_JSONL="${PAIR_RUN_DIR}/validation_pairs.jsonl"

python - "$ARTIFACT_JSON" "$PAIR_RUN_DIR" "$TRAIN_PAIRS_JSONL" "$VALIDATION_PAIRS_JSONL" "$AUDIT_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "pair_run_dir": sys.argv[2],
    "train_pairs_jsonl": sys.argv[3],
    "validation_pairs_jsonl": sys.argv[4],
    "audit_summary": sys.argv[5],
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

CONFIG_SPECS=(
  "C1_LINEAR_SCORE_1536|ranking_only|score|linear|1536"
  "C2_LINEAR_LOGIT_1536|ranking_only|logit|linear|1536"
  "C3_MLP_SCORE_1536|ranking_only|score|mlp|1536"
  "C4_MLP_LOGIT_1536|ranking_only|logit|mlp|1536"
  "C5_MLP_JOINT_SCORE_1536|joint|score|mlp|1536"
  "C6_MLP_JOINT_LOGIT_1536|joint|logit|mlp|1536"
  "C7_MLP_BCE_1536|pair_bce_only|score|mlp|1536"
  "C8_MLP_JOINT_LOGIT_2048|joint|logit|mlp|2048"
  "C9_MLP_BCE_2048|pair_bce_only|score|mlp|2048"
)

for spec in "${CONFIG_SPECS[@]}"; do
  IFS='|' read -r config_id objective_mode ranking_target_space head_architecture max_length <<<"$spec"
  CURRENT_STAGE="train_${config_id,,}"
  train_batch_size="$(length_train_batch_size "$max_length")"
  eval_batch_size="$(length_eval_batch_size "$max_length")"
  run_name="${RUN_PREFIX}_${config_id,,}"
  train_cmd=(
    "$PYTHON_BIN" -u scripts/phase_e_train_value.py
    --train-pairs-jsonl "$TRAIN_PAIRS_JSONL"
    --eval-pairs-jsonl "$VALIDATION_PAIRS_JSONL"
    --model-path "$MODEL_PATH"
    --run-name "$run_name"
    --output-root assets/artifacts/phase_e_runs
    --objective-mode "$objective_mode"
    --learning-rate 3e-5
    --num-train-epochs 6
    --per-device-train-batch-size "$train_batch_size"
    --per-device-eval-batch-size "$eval_batch_size"
    --pair-weight-mode none
    --source-balance none
    --permutation-mode stable_hash
    --checkpoint-selection-metric pair_acc
    --seed "$PAIR_SEED"
    --dtype "$DTYPE"
    --device-map "$DEVICE_MAP"
    --feature-cache-root "$FEATURE_CACHE_ROOT"
    --feature-cache-mode "$FEATURE_CACHE_MODE"
    --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC"
    --max-length "$max_length"
    --ranking-target-space "$ranking_target_space"
    --ranking-margin 0.02
    --head-architecture "$head_architecture"
    --strict-determinism
    --recipe-risk-policy "$RECIPE_RISK_POLICY"
    --require-cuda
  )
  if [[ "$objective_mode" == "joint" ]]; then
    train_cmd+=(--lambda-ranking 1.0 --lambda-bce 1.0)
  elif [[ "$objective_mode" == "pair_bce_only" ]]; then
    train_cmd+=(--lambda-ranking 0.0 --lambda-bce 1.0)
  else
    train_cmd+=(--lambda-ranking 1.0 --lambda-bce 0.0)
  fi
  if [[ "$head_architecture" == "mlp" ]]; then
    train_cmd+=(--head-dropout-prob 0.05 --head-init-std 0.02 --head-mlp-hidden-size 1024 --head-activation gelu)
  else
    train_cmd+=(--head-dropout-prob 0.0 --head-init-std 0.02 --head-activation gelu)
  fi
  if [[ -n "$ADAPTER_PATH" ]]; then
    train_cmd+=(--adapter-path "$ADAPTER_PATH")
  fi
  log_line "RUN: ${train_cmd[*]}" | tee -a "$SUITE_LOG_FILE"
  train_output=""
  run_and_capture train_output "${train_cmd[@]}"
  run_dir="$(printf '%s\n' "$train_output" | awk -F': ' '/^run_dir/{print $2}' | tail -n1)"
  summary_path="${run_dir}/summary.json"
  eval_metrics_path="${run_dir}/eval_metrics.json"
  eval_pair_scores_path="${run_dir}/eval_pair_scores.jsonl"
  ok_payload_json="$(python - "$config_id" "$objective_mode" "$ranking_target_space" "$head_architecture" "$max_length" "$train_batch_size" "$eval_batch_size" "$run_dir" "$summary_path" "$eval_metrics_path" "$eval_pair_scores_path" "$VALIDATION_PAIRS_JSONL" <<'PY'
import json
import sys
from pathlib import Path

config_id = sys.argv[1]
objective_mode = sys.argv[2]
ranking_target_space = sys.argv[3]
head_architecture = sys.argv[4]
max_length = int(sys.argv[5])
train_batch_size = int(sys.argv[6])
eval_batch_size = int(sys.argv[7])
run_dir = sys.argv[8]
summary_path = Path(sys.argv[9])
eval_metrics_path = Path(sys.argv[10])
eval_pair_scores_path = Path(sys.argv[11])
validation_pairs_path = Path(sys.argv[12])

summary = json.loads(summary_path.read_text(encoding="utf-8"))
eval_metrics = json.loads(eval_metrics_path.read_text(encoding="utf-8"))
pair_metrics = dict(eval_metrics["eval_pairs"])
pair_meta = {}
prompt_lengths = []
for raw in validation_pairs_path.read_text(encoding="utf-8").splitlines():
    raw = raw.strip()
    if not raw:
        continue
    row = json.loads(raw)
    pair_meta[row["pair_id"]] = {
        "chosen_verdict": row.get("metadata", {}).get("chosen_verdict"),
        "prompt_chars": len(str(row.get("prompt_text", ""))),
    }
    prompt_lengths.append(len(str(row.get("prompt_text", ""))))
if not prompt_lengths:
    raise ValueError("No validation prompt lengths found")
sorted_lengths = sorted(prompt_lengths)
q1 = sorted_lengths[int(round(0.25 * (len(sorted_lengths) - 1)))]
q4 = sorted_lengths[int(round(0.75 * (len(sorted_lengths) - 1)))]

yes_rows = []
no_rows = []
short_rows = []
long_rows = []
for raw in eval_pair_scores_path.read_text(encoding="utf-8").splitlines():
    raw = raw.strip()
    if not raw:
        continue
    row = json.loads(raw)
    meta = dict(pair_meta[row["pair_id"]])
    enriched = {
        "margin": float(row["margin"]),
        "chosen_verdict": meta.get("chosen_verdict"),
        "prompt_chars": int(meta.get("prompt_chars", 0)),
    }
    if enriched["chosen_verdict"] == "yes":
        yes_rows.append(enriched)
    elif enriched["chosen_verdict"] == "no":
        no_rows.append(enriched)
    if enriched["prompt_chars"] <= q1:
        short_rows.append(enriched)
    if enriched["prompt_chars"] >= q4:
        long_rows.append(enriched)

def bucket_metrics(rows):
    if not rows:
        return {"n": 0, "pair_acc": None, "mean_margin": None}
    margins = [float(item["margin"]) for item in rows]
    positives = [1.0 if float(item["margin"]) > 0.0 else 0.0 for item in rows]
    return {
        "n": int(len(rows)),
        "pair_acc": float(sum(positives) / len(positives)),
        "mean_margin": float(sum(margins) / len(margins)),
    }

yes_payload = bucket_metrics(yes_rows)
no_payload = bucket_metrics(no_rows)
short_payload = bucket_metrics(short_rows)
long_payload = bucket_metrics(long_rows)

payload = {
    "config_id": config_id,
    "objective_mode": objective_mode,
    "ranking_target_space": ranking_target_space,
    "head_architecture": head_architecture,
    "max_length": int(max_length),
    "train_batch_size": int(train_batch_size),
    "eval_batch_size": int(eval_batch_size),
    "run_dir": run_dir,
    "summary_path": str(summary_path),
    "eval_metrics_path": str(eval_metrics_path),
    "heldout_pair_acc": float(pair_metrics["pair_accuracy"]),
    "heldout_auc": float(pair_metrics["auc"]),
    "heldout_ranking_score": float(pair_metrics["ranking_score"]),
    "yes_pair_acc": yes_payload["pair_acc"],
    "no_pair_acc": no_payload["pair_acc"],
    "yes_mean_margin": yes_payload["mean_margin"],
    "no_mean_margin": no_payload["mean_margin"],
    "yes_no_gap": (
        abs(float(yes_payload["pair_acc"]) - float(no_payload["pair_acc"]))
        if yes_payload["pair_acc"] is not None and no_payload["pair_acc"] is not None
        else None
    ),
    "short_q1_pair_acc": short_payload["pair_acc"],
    "long_q4_pair_acc": long_payload["pair_acc"],
    "short_q1_mean_margin": short_payload["mean_margin"],
    "long_q4_mean_margin": long_payload["mean_margin"],
    "q1_q4_gap": (
        abs(float(short_payload["pair_acc"]) - float(long_payload["pair_acc"]))
        if short_payload["pair_acc"] is not None and long_payload["pair_acc"] is not None
        else None
    ),
    "train_over_limit": float(summary["train_truncation_diagnostics"]["overall"]["frac_pairs_over_limit"]),
}
print(json.dumps(payload, ensure_ascii=False))
PY
)"
  append_result_row "$ok_payload_json"
done

CURRENT_STAGE="render_summary"
render_summary
log_line "Summary file   : ${SUMMARY_FILE}" | tee -a "$SUITE_LOG_FILE"
log_line "Group complete" | tee -a "$SUITE_LOG_FILE"
