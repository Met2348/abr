#!/usr/bin/env bash
# Phase E PBR40 semantic-consensus formal suite.
#
# English
# -------
# This suite turns the new semantic-consensus pair pool into a reproducible
# formal experiment contract.  It is intentionally narrower than the general
# Phase E registry: the question here is not "which knob exists", but
# "does the repaired pool actually beat the current frontier references?"
#
# 中文
# ----
# 这个 suite 把新的 semantic-consensus pair pool 收束成一条正式可复现的实验链。
# 它比通用 Phase E registry 更窄，重点不是“还能调哪些旋钮”，而是
# “这个修过的数据池到底能不能实打实超过当前 frontier 参考线”。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACTIVE_PHASE_E_PBR40_MODE="${ACTIVE_PHASE_E_PBR40_MODE:-all}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_pbr40_semcons_formal}"
MODEL_PATH="${MODEL_PATH:-assets/models/Qwen2.5-Math-PRM-7B}"
PAIR_ARTIFACT_DIR="${PAIR_ARTIFACT_DIR:-assets/artifacts/phase_e_pairs/phase_e_pbr40_semantic_consensus_v1_20260312T092021Z}"
VALUE_OUTPUT_ROOT="${VALUE_OUTPUT_ROOT:-assets/artifacts/phase_e_runs}"
SAMEFAMILY_OUTPUT_ROOT="${SAMEFAMILY_OUTPUT_ROOT:-assets/artifacts/phase_e_samefamily_eval}"
BENCH_OUTPUT_ROOT="${BENCH_OUTPUT_ROOT:-assets/artifacts/phase_e_eval}"
PROMOTION_OUTPUT_ROOT="${PROMOTION_OUTPUT_ROOT:-assets/artifacts/phase_e_rl_promotion_diag}"
IMPLICIT_OUTPUT_ROOT="${IMPLICIT_OUTPUT_ROOT:-assets/artifacts/phase_f_implicit_prm_v2}"

REFERENCE_PBR26_VALUE_RUN_DIR="${REFERENCE_PBR26_VALUE_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_pbr26_dpo_ms_full_s42_value_20260311T134542Z}"
REFERENCE_PBR26_SAMEFAMILY_DIR="${REFERENCE_PBR26_SAMEFAMILY_DIR:-assets/artifacts/phase_e_samefamily_eval/pbr26_samefamily_verify_0312_20260311T165945Z}"
REFERENCE_PBR26_PB_GSM_DIR="${REFERENCE_PBR26_PB_GSM_DIR:-assets/artifacts/phase_e_eval/pbr26_dpo_ms_full_gsm8k_eval_20260311T140510Z}"
REFERENCE_PBR26_PB_MATH_DIR="${REFERENCE_PBR26_PB_MATH_DIR:-assets/artifacts/phase_e_eval/pbr26_dpo_ms_full_math_fulleval_0312_20260311T140557Z}"
REFERENCE_PBR31_VALUE_RUN_DIR="${REFERENCE_PBR31_VALUE_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_pbr31_lora_mathprm_pbr12data_s42_20260311T150316Z}"
REFERENCE_PBR31_SAMEFAMILY_DIR="${REFERENCE_PBR31_SAMEFAMILY_DIR:-assets/artifacts/phase_e_samefamily_eval/pbr31_samefamily_verify_0312_20260311T170244Z}"
REFERENCE_PBR31_PB_GSM_DIR="${REFERENCE_PBR31_PB_GSM_DIR:-assets/artifacts/phase_e_eval/pbr31_verify_gsm_0312_20260311T170309Z}"
REFERENCE_PBR31_PB_MATH_DIR="${REFERENCE_PBR31_PB_MATH_DIR:-assets/artifacts/phase_e_eval/pbr31_verify_math_0312_20260311T170630Z}"
REFERENCE_PBR32_VALUE_RUN_DIR="${REFERENCE_PBR32_VALUE_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_pbr32_lora_mathprm_alllayers_pbr12data_s42_20260311T152656Z}"
REFERENCE_PBR32_SAMEFAMILY_DIR="${REFERENCE_PBR32_SAMEFAMILY_DIR:-}"
REFERENCE_PBR32_PB_GSM_DIR="${REFERENCE_PBR32_PB_GSM_DIR:-assets/artifacts/phase_e_eval/pbr32_lora_mathprm_alllayers_pb_gsm_20260311T171442Z}"
REFERENCE_PBR32_PB_MATH_DIR="${REFERENCE_PBR32_PB_MATH_DIR:-assets/artifacts/phase_e_eval/pbr32_lora_mathprm_alllayers_pb_math_20260311T171442Z}"

FROZEN_MAX_GPU_MEMORY_GIB="${FROZEN_MAX_GPU_MEMORY_GIB:-48}"
EVAL_MAX_GPU_MEMORY_GIB="${EVAL_MAX_GPU_MEMORY_GIB:-48}"
BENCH_F1_POLICY="${BENCH_F1_POLICY:-oracle_sweep}"
BENCH_F1_THRESHOLD="${BENCH_F1_THRESHOLD:-}"
BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
SAMEFAMILY_BATCH_SIZE="${SAMEFAMILY_BATCH_SIZE:-64}"
BENCH_BATCH_SIZE="${BENCH_BATCH_SIZE:-48}"
BENCH_MAX_LENGTH="${BENCH_MAX_LENGTH:-1024}"
IMPLICIT_MAX_SAMPLES="${IMPLICIT_MAX_SAMPLES:-}"
IMPLICIT_BETA="${IMPLICIT_BETA:-0.5}"
IMPLICIT_FIXED_THRESHOLD="${IMPLICIT_FIXED_THRESHOLD:-0.0}"

LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"
RESULTS_JSONL="${LOG_ROOT}/results.jsonl"
IMPLICIT_RESULTS_JSONL="${LOG_ROOT}/implicit_results.jsonl"
CURRENT_STAGE="bootstrap"

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
    echo "# Phase E PBR40 Semantic-Consensus Suite"
    echo
    echo "- status: failed"
    echo "- failed_stage: ${CURRENT_STAGE}"
    echo "- suite_log_file: ${SUITE_LOG_FILE}"
  } > "$SUMMARY_FILE"
}
trap on_exit EXIT

latest_timestamped_dir() {
  local prefix="$1"
  "$PYTHON_BIN" - "$prefix" <<'PY'
from pathlib import Path
import sys
prefix = Path(sys.argv[1])
matches = sorted(prefix.parent.glob(prefix.name + "_*"))
if not matches:
    raise SystemExit(f"No timestamped artifact matches prefix: {prefix}")
print(matches[-1])
PY
}

run_cmd() {
  CURRENT_STAGE="$1"
  shift
  log_line "RUN: $*" | tee -a "$SUITE_LOG_FILE" >&2
  "$@" | tee -a "$SUITE_LOG_FILE" >&2
}

append_benchmark_args() {
  local -n target_ref="$1"
  target_ref+=(--processbench-f1-threshold-policy "$BENCH_F1_POLICY")
  if [[ -n "${BENCH_F1_THRESHOLD:-}" ]]; then
    target_ref+=(--processbench-f1-threshold "$BENCH_F1_THRESHOLD")
  fi
  if [[ -n "${BENCH_MAX_SAMPLES:-}" ]]; then
    target_ref+=(--max-samples "$BENCH_MAX_SAMPLES")
  fi
}

train_frozen_case() {
  local case_id="$1"
  local run_name="${RUN_PREFIX}_${case_id}_value"
  run_cmd "train_${case_id}" \
    "$PYTHON_BIN" -u scripts/phase_e_train_value.py \
    --train-pairs-jsonl "${PAIR_ARTIFACT_DIR}/train_pairs.jsonl" \
    --eval-pairs-jsonl "${PAIR_ARTIFACT_DIR}/validation_pairs.jsonl" \
    --model-path "$MODEL_PATH" \
    --run-name "$run_name" \
    --output-root "$VALUE_OUTPUT_ROOT" \
    --objective-mode joint \
    --learning-rate 5e-5 \
    --num-train-epochs 5 \
    --per-device-train-batch-size 96 \
    --per-device-eval-batch-size 96 \
    --max-length 1024 \
    --lambda-ranking 1.0 \
    --lambda-bce 0.5 \
    --terminal-bce-lambda 0.25 \
    --ranking-margin 0.1 \
    --ranking-target-space score \
    --pair-weight-mode confidence \
    --source-balance none \
    --permutation-mode stable_hash \
    --checkpoint-selection-metric pair_acc \
    --recipe-risk-policy error \
    --head-architecture mlp \
    --head-mlp-hidden-size 512 \
    --head-dropout-prob 0.1 \
    --head-init-std 0.02 \
    --head-activation gelu \
    --anti-saturation-weight 0.0 \
    --anti-saturation-logit-threshold 4.0 \
    --feature-cache-mode read_write \
    --max-gpu-memory-gib "$FROZEN_MAX_GPU_MEMORY_GIB" \
    --require-cuda \
    --seed 42
  latest_timestamped_dir "${VALUE_OUTPUT_ROOT}/${run_name}"
}

train_lora_case() {
  local case_id="$1"
  local run_name="${RUN_PREFIX}_${case_id}_value"
  run_cmd "train_${case_id}" \
    "$PYTHON_BIN" -u scripts/phase_e_train_value_lora.py \
    --train-pairs-jsonl "${PAIR_ARTIFACT_DIR}/train_pairs.jsonl" \
    --eval-pairs-jsonl "${PAIR_ARTIFACT_DIR}/validation_pairs.jsonl" \
    --model-path "$MODEL_PATH" \
    --run-name "$run_name" \
    --output-root "$VALUE_OUTPUT_ROOT" \
    --objective-mode joint \
    --learning-rate 3e-5 \
    --num-train-epochs 5 \
    --per-device-train-batch-size 4 \
    --per-device-eval-batch-size 4 \
    --gradient-accumulation-steps 24 \
    --max-length 1024 \
    --lambda-ranking 1.0 \
    --lambda-bce 0.5 \
    --terminal-bce-lambda 0.25 \
    --ranking-margin 0.02 \
    --ranking-target-space score \
    --pair-weight-mode none \
    --source-balance uniform \
    --permutation-mode stable_hash \
    --checkpoint-selection-metric pair_acc \
    --recipe-risk-policy error \
    --head-architecture mlp \
    --head-mlp-hidden-size 512 \
    --head-dropout-prob 0.05 \
    --head-init-std 0.02 \
    --head-activation gelu \
    --anti-saturation-weight 5e-4 \
    --anti-saturation-logit-threshold 3.5 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-dropout 0.05 \
    --lora-target-modules q_proj,v_proj \
    --lora-top-k-layers 0 \
    --gradient-checkpointing \
    --require-cuda \
    --seed 42
  latest_timestamped_dir "${VALUE_OUTPUT_ROOT}/${run_name}"
}

eval_samefamily() {
  local case_id="$1"
  local value_run_dir="$2"
  local run_name="${RUN_PREFIX}_${case_id}_samefamily"
  run_cmd "samefamily_${case_id}" \
    "$PYTHON_BIN" -u scripts/phase_e_eval_samefamily_trust.py \
    --value-run-dir "$value_run_dir" \
    --eval-pairs-jsonl "${PAIR_ARTIFACT_DIR}/validation_pairs.jsonl" \
    --run-name "$run_name" \
    --output-root "$SAMEFAMILY_OUTPUT_ROOT" \
    --batch-size "$SAMEFAMILY_BATCH_SIZE" \
    --max-length "$BENCH_MAX_LENGTH" \
    --device-map auto \
    --max-gpu-memory-gib "$EVAL_MAX_GPU_MEMORY_GIB" \
    --feature-cache-mode read_write \
    --require-cuda
  latest_timestamped_dir "${SAMEFAMILY_OUTPUT_ROOT}/${run_name}"
}

eval_benchmark() {
  local case_id="$1"
  local bench_id="$2"
  local value_run_dir="$3"
  local run_name="${RUN_PREFIX}_${case_id}_${bench_id}"
  local cmd=(
    "$PYTHON_BIN" -u scripts/phase_e_eval_benchmark.py
    --value-run-dir "$value_run_dir"
    --benchmark-id "$bench_id"
    --run-name "$run_name"
    --output-root "$BENCH_OUTPUT_ROOT"
    --batch-size "$BENCH_BATCH_SIZE"
    --max-length "$BENCH_MAX_LENGTH"
    --device-map auto
    --max-gpu-memory-gib "$EVAL_MAX_GPU_MEMORY_GIB"
    --feature-cache-mode read_write
    --require-cuda
  )
  append_benchmark_args cmd
  run_cmd "bench_${case_id}_${bench_id}" "${cmd[@]}"
  latest_timestamped_dir "${BENCH_OUTPUT_ROOT}/${run_name}"
}

append_result_row() {
  local case_id="$1"
  local variant="$2"
  local value_run_dir="$3"
  local samefamily_dir="$4"
  local pb_gsm_dir="$5"
  local pb_math_dir="$6"
  "$PYTHON_BIN" - "$case_id" "$variant" "$value_run_dir" "$samefamily_dir" "$pb_gsm_dir" "$pb_math_dir" "$RESULTS_JSONL" <<'PY'
import json
import sys
from pathlib import Path

case_id, variant, value_run_dir, samefamily_dir, pb_gsm_dir, pb_math_dir, out_path = sys.argv[1:]
value_summary = json.loads((Path(value_run_dir) / "summary.json").read_text(encoding="utf-8"))
samefamily_summary = json.loads((Path(samefamily_dir) / "summary.json").read_text(encoding="utf-8"))
pb_gsm_summary = json.loads((Path(pb_gsm_dir) / "summary.json").read_text(encoding="utf-8"))
pb_math_summary = json.loads((Path(pb_math_dir) / "summary.json").read_text(encoding="utf-8"))

row = {
    "case_id": case_id,
    "variant": variant,
    "value_run_dir": value_run_dir,
    "samefamily_dir": samefamily_dir,
    "pb_gsm_dir": pb_gsm_dir,
    "pb_math_dir": pb_math_dir,
    "heldout_pair_acc": float(value_summary["eval_pairs"]["pair_accuracy"]),
    "heldout_auc": float(value_summary["eval_pairs"]["auc"]),
    "samefamily_top1": float(samefamily_summary["prompt_pool_top1_accuracy"]),
    "samefamily_local": float(samefamily_summary.get("local_first_bad_edge_accuracy", 0.0)),
    "gsm_auc": float(pb_gsm_summary["metrics"]["pair_auc_good_vs_bad"]),
    "gsm_first_edge": float(pb_gsm_summary["metrics"]["first_error_edge_accuracy"]),
    "gsm_f1": float(pb_gsm_summary["metrics"].get("processbench_f1", 0.0)),
    "math_auc": float(pb_math_summary["metrics"]["pair_auc_good_vs_bad"]),
    "math_first_edge": float(pb_math_summary["metrics"]["first_error_edge_accuracy"]),
    "math_f1": float(pb_math_summary["metrics"].get("processbench_f1", 0.0)),
}
with Path(out_path).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

append_implicit_row() {
  local case_id="$1"
  local bench_name="$2"
  local implicit_dir="$3"
  "$PYTHON_BIN" - "$case_id" "$bench_name" "$implicit_dir" "$IMPLICIT_RESULTS_JSONL" <<'PY'
import json
import sys
from pathlib import Path

case_id, bench_name, implicit_dir, out_path = sys.argv[1:]
summary = json.loads((Path(implicit_dir) / "summary.json").read_text(encoding="utf-8"))
fixed_metrics = dict(summary.get("metrics_fixed_threshold", {}))
oracle_metrics = dict(summary.get("metrics_oracle_sweep", {}))
row = {
    "case_id": case_id,
    "benchmark": bench_name,
    "implicit_dir": implicit_dir,
    "pair_auc": float(fixed_metrics.get("pair_auc_good_vs_bad", 0.0)),
    "first_edge": float(fixed_metrics.get("first_error_edge_accuracy", 0.0)),
    "f1_fixed": float(fixed_metrics.get("processbench_f1", 0.0)),
    "f1_oracle": float(oracle_metrics.get("processbench_f1", 0.0)),
}
with Path(out_path).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

run_rl_diag() {
  local case_id="$1"
  local samefamily_dir="$2"
  local pb_gsm_dir="$3"
  local pb_math_dir="$4"
  local run_name="${RUN_PREFIX}_${case_id}_rl_diag"
  run_cmd "rl_diag_${case_id}" \
    "$PYTHON_BIN" -u scripts/phase_e_diagnose_rl_promotion.py \
    --run-name "$run_name" \
    --output-root "$PROMOTION_OUTPUT_ROOT" \
    --audit-spec "${case_id}|math_shepherd|${samefamily_dir}|${pb_gsm_dir}|${pb_math_dir}"
  latest_timestamped_dir "${PROMOTION_OUTPUT_ROOT}/${run_name}"
}

run_implicit_eval() {
  local case_id="$1"
  local value_run_dir="$2"
  local bench_path="$3"
  local bench_name="$4"
  local run_name="${RUN_PREFIX}_${case_id}_implicit_${bench_name}"
  local cmd=(
    "$PYTHON_BIN" -u scripts/phase_f_implicit_prm_eval_v2.py
    --lora-run-dir "$value_run_dir"
    --base-model-path "$MODEL_PATH"
    --benchmark-path "$bench_path"
    --run-name "$run_name"
    --output-root "$IMPLICIT_OUTPUT_ROOT"
    --beta "$IMPLICIT_BETA"
    --fixed-threshold "$IMPLICIT_FIXED_THRESHOLD"
    --max-length "$BENCH_MAX_LENGTH"
    --device-map auto
    --max-gpu-memory-gib "$EVAL_MAX_GPU_MEMORY_GIB"
    --require-cuda
  )
  if [[ -n "${IMPLICIT_MAX_SAMPLES:-}" ]]; then
    cmd+=(--max-samples "$IMPLICIT_MAX_SAMPLES")
  fi
  run_cmd "implicit_${case_id}_${bench_name}" "${cmd[@]}"
  latest_timestamped_dir "${IMPLICIT_OUTPUT_ROOT}/${run_name}"
}

mkdir -p "$LOG_ROOT"
: > "$SUITE_LOG_FILE"
: > "$RESULTS_JSONL"
: > "$IMPLICIT_RESULTS_JSONL"

if [[ ! -f "${PAIR_ARTIFACT_DIR}/train_pairs.jsonl" || ! -f "${PAIR_ARTIFACT_DIR}/validation_pairs.jsonl" ]]; then
  echo "ERROR: PBR40 pair artifact is incomplete: ${PAIR_ARTIFACT_DIR}" >&2
  exit 1
fi

case "$ACTIVE_PHASE_E_PBR40_MODE" in
  frozen)
    FROZEN_RUN_DIR="$(train_frozen_case frozen)"
    FROZEN_SAMEFAMILY_DIR="$(eval_samefamily frozen "$FROZEN_RUN_DIR")"
    FROZEN_PB_GSM_DIR="$(eval_benchmark frozen processbench_gsm8k "$FROZEN_RUN_DIR")"
    FROZEN_PB_MATH_DIR="$(eval_benchmark frozen processbench_math "$FROZEN_RUN_DIR")"
    append_result_row frozen frozen "$FROZEN_RUN_DIR" "$FROZEN_SAMEFAMILY_DIR" "$FROZEN_PB_GSM_DIR" "$FROZEN_PB_MATH_DIR"
    run_rl_diag frozen "$FROZEN_SAMEFAMILY_DIR" "$FROZEN_PB_GSM_DIR" "$FROZEN_PB_MATH_DIR" >/dev/null
    ;;
  lora)
    LORA_RUN_DIR="$(train_lora_case lora)"
    LORA_SAMEFAMILY_DIR="$(eval_samefamily lora "$LORA_RUN_DIR")"
    LORA_PB_GSM_DIR="$(eval_benchmark lora processbench_gsm8k "$LORA_RUN_DIR")"
    LORA_PB_MATH_DIR="$(eval_benchmark lora processbench_math "$LORA_RUN_DIR")"
    append_result_row lora lora "$LORA_RUN_DIR" "$LORA_SAMEFAMILY_DIR" "$LORA_PB_GSM_DIR" "$LORA_PB_MATH_DIR"
    run_rl_diag lora "$LORA_SAMEFAMILY_DIR" "$LORA_PB_GSM_DIR" "$LORA_PB_MATH_DIR" >/dev/null
    LORA_IMPLICIT_MATH_DIR="$(run_implicit_eval lora "$LORA_RUN_DIR" assets/external_datasets/qwen_processbench/math.json math)"
    append_implicit_row lora math "$LORA_IMPLICIT_MATH_DIR"
    LORA_IMPLICIT_GSM_DIR="$(run_implicit_eval lora "$LORA_RUN_DIR" assets/external_datasets/qwen_processbench/gsm8k.json gsm8k)"
    append_implicit_row lora gsm8k "$LORA_IMPLICIT_GSM_DIR"
    ;;
  all)
    FROZEN_RUN_DIR="$(train_frozen_case frozen)"
    FROZEN_SAMEFAMILY_DIR="$(eval_samefamily frozen "$FROZEN_RUN_DIR")"
    FROZEN_PB_GSM_DIR="$(eval_benchmark frozen processbench_gsm8k "$FROZEN_RUN_DIR")"
    FROZEN_PB_MATH_DIR="$(eval_benchmark frozen processbench_math "$FROZEN_RUN_DIR")"
    append_result_row frozen frozen "$FROZEN_RUN_DIR" "$FROZEN_SAMEFAMILY_DIR" "$FROZEN_PB_GSM_DIR" "$FROZEN_PB_MATH_DIR"
    run_rl_diag frozen "$FROZEN_SAMEFAMILY_DIR" "$FROZEN_PB_GSM_DIR" "$FROZEN_PB_MATH_DIR" >/dev/null

    LORA_RUN_DIR="$(train_lora_case lora)"
    LORA_SAMEFAMILY_DIR="$(eval_samefamily lora "$LORA_RUN_DIR")"
    LORA_PB_GSM_DIR="$(eval_benchmark lora processbench_gsm8k "$LORA_RUN_DIR")"
    LORA_PB_MATH_DIR="$(eval_benchmark lora processbench_math "$LORA_RUN_DIR")"
    append_result_row lora lora "$LORA_RUN_DIR" "$LORA_SAMEFAMILY_DIR" "$LORA_PB_GSM_DIR" "$LORA_PB_MATH_DIR"
    run_rl_diag lora "$LORA_SAMEFAMILY_DIR" "$LORA_PB_GSM_DIR" "$LORA_PB_MATH_DIR" >/dev/null
    LORA_IMPLICIT_MATH_DIR="$(run_implicit_eval lora "$LORA_RUN_DIR" assets/external_datasets/qwen_processbench/math.json math)"
    append_implicit_row lora math "$LORA_IMPLICIT_MATH_DIR"
    LORA_IMPLICIT_GSM_DIR="$(run_implicit_eval lora "$LORA_RUN_DIR" assets/external_datasets/qwen_processbench/gsm8k.json gsm8k)"
    append_implicit_row lora gsm8k "$LORA_IMPLICIT_GSM_DIR"
    ;;
  *)
    echo "ERROR: Unknown ACTIVE_PHASE_E_PBR40_MODE=${ACTIVE_PHASE_E_PBR40_MODE}" >&2
    exit 1
    ;;
esac

"$PYTHON_BIN" - "$RESULTS_JSONL" "$IMPLICIT_RESULTS_JSONL" "$SUMMARY_FILE" \
  "$REFERENCE_PBR26_VALUE_RUN_DIR" "$REFERENCE_PBR26_SAMEFAMILY_DIR" "$REFERENCE_PBR26_PB_GSM_DIR" "$REFERENCE_PBR26_PB_MATH_DIR" \
  "$REFERENCE_PBR31_VALUE_RUN_DIR" "$REFERENCE_PBR31_SAMEFAMILY_DIR" "$REFERENCE_PBR31_PB_GSM_DIR" "$REFERENCE_PBR31_PB_MATH_DIR" \
  "$REFERENCE_PBR32_VALUE_RUN_DIR" "$REFERENCE_PBR32_SAMEFAMILY_DIR" "$REFERENCE_PBR32_PB_GSM_DIR" "$REFERENCE_PBR32_PB_MATH_DIR" <<'PY'
import json
import sys
from pathlib import Path

results_path = Path(sys.argv[1])
implicit_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
refs = [
    ("PBR26", *sys.argv[4:8]),
    ("PBR31", *sys.argv[8:12]),
    ("PBR32", *sys.argv[12:16]),
]

rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
implicit_rows = [json.loads(line) for line in implicit_path.read_text(encoding="utf-8").splitlines() if line.strip()]

def maybe_load(path_text: str):
    path_text = str(path_text).strip()
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        return None
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))

ref_rows = []
for ref_id, value_dir, samefamily_dir, pb_gsm_dir, pb_math_dir in refs:
    value_summary = maybe_load(value_dir)
    pb_gsm_summary = maybe_load(pb_gsm_dir)
    pb_math_summary = maybe_load(pb_math_dir)
    samefamily_summary = maybe_load(samefamily_dir)
    if value_summary is None or pb_gsm_summary is None or pb_math_summary is None:
        continue
    ref_rows.append({
        "case_id": ref_id,
        "variant": "reference",
        "heldout_pair_acc": float(value_summary["eval_pairs"]["pair_accuracy"]),
        "heldout_auc": float(value_summary["eval_pairs"]["auc"]),
        "samefamily_top1": (
            float(samefamily_summary["prompt_pool_top1_accuracy"])
            if samefamily_summary is not None and "prompt_pool_top1_accuracy" in samefamily_summary
            else None
        ),
        "samefamily_local": (
            float(samefamily_summary.get("local_first_bad_edge_accuracy", 0.0))
            if samefamily_summary is not None
            else None
        ),
        "gsm_auc": float(pb_gsm_summary["metrics"]["pair_auc_good_vs_bad"]),
        "gsm_first_edge": float(pb_gsm_summary["metrics"]["first_error_edge_accuracy"]),
        "gsm_f1": float(pb_gsm_summary["metrics"].get("processbench_f1", 0.0)),
        "math_auc": float(pb_math_summary["metrics"]["pair_auc_good_vs_bad"]),
        "math_first_edge": float(pb_math_summary["metrics"]["first_error_edge_accuracy"]),
        "math_f1": float(pb_math_summary["metrics"].get("processbench_f1", 0.0)),
    })

def fmt(value):
    if value is None:
        return "NA"
    return f"{float(value):.4f}"

lines = [
    "# Phase E PBR40 Semantic-Consensus Suite",
    "",
    "## New Runs",
    "",
    "| case | variant | heldout_pair | heldout_auc | sf_top1 | sf_local | gsm_auc | gsm_f1 | math_auc | math_f1 |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in sorted(rows, key=lambda item: item["case_id"]):
    lines.append(
        f"| {row['case_id']} | {row['variant']} | {fmt(row['heldout_pair_acc'])} | {fmt(row['heldout_auc'])} | "
        f"{fmt(row['samefamily_top1'])} | {fmt(row['samefamily_local'])} | {fmt(row['gsm_auc'])} | "
        f"{fmt(row['gsm_f1'])} | {fmt(row['math_auc'])} | {fmt(row['math_f1'])} |"
    )

lines.extend([
    "",
    "## Frontier References",
    "",
    "| case | heldout_pair | heldout_auc | sf_top1 | sf_local | gsm_auc | gsm_f1 | math_auc | math_f1 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
])
for row in ref_rows:
    lines.append(
        f"| {row['case_id']} | {fmt(row['heldout_pair_acc'])} | {fmt(row['heldout_auc'])} | "
        f"{fmt(row['samefamily_top1'])} | {fmt(row['samefamily_local'])} | {fmt(row['gsm_auc'])} | "
        f"{fmt(row['gsm_f1'])} | {fmt(row['math_auc'])} | {fmt(row['math_f1'])} |"
    )

if implicit_rows:
    lines.extend([
        "",
        "## Implicit PRM v2",
        "",
        "| case | benchmark | pair_auc | first_edge | f1_fixed | f1_oracle |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in sorted(implicit_rows, key=lambda item: (item["case_id"], item["benchmark"])):
        lines.append(
            f"| {row['case_id']} | {row['benchmark']} | {fmt(row['pair_auc'])} | {fmt(row['first_edge'])} | "
            f"{fmt(row['f1_fixed'])} | {fmt(row['f1_oracle'])} |"
        )

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

log_line "Suite complete. Summary: ${SUMMARY_FILE}" | tee -a "$SUITE_LOG_FILE" >&2
