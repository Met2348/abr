#!/usr/bin/env bash
# Phase E pilot suite: frozen vs LoRA, with and without judge-filtered data.
#
# English
# -------
# This suite is a controlled pilot for two Phase E questions:
# 1. does loosening the frozen-backbone assumption help on the same curated pair slice?
# 2. can a local prefix-correctness judge clean up enough pair noise to improve transfer?
#
# 中文
# ----
# 这个 suite 是一个受控 pilot，专门回答 Phase E 的两个问题：
# 1. 在同一个 curated pair 切片上，放开 frozen backbone 是否有帮助？
# 2. 一个本地 prefix-correctness judge 能不能清理掉足够多的 pair 噪声，从而改善迁移？
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_backbone_judge_pilot}"
MODEL_PATH="${MODEL_PATH:-assets/models/Qwen2.5-7B-Instruct}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-assets/models/Qwen2.5-Math-7B-Instruct}"
SOURCE_PAIR_ARTIFACT="${SOURCE_PAIR_ARTIFACT:-assets/artifacts/phase_e_pairs/phase_e_processbench_research_v2_pbr2_ms_align_gated_ms_align_v1_pairs__79c6e734325c}"
PAIR_OUTPUT_ROOT="${PAIR_OUTPUT_ROOT:-assets/artifacts/phase_e_pairs}"
VALUE_OUTPUT_ROOT="${VALUE_OUTPUT_ROOT:-assets/artifacts/phase_e_runs}"
SAMEFAMILY_OUTPUT_ROOT="${SAMEFAMILY_OUTPUT_ROOT:-assets/artifacts/phase_e_samefamily_eval}"
BENCH_OUTPUT_ROOT="${BENCH_OUTPUT_ROOT:-assets/artifacts/phase_e_eval}"
TRANSFER_COMPARE_ROOT="${TRANSFER_COMPARE_ROOT:-assets/artifacts/phase_e_transfer_compare}"
PROMOTION_OUTPUT_ROOT="${PROMOTION_OUTPUT_ROOT:-assets/artifacts/phase_e_rl_promotion_diag}"
LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"
RESULTS_JSONL="${LOG_ROOT}/results.jsonl"
CURRENT_STAGE="bootstrap"

TARGET_TRAIN_PAIRS="${TARGET_TRAIN_PAIRS:-384}"
TARGET_VAL_PAIRS="${TARGET_VAL_PAIRS:-127}"
BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-64}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-8}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-96}"
JUDGE_LOGGING_BATCHES="${JUDGE_LOGGING_BATCHES:-2}"
FROZEN_HEAD_ARCH="${FROZEN_HEAD_ARCH:-mlp}"
FROZEN_EPOCHS="${FROZEN_EPOCHS:-4}"
FROZEN_TRAIN_BATCH="${FROZEN_TRAIN_BATCH:-64}"
FROZEN_EVAL_BATCH="${FROZEN_EVAL_BATCH:-96}"
FROZEN_LR="${FROZEN_LR:-3e-5}"
# Default frozen runs to no-cache because stale Phase E feature caches can
# silently poison small pilots with non-finite pooled features.
# frozen 默认关缓存，因为旧的 Phase E feature cache 可能静默污染小规模 pilot，
# 导致 pooled feature 出现 non-finite。
FROZEN_FEATURE_CACHE_MODE="${FROZEN_FEATURE_CACHE_MODE:-off}"
FROZEN_MAX_GPU_MEMORY_GIB="${FROZEN_MAX_GPU_MEMORY_GIB:-45}"
LORA_HEAD_ARCH="${LORA_HEAD_ARCH:-mlp}"
LORA_EPOCHS="${LORA_EPOCHS:-2}"
LORA_TRAIN_BATCH="${LORA_TRAIN_BATCH:-1}"
LORA_EVAL_BATCH="${LORA_EVAL_BATCH:-8}"
LORA_GRAD_ACCUM="${LORA_GRAD_ACCUM:-16}"
LORA_LR="${LORA_LR:-2e-5}"
LORA_MAX_LENGTH="${LORA_MAX_LENGTH:-768}"
# LoRA training can use a shorter max length for memory, but benchmark-facing
# evaluation should stay on the longer budget unless truncation diagnostics say otherwise.
# LoRA 训练时可以为了显存用更短上下文，但 benchmark 评测默认应保持更长预算，
# 除非 truncation diagnostics 明确表明可以缩短。
LORA_EVAL_MAX_LENGTH="${LORA_EVAL_MAX_LENGTH:-1024}"
LORA_RANK="${LORA_RANK:-4}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,v_proj,gate_proj,up_proj,down_proj}"
LORA_TOP_K_LAYERS="${LORA_TOP_K_LAYERS:-4}"

REFERENCE_PBR2_VALUE_RUN_DIR="${REFERENCE_PBR2_VALUE_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_processbench_research_v2_pbr2_ms_align_gated_value_20260311T043818Z}"
REFERENCE_PBR2_SAMEFAMILY_DIR="${REFERENCE_PBR2_SAMEFAMILY_DIR:-assets/artifacts/phase_e_samefamily_eval/phase_e_processbench_research_v2_pbr2_ms_align_gated_samefamily_20260311T043905Z}"
REFERENCE_PBR2_PB_GSM_DIR="${REFERENCE_PBR2_PB_GSM_DIR:-assets/artifacts/phase_e_eval/phase_e_processbench_research_v2_pbr2_ms_align_gated_processbench_gsm8k_20260311T043919Z}"
REFERENCE_PBR2_PB_MATH_DIR="${REFERENCE_PBR2_PB_MATH_DIR:-assets/artifacts/phase_e_eval/phase_e_processbench_research_v2_pbr2_ms_align_gated_processbench_math_20260311T043935Z}"

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
    echo "# Phase E Backbone/Judge Pilot Suite"
    echo
    echo "- status: failed"
    echo "- failed_stage: ${CURRENT_STAGE}"
    echo "- suite_log_file: ${SUITE_LOG_FILE}"
  } > "$SUMMARY_FILE"
}
trap on_exit EXIT

latest_run_dir() {
  local prefix="$1"
  "$PYTHON_BIN" - "$prefix" <<'PY'
from pathlib import Path
import sys
prefix = sys.argv[1]
matches = sorted(Path(prefix).parent.glob(Path(prefix).name + "__*"))
if not matches:
    raise SystemExit(f"No artifact directory matches prefix: {prefix}")
print(matches[-1])
PY
}

latest_timestamped_dir() {
  local prefix="$1"
  "$PYTHON_BIN" - "$prefix" <<'PY'
from pathlib import Path
import sys
prefix = sys.argv[1]
matches = sorted(Path(prefix).parent.glob(Path(prefix).name + "_*"))
if not matches:
    raise SystemExit(f"No artifact directory matches prefix: {prefix}")
print(matches[-1])
PY
}

run_cmd() {
  CURRENT_STAGE="$1"
  shift
  log_line "RUN: $*" | tee -a "$SUITE_LOG_FILE" >&2
  "$@" | tee -a "$SUITE_LOG_FILE" >&2
}

mkdir -p "$LOG_ROOT"
: > "$SUITE_LOG_FILE"
: > "$RESULTS_JSONL"

SUBSET_RUN_NAME="${RUN_PREFIX}_subset"
run_cmd subset_curate \
  "$PYTHON_BIN" -u scripts/phase_e_curate_semantic_pairs.py \
  --slice "pilot=${SOURCE_PAIR_ARTIFACT}|*|*|${TARGET_TRAIN_PAIRS}|${TARGET_VAL_PAIRS}" \
  --run-name "$SUBSET_RUN_NAME" \
  --output-root "$PAIR_OUTPUT_ROOT" \
  --seed 42
SUBSET_ARTIFACT_DIR="$(latest_run_dir "${PAIR_OUTPUT_ROOT}/${SUBSET_RUN_NAME}")"

JUDGE_RUN_NAME="${RUN_PREFIX}_judgefilter"
run_cmd judge_filter \
  "$PYTHON_BIN" -u scripts/phase_e_judge_filter_pairs.py \
  --train-pairs-jsonl "${SUBSET_ARTIFACT_DIR}/train_pairs.jsonl" \
  --eval-pairs-jsonl "${SUBSET_ARTIFACT_DIR}/validation_pairs.jsonl" \
  --model-path "$JUDGE_MODEL_PATH" \
  --run-name "$JUDGE_RUN_NAME" \
  --output-root "$PAIR_OUTPUT_ROOT" \
  --batch-size "$JUDGE_BATCH_SIZE" \
  --max-new-tokens "$JUDGE_MAX_NEW_TOKENS" \
  --logging-batches "$JUDGE_LOGGING_BATCHES" \
  --dtype bfloat16 \
  --device-map auto \
  --min-confidence 0.0
JUDGE_ARTIFACT_DIR="$(latest_run_dir "${PAIR_OUTPUT_ROOT}/${JUDGE_RUN_NAME}")"

train_frozen_case() {
  local case_id="$1"
  local train_pairs_jsonl="$2"
  local eval_pairs_jsonl="$3"
  local run_name="${RUN_PREFIX}_${case_id}_value"
  run_cmd "train_${case_id}" \
    "$PYTHON_BIN" -u scripts/phase_e_train_value.py \
    --train-pairs-jsonl "$train_pairs_jsonl" \
    --eval-pairs-jsonl "$eval_pairs_jsonl" \
    --model-path "$MODEL_PATH" \
    --run-name "$run_name" \
    --output-root "$VALUE_OUTPUT_ROOT" \
    --objective-mode joint \
    --learning-rate "$FROZEN_LR" \
    --num-train-epochs "$FROZEN_EPOCHS" \
    --per-device-train-batch-size "$FROZEN_TRAIN_BATCH" \
    --per-device-eval-batch-size "$FROZEN_EVAL_BATCH" \
    --max-length 1024 \
    --lambda-ranking 1.0 \
    --lambda-bce 1.0 \
    --ranking-margin 0.02 \
    --ranking-target-space logit \
    --pair-weight-mode confidence_semantic \
    --source-balance none \
    --permutation-mode stable_hash \
    --checkpoint-selection-metric ranking_score \
    --head-architecture "$FROZEN_HEAD_ARCH" \
    --head-mlp-hidden-size 1024 \
    --head-dropout-prob 0.05 \
    --head-init-std 0.02 \
    --head-activation gelu \
    --anti-saturation-weight 5e-4 \
    --anti-saturation-logit-threshold 3.5 \
    --feature-cache-mode "$FROZEN_FEATURE_CACHE_MODE" \
    --max-gpu-memory-gib "$FROZEN_MAX_GPU_MEMORY_GIB"
  latest_timestamped_dir "${VALUE_OUTPUT_ROOT}/${run_name}"
}

train_lora_case() {
  local case_id="$1"
  local train_pairs_jsonl="$2"
  local eval_pairs_jsonl="$3"
  local run_name="${RUN_PREFIX}_${case_id}_value"
  run_cmd "train_${case_id}" \
    "$PYTHON_BIN" -u scripts/phase_e_train_value_lora.py \
    --train-pairs-jsonl "$train_pairs_jsonl" \
    --eval-pairs-jsonl "$eval_pairs_jsonl" \
    --model-path "$MODEL_PATH" \
    --run-name "$run_name" \
    --output-root "$VALUE_OUTPUT_ROOT" \
    --objective-mode joint \
    --learning-rate "$LORA_LR" \
    --num-train-epochs "$LORA_EPOCHS" \
    --per-device-train-batch-size "$LORA_TRAIN_BATCH" \
    --per-device-eval-batch-size "$LORA_EVAL_BATCH" \
    --gradient-accumulation-steps "$LORA_GRAD_ACCUM" \
    --max-length "$LORA_MAX_LENGTH" \
    --lambda-ranking 1.0 \
    --lambda-bce 1.0 \
    --ranking-margin 0.02 \
    --ranking-target-space logit \
    --pair-weight-mode confidence_semantic \
    --source-balance none \
    --permutation-mode stable_hash \
    --checkpoint-selection-metric ranking_score \
    --head-architecture "$LORA_HEAD_ARCH" \
    --head-mlp-hidden-size 1024 \
    --head-dropout-prob 0.05 \
    --head-init-std 0.02 \
    --head-activation gelu \
    --anti-saturation-weight 5e-4 \
    --anti-saturation-logit-threshold 3.5 \
    --lora-rank "$LORA_RANK" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-dropout "$LORA_DROPOUT" \
    --lora-target-modules "$LORA_TARGET_MODULES" \
    --lora-top-k-layers "$LORA_TOP_K_LAYERS" \
    --gradient-checkpointing
  latest_timestamped_dir "${VALUE_OUTPUT_ROOT}/${run_name}"
}

eval_case() {
  local case_id="$1"
  local value_run_dir="$2"
  local eval_pairs_jsonl="$3"

  local samefamily_run_name="${RUN_PREFIX}_${case_id}_samefamily"
  run_cmd "samefamily_${case_id}" \
    "$PYTHON_BIN" -u scripts/phase_e_eval_samefamily_trust.py \
    --value-run-dir "$value_run_dir" \
    --eval-pairs-jsonl "$eval_pairs_jsonl" \
    --run-name "$samefamily_run_name" \
    --output-root "$SAMEFAMILY_OUTPUT_ROOT" \
    --batch-size 64 \
    --max-length "$LORA_EVAL_MAX_LENGTH" \
    --device-map auto
  local samefamily_dir
  samefamily_dir="$(latest_timestamped_dir "${SAMEFAMILY_OUTPUT_ROOT}/${samefamily_run_name}")"

  local pb_gsm_run_name="${RUN_PREFIX}_${case_id}_pb_gsm8k"
  run_cmd "pb_gsm_${case_id}" \
    "$PYTHON_BIN" -u scripts/phase_e_eval_benchmark.py \
    --value-run-dir "$value_run_dir" \
    --benchmark-id processbench_gsm8k \
    --run-name "$pb_gsm_run_name" \
    --output-root "$BENCH_OUTPUT_ROOT" \
    --max-samples "$BENCH_MAX_SAMPLES" \
    --batch-size 96 \
    --max-length "$LORA_EVAL_MAX_LENGTH" \
    --device-map auto
  local pb_gsm_dir
  pb_gsm_dir="$(latest_timestamped_dir "${BENCH_OUTPUT_ROOT}/${pb_gsm_run_name}")"

  local pb_math_run_name="${RUN_PREFIX}_${case_id}_pb_math"
  run_cmd "pb_math_${case_id}" \
    "$PYTHON_BIN" -u scripts/phase_e_eval_benchmark.py \
    --value-run-dir "$value_run_dir" \
    --benchmark-id processbench_math \
    --run-name "$pb_math_run_name" \
    --output-root "$BENCH_OUTPUT_ROOT" \
    --max-samples "$BENCH_MAX_SAMPLES" \
    --batch-size 96 \
    --max-length "$LORA_EVAL_MAX_LENGTH" \
    --device-map auto
  local pb_math_dir
  pb_math_dir="$(latest_timestamped_dir "${BENCH_OUTPUT_ROOT}/${pb_math_run_name}")"

  printf '%s\n%s\n%s\n' "$samefamily_dir" "$pb_gsm_dir" "$pb_math_dir"
}

append_result_row() {
  local case_id="$1"
  local variant="$2"
  local data_mode="$3"
  local value_run_dir="$4"
  local samefamily_dir="$5"
  local pb_gsm_dir="$6"
  local pb_math_dir="$7"
  "$PYTHON_BIN" - "$case_id" "$variant" "$data_mode" "$value_run_dir" "$samefamily_dir" "$pb_gsm_dir" "$pb_math_dir" "$RESULTS_JSONL" <<'PY'
import json
import sys
from pathlib import Path
case_id, variant, data_mode, value_run_dir, samefamily_dir, pb_gsm_dir, pb_math_dir, out_path = sys.argv[1:]
value_summary = json.loads((Path(value_run_dir) / 'summary.json').read_text())
samefamily_summary = json.loads((Path(samefamily_dir) / 'summary.json').read_text())
pb_gsm_summary = json.loads((Path(pb_gsm_dir) / 'summary.json').read_text())
pb_math_summary = json.loads((Path(pb_math_dir) / 'summary.json').read_text())
row = {
    'case_id': case_id,
    'variant': variant,
    'data_mode': data_mode,
    'value_run_dir': value_run_dir,
    'samefamily_dir': samefamily_dir,
    'pb_gsm_dir': pb_gsm_dir,
    'pb_math_dir': pb_math_dir,
    'heldout_pair_acc': float(value_summary['eval_pairs']['pair_accuracy']),
    'heldout_auc': float(value_summary['eval_pairs']['auc']),
    'samefamily_top1': float(samefamily_summary['prompt_pool_top1_accuracy']),
    'samefamily_local': float(samefamily_summary.get('local_first_bad_edge_accuracy', 0.0)),
    'gsm_auc': float(pb_gsm_summary['metrics']['pair_auc_good_vs_bad']),
    'gsm_first_edge': float(pb_gsm_summary['metrics']['first_error_edge_accuracy']),
    'gsm_terminal_top1': float(pb_gsm_summary['metrics'].get('mean_all_correct_last_score', 0.0)),
    'math_auc': float(pb_math_summary['metrics']['pair_auc_good_vs_bad']),
    'math_first_edge': float(pb_math_summary['metrics']['first_error_edge_accuracy']),
    'math_terminal_top1': float(pb_math_summary['metrics'].get('mean_all_correct_last_score', 0.0)),
}
with Path(out_path).open('a', encoding='utf-8') as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
PY
}

FROZEN_BASE_RUN_DIR="$(train_frozen_case base_frozen "${SUBSET_ARTIFACT_DIR}/train_pairs.jsonl" "${SUBSET_ARTIFACT_DIR}/validation_pairs.jsonl")"
mapfile -t _base_frozen_eval < <(eval_case base_frozen "$FROZEN_BASE_RUN_DIR" "${SUBSET_ARTIFACT_DIR}/validation_pairs.jsonl")
append_result_row base_frozen frozen raw "$FROZEN_BASE_RUN_DIR" "${_base_frozen_eval[0]}" "${_base_frozen_eval[1]}" "${_base_frozen_eval[2]}"

LORA_BASE_RUN_DIR="$(train_lora_case base_lora "${SUBSET_ARTIFACT_DIR}/train_pairs.jsonl" "${SUBSET_ARTIFACT_DIR}/validation_pairs.jsonl")"
mapfile -t _base_lora_eval < <(eval_case base_lora "$LORA_BASE_RUN_DIR" "${SUBSET_ARTIFACT_DIR}/validation_pairs.jsonl")
append_result_row base_lora lora raw "$LORA_BASE_RUN_DIR" "${_base_lora_eval[0]}" "${_base_lora_eval[1]}" "${_base_lora_eval[2]}"

FROZEN_JUDGE_RUN_DIR="$(train_frozen_case judge_frozen "${JUDGE_ARTIFACT_DIR}/train_pairs.jsonl" "${JUDGE_ARTIFACT_DIR}/validation_pairs.jsonl")"
mapfile -t _judge_frozen_eval < <(eval_case judge_frozen "$FROZEN_JUDGE_RUN_DIR" "${JUDGE_ARTIFACT_DIR}/validation_pairs.jsonl")
append_result_row judge_frozen frozen judge_filtered "$FROZEN_JUDGE_RUN_DIR" "${_judge_frozen_eval[0]}" "${_judge_frozen_eval[1]}" "${_judge_frozen_eval[2]}"

LORA_JUDGE_RUN_DIR="$(train_lora_case judge_lora "${JUDGE_ARTIFACT_DIR}/train_pairs.jsonl" "${JUDGE_ARTIFACT_DIR}/validation_pairs.jsonl")"
mapfile -t _judge_lora_eval < <(eval_case judge_lora "$LORA_JUDGE_RUN_DIR" "${JUDGE_ARTIFACT_DIR}/validation_pairs.jsonl")
append_result_row judge_lora lora judge_filtered "$LORA_JUDGE_RUN_DIR" "${_judge_lora_eval[0]}" "${_judge_lora_eval[1]}" "${_judge_lora_eval[2]}"

run_cmd compare_math \
  "$PYTHON_BIN" -u scripts/phase_e_compare_processbench_transfer.py \
  --run-name "${RUN_PREFIX}_math_compare" \
  --output-root "$TRANSFER_COMPARE_ROOT" \
  --case "ref_pbr2=${REFERENCE_PBR2_VALUE_RUN_DIR}::${REFERENCE_PBR2_PB_MATH_DIR}" \
  --case "base_frozen=${FROZEN_BASE_RUN_DIR}::${_base_frozen_eval[2]}" \
  --case "base_lora=${LORA_BASE_RUN_DIR}::${_base_lora_eval[2]}" \
  --case "judge_frozen=${FROZEN_JUDGE_RUN_DIR}::${_judge_frozen_eval[2]}" \
  --case "judge_lora=${LORA_JUDGE_RUN_DIR}::${_judge_lora_eval[2]}"
MATH_COMPARE_DIR="$(latest_timestamped_dir "${TRANSFER_COMPARE_ROOT}/${RUN_PREFIX}_math_compare")"

run_cmd compare_gsm \
  "$PYTHON_BIN" -u scripts/phase_e_compare_processbench_transfer.py \
  --run-name "${RUN_PREFIX}_gsm_compare" \
  --output-root "$TRANSFER_COMPARE_ROOT" \
  --case "ref_pbr2=${REFERENCE_PBR2_VALUE_RUN_DIR}::${REFERENCE_PBR2_PB_GSM_DIR}" \
  --case "base_frozen=${FROZEN_BASE_RUN_DIR}::${_base_frozen_eval[1]}" \
  --case "base_lora=${LORA_BASE_RUN_DIR}::${_base_lora_eval[1]}" \
  --case "judge_frozen=${FROZEN_JUDGE_RUN_DIR}::${_judge_frozen_eval[1]}" \
  --case "judge_lora=${LORA_JUDGE_RUN_DIR}::${_judge_lora_eval[1]}"
GSM_COMPARE_DIR="$(latest_timestamped_dir "${TRANSFER_COMPARE_ROOT}/${RUN_PREFIX}_gsm_compare")"

run_cmd rl_diag \
  "$PYTHON_BIN" -u scripts/phase_e_diagnose_rl_promotion.py \
  --run-name "${RUN_PREFIX}_rl_diag" \
  --output-root "$PROMOTION_OUTPUT_ROOT" \
  --audit-spec "ref_pbr2|math_shepherd|${REFERENCE_PBR2_SAMEFAMILY_DIR}|${REFERENCE_PBR2_PB_GSM_DIR}|${REFERENCE_PBR2_PB_MATH_DIR}" \
  --audit-spec "base_frozen|math_shepherd|${_base_frozen_eval[0]}|${_base_frozen_eval[1]}|${_base_frozen_eval[2]}" \
  --audit-spec "base_lora|math_shepherd|${_base_lora_eval[0]}|${_base_lora_eval[1]}|${_base_lora_eval[2]}" \
  --audit-spec "judge_frozen|math_shepherd|${_judge_frozen_eval[0]}|${_judge_frozen_eval[1]}|${_judge_frozen_eval[2]}" \
  --audit-spec "judge_lora|math_shepherd|${_judge_lora_eval[0]}|${_judge_lora_eval[1]}|${_judge_lora_eval[2]}"
RL_DIAG_DIR="$(latest_timestamped_dir "${PROMOTION_OUTPUT_ROOT}/${RUN_PREFIX}_rl_diag")"

"$PYTHON_BIN" - "$RESULTS_JSONL" "$SUMMARY_FILE" "$SUBSET_ARTIFACT_DIR" "$JUDGE_ARTIFACT_DIR" "$MATH_COMPARE_DIR" "$GSM_COMPARE_DIR" "$RL_DIAG_DIR" <<'PY'
import json
import sys
from pathlib import Path
rows_path, summary_path, subset_dir, judge_dir, math_compare_dir, gsm_compare_dir, rl_diag_dir = sys.argv[1:]
rows = [json.loads(line) for line in Path(rows_path).read_text(encoding='utf-8').splitlines() if line.strip()]
rows.sort(key=lambda item: item['case_id'])
lines = [
    '# Phase E Backbone/Judge Pilot Suite',
    '',
    f'- subset_artifact_dir: `{subset_dir}`',
    f'- judge_filtered_artifact_dir: `{judge_dir}`',
    f'- math_compare_dir: `{math_compare_dir}`',
    f'- gsm_compare_dir: `{gsm_compare_dir}`',
    f'- rl_diag_dir: `{rl_diag_dir}`',
    '',
    '| case | variant | data | heldout_pair | heldout_auc | samefamily_top1 | samefamily_local | gsm_auc | gsm_first_edge | math_auc | math_first_edge |',
    '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
]
for row in rows:
    lines.append(
        '| {case_id} | {variant} | {data_mode} | {heldout_pair_acc:.4f} | {heldout_auc:.4f} | {samefamily_top1:.4f} | {samefamily_local:.4f} | {gsm_auc:.4f} | {gsm_first_edge:.4f} | {math_auc:.4f} | {math_first_edge:.4f} |'.format(**row)
    )
lines.append('')
Path(summary_path).write_text('\n'.join(lines) + '\n', encoding='utf-8')
PY

log_line "Suite complete. Summary: ${SUMMARY_FILE}" | tee -a "$SUITE_LOG_FILE" >&2
