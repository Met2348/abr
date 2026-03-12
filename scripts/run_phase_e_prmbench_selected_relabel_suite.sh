#!/usr/bin/env bash
# Phase E formal selected-relabel experiment on PRMBench_Preview.
#
# English
# -------
# 1. score full PRMBench train pairs with the strongest current same-source model,
# 2. select the lowest-margin slice,
# 3. run pairwise + swap-debias judge only on that slice,
# 4. merge judge-kept rows back with untouched pairs,
# 5. retrain one selected-relabel value head and compare against the baseline.
#
# 中文
# ----
# 1. 先用当前最强的 PRMBench same-source 模型给全量训练 pair 打分，
# 2. 选出最低 margin 的困难子集，
# 3. 只对这部分跑 pairwise + swap-debias judge，
# 4. 再把 judge 保留的样本并回未触碰训练集，
# 5. 训练一个 selected-relabel 版本并和 baseline 对照。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_prmbench_selected_relabel_$(date +%m%d_%H%M)}"
PAIR_ARTIFACT_DIR="${PAIR_ARTIFACT_DIR:-assets/artifacts/phase_e_pairs/phase_e_prmbench_acc95_push_0310_2359_e78_prmbench_acc95_joint_overfit_seed42_e78_prmbench_acc95_joint_overfit_seed42_sharedsplit_s42_pairs__ee938d92db9f}"
BASELINE_VALUE_RUN_DIR="${BASELINE_VALUE_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_prmbench_acc95_push_0310_2359_e78_prmbench_acc95_joint_overfit_seed42_e78_prmbench_acc95_joint_overfit_seed42_s42_value_20260310T153050Z}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-assets/models/Qwen2.5-Math-7B-Instruct}"
SELECTION_SIZE="${SELECTION_SIZE:-64}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-4}"
JUDGE_MAX_INPUT_LENGTH="${JUDGE_MAX_INPUT_LENGTH:-2048}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-96}"
MIN_FILTER_CONFIDENCE="${MIN_FILTER_CONFIDENCE:-0.60}"
SLICE_GPU="${SLICE_GPU:-0}"
JUDGE_GPU="${JUDGE_GPU:-1}"
TRAIN_GPU="${TRAIN_GPU:-0}"
RECIPE_RISK_POLICY="${RECIPE_RISK_POLICY:-error}"
MODEL_PATH="${MODEL_PATH:-assets/models/Qwen2.5-7B-Instruct}"
SLICE_BATCH_SIZE="${SLICE_BATCH_SIZE:-128}"
SLICE_MAX_LENGTH="${SLICE_MAX_LENGTH:-1024}"
SLICE_MAX_GPU_MEMORY_GIB="${SLICE_MAX_GPU_MEMORY_GIB:-45}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-14}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
TRAIN_EVAL_BATCH_SIZE="${TRAIN_EVAL_BATCH_SIZE:-128}"
TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-1024}"
TRAIN_MAX_GPU_MEMORY_GIB="${TRAIN_MAX_GPU_MEMORY_GIB:-45}"
TRAIN_MAX_CPU_MEMORY_GIB="${TRAIN_MAX_CPU_MEMORY_GIB:-96}"
TRAIN_FEATURE_CACHE_MODE="${TRAIN_FEATURE_CACHE_MODE:-read_write}"

LOG_DIR="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
LOG_FILE="${LOG_DIR}/suite.log"
SUMMARY_FILE="${LOG_DIR}/final_summary.md"
mkdir -p "$LOG_DIR"
: > "$LOG_FILE"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1" | tee -a "$LOG_FILE" >&2
}

latest_timestamped_dir() {
  local prefix="$1"
  "$PYTHON_BIN" - "$prefix" <<'PY'
from pathlib import Path
import sys
prefix = Path(sys.argv[1])
matches = sorted(prefix.parent.glob(prefix.name + '__*'))
if not matches:
    raise SystemExit(f'No artifact dir found for prefix: {prefix}')
print(matches[-1])
PY
}

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_path "$PAIR_ARTIFACT_DIR/train_pairs.jsonl" "PRMBench train pairs"
require_path "$PAIR_ARTIFACT_DIR/validation_pairs.jsonl" "PRMBench validation pairs"
require_path "$BASELINE_VALUE_RUN_DIR/manifest.json" "baseline value run"
require_path "$JUDGE_MODEL_PATH" "judge model"

SLICE_RUN_NAME="${RUN_PREFIX}_slice"
log "Slice low-margin PRMBench train pairs"
CUDA_VISIBLE_DEVICES="$SLICE_GPU" "$PYTHON_BIN" -u scripts/phase_e_slice_pairs_by_margin.py \
  --train-pairs-jsonl "$PAIR_ARTIFACT_DIR/train_pairs.jsonl" \
  --eval-pairs-jsonl "$PAIR_ARTIFACT_DIR/validation_pairs.jsonl" \
  --value-run-dir "$BASELINE_VALUE_RUN_DIR" \
  --selection-size "$SELECTION_SIZE" \
  --selection-mode lowest_abs_margin \
  --run-name "$SLICE_RUN_NAME" \
  --output-root assets/artifacts/phase_e_pairs \
  --batch-size "$SLICE_BATCH_SIZE" \
  --max-length "$SLICE_MAX_LENGTH" \
  --dtype bfloat16 \
  --device-map auto \
  --max-gpu-memory-gib "$SLICE_MAX_GPU_MEMORY_GIB" | tee -a "$LOG_FILE"
SLICE_DIR="$(latest_timestamped_dir "assets/artifacts/phase_e_pairs/${SLICE_RUN_NAME}")"

JUDGE_RUN_NAME="${RUN_PREFIX}_pairjudge"
log "Judge only the selected low-margin slice"
CUDA_VISIBLE_DEVICES="$JUDGE_GPU" "$PYTHON_BIN" -u scripts/phase_e_pairwise_judge_benchmark.py \
  --pairs-jsonl "$SLICE_DIR/selected_train_pairs.jsonl" \
  --model-path "$JUDGE_MODEL_PATH" \
  --run-name "$JUDGE_RUN_NAME" \
  --dataset-label prmbench_preview_selected_margin64 \
  --max-samples "$SELECTION_SIZE" \
  --batch-size "$JUDGE_BATCH_SIZE" \
  --max-input-length "$JUDGE_MAX_INPUT_LENGTH" \
  --max-new-tokens "$JUDGE_MAX_NEW_TOKENS" \
  --min-filter-confidence "$MIN_FILTER_CONFIDENCE" \
  --dtype bfloat16 \
  --device-map auto \
  --write-filtered-jsonl | tee -a "$LOG_FILE"
JUDGE_DIR="$(python - <<PY
from pathlib import Path
matches = sorted(Path('assets/artifacts/phase_e_pairwise_judge').glob('${JUDGE_RUN_NAME}_*'))
print(matches[-1])
PY
)"

MERGE_RUN_NAME="${RUN_PREFIX}_materialized"
log "Merge judge-kept rows back with untouched PRMBench train pairs"
"$PYTHON_BIN" -u scripts/phase_e_materialize_selected_relabel_pairs.py \
  --retained-train-pairs-jsonl "$SLICE_DIR/retained_train_pairs.jsonl" \
  --judge-kept-pairs-jsonl "$JUDGE_DIR/filtered_pairs.jsonl" \
  --validation-pairs-jsonl "$SLICE_DIR/validation_pairs.jsonl" \
  --selected-train-pairs-jsonl "$SLICE_DIR/selected_train_pairs.jsonl" \
  --run-name "$MERGE_RUN_NAME" \
  --output-root assets/artifacts/phase_e_pairs | tee -a "$LOG_FILE"
MERGED_DIR="$(latest_timestamped_dir "assets/artifacts/phase_e_pairs/${MERGE_RUN_NAME}")"

TRAIN_RUN_NAME="${RUN_PREFIX}_value"
log "Train selected-relabel PRMBench value head"
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$PYTHON_BIN" -u scripts/phase_e_train_value.py \
  --train-pairs-jsonl "$MERGED_DIR/train_pairs.jsonl" \
  --eval-pairs-jsonl "$MERGED_DIR/validation_pairs.jsonl" \
  --model-path "$MODEL_PATH" \
  --run-name "$TRAIN_RUN_NAME" \
  --output-root assets/artifacts/phase_e_runs \
  --objective-mode joint \
  --learning-rate 3e-5 \
  --num-train-epochs "$TRAIN_EPOCHS" \
  --per-device-train-batch-size "$TRAIN_BATCH_SIZE" \
  --per-device-eval-batch-size "$TRAIN_EVAL_BATCH_SIZE" \
  --max-length "$TRAIN_MAX_LENGTH" \
  --lambda-ranking 1.0 \
  --lambda-bce 1.0 \
  --ranking-margin 0.02 \
  --ranking-target-space logit \
  --pair-weight-mode none \
  --source-balance none \
  --permutation-mode stable_hash \
  --checkpoint-selection-metric pair_acc \
  --recipe-risk-policy "$RECIPE_RISK_POLICY" \
  --head-architecture mlp \
  --head-dropout-prob 0.0 \
  --head-init-std 0.02 \
  --head-mlp-hidden-size 1024 \
  --head-activation gelu \
  --anti-saturation-weight 0.0 \
  --anti-saturation-logit-threshold 4.0 \
  --max-gpu-memory-gib "$TRAIN_MAX_GPU_MEMORY_GIB" \
  --max-cpu-memory-gib "$TRAIN_MAX_CPU_MEMORY_GIB" \
  --feature-cache-mode "$TRAIN_FEATURE_CACHE_MODE" | tee -a "$LOG_FILE"
SELECTED_VALUE_DIR="$(python - <<PY
from pathlib import Path
matches = sorted(Path('assets/artifacts/phase_e_runs').glob('${TRAIN_RUN_NAME}_*'))
print(matches[-1])
PY
)"

"$PYTHON_BIN" - "$BASELINE_VALUE_RUN_DIR" "$SELECTED_VALUE_DIR" "$SLICE_DIR" "$JUDGE_DIR" "$MERGED_DIR" "$SUMMARY_FILE" <<'PY'
import json
import sys
from pathlib import Path
baseline_dir, selected_dir, slice_dir, judge_dir, merged_dir, summary_file = [Path(x) for x in sys.argv[1:]]
baseline = json.loads((baseline_dir / 'summary.json').read_text())
selected = json.loads((selected_dir / 'summary.json').read_text())
slice_summary = json.loads((slice_dir / 'summary.json').read_text())
judge_summary = json.loads((judge_dir / 'summary.json').read_text())
merged_summary = json.loads((merged_dir / 'summary.json').read_text())
lines = [
    '# Phase E PRMBench Selected Relabel Suite',
    '',
    f'- baseline_value_run_dir: `{baseline_dir}`',
    f'- selected_value_run_dir: `{selected_dir}`',
    f'- slice_dir: `{slice_dir}`',
    f'- judge_dir: `{judge_dir}`',
    f'- merged_dir: `{merged_dir}`',
    '',
    '| case | heldout_pair_acc | heldout_auc | train_pairs |',
    '|---|---:|---:|---:|',
    f"| baseline_e78 | {baseline['eval_pairs']['pair_accuracy']:.4f} | {baseline['eval_pairs']['auc']:.4f} | {baseline['train_pairs']} |",
    f"| selected_relabel | {selected['eval_pairs']['pair_accuracy']:.4f} | {selected['eval_pairs']['auc']:.4f} | {selected['train_pairs']} |",
    '',
    '## Judge Slice',
    '',
    f"- selection_size: `{slice_summary['selection_size']}`",
    f"- selected_negative_margin_pairs: `{slice_summary['selected_margin_summary']['num_negative_margin_pairs']}`",
    f"- pair_acc_majority: `{judge_summary['pair_acc_majority']:.4f}`",
    f"- swap_consistency_rate: `{judge_summary['swap_consistency_rate']:.4f}`",
    f"- label_preserving_keep_rate: `{judge_summary['label_preserving_keep_rate']:.4f}`",
    f"- judge_contradiction_rate: `{judge_summary['judge_contradiction_rate']:.4f}`",
    '',
    '## Materialized Train Set',
    '',
    f"- num_merged_rows: `{merged_summary['num_merged_rows']}`",
    f"- selected_keep_rate: `{merged_summary['selected_keep_rate'] if merged_summary['selected_keep_rate'] is not None else 'n/a'}`",
]
summary_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
PY

log "Suite complete: ${SUMMARY_FILE}"
