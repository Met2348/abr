#!/usr/bin/env bash
# Phase E formal selected-relabel experiment on PRMBench_Preview.
#
# English
# -------
# 1. score full PRMBench train pairs with the strongest current same-source model,
# 2. select the lowest-margin slice,
# 3. run pairwise + swap-debias judge only on that slice,
# 4. merge judge-kept rows back with untouched pairs,
# 5. retrain one selected-relabel value head,
# 6. run same-family + ProcessBench eval so the repair is judged on benchmark-facing behavior,
# 7. compare against the current balanced baseline rather than the older overfit line.
#
# 中文
# ----
# 1. 先用当前更平衡的 PRMBench same-source 基线给全量训练 pair 打分，
# 2. 选出最低 margin 的困难子集，
# 3. 只对这部分跑 pairwise + swap-debias judge，
# 4. 再把 judge 保留的样本并回未触碰训练集，
# 5. 训练一个 selected-relabel 版本，
# 6. 追加 same-family 与 ProcessBench 评测，避免只看同分布 held-out，
# 7. 默认和 `E46` 平衡基线比，而不是沿用旧的 `E78` 过拟合基线。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_prmbench_selected_relabel_$(date +%m%d_%H%M)}"
PAIR_ARTIFACT_DIR="${PAIR_ARTIFACT_DIR:-assets/artifacts/phase_e_pairs/phase_e_prmbench_acc90_full_0310_1914_e46_prmbench_acc90_mlp_joint_seed3_e46_prmbench_acc90_mlp_joint_seed3_sharedsplit_s42_pairs__8886075f9c6e}"
BASELINE_VALUE_RUN_DIR="${BASELINE_VALUE_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_prmbench_acc90_full_0310_1914_e46_prmbench_acc90_mlp_joint_seed3_e46_prmbench_acc90_mlp_joint_seed3_s43_value_20260310T113737Z}"
BASELINE_SAMEFAMILY_DIR="${BASELINE_SAMEFAMILY_DIR:-assets/artifacts/phase_e_samefamily_eval/phase_e_rltops_0311_1124_prm_e46_samefamily_20260311T032656Z}"
BASELINE_GSM_EVAL_DIR="${BASELINE_GSM_EVAL_DIR:-assets/artifacts/phase_e_eval/phase_e_rltops_0311_1124_prm_e46_processbench_gsm8k_20260311T032704Z}"
BASELINE_MATH_EVAL_DIR="${BASELINE_MATH_EVAL_DIR:-assets/artifacts/phase_e_eval/phase_e_rltops_0311_1124_prm_e46_processbench_math_20260311T032713Z}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-assets/models/Qwen2.5-Math-7B-Instruct}"
SELECTION_SIZE="${SELECTION_SIZE:-64}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-4}"
JUDGE_MAX_INPUT_LENGTH="${JUDGE_MAX_INPUT_LENGTH:-2048}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-96}"
MIN_FILTER_CONFIDENCE="${MIN_FILTER_CONFIDENCE:-0.60}"
SLICE_GPU="${SLICE_GPU:-1}"
JUDGE_GPU="${JUDGE_GPU:-1}"
TRAIN_GPU="${TRAIN_GPU:-1}"
RECIPE_RISK_POLICY="${RECIPE_RISK_POLICY:-error}"
MODEL_PATH="${MODEL_PATH:-assets/models/Qwen2.5-7B-Instruct}"
SLICE_BATCH_SIZE="${SLICE_BATCH_SIZE:-128}"
SLICE_MAX_LENGTH="${SLICE_MAX_LENGTH:-1024}"
SLICE_MAX_GPU_MEMORY_GIB="${SLICE_MAX_GPU_MEMORY_GIB:-45}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-14}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-96}"
TRAIN_EVAL_BATCH_SIZE="${TRAIN_EVAL_BATCH_SIZE:-128}"
TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-1024}"
TRAIN_MAX_GPU_MEMORY_GIB="${TRAIN_MAX_GPU_MEMORY_GIB:-45}"
TRAIN_MAX_CPU_MEMORY_GIB="${TRAIN_MAX_CPU_MEMORY_GIB:-96}"
TRAIN_FEATURE_CACHE_MODE="${TRAIN_FEATURE_CACHE_MODE:-read_write}"
TRAIN_OBJECTIVE_MODE="${TRAIN_OBJECTIVE_MODE:-joint}"
TRAIN_RANKING_TARGET_SPACE="${TRAIN_RANKING_TARGET_SPACE:-score}"
BENCH_EVAL_BATCH_SIZE="${BENCH_EVAL_BATCH_SIZE:-128}"
BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-256}"

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
matches = sorted(set(prefix.parent.glob(prefix.name + '__*')) | set(prefix.parent.glob(prefix.name + '_*')))
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
require_path "$BASELINE_SAMEFAMILY_DIR/metrics.json" "baseline samefamily eval"
require_path "$BASELINE_GSM_EVAL_DIR/summary.json" "baseline ProcessBench GSM eval"
require_path "$BASELINE_MATH_EVAL_DIR/summary.json" "baseline ProcessBench Math eval"
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
  --objective-mode "$TRAIN_OBJECTIVE_MODE" \
  --learning-rate 3e-5 \
  --num-train-epochs "$TRAIN_EPOCHS" \
  --per-device-train-batch-size "$TRAIN_BATCH_SIZE" \
  --per-device-eval-batch-size "$TRAIN_EVAL_BATCH_SIZE" \
  --max-length "$TRAIN_MAX_LENGTH" \
  --lambda-ranking 1.0 \
  --lambda-bce 1.0 \
  --ranking-margin 0.02 \
  --ranking-target-space "$TRAIN_RANKING_TARGET_SPACE" \
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

SELECTED_SAMEFAMILY_RUN_NAME="${RUN_PREFIX}_samefamily"
log "Run same-family trust eval on selected-relabel checkpoint"
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$PYTHON_BIN" -u scripts/phase_e_eval_samefamily_trust.py \
  --value-run-dir "$SELECTED_VALUE_DIR" \
  --eval-pairs-jsonl "$MERGED_DIR/validation_pairs.jsonl" \
  --run-name "$SELECTED_SAMEFAMILY_RUN_NAME" \
  --output-root assets/artifacts/phase_e_samefamily_eval \
  --checkpoint-name best \
  --batch-size "$BENCH_EVAL_BATCH_SIZE" \
  --max-length "$TRAIN_MAX_LENGTH" \
  --feature-cache-root assets/artifacts/phase_e_feature_cache \
  --feature-cache-mode read_write \
  --feature-cache-lock-timeout-sec 600 \
  --edge-weight-mode confidence \
  --require-cuda | tee -a "$LOG_FILE"
SELECTED_SAMEFAMILY_DIR="$(python - <<PY
from pathlib import Path
matches = sorted(Path('assets/artifacts/phase_e_samefamily_eval').glob('${SELECTED_SAMEFAMILY_RUN_NAME}_*'))
print(matches[-1])
PY
)"

for benchmark_id in gsm8k math; do
  eval_run_name="${RUN_PREFIX}_processbench_${benchmark_id}"
  log "Run ProcessBench ${benchmark_id} eval on selected-relabel checkpoint"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$PYTHON_BIN" -u scripts/phase_e_eval_benchmark.py \
    --value-run-dir "$SELECTED_VALUE_DIR" \
    --benchmark-id "processbench_${benchmark_id}" \
    --run-name "$eval_run_name" \
    --output-root assets/artifacts/phase_e_eval \
    --checkpoint-name best \
    --max-samples "$BENCH_MAX_SAMPLES" \
    --batch-size "$BENCH_EVAL_BATCH_SIZE" \
    --max-length "$TRAIN_MAX_LENGTH" \
    --feature-cache-root assets/artifacts/phase_e_feature_cache \
    --feature-cache-mode read_write \
    --feature-cache-lock-timeout-sec 600 \
    --require-cuda | tee -a "$LOG_FILE"
done
SELECTED_GSM_EVAL_DIR="$(python - <<PY
from pathlib import Path
matches = sorted(Path('assets/artifacts/phase_e_eval').glob('${RUN_PREFIX}_processbench_gsm8k_*'))
print(matches[-1])
PY
)"
SELECTED_MATH_EVAL_DIR="$(python - <<PY
from pathlib import Path
matches = sorted(Path('assets/artifacts/phase_e_eval').glob('${RUN_PREFIX}_processbench_math_*'))
print(matches[-1])
PY
)"

"$PYTHON_BIN" - "$BASELINE_VALUE_RUN_DIR" "$BASELINE_SAMEFAMILY_DIR" "$BASELINE_GSM_EVAL_DIR" "$BASELINE_MATH_EVAL_DIR" "$SELECTED_VALUE_DIR" "$SELECTED_SAMEFAMILY_DIR" "$SELECTED_GSM_EVAL_DIR" "$SELECTED_MATH_EVAL_DIR" "$SLICE_DIR" "$JUDGE_DIR" "$MERGED_DIR" "$SUMMARY_FILE" <<'PY'
import json
import sys
from pathlib import Path

(
    baseline_dir,
    baseline_samefamily_dir,
    baseline_gsm_eval_dir,
    baseline_math_eval_dir,
    selected_dir,
    selected_samefamily_dir,
    selected_gsm_eval_dir,
    selected_math_eval_dir,
    slice_dir,
    judge_dir,
    merged_dir,
    summary_file,
) = [Path(x) for x in sys.argv[1:]]

baseline = json.loads((baseline_dir / 'summary.json').read_text())
baseline_samefamily = json.loads((baseline_samefamily_dir / 'metrics.json').read_text())
baseline_gsm = json.loads((baseline_gsm_eval_dir / 'summary.json').read_text())
baseline_math = json.loads((baseline_math_eval_dir / 'summary.json').read_text())
selected = json.loads((selected_dir / 'summary.json').read_text())
selected_samefamily = json.loads((selected_samefamily_dir / 'metrics.json').read_text())
selected_gsm = json.loads((selected_gsm_eval_dir / 'summary.json').read_text())
selected_math = json.loads((selected_math_eval_dir / 'summary.json').read_text())
slice_summary = json.loads((slice_dir / 'summary.json').read_text())
judge_summary = json.loads((judge_dir / 'summary.json').read_text())
merged_summary = json.loads((merged_dir / 'summary.json').read_text())

def local_metric(metrics: dict) -> float:
    return float(metrics.get('local_first_bad_edge_accuracy') or metrics.get('local_safe_vs_bad_pair_accuracy') or 0.0)

lines = [
    '# Phase E PRMBench Selected Relabel Suite',
    '',
    f'- baseline_value_run_dir: `{baseline_dir}`',
    f'- baseline_samefamily_dir: `{baseline_samefamily_dir}`',
    f'- baseline_gsm_eval_dir: `{baseline_gsm_eval_dir}`',
    f'- baseline_math_eval_dir: `{baseline_math_eval_dir}`',
    f'- selected_value_run_dir: `{selected_dir}`',
    f'- selected_samefamily_dir: `{selected_samefamily_dir}`',
    f'- selected_gsm_eval_dir: `{selected_gsm_eval_dir}`',
    f'- selected_math_eval_dir: `{selected_math_eval_dir}`',
    f'- slice_dir: `{slice_dir}`',
    f'- judge_dir: `{judge_dir}`',
    f'- merged_dir: `{merged_dir}`',
    '',
    '| case | heldout_pair_acc | heldout_auc | samefamily_top1 | samefamily_local | pb_gsm_auc | pb_math_auc | train_pairs |',
    '|---|---:|---:|---:|---:|---:|---:|---:|',
    (
        f"| baseline_e46 | {baseline['eval_pairs']['pair_accuracy']:.4f} | {baseline['eval_pairs']['auc']:.4f} | "
        f"{baseline_samefamily.get('prompt_pool_top1_accuracy', 0.0):.4f} | "
        f"{local_metric(baseline_samefamily):.4f} | "
        f"{baseline_gsm['metrics'].get('pair_auc_good_vs_bad', 0.0):.4f} | "
        f"{baseline_math['metrics'].get('pair_auc_good_vs_bad', 0.0):.4f} | {baseline['train_pairs']} |"
    ),
    (
        f"| selected_relabel | {selected['eval_pairs']['pair_accuracy']:.4f} | {selected['eval_pairs']['auc']:.4f} | "
        f"{selected_samefamily.get('prompt_pool_top1_accuracy', 0.0):.4f} | "
        f"{local_metric(selected_samefamily):.4f} | "
        f"{selected_gsm['metrics'].get('pair_auc_good_vs_bad', 0.0):.4f} | "
        f"{selected_math['metrics'].get('pair_auc_good_vs_bad', 0.0):.4f} | {selected['train_pairs']} |"
    ),
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
    '',
    '## Interpretation',
    '',
    '- This suite is now benchmark-facing by default: the main question is whether selective relabel improves ProcessBench while preserving PRMBench same-family trust.',
    '- The baseline is `E46` because it is the current balanced PRMBench candidate; the older `E78` line is not used as the default reference anymore.',
]
summary_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
PY

log "Suite complete: ${SUMMARY_FILE}"
