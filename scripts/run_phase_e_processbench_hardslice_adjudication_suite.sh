#!/usr/bin/env bash
# Benchmark-side ProcessBench hard-slice adjudication experiment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_processbench_hardslice_adj_$(date +%m%d_%H%M)}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-assets/models/Qwen2.5-Math-7B-Instruct}"
MATH_EVAL_DIR="${MATH_EVAL_DIR:-assets/artifacts/phase_e_eval/phase_e_processbench_research_v2_pbr2_ms_align_gated_processbench_math_20260311T043935Z}"
GSM_EVAL_DIR="${GSM_EVAL_DIR:-assets/artifacts/phase_e_eval/phase_e_processbench_research_v2_pbr2_ms_align_gated_processbench_gsm8k_20260311T043919Z}"
PREP_GPU="${PREP_GPU:-0}"
JUDGE_GPU="${JUDGE_GPU:-1}"
MAX_EDGE_PAIRS="${MAX_EDGE_PAIRS:-16}"
MAX_LATERBAD_PAIRS="${MAX_LATERBAD_PAIRS:-16}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-4}"
JUDGE_MAX_INPUT_LENGTH="${JUDGE_MAX_INPUT_LENGTH:-2048}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-96}"

LOG_DIR="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
LOG_FILE="${LOG_DIR}/suite.log"
SUMMARY_FILE="${LOG_DIR}/final_summary.md"
mkdir -p "$LOG_DIR"
: > "$LOG_FILE"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1" | tee -a "$LOG_FILE" >&2
}

latest_dir_prefix() {
  local root="$1"
  local prefix="$2"
  "$PYTHON_BIN" - "$root" "$prefix" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]); prefix=sys.argv[2]
matches = sorted(root.glob(prefix + '__*')) or sorted(root.glob(prefix + '_*'))
if not matches:
    raise SystemExit(f'No dir for prefix {prefix} under {root}')
print(matches[-1])
PY
}

run_prepare() {
  local eval_dir="$1"
  local suffix="$2"
  local run_name="${RUN_PREFIX}_${suffix}_pairs"
  log "Prepare hard-slice pairs for ${suffix}"
  CUDA_VISIBLE_DEVICES="$PREP_GPU" "$PYTHON_BIN" -u scripts/phase_e_prepare_processbench_hardslice_pairs.py \
    --benchmark-eval-dir "$eval_dir" \
    --max-edge-pairs "$MAX_EDGE_PAIRS" \
    --max-laterbad-pairs "$MAX_LATERBAD_PAIRS" \
    --run-name "$run_name" \
    --output-root assets/artifacts/phase_e_pairs | tee -a "$LOG_FILE" >&2
}

run_judge() {
  local pairs_jsonl="$1"
  local suffix="$2"
  local run_name="${RUN_PREFIX}_${suffix}_judge"
  log "Judge hard slices for ${suffix}"
  CUDA_VISIBLE_DEVICES="$JUDGE_GPU" "$PYTHON_BIN" -u scripts/phase_e_pairwise_judge_benchmark.py \
    --pairs-jsonl "$pairs_jsonl" \
    --model-path "$JUDGE_MODEL_PATH" \
    --run-name "$run_name" \
    --dataset-label "$suffix" \
    --max-samples 999999 \
    --batch-size "$JUDGE_BATCH_SIZE" \
    --max-input-length "$JUDGE_MAX_INPUT_LENGTH" \
    --max-new-tokens "$JUDGE_MAX_NEW_TOKENS" \
    --min-filter-confidence 0.60 \
    --dtype bfloat16 \
    --device-map auto \
    --write-filtered-jsonl | tee -a "$LOG_FILE" >&2
}

run_prepare "$MATH_EVAL_DIR" math
MATH_PAIRS_DIR="$(latest_dir_prefix assets/artifacts/phase_e_pairs "${RUN_PREFIX}_math_pairs")"
run_judge "$MATH_PAIRS_DIR/pairs.jsonl" processbench_math_hardslice
MATH_JUDGE_DIR="$(latest_dir_prefix assets/artifacts/phase_e_pairwise_judge "${RUN_PREFIX}_processbench_math_hardslice_judge")"
run_prepare "$GSM_EVAL_DIR" gsm8k
GSM_PAIRS_DIR="$(latest_dir_prefix assets/artifacts/phase_e_pairs "${RUN_PREFIX}_gsm8k_pairs")"
run_judge "$GSM_PAIRS_DIR/pairs.jsonl" processbench_gsm8k_hardslice
GSM_JUDGE_DIR="$(latest_dir_prefix assets/artifacts/phase_e_pairwise_judge "${RUN_PREFIX}_processbench_gsm8k_hardslice_judge")"

"$PYTHON_BIN" - "$MATH_PAIRS_DIR" "$MATH_JUDGE_DIR" "$GSM_PAIRS_DIR" "$GSM_JUDGE_DIR" "$SUMMARY_FILE" <<'PY'
import json
import sys
from pathlib import Path
math_pairs_dir, math_judge_dir, gsm_pairs_dir, gsm_judge_dir, summary_file = [Path(x) for x in sys.argv[1:]]
math_pairs = json.loads((math_pairs_dir / 'summary.json').read_text())
math_judge = json.loads((math_judge_dir / 'summary.json').read_text())
gsm_pairs = json.loads((gsm_pairs_dir / 'summary.json').read_text())
gsm_judge = json.loads((gsm_judge_dir / 'summary.json').read_text())
lines = [
    '# Phase E ProcessBench Hard-Slice Adjudication Suite',
    '',
    '| benchmark | num_pairs | both_parse_ok | pair_acc_majority | swap_consistency | label_preserving_keep | contradiction | tie_rate |',
    '|---|---:|---:|---:|---:|---:|---:|---:|',
    f"| math | {math_judge['num_pairs']} | {math_judge['both_parse_ok_rate']:.4f} | {math_judge['pair_acc_majority']:.4f} | {math_judge['swap_consistency_rate']:.4f} | {math_judge['label_preserving_keep_rate']:.4f} | {math_judge['judge_contradiction_rate']:.4f} | {math_judge['tie_rate']:.4f} |",
    f"| gsm8k | {gsm_judge['num_pairs']} | {gsm_judge['both_parse_ok_rate']:.4f} | {gsm_judge['pair_acc_majority']:.4f} | {gsm_judge['swap_consistency_rate']:.4f} | {gsm_judge['label_preserving_keep_rate']:.4f} | {gsm_judge['judge_contradiction_rate']:.4f} | {gsm_judge['tie_rate']:.4f} |",
    '',
    '## Slice Construction',
    '',
    f"- math edge/laterbad selected: `{math_pairs['num_edge_selected']}` / `{math_pairs['num_laterbad_selected']}`",
    f"- gsm8k edge/laterbad selected: `{gsm_pairs['num_edge_selected']}` / `{gsm_pairs['num_laterbad_selected']}`",
    '',
    '## Artifacts',
    '',
    f"- math_pairs_dir: `{math_pairs_dir}`",
    f"- math_judge_dir: `{math_judge_dir}`",
    f"- gsm_pairs_dir: `{gsm_pairs_dir}`",
    f"- gsm_judge_dir: `{gsm_judge_dir}`",
]
summary_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
PY

log "Suite complete: ${SUMMARY_FILE}"
