#!/usr/bin/env bash
# Phase E dedicated diagnostics for the repaired compact R-PRM source.
#
# Why this file exists / 为什么需要这个脚本
# ------------------------------------------
# Earlier Phase E runs already proved one thing: compact R-PRM is no longer
# the old truncation-destroyed contract.  But that is not enough to explain why
# same-source fit is still weak.
#
# This wrapper therefore fixes one experiment contract and answers three tighter
# questions:
# 1. what structural signal survives the compact rewrite,
# 2. which recipe works best once we force a safe max_length,
# 3. and whether the best repaired run is useful inside its *own* family.
#
# 之前的 Phase E 已经证明：compact R-PRM 不再是“长文本一开始就被截断毁掉”的
# 合同。但这还不足以解释为什么 same-source 拟合仍然偏弱。
#
# 因此这个 wrapper 固定一套实验契约，专门回答三个更窄的问题：
# 1. compact 重写之后到底还保留了什么结构信号；
# 2. 在安全长度下，哪种训练 recipe 最适合 R-PRM；
# 3. 最好的 repaired run 在“自己数据家族内部”到底有没有 utility。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ACTIVE_PHASE_E_RPRM_DIAG_GROUP="${ACTIVE_PHASE_E_RPRM_DIAG_GROUP:-RD2_RPRM_RECIPE_MATRIX_SMOKE}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_rprm_diag}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"
ARTIFACT_JSON="${LOG_ROOT}/artifact_paths.json"
RESULTS_JSONL="${LOG_ROOT}/recipe_rows.jsonl"
CURRENT_STAGE="bootstrap"

MODEL_PATH="${MODEL_PATH:-assets/models/Qwen2.5-7B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
R_PRM_ROOT="${R_PRM_ROOT:-assets/external_datasets/kevinpro_r_prm}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-assets/artifacts/phase_e_feature_cache}"
FEATURE_CACHE_MODE="${FEATURE_CACHE_MODE:-read_write}"
FEATURE_CACHE_LOCK_TIMEOUT_SEC="${FEATURE_CACHE_LOCK_TIMEOUT_SEC:-600}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

GROUP_TITLE=""
GROUP_INTENTION=""
GROUP_OBSERVE=""
GROUP_EXPECT=""
PAIR_MAX_TOTAL=3000
PAIR_MAX_PER_SOURCE=3000
PAIR_MIN_CONFIDENCE=0.75
PAIR_SPLIT_GRANULARITY="pair_id"
SAFE_MAX_LENGTH=2048
RAW_AUDIT_MAX_ROWS=4000
TRUST_REJECTION_COVERAGES="1.0,0.8,0.6,0.4,0.2"
TRUST_PRESSURE_SIZES="2"
TRUST_PRESSURE_REPEATS=8
TRUST_BATCH_SIZE="${TRUST_BATCH_SIZE:-16}"

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
    echo "# Phase E R-PRM Diagnostic Summary"
    echo
    echo "- group_id: ${ACTIVE_PHASE_E_RPRM_DIAG_GROUP}"
    echo "- group_title: ${GROUP_TITLE}"
    echo "- run_prefix: ${RUN_PREFIX}"
    echo "- status: failed"
    echo "- failed_stage: ${CURRENT_STAGE}"
    echo "- suite_log_file: ${SUITE_LOG_FILE}"
  } > "$SUMMARY_FILE"
}
trap on_exit EXIT

resolve_group() {
  case "$ACTIVE_PHASE_E_RPRM_DIAG_GROUP" in
    RD1_RPRM_LENGTH_SWEEP_SMOKE)
      GROUP_TITLE="RD1 R-PRM Length Sweep Smoke"
      GROUP_INTENTION="Diagnose whether residual truncation is still the first-order blocker for compact R-PRM."
      GROUP_OBSERVE="Audit compact R-PRM, measure cutoff risk, then rerun the strongest current repaired recipe at safe lengths."
      GROUP_EXPECT="If length is the main blocker, 1536/2048 should materially outperform the shorter baseline."
      ;;
    RD2_RPRM_RECIPE_MATRIX_SMOKE)
      GROUP_TITLE="RD2 R-PRM Recipe Matrix Smoke"
      GROUP_INTENTION="Hold the repaired compact contract fixed and separate recipe failure from source failure."
      GROUP_OBSERVE="Audit one deterministic compact artifact, then compare linear/rank, MLP/rank, MLP/joint, and MLP/BCE-only under the same 2048-length supervision pool."
      GROUP_EXPECT="If the source is usable, at least one recipe should clearly beat the old ACC90 matrix and show positive same-family utility."
      ;;
    RD3_RPRM_SPLIT_TRUST_SMOKE)
      GROUP_TITLE="RD3 R-PRM Split Stress + Trust Smoke"
      GROUP_INTENTION="Check whether split granularity or prompt-level leakage explains repaired R-PRM behavior."
      GROUP_OBSERVE="Compare pair_id split and source_sample split under one strong repaired recipe, then run same-family trust evaluation on both."
      GROUP_EXPECT="If split leakage is a hidden issue, source_sample splitting should noticeably reduce held-out metrics or utility."
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_E_RPRM_DIAG_GROUP=$ACTIVE_PHASE_E_RPRM_DIAG_GROUP" >&2
      exit 1
      ;;
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

find_latest_run_dir() {
  local root_dir="$1"
  local run_name="$2"
  python - "$root_dir" "$run_name" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_name = sys.argv[2]
candidates = [path for path in root.glob(f"{run_name}*") if path.is_dir()]
if not candidates:
    raise SystemExit(1)
candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
print(str(candidates[0]))
PY
}

append_result_row() {
  local payload_json="$1"
  python - "$RESULTS_JSONL" "$payload_json" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(sys.argv[2])
with path.open('a', encoding='utf-8') as handle:
    handle.write(json.dumps(payload, ensure_ascii=False) + '\n')
PY
}

prepare_pairs() {
  local pair_run_name="$1"
  local pair_min_conf="$2"
  local split_granularity="$3"
  local prep_cmd=(
    "$PYTHON_BIN" -u scripts/phase_e_prepare_pairs.py
    --source-bundle r_prm_train
    --run-name "$pair_run_name"
    --output-root assets/artifacts/phase_e_pairs
    --seed 42
    --split-granularity "$split_granularity"
    --max-pairs-total "$PAIR_MAX_TOTAL"
    --max-pairs-per-source "$PAIR_MAX_PER_SOURCE"
    --min-pair-confidence "$pair_min_conf"
    --r-prm-pair-mode compact_verdict
  )
  log_line "RUN: ${prep_cmd[*]}" | tee -a "$SUITE_LOG_FILE" >&2
  local prep_output=""
  run_and_capture prep_output "${prep_cmd[@]}" >/dev/null
  local pair_run_dir
  pair_run_dir="$(find_latest_run_dir "assets/artifacts/phase_e_pairs" "$pair_run_name")"
  if [[ -z "$pair_run_dir" ]]; then
    echo "ERROR: Failed to parse pair artifact run_dir" >&2
    exit 1
  fi
  printf '%s\n' "$pair_run_dir"
}

audit_raw_contract() {
  local audit_name="$1"
  local raw_cmd=(
    "$PYTHON_BIN" -u scripts/phase_e_audit_rprm_contract.py
    --r-prm-root "$R_PRM_ROOT"
    --split train
    --model-path "$MODEL_PATH"
    --run-name "$audit_name"
    --output-root "$LOG_ROOT/raw_audit"
    --max-rows "$RAW_AUDIT_MAX_ROWS"
    --max-lengths 1024 1536 2048
  )
  if [[ -n "$ADAPTER_PATH" ]]; then
    raw_cmd+=(--adapter-path "$ADAPTER_PATH")
  fi
  log_line "RUN: ${raw_cmd[*]}" | tee -a "$SUITE_LOG_FILE" >&2
  local raw_output=""
  run_and_capture raw_output "${raw_cmd[@]}" >/dev/null
  local summary_json
  summary_json="$(find_latest_run_dir "${LOG_ROOT}/raw_audit" "$audit_name")/summary.json"
  if [[ -z "$summary_json" ]]; then
    echo "ERROR: Failed to parse raw audit summary_json" >&2
    exit 1
  fi
  printf '%s\n' "$summary_json"
}

audit_pair_artifact() {
  local pair_run_dir="$1"
  local audit_out="$2"
  python - "$pair_run_dir" "$audit_out" <<'PY'
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

pair_run_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])

def pct(values, q):
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[idx]

def audit_file(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    prompts = defaultdict(list)
    chosen_counter = Counter()
    rejected_counter = Counter()
    for row in rows:
        prompts[str(row.get('prompt_text', ''))].append(row)
        meta = dict(row.get('metadata') or {})
        chosen_counter[str(meta.get('chosen_verdict', '<missing>'))] += 1
        rejected_counter[str(meta.get('rejected_verdict', '<missing>'))] += 1
    prompt_sizes = [len(v) for v in prompts.values()]
    conflicting_chosen = 0
    conflicting_rejected = 0
    for group in prompts.values():
        chosen_set = {str((item.get('metadata') or {}).get('chosen_verdict', '<missing>')) for item in group}
        rejected_set = {str((item.get('metadata') or {}).get('rejected_verdict', '<missing>')) for item in group}
        if len(chosen_set) > 1:
            conflicting_chosen += 1
        if len(rejected_set) > 1:
            conflicting_rejected += 1
    return {
        'num_pairs': len(rows),
        'num_unique_prompts': len(prompts),
        'mean_pairs_per_prompt': float(len(rows) / len(prompts)) if prompts else 0.0,
        'prompt_size_p95': int(pct(prompt_sizes, 0.95)) if prompt_sizes else 0,
        'max_pairs_per_prompt': int(max(prompt_sizes)) if prompt_sizes else 0,
        'num_multi_pair_prompts': int(sum(1 for size in prompt_sizes if size > 1)),
        'num_conflicting_chosen_prompts': int(conflicting_chosen),
        'num_conflicting_rejected_prompts': int(conflicting_rejected),
        'chosen_verdict_counts': dict(sorted(chosen_counter.items())),
        'rejected_verdict_counts': dict(sorted(rejected_counter.items())),
    }

payload = {
    'generated_at': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
    'pair_run_dir': str(pair_run_dir),
    'train': audit_file(pair_run_dir / 'train_pairs.jsonl'),
    'validation': audit_file(pair_run_dir / 'validation_pairs.jsonl'),
}
out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
PY
}

train_recipe() {
  local label="$1"
  local train_pairs_jsonl="$2"
  local eval_pairs_jsonl="$3"
  local max_length="$4"
  local objective_mode="$5"
  local head_arch="$6"
  local learning_rate="$7"
  local num_epochs="$8"
  local train_batch_size="$9"
  local eval_batch_size="${10}"
  local checkpoint_metric="${11}"
  local lambda_ranking="${12}"
  local lambda_bce="${13}"
  local anti_sat_weight="${14}"
  local head_dropout="${15}"

  local run_name="${RUN_PREFIX}_${label}"
  local train_cmd=(
    "$PYTHON_BIN" -u scripts/phase_e_train_value.py
    --train-pairs-jsonl "$train_pairs_jsonl"
    --eval-pairs-jsonl "$eval_pairs_jsonl"
    --model-path "$MODEL_PATH"
    --run-name "$run_name"
    --output-root assets/artifacts/phase_e_runs
    --objective-mode "$objective_mode"
    --learning-rate "$learning_rate"
    --num-train-epochs "$num_epochs"
    --per-device-train-batch-size "$train_batch_size"
    --per-device-eval-batch-size "$eval_batch_size"
    --pair-weight-mode none
    --source-balance none
    --permutation-mode stable_hash
    --checkpoint-selection-metric "$checkpoint_metric"
    --seed 42
    --dtype "$DTYPE"
    --device-map "$DEVICE_MAP"
    --feature-cache-root "$FEATURE_CACHE_ROOT"
    --feature-cache-mode "$FEATURE_CACHE_MODE"
    --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC"
    --max-length "$max_length"
    --lambda-ranking "$lambda_ranking"
    --lambda-bce "$lambda_bce"
    --ranking-margin 0.02
    --anti-saturation-weight "$anti_sat_weight"
    --anti-saturation-logit-threshold 3.0
    --head-architecture "$head_arch"
    --head-mlp-hidden-size 1024
    --head-dropout-prob "$head_dropout"
    --head-init-std 0.02
    --head-activation gelu
    --strict-determinism
    --require-cuda
  )
  if [[ -n "$ADAPTER_PATH" ]]; then
    train_cmd+=(--adapter-path "$ADAPTER_PATH")
  fi
  log_line "RUN: ${train_cmd[*]}" | tee -a "$SUITE_LOG_FILE"
  local train_output=""
  set +e
  run_and_capture train_output "${train_cmd[@]}"
  local exit_code=$?
  set -e
  if [[ $exit_code -ne 0 ]]; then
    local error_message
    error_message="$(printf '%s\n' "$train_output" | tail -n 20 | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')"
    local failed_payload_json
    failed_payload_json="$(python - "$label" "$exit_code" "$error_message" <<'PY'
import json, sys
print(json.dumps({
  'label': sys.argv[1],
  'status': 'failed',
  'exit_code': int(sys.argv[2]),
  'error_message': sys.argv[3],
}, ensure_ascii=False))
PY
)"
    append_result_row "$failed_payload_json"
    return 0
  fi

  local run_dir
  run_dir="$(find_latest_run_dir "assets/artifacts/phase_e_runs" "$run_name")"
  if [[ -z "$run_dir" ]]; then
    echo "ERROR: Failed to parse train run_dir for $label" >&2
    exit 1
  fi

  local trust_cmd=(
    "$PYTHON_BIN" -u scripts/phase_e_eval_samefamily_trust.py
    --value-run-dir "$run_dir"
    --eval-pairs-jsonl "$eval_pairs_jsonl"
    --run-name "${run_name}_samefamily"
    --batch-size "$TRUST_BATCH_SIZE"
    --max-length "$max_length"
    --dtype "$DTYPE"
    --device-map "$DEVICE_MAP"
    --feature-cache-root "$FEATURE_CACHE_ROOT"
    --feature-cache-mode off
    --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC"
    --edge-weight-mode unit
    --rejection-coverages "$TRUST_REJECTION_COVERAGES"
    --pressure-sizes "$TRUST_PRESSURE_SIZES"
    --pressure-repeats "$TRUST_PRESSURE_REPEATS"
    --require-cuda
  )
  log_line "RUN: ${trust_cmd[*]}" | tee -a "$SUITE_LOG_FILE"
  local trust_output=""
  run_and_capture trust_output "${trust_cmd[@]}"
  local trust_metrics_path
  trust_metrics_path="$(find_latest_run_dir "assets/artifacts/phase_e_samefamily_eval" "${run_name}_samefamily")/metrics.json"
  if [[ -z "$trust_metrics_path" ]]; then
    echo "ERROR: Failed to parse same-family trust metrics_path for $label" >&2
    exit 1
  fi

  local row_json
  row_json="$(python - "$label" "$run_dir" "$trust_metrics_path" <<'PY'
import json, sys
from pathlib import Path
label = sys.argv[1]
run_dir = Path(sys.argv[2])
trust_metrics = json.loads(Path(sys.argv[3]).read_text(encoding='utf-8'))
eval_metrics = json.loads((run_dir / 'eval_metrics.json').read_text(encoding='utf-8'))
pair = dict(eval_metrics['eval_pairs'])
rejection = list(trust_metrics.get('rejection_curve', []))
rejection_cov06 = None
for point in rejection:
    if abs(float(point.get('target_coverage', 0.0)) - 0.6) < 1e-9:
        rejection_cov06 = point
        break
payload = {
    'label': label,
    'status': 'ok',
    'run_dir': str(run_dir),
    'heldout_pair_acc': float(pair['pair_accuracy']),
    'heldout_auc': float(pair['auc']),
    'heldout_ranking_score': float(pair['ranking_score']),
    'heldout_mean_margin': float(pair['mean_margin']),
    'trust_num_prompt_pools': int(trust_metrics['num_prompt_pools']),
    'trust_prompt_top1_accuracy': float(trust_metrics['prompt_pool_top1_accuracy']),
    'trust_prompt_mean_regret': float(trust_metrics['prompt_pool_mean_regret']),
    'trust_prompt_mean_score_gap': float(trust_metrics['prompt_pool_mean_score_gap']),
    'trust_reject_cov06_top1': None if rejection_cov06 is None else float(rejection_cov06['top1_accuracy']),
    'trust_reject_cov06_actual_cov': None if rejection_cov06 is None else float(rejection_cov06['actual_coverage']),
    'trust_local_first_bad_edge_accuracy': trust_metrics.get('local_first_bad_edge_accuracy'),
    'trust_metrics_path': sys.argv[3],
    'eval_metrics_path': str(run_dir / 'eval_metrics.json'),
    'summary_path': str(run_dir / 'summary.md'),
}
print(json.dumps(payload, ensure_ascii=False))
PY
)"
  append_result_row "$row_json"
}

render_summary_rd1() {
  python - "$ARTIFACT_JSON" "$RESULTS_JSONL" "$SUMMARY_FILE" "$ACTIVE_PHASE_E_RPRM_DIAG_GROUP" "$GROUP_TITLE" "$RUN_PREFIX" "$SUITE_LOG_FILE" "$GROUP_INTENTION" "$GROUP_OBSERVE" "$GROUP_EXPECT" <<'PY'
import json, sys
from pathlib import Path
artifact = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
rows = [json.loads(line) for line in Path(sys.argv[2]).read_text(encoding='utf-8').splitlines() if line.strip()]
summary_path = Path(sys.argv[3])
lines = [
    '# Phase E R-PRM Diagnostic Summary',
    '',
    f'- group_id: {sys.argv[4]}',
    f'- group_title: {sys.argv[5]}',
    f'- run_prefix: {sys.argv[6]}',
    '- status: ok',
    f'- suite_log_file: {sys.argv[7]}',
    f'- group_intention: {sys.argv[8]}',
    f'- observe: {sys.argv[9]}',
    f'- expect: {sys.argv[10]}',
    f"- raw_audit_summary: `{artifact['raw_audit_summary']}`",
    f"- pair_artifact_dir: `{artifact['pair_run_dir']}`",
    '',
    '## Length Sweep',
    '',
    '| max_length | status | heldout_pair_acc | heldout_auc | ranking_score |',
    '|---:|---|---:|---:|---:|',
]
for row in rows:
    lines.append(
        f"| {int(row['max_length'])} | {row['status']} | "
        f"{float(row.get('heldout_pair_acc', 0.0)):.4f} | {float(row.get('heldout_auc', 0.0)):.4f} | "
        f"{float(row.get('heldout_ranking_score', 0.0)):.4f} |"
    )
summary_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
PY
}

render_summary_rd2() {
  python - "$ARTIFACT_JSON" "$RESULTS_JSONL" "$SUMMARY_FILE" "$ACTIVE_PHASE_E_RPRM_DIAG_GROUP" "$GROUP_TITLE" "$RUN_PREFIX" "$SUITE_LOG_FILE" "$GROUP_INTENTION" "$GROUP_OBSERVE" "$GROUP_EXPECT" <<'PY'
import json, sys
from pathlib import Path
artifact = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
rows = [json.loads(line) for line in Path(sys.argv[2]).read_text(encoding='utf-8').splitlines() if line.strip()]
summary_path = Path(sys.argv[3])
raw_audit = json.loads(Path(artifact['raw_audit_summary']).read_text(encoding='utf-8'))
pair_audit = json.loads(Path(artifact['pair_audit_summary']).read_text(encoding='utf-8'))
lines = [
    '# Phase E R-PRM Diagnostic Summary',
    '',
    f'- group_id: {sys.argv[4]}',
    f'- group_title: {sys.argv[5]}',
    f'- run_prefix: {sys.argv[6]}',
    '- status: ok',
    f'- suite_log_file: {sys.argv[7]}',
    f'- group_intention: {sys.argv[8]}',
    f'- observe: {sys.argv[9]}',
    f'- expect: {sys.argv[10]}',
    f"- raw_audit_summary: `{artifact['raw_audit_summary']}`",
    f"- pair_artifact_dir: `{artifact['pair_run_dir']}`",
    f"- pair_audit_summary: `{artifact['pair_audit_summary']}`",
    '',
    '## Raw Contract Audit',
    '',
    f"- acceptance_rate: `{float(raw_audit['acceptance_rate']):.4f}`",
    f"- chosen_yes: `{int(raw_audit['chosen_verdict_counts'].get('yes', 0))}`",
    f"- chosen_no: `{int(raw_audit['chosen_verdict_counts'].get('no', 0))}`",
    f"- prompt_token_p95: `{int(raw_audit['prompt_token_lengths']['p95'])}`",
    f"- first_diff_token_p95: `{int(raw_audit['first_diff_token_index']['p95'])}`",
    '',
    '## Pair Artifact Audit',
    '',
    f"- train_unique_prompts: `{int(pair_audit['train']['num_unique_prompts'])}`",
    f"- train_mean_pairs_per_prompt: `{float(pair_audit['train']['mean_pairs_per_prompt']):.4f}`",
    f"- train_prompt_size_p95: `{int(pair_audit['train']['prompt_size_p95'])}`",
    f"- train_conflicting_chosen_prompts: `{int(pair_audit['train']['num_conflicting_chosen_prompts'])}`",
    f"- validation_unique_prompts: `{int(pair_audit['validation']['num_unique_prompts'])}`",
    f"- validation_mean_pairs_per_prompt: `{float(pair_audit['validation']['mean_pairs_per_prompt']):.4f}`",
    '',
    '## Recipe Matrix @ Safe Length',
    '',
    '| label | status | heldout_pair_acc | heldout_auc | ranking_score | mean_margin | prompt_top1 | reject@0.6 top1 | prompt_regret |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|',
]
for row in rows:
    if row.get('status') != 'ok':
        lines.append(f"| {row['label']} | failed | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
        lines.append(f"Error: `{row.get('error_message', '')}`")
        continue
    def fmt(key):
        value = row.get(key)
        return 'N/A' if value is None else f"{float(value):.4f}"
    lines.append(
        '| ' + ' | '.join([
            str(row['label']),
            str(row['status']),
            fmt('heldout_pair_acc'),
            fmt('heldout_auc'),
            fmt('heldout_ranking_score'),
            fmt('heldout_mean_margin'),
            fmt('trust_prompt_top1_accuracy'),
            fmt('trust_reject_cov06_top1'),
            fmt('trust_prompt_mean_regret'),
        ]) + ' |'
    )
    lines.append(f"Run: `{row['run_dir']}`")
    lines.append(f"Trust: `{row['trust_metrics_path']}`")
summary_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
PY
}

render_summary_rd3() {
  python - "$ARTIFACT_JSON" "$RESULTS_JSONL" "$SUMMARY_FILE" "$ACTIVE_PHASE_E_RPRM_DIAG_GROUP" "$GROUP_TITLE" "$RUN_PREFIX" "$SUITE_LOG_FILE" "$GROUP_INTENTION" "$GROUP_OBSERVE" "$GROUP_EXPECT" <<'PY'
import json, sys
from pathlib import Path
artifact = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
rows = [json.loads(line) for line in Path(sys.argv[2]).read_text(encoding='utf-8').splitlines() if line.strip()]
summary_path = Path(sys.argv[3])
lines = [
    '# Phase E R-PRM Diagnostic Summary',
    '',
    f'- group_id: {sys.argv[4]}',
    f'- group_title: {sys.argv[5]}',
    f'- run_prefix: {sys.argv[6]}',
    '- status: ok',
    f'- suite_log_file: {sys.argv[7]}',
    f'- group_intention: {sys.argv[8]}',
    f'- observe: {sys.argv[9]}',
    f'- expect: {sys.argv[10]}',
    f"- pair_id_artifact_dir: `{artifact['pair_id_run_dir']}`",
    f"- source_sample_artifact_dir: `{artifact['source_sample_run_dir']}`",
    '',
    '## Split Comparison',
    '',
    '| label | heldout_pair_acc | heldout_auc | ranking_score | prompt_top1 | reject@0.6 top1 | prompt_regret |',
    '|---|---:|---:|---:|---:|---:|---:|',
]
for row in rows:
    def fmt(key):
        value = row.get(key)
        return 'N/A' if value is None else f"{float(value):.4f}"
    lines.append(
        '| ' + ' | '.join([
            str(row['label']),
            fmt('heldout_pair_acc'),
            fmt('heldout_auc'),
            fmt('heldout_ranking_score'),
            fmt('trust_prompt_top1_accuracy'),
            fmt('trust_reject_cov06_top1'),
            fmt('trust_prompt_mean_regret'),
        ]) + ' |'
    )
summary_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
PY
}

resolve_group
mkdir -p "$LOG_ROOT"
: > "$SUITE_LOG_FILE"
: > "$RESULTS_JSONL"

{
  log_line "Phase E R-PRM Diagnostic Suite"
  log_line "group_id=${ACTIVE_PHASE_E_RPRM_DIAG_GROUP}"
  log_line "group_title=${GROUP_TITLE}"
  log_line "group_intention=${GROUP_INTENTION}"
  log_line "group_observe=${GROUP_OBSERVE}"
  log_line "group_expect=${GROUP_EXPECT}"
  log_line "run_prefix=${RUN_PREFIX}"
  log_line "model_path=${MODEL_PATH}"
  log_line "r_prm_root=${R_PRM_ROOT}"
  log_line "safe_max_length=${SAFE_MAX_LENGTH}"
} | tee -a "$SUITE_LOG_FILE"

case "$ACTIVE_PHASE_E_RPRM_DIAG_GROUP" in
  RD1_RPRM_LENGTH_SWEEP_SMOKE)
    CURRENT_STAGE="raw_audit"
    RAW_AUDIT_SUMMARY="$(audit_raw_contract "${RUN_PREFIX}_raw_audit" | tail -n1)"
    CURRENT_STAGE="prepare_pairs"
    PAIR_RUN_DIR="$(prepare_pairs "${RUN_PREFIX}_pairs" "$PAIR_MIN_CONFIDENCE" "pair_id" | tail -n1)"
    TRAIN_PAIRS_JSONL="${PAIR_RUN_DIR}/train_pairs.jsonl"
    VALIDATION_PAIRS_JSONL="${PAIR_RUN_DIR}/validation_pairs.jsonl"
    python - "$ARTIFACT_JSON" "$PAIR_RUN_DIR" "$RAW_AUDIT_SUMMARY" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
  'pair_run_dir': sys.argv[2],
  'raw_audit_summary': sys.argv[3],
}, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
PY
    for max_length in 1536 2048; do
      CURRENT_STAGE="train_len_${max_length}"
      train_batch_size=48
      eval_batch_size=96
      if [[ "$max_length" -eq 2048 ]]; then
        train_batch_size=32
        eval_batch_size=96
      fi
      train_recipe "len${max_length}_mlp_joint" "$TRAIN_PAIRS_JSONL" "$VALIDATION_PAIRS_JSONL" "$max_length" joint mlp 3e-5 6 "$train_batch_size" "$eval_batch_size" pair_acc 1.0 1.0 5e-4 0.05
      # The generic recipe row carries the real max_length in the label.
      python - <<'PY'
PY
    done
    # Patch recipe labels into a shorter RD1-style payload with max_length field.
    python - "$RESULTS_JSONL" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
out = []
for row in rows:
    label = str(row.get('label',''))
    if label.startswith('len1536_'):
        row['max_length'] = 1536
    elif label.startswith('len2048_'):
        row['max_length'] = 2048
    out.append(row)
path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out), encoding='utf-8')
PY
    CURRENT_STAGE="render_summary"
    render_summary_rd1
    ;;
  RD2_RPRM_RECIPE_MATRIX_SMOKE)
    CURRENT_STAGE="raw_audit"
    RAW_AUDIT_SUMMARY="$(audit_raw_contract "${RUN_PREFIX}_raw_audit" | tail -n1)"
    CURRENT_STAGE="prepare_pairs"
    PAIR_RUN_DIR="$(prepare_pairs "${RUN_PREFIX}_pairs" "$PAIR_MIN_CONFIDENCE" "pair_id" | tail -n1)"
    TRAIN_PAIRS_JSONL="${PAIR_RUN_DIR}/train_pairs.jsonl"
    VALIDATION_PAIRS_JSONL="${PAIR_RUN_DIR}/validation_pairs.jsonl"
    CURRENT_STAGE="pair_audit"
    PAIR_AUDIT_SUMMARY="${LOG_ROOT}/pair_audit_summary.json"
    audit_pair_artifact "$PAIR_RUN_DIR" "$PAIR_AUDIT_SUMMARY"
    python - "$ARTIFACT_JSON" "$PAIR_RUN_DIR" "$RAW_AUDIT_SUMMARY" "$PAIR_AUDIT_SUMMARY" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
  'pair_run_dir': sys.argv[2],
  'raw_audit_summary': sys.argv[3],
  'pair_audit_summary': sys.argv[4],
}, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
PY
    CURRENT_STAGE="train_linear_rank"
    train_recipe "linear_rank_2048" "$TRAIN_PAIRS_JSONL" "$VALIDATION_PAIRS_JSONL" "$SAFE_MAX_LENGTH" ranking_only linear 3e-5 8 48 128 pair_acc 1.0 0.0 1e-3 0.0
    CURRENT_STAGE="train_mlp_rank"
    train_recipe "mlp_rank_2048" "$TRAIN_PAIRS_JSONL" "$VALIDATION_PAIRS_JSONL" "$SAFE_MAX_LENGTH" ranking_only mlp 5e-5 10 48 128 pair_acc 1.0 0.0 1e-3 0.05
    CURRENT_STAGE="train_mlp_joint"
    train_recipe "mlp_joint_2048" "$TRAIN_PAIRS_JSONL" "$VALIDATION_PAIRS_JSONL" "$SAFE_MAX_LENGTH" joint mlp 3e-5 10 48 128 pair_acc 1.0 1.0 5e-4 0.05
    CURRENT_STAGE="train_mlp_bce"
    train_recipe "mlp_bce_2048" "$TRAIN_PAIRS_JSONL" "$VALIDATION_PAIRS_JSONL" "$SAFE_MAX_LENGTH" pair_bce_only mlp 3e-5 10 48 128 pair_acc 0.0 1.0 0.0 0.05
    CURRENT_STAGE="render_summary"
    render_summary_rd2
    ;;
  RD3_RPRM_SPLIT_TRUST_SMOKE)
    CURRENT_STAGE="prepare_pair_id_pairs"
    PAIR_ID_RUN_DIR="$(prepare_pairs "${RUN_PREFIX}_pairid_pairs" "$PAIR_MIN_CONFIDENCE" "pair_id" | tail -n1)"
    CURRENT_STAGE="prepare_source_sample_pairs"
    SOURCE_SAMPLE_RUN_DIR="$(prepare_pairs "${RUN_PREFIX}_sourcesample_pairs" "$PAIR_MIN_CONFIDENCE" "source_sample" | tail -n1)"
    python - "$ARTIFACT_JSON" "$PAIR_ID_RUN_DIR" "$SOURCE_SAMPLE_RUN_DIR" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
  'pair_id_run_dir': sys.argv[2],
  'source_sample_run_dir': sys.argv[3],
}, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
PY
    CURRENT_STAGE="train_pairid"
    train_recipe "pairid_joint_2048" "${PAIR_ID_RUN_DIR}/train_pairs.jsonl" "${PAIR_ID_RUN_DIR}/validation_pairs.jsonl" "$SAFE_MAX_LENGTH" joint mlp 3e-5 10 48 128 pair_acc 1.0 1.0 5e-4 0.05
    CURRENT_STAGE="train_sourcesample"
    train_recipe "sourcesample_joint_2048" "${SOURCE_SAMPLE_RUN_DIR}/train_pairs.jsonl" "${SOURCE_SAMPLE_RUN_DIR}/validation_pairs.jsonl" "$SAFE_MAX_LENGTH" joint mlp 3e-5 10 48 128 pair_acc 1.0 1.0 5e-4 0.05
    CURRENT_STAGE="render_summary"
    render_summary_rd3
    ;;
esac

log_line "Summary file   : ${SUMMARY_FILE}" | tee -a "$SUITE_LOG_FILE"
log_line "Group complete" | tee -a "$SUITE_LOG_FILE"
