#!/usr/bin/env bash
# Phase D bridge suite: external ranking pretrain -> in-domain continue training.
#
# Why this file exists:
# - DT2 stable proved that external Math-Shepherd triplets can train a stable
#   ranking head on held-out triplets.
# - Transfer runs then showed that this ability does not automatically move back
#   to StrategyQA C1 corruption ranking.
# - Therefore the next method step is a bridge experiment, not more LR search:
#   first learn a ranking prior on external triplets, then continue training on
#   in-domain C1/CQR pairs and check whether corruption metrics improve.
#
# What this suite does:
# 1) Build/reuse one shared Math-Shepherd pair split.
# 2) Stage-1: run the stable external ranking pretrain recipe.
# 3) Stage-2: warm-start from stage-1 best_value_head.pt and continue on
#    StrategyQA in-domain CQR/C1 supervision.
# 4) Summarize both the external-stage metrics and final in-domain metrics.
#
# Example:
#   ACTIVE_PHASE_DBR_GROUP=DB1_STRATEGYQA_BRIDGE_SMOKE_RANK \
#   RUN_PREFIX=phase_d_bridge_smoke_rank \
#   CUDA_VISIBLE_DEVICES=1 \
#   bash scripts/run_phase_d_bridge_suite.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACTIVE_PHASE_DBR_GROUP="${ACTIVE_PHASE_DBR_GROUP:-DB1_STRATEGYQA_BRIDGE_SMOKE_RANK}"
RUN_PREFIX="${RUN_PREFIX:-phase_d_bridge_suite}"

# Stage-1 uses the original full k=8 StrategyQA rollout artifacts, matching the
# proven DT2 stable recipe.
PHASE_C_STAGE1_TRAIN_DIR="${PHASE_C_STAGE1_TRAIN_DIR:-assets/artifacts/phase_c_data/strategyqa/strategyqa_value_rollouts_k8_train_full__59aed6ac5f99}"
PHASE_C_STAGE1_EVAL_DIR="${PHASE_C_STAGE1_EVAL_DIR:-assets/artifacts/phase_c_data/strategyqa/strategyqa_value_rollouts_k8_val_full__c5cb0e294b2c}"

# Stage-2 uses the current strongest full StrategyQA CQR artifacts.
PHASE_C_STAGE2_TRAIN_DIR="${PHASE_C_STAGE2_TRAIN_DIR:-assets/artifacts/phase_c_data/strategyqa/phase_cd_full_0304_2347_c_c2_strategyqa_cqr_full_c1_train__7696783f831e}"
PHASE_C_STAGE2_EVAL_DIR="${PHASE_C_STAGE2_EVAL_DIR:-assets/artifacts/phase_c_data/strategyqa/phase_cd_full_0304_2347_c_c2_strategyqa_cqr_full_c1_eval__909c43d9b7d0}"
REFERENCE_C2_METRICS="${REFERENCE_C2_METRICS:-assets/artifacts/phase_c_eval/phase_cd_full_0304_2347_c_c2_strategyqa_cqr_full_c2_eval_20260305T005709Z/metrics.json}"

MATH_SHEPHERD_PATH="${MATH_SHEPHERD_PATH:-assets/external_datasets/peiyi_math_shepherd/math-shepherd.jsonl}"
PAIR_OUTPUT_ROOT="${PAIR_OUTPUT_ROOT:-assets/artifacts/phase_d_external_pairs}"
STAGE1_OUTPUT_ROOT="${STAGE1_OUTPUT_ROOT:-assets/artifacts/phase_c_runs}"
STAGE1_EVAL_OUTPUT_ROOT="${STAGE1_EVAL_OUTPUT_ROOT:-assets/artifacts/phase_d_triplet_eval}"
STAGE2_OUTPUT_ROOT="${STAGE2_OUTPUT_ROOT:-assets/artifacts/phase_c_runs}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-assets/artifacts/phase_c_feature_cache}"
FEATURE_CACHE_MODE="${FEATURE_CACHE_MODE:-read_write}"
FEATURE_CACHE_LOCK_TIMEOUT_SEC="${FEATURE_CACHE_LOCK_TIMEOUT_SEC:-600}"

LOG_ROOT="assets/artifacts/phase_d_bridge_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"
SEED_RESULTS_JSONL="${LOG_ROOT}/seed_results.jsonl"

CURRENT_STAGE="bootstrap"
SHARED_PAIR_RUN_DIR=""

# Global defaults. Group resolution may override them.
DPAIR_SPLIT_MODE="${DPAIR_SPLIT_MODE:-shared}"
DPAIR_SPLIT_SEED="${DPAIR_SPLIT_SEED:-42}"
STRICT_DETERMINISM="${STRICT_DETERMINISM:-1}"
STAGE1_EXTERNAL_PAIR_PERM_MODE="${STAGE1_EXTERNAL_PAIR_PERM_MODE:-stable_hash}"
VALIDATION_RATIO="${VALIDATION_RATIO:-0.1}"

GROUP_TITLE=""
GROUP_INTENTION=""
GROUP_OBSERVE=""
GROUP_EXPECT=""
SEEDS=()
REFERENCE_COMPARE="0"

log_line() {
  local text="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$text"
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: Missing ${label}: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    echo "ERROR: Missing ${label}: $path" >&2
    exit 1
  fi
}

latest_dir_for_prefix() {
  local pattern="$1"
  python - "$pattern" <<'PY'
from pathlib import Path
import sys
pattern = sys.argv[1]
paths = sorted(Path('.').glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
if paths:
    print(paths[0])
PY
}

append_extra_args() {
  local -n _dest_ref="$1"
  local raw="${2:-}"
  if [[ -z "$raw" ]]; then
    return 0
  fi
  local pieces=()
  # shellcheck disable=SC2206
  pieces=($raw)
  if [[ ${#pieces[@]} -gt 0 ]]; then
    _dest_ref+=("${pieces[@]}")
  fi
}

summary_looks_complete() {
  local path="$1"
  [[ -f "$path" ]] || return 1
  grep -q "## Aggregated Bridge" "$path"
}

on_exit() {
  local exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    return
  fi
  mkdir -p "$LOG_ROOT"
  if summary_looks_complete "$SUMMARY_FILE"; then
    log_line "Bridge suite exited after a complete summary was already written; preserving summary" | tee -a "$SUITE_LOG_FILE" >/dev/null || true
    return
  fi
  {
    echo "# Phase D Bridge Suite Summary"
    echo
    echo "- generated_at: $(python - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)"
    echo "- group_id: ${ACTIVE_PHASE_DBR_GROUP}"
    echo "- group_title: ${GROUP_TITLE}"
    echo "- run_prefix: ${RUN_PREFIX}"
    echo "- status: failed"
    echo "- exit_code: ${exit_code}"
    echo "- failed_stage: ${CURRENT_STAGE}"
    echo "- suite_log_file: ${SUITE_LOG_FILE}"
  } > "$SUMMARY_FILE"
}
trap on_exit EXIT

resolve_group() {
  case "$ACTIVE_PHASE_DBR_GROUP" in
    DB1_STRATEGYQA_BRIDGE_SMOKE_RANK)
      GROUP_TITLE="DB1 StrategyQA Bridge Smoke (Ranking-Only)"
      GROUP_INTENTION="Fast proof that warm-starting from stable external ranking can be continued on in-domain StrategyQA CQR pairs."
      GROUP_OBSERVE="Use one seed and capped sample counts to validate the bridge wiring before paying full cost."
      GROUP_EXPECT="If the bridge idea is viable, stage-2 corr_pair_acc/corr_auc should at least beat plain near-random transfer runs."
      SEEDS=(42)
      STAGE1_MAX_PAIRS_PER_SOURCE="${STAGE1_MAX_PAIRS_PER_SOURCE:-2500}"
      STAGE1_MAX_PAIRS_TOTAL="${STAGE1_MAX_PAIRS_TOTAL:-2500}"
      STAGE1_MIN_PAIR_CONFIDENCE="${STAGE1_MIN_PAIR_CONFIDENCE:-0.55}"
      STAGE1_TRAIN_BATCH_SIZE="${STAGE1_TRAIN_BATCH_SIZE:-64}"
      STAGE1_EVAL_BATCH_SIZE="${STAGE1_EVAL_BATCH_SIZE:-64}"
      STAGE1_LR="${STAGE1_LR:-5e-5}"
      STAGE1_EPOCHS="${STAGE1_EPOCHS:-3}"
      STAGE1_MAX_TRAIN_SAMPLES="${STAGE1_MAX_TRAIN_SAMPLES:-512}"
      STAGE1_MAX_EVAL_SAMPLES="${STAGE1_MAX_EVAL_SAMPLES:-215}"
      STAGE2_MAX_TRAIN_SAMPLES="${STAGE2_MAX_TRAIN_SAMPLES:-512}"
      STAGE2_MAX_EVAL_SAMPLES="${STAGE2_MAX_EVAL_SAMPLES:-215}"
      STAGE2_TRAIN_MODE="ranking_only"
      STAGE2_TRAIN_BATCH_SIZE="${STAGE2_TRAIN_BATCH_SIZE:-192}"
      STAGE2_EVAL_BATCH_SIZE="${STAGE2_EVAL_BATCH_SIZE:-192}"
      STAGE2_LR="${STAGE2_LR:-5e-5}"
      STAGE2_EPOCHS="${STAGE2_EPOCHS:-4}"
      STAGE2_CHECKPOINT_METRIC="${STAGE2_CHECKPOINT_METRIC:-corr_auc}"
      STAGE2_TRAIN_EXTRA_DEFAULT="${STAGE2_TRAIN_EXTRA_DEFAULT:---lambda-contrastive 0.08 --contrastive-margin 0.02 --contrastive-pair-filter confidence_parseable_label --contrastive-confidence-threshold 0.2 --contrastive-parseable-threshold 0.75 --contrastive-label-delta-q-min 0.2 --contrastive-label-z-min 0.5 --contrastive-label-pair-weight-min 0.3 --contrastive-require-pair-pass-gate --contrastive-use-pair-weights --contrastive-stratified-sampling --contrastive-stratify-step-bucket-size 2 --contrastive-stratify-include-no-corruption}"
      ;;
    DB2_STRATEGYQA_BRIDGE_SMOKE_JOINT)
      GROUP_TITLE="DB2 StrategyQA Bridge Smoke (Joint)"
      GROUP_INTENTION="Check whether a gentler external warm-start still helps when stage-2 keeps calibration+ranking jointly active."
      GROUP_OBSERVE="Same smoke-scale bridge as DB1, but stage-2 uses a CQR-style joint objective instead of ranking-only finetune."
      GROUP_EXPECT="If external pretraining mostly helps ranking geometry, joint stage-2 should remain competitive without collapsing CQR metrics."
      SEEDS=(42)
      STAGE1_MAX_PAIRS_PER_SOURCE="${STAGE1_MAX_PAIRS_PER_SOURCE:-2500}"
      STAGE1_MAX_PAIRS_TOTAL="${STAGE1_MAX_PAIRS_TOTAL:-2500}"
      STAGE1_MIN_PAIR_CONFIDENCE="${STAGE1_MIN_PAIR_CONFIDENCE:-0.55}"
      STAGE1_TRAIN_BATCH_SIZE="${STAGE1_TRAIN_BATCH_SIZE:-64}"
      STAGE1_EVAL_BATCH_SIZE="${STAGE1_EVAL_BATCH_SIZE:-64}"
      STAGE1_LR="${STAGE1_LR:-5e-5}"
      STAGE1_EPOCHS="${STAGE1_EPOCHS:-3}"
      STAGE1_MAX_TRAIN_SAMPLES="${STAGE1_MAX_TRAIN_SAMPLES:-512}"
      STAGE1_MAX_EVAL_SAMPLES="${STAGE1_MAX_EVAL_SAMPLES:-215}"
      STAGE2_MAX_TRAIN_SAMPLES="${STAGE2_MAX_TRAIN_SAMPLES:-512}"
      STAGE2_MAX_EVAL_SAMPLES="${STAGE2_MAX_EVAL_SAMPLES:-215}"
      STAGE2_TRAIN_MODE="joint"
      STAGE2_TRAIN_BATCH_SIZE="${STAGE2_TRAIN_BATCH_SIZE:-192}"
      STAGE2_EVAL_BATCH_SIZE="${STAGE2_EVAL_BATCH_SIZE:-192}"
      STAGE2_LR="${STAGE2_LR:-5e-5}"
      STAGE2_EPOCHS="${STAGE2_EPOCHS:-4}"
      STAGE2_CHECKPOINT_METRIC="${STAGE2_CHECKPOINT_METRIC:-corr_auc}"
      STAGE2_TRAIN_EXTRA_DEFAULT="${STAGE2_TRAIN_EXTRA_DEFAULT:---calibration-loss bce_mse --calibration-bce-weight 1.0 --calibration-mse-weight 0.5 --calibration-sample-weighting q_weight_parseable --calibration-weight-floor 0.1 --calibration-weight-gamma 1.0 --lambda-contrastive 0.08 --contrastive-margin 0.02 --contrastive-pair-filter confidence_parseable_label --contrastive-confidence-threshold 0.2 --contrastive-parseable-threshold 0.75 --contrastive-label-delta-q-min 0.2 --contrastive-label-z-min 0.5 --contrastive-label-pair-weight-min 0.3 --contrastive-require-pair-pass-gate --contrastive-use-pair-weights --contrastive-stratified-sampling --contrastive-stratify-step-bucket-size 2 --contrastive-stratify-include-no-corruption}"
      ;;
    DB3_STRATEGYQA_BRIDGE_FULL_RANK_SEED3)
      GROUP_TITLE="DB3 StrategyQA Bridge Full Seed3 (Ranking-Only)"
      GROUP_INTENTION="Main bridge test: external ranking pretrain followed by full in-domain StrategyQA ranking finetune."
      GROUP_OBSERVE="Use the proven stable Math-Shepherd stage-1 recipe, then continue on full CQR train/eval with 3 seeds."
      GROUP_EXPECT="If bridge works, mean in-domain corr_pair_acc/corr_auc should improve over direct transfer and challenge the strongest CQR baseline."
      SEEDS=(42 43 44)
      REFERENCE_COMPARE="1"
      STAGE1_MAX_PAIRS_PER_SOURCE="${STAGE1_MAX_PAIRS_PER_SOURCE:-6000}"
      STAGE1_MAX_PAIRS_TOTAL="${STAGE1_MAX_PAIRS_TOTAL:-12000}"
      STAGE1_MIN_PAIR_CONFIDENCE="${STAGE1_MIN_PAIR_CONFIDENCE:-0.55}"
      STAGE1_TRAIN_BATCH_SIZE="${STAGE1_TRAIN_BATCH_SIZE:-64}"
      STAGE1_EVAL_BATCH_SIZE="${STAGE1_EVAL_BATCH_SIZE:-64}"
      STAGE1_LR="${STAGE1_LR:-5e-5}"
      STAGE1_EPOCHS="${STAGE1_EPOCHS:-3}"
      STAGE1_MAX_TRAIN_SAMPLES="${STAGE1_MAX_TRAIN_SAMPLES:-}"
      STAGE1_MAX_EVAL_SAMPLES="${STAGE1_MAX_EVAL_SAMPLES:-}"
      STAGE2_MAX_TRAIN_SAMPLES="${STAGE2_MAX_TRAIN_SAMPLES:-}"
      STAGE2_MAX_EVAL_SAMPLES="${STAGE2_MAX_EVAL_SAMPLES:-}"
      STAGE2_TRAIN_MODE="ranking_only"
      STAGE2_TRAIN_BATCH_SIZE="${STAGE2_TRAIN_BATCH_SIZE:-192}"
      STAGE2_EVAL_BATCH_SIZE="${STAGE2_EVAL_BATCH_SIZE:-192}"
      STAGE2_LR="${STAGE2_LR:-5e-5}"
      STAGE2_EPOCHS="${STAGE2_EPOCHS:-4}"
      STAGE2_CHECKPOINT_METRIC="${STAGE2_CHECKPOINT_METRIC:-corr_auc}"
      STAGE2_TRAIN_EXTRA_DEFAULT="${STAGE2_TRAIN_EXTRA_DEFAULT:---lambda-contrastive 0.08 --contrastive-margin 0.02 --contrastive-pair-filter confidence_parseable_label --contrastive-confidence-threshold 0.2 --contrastive-parseable-threshold 0.75 --contrastive-label-delta-q-min 0.2 --contrastive-label-z-min 0.5 --contrastive-label-pair-weight-min 0.3 --contrastive-require-pair-pass-gate --contrastive-use-pair-weights --contrastive-stratified-sampling --contrastive-stratify-step-bucket-size 2 --contrastive-stratify-include-no-corruption}"
      ;;
    DB4_STRATEGYQA_BRIDGE_FULL_JOINT_SEED3)
      GROUP_TITLE="DB4 StrategyQA Bridge Full Seed3 (Joint)"
      GROUP_INTENTION="Main bridge variant: external ranking pretrain followed by full in-domain joint CQR-style training."
      GROUP_OBSERVE="Same stage-1 as DB3, but stage-2 keeps calibration auxiliary active while still selecting checkpoints by corruption ranking."
      GROUP_EXPECT="If bridge helps as representation shaping rather than pure ranking memorization, joint stage-2 may retain or improve CQR metrics."
      SEEDS=(42 43 44)
      REFERENCE_COMPARE="1"
      STAGE1_MAX_PAIRS_PER_SOURCE="${STAGE1_MAX_PAIRS_PER_SOURCE:-6000}"
      STAGE1_MAX_PAIRS_TOTAL="${STAGE1_MAX_PAIRS_TOTAL:-12000}"
      STAGE1_MIN_PAIR_CONFIDENCE="${STAGE1_MIN_PAIR_CONFIDENCE:-0.55}"
      STAGE1_TRAIN_BATCH_SIZE="${STAGE1_TRAIN_BATCH_SIZE:-64}"
      STAGE1_EVAL_BATCH_SIZE="${STAGE1_EVAL_BATCH_SIZE:-64}"
      STAGE1_LR="${STAGE1_LR:-5e-5}"
      STAGE1_EPOCHS="${STAGE1_EPOCHS:-3}"
      STAGE1_MAX_TRAIN_SAMPLES="${STAGE1_MAX_TRAIN_SAMPLES:-}"
      STAGE1_MAX_EVAL_SAMPLES="${STAGE1_MAX_EVAL_SAMPLES:-}"
      STAGE2_MAX_TRAIN_SAMPLES="${STAGE2_MAX_TRAIN_SAMPLES:-}"
      STAGE2_MAX_EVAL_SAMPLES="${STAGE2_MAX_EVAL_SAMPLES:-}"
      STAGE2_TRAIN_MODE="joint"
      STAGE2_TRAIN_BATCH_SIZE="${STAGE2_TRAIN_BATCH_SIZE:-192}"
      STAGE2_EVAL_BATCH_SIZE="${STAGE2_EVAL_BATCH_SIZE:-192}"
      STAGE2_LR="${STAGE2_LR:-5e-5}"
      STAGE2_EPOCHS="${STAGE2_EPOCHS:-4}"
      STAGE2_CHECKPOINT_METRIC="${STAGE2_CHECKPOINT_METRIC:-corr_auc}"
      STAGE2_TRAIN_EXTRA_DEFAULT="${STAGE2_TRAIN_EXTRA_DEFAULT:---calibration-loss bce_mse --calibration-bce-weight 1.0 --calibration-mse-weight 0.5 --calibration-sample-weighting q_weight_parseable --calibration-weight-floor 0.1 --calibration-weight-gamma 1.0 --lambda-contrastive 0.08 --contrastive-margin 0.02 --contrastive-pair-filter confidence_parseable_label --contrastive-confidence-threshold 0.2 --contrastive-parseable-threshold 0.75 --contrastive-label-delta-q-min 0.2 --contrastive-label-z-min 0.5 --contrastive-label-pair-weight-min 0.3 --contrastive-require-pair-pass-gate --contrastive-use-pair-weights --contrastive-stratified-sampling --contrastive-stratify-step-bucket-size 2 --contrastive-stratify-include-no-corruption}"
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_DBR_GROUP=$ACTIVE_PHASE_DBR_GROUP" >&2
      echo "Supported groups:" >&2
      echo "  DB1_STRATEGYQA_BRIDGE_SMOKE_RANK" >&2
      echo "  DB2_STRATEGYQA_BRIDGE_SMOKE_JOINT" >&2
      echo "  DB3_STRATEGYQA_BRIDGE_FULL_RANK_SEED3" >&2
      echo "  DB4_STRATEGYQA_BRIDGE_FULL_JOINT_SEED3" >&2
      exit 1
      ;;
  esac
}

append_seed_result() {
  local seed="$1"
  local pair_run_dir="$2"
  local stage1_run_dir="$3"
  local stage1_ext_eval_dir="$4"
  local stage2_run_dir="$5"
  local reference_metrics_path="$6"
  "$PYTHON_BIN" - "$seed" "$pair_run_dir" "$stage1_run_dir" "$stage1_ext_eval_dir" "$stage2_run_dir" "$reference_metrics_path" "$SEED_RESULTS_JSONL" <<'PY'
import json
import sys
from pathlib import Path

seed = int(sys.argv[1])
pair_run_dir = Path(sys.argv[2])
stage1_run_dir = Path(sys.argv[3])
stage1_ext_eval_dir = Path(sys.argv[4])
stage2_run_dir = Path(sys.argv[5])
reference_metrics_path_raw = sys.argv[6]
out_path = Path(sys.argv[7])

pair_summary = json.loads((pair_run_dir / 'summary.json').read_text(encoding='utf-8'))
stage1_ext = json.loads((stage1_ext_eval_dir / 'metrics.json').read_text(encoding='utf-8'))
stage2_eval = json.loads((stage2_run_dir / 'eval_metrics.json').read_text(encoding='utf-8'))
stage2_best = dict(stage2_eval.get('best') or stage2_eval.get('final') or {})
corruption = dict(stage2_best.get('corruption') or {})
calibration = dict(stage2_best.get('calibration') or {})
reference = None
if reference_metrics_path_raw.strip():
    ref_path = Path(reference_metrics_path_raw)
    if ref_path.exists():
        ref_payload = json.loads(ref_path.read_text(encoding='utf-8'))
        ref_corruption = dict(ref_payload.get('corruption') or {})
        reference = {
            'pair_acc': ref_corruption.get('pair_accuracy'),
            'auc': ref_corruption.get('auc_clean_vs_corrupt'),
        }

row = {
    'seed': seed,
    'pair_run_dir': str(pair_run_dir),
    'stage1_run_dir': str(stage1_run_dir),
    'stage1_ext_eval_dir': str(stage1_ext_eval_dir),
    'stage2_run_dir': str(stage2_run_dir),
    'num_train_pairs': int(pair_summary.get('num_train_rows', 0)),
    'num_val_pairs': int(pair_summary.get('num_validation_rows', 0)),
    'pair_sources': dict((pair_summary.get('train_summary') or {}).get('by_source') or {}),
    'pair_mean_conf': float((pair_summary.get('train_summary') or {}).get('mean_pair_confidence', 0.0)),
    'stage1_ext_pair_acc': float(stage1_ext.get('pair_accuracy', 0.0)),
    'stage1_ext_auc': float(stage1_ext.get('auc_chosen_vs_rejected', 0.0)),
    'stage1_ext_mean_margin': float(stage1_ext.get('mean_margin', 0.0)),
    'stage2_corr_pair_acc': float(corruption.get('pair_accuracy')) if corruption.get('pair_accuracy') is not None else None,
    'stage2_corr_auc': float(corruption.get('auc_clean_vs_corrupt')) if corruption.get('auc_clean_vs_corrupt') is not None else None,
    'stage2_corr_mean_margin': float(corruption.get('mean_margin')) if corruption.get('mean_margin') is not None else None,
    'stage2_brier': float(calibration.get('brier_score')) if calibration.get('brier_score') is not None else None,
    'reference_pair_acc': (float(reference['pair_acc']) if reference and reference['pair_acc'] is not None else None),
    'reference_auc': (float(reference['auc']) if reference and reference['auc'] is not None else None),
}
if row['reference_pair_acc'] is not None and row['stage2_corr_pair_acc'] is not None:
    row['delta_pair_vs_reference'] = row['stage2_corr_pair_acc'] - row['reference_pair_acc']
else:
    row['delta_pair_vs_reference'] = None
if row['reference_auc'] is not None and row['stage2_corr_auc'] is not None:
    row['delta_auc_vs_reference'] = row['stage2_corr_auc'] - row['reference_auc']
else:
    row['delta_auc_vs_reference'] = None
with out_path.open('a', encoding='utf-8') as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
PY
}

render_final_summary() {
  "$PYTHON_BIN" - "$ACTIVE_PHASE_DBR_GROUP" "$GROUP_TITLE" "$RUN_PREFIX" "$GROUP_INTENTION" "$GROUP_OBSERVE" "$GROUP_EXPECT" "$SEED_RESULTS_JSONL" <<'PY'
import json
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

group_id, group_title, run_prefix, group_intention, group_observe, group_expect, seed_results_path = sys.argv[1:8]
rows = [json.loads(line) for line in Path(seed_results_path).read_text(encoding='utf-8').splitlines() if line.strip()]

def mean_or_none(values):
    vals = [v for v in values if v is not None]
    return (st.mean(vals) if vals else None)

def pstdev_or_none(values):
    vals = [v for v in values if v is not None]
    return (st.pstdev(vals) if len(vals) > 1 else 0.0 if len(vals) == 1 else None)

mean_stage1_pair = mean_or_none([r.get('stage1_ext_pair_acc') for r in rows])
mean_stage1_auc = mean_or_none([r.get('stage1_ext_auc') for r in rows])
mean_stage2_pair = mean_or_none([r.get('stage2_corr_pair_acc') for r in rows])
mean_stage2_auc = mean_or_none([r.get('stage2_corr_auc') for r in rows])
std_stage2_pair = pstdev_or_none([r.get('stage2_corr_pair_acc') for r in rows])
std_stage2_auc = pstdev_or_none([r.get('stage2_corr_auc') for r in rows])
mean_delta_pair = mean_or_none([r.get('delta_pair_vs_reference') for r in rows])
mean_delta_auc = mean_or_none([r.get('delta_auc_vs_reference') for r in rows])
reference_available = any(r.get('reference_pair_acc') is not None for r in rows)

print('# Phase D Bridge Suite Summary')
print()
print(f'- generated_at: {datetime.now(timezone.utc).isoformat()}')
print(f'- group_id: {group_id}')
print(f'- group_title: {group_title}')
print(f'- run_prefix: {run_prefix}')
print('- status: ok')
print(f'- seed_results_path: {seed_results_path}')
print(f'- group_intention: {group_intention}')
print(f'- observe: {group_observe}')
print(f'- expect: {group_expect}')
print()
print('## Per-Seed Metrics')
print()
print('| seed | train_pairs | val_pairs | stage1_ext_pair_acc | stage1_ext_auc | stage2_corr_pair_acc | stage2_corr_auc | delta_pair_vs_ref | delta_auc_vs_ref |')
print('|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for row in rows:
    def fmt(v):
        return 'N/A' if v is None else f'{float(v):.4f}'
    print(
        f"| {int(row['seed'])} | {int(row['num_train_pairs'])} | {int(row['num_val_pairs'])} | "
        f"{fmt(row.get('stage1_ext_pair_acc'))} | {fmt(row.get('stage1_ext_auc'))} | "
        f"{fmt(row.get('stage2_corr_pair_acc'))} | {fmt(row.get('stage2_corr_auc'))} | "
        f"{fmt(row.get('delta_pair_vs_reference'))} | {fmt(row.get('delta_auc_vs_reference'))} |"
    )
print()
print('## Aggregated Bridge')
print()
print(f"- mean_stage1_ext_pair_acc: `{mean_stage1_pair:.6f}`" if mean_stage1_pair is not None else '- mean_stage1_ext_pair_acc: `N/A`')
print(f"- mean_stage1_ext_auc: `{mean_stage1_auc:.6f}`" if mean_stage1_auc is not None else '- mean_stage1_ext_auc: `N/A`')
print(f"- mean_stage2_corr_pair_acc: `{mean_stage2_pair:.6f}`" if mean_stage2_pair is not None else '- mean_stage2_corr_pair_acc: `N/A`')
print(f"- mean_stage2_corr_auc: `{mean_stage2_auc:.6f}`" if mean_stage2_auc is not None else '- mean_stage2_corr_auc: `N/A`')
print(f"- std_stage2_corr_pair_acc: `{std_stage2_pair:.6f}`" if std_stage2_pair is not None else '- std_stage2_corr_pair_acc: `N/A`')
print(f"- std_stage2_corr_auc: `{std_stage2_auc:.6f}`" if std_stage2_auc is not None else '- std_stage2_corr_auc: `N/A`')
if reference_available:
    print(f"- mean_delta_pair_vs_reference: `{mean_delta_pair:.6f}`" if mean_delta_pair is not None else '- mean_delta_pair_vs_reference: `N/A`')
    print(f"- mean_delta_auc_vs_reference: `{mean_delta_auc:.6f}`" if mean_delta_auc is not None else '- mean_delta_auc_vs_reference: `N/A`')
    if mean_delta_pair is not None and mean_delta_auc is not None:
        bridge_pass = (mean_delta_pair > 0.0) and (mean_delta_auc > 0.0)
        print(f'- bridge_beats_reference: `{bridge_pass}`')
PY
}

run_one_seed() {
  local seed="$1"
  local split_seed="$seed"
  if [[ "$DPAIR_SPLIT_MODE" == "shared" ]]; then
    split_seed="$DPAIR_SPLIT_SEED"
  fi

  local pair_run_name="${RUN_PREFIX}_$(echo "$ACTIVE_PHASE_DBR_GROUP" | tr '[:upper:]' '[:lower:]')_sharedsplit_s${split_seed}_pairs"
  local pair_run_dir=""
  if [[ "$DPAIR_SPLIT_MODE" == "shared" && -n "$SHARED_PAIR_RUN_DIR" ]]; then
    pair_run_dir="$SHARED_PAIR_RUN_DIR"
  else
    local prep_args=(
      -u scripts/phase_d_prepare_external_pairs.py
      --run-name "$pair_run_name"
      --output-root "$PAIR_OUTPUT_ROOT"
      --seed "$split_seed"
      --validation-ratio "$VALIDATION_RATIO"
      --build-step-converted
      --math-shepherd-path "$MATH_SHEPHERD_PATH"
      --max-pairs-per-source "$STAGE1_MAX_PAIRS_PER_SOURCE"
      --max-pairs-total "$STAGE1_MAX_PAIRS_TOTAL"
      --min-pair-confidence "$STAGE1_MIN_PAIR_CONFIDENCE"
      --min-chars "12"
      --max-length-ratio "3.0"
      --max-token-overlap "0.99"
      --max-pairs-per-sample "1"
    )
    CURRENT_STAGE="seed_${seed}_prepare_pairs"
    {
      log_line "[seed=${seed}] Prepare shared Math-Shepherd pairs"
      log_line "[seed=${seed}] pair_split_mode=${DPAIR_SPLIT_MODE} split_seed=${split_seed}"
      log_line "[seed=${seed}] Command: $PYTHON_BIN ${prep_args[*]}"
    } | tee -a "$SUITE_LOG_FILE" >/dev/null
    "$PYTHON_BIN" "${prep_args[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"
    pair_run_dir="$(latest_dir_for_prefix "${PAIR_OUTPUT_ROOT}/${pair_run_name}__*")"
    if [[ "$DPAIR_SPLIT_MODE" == "shared" ]]; then
      SHARED_PAIR_RUN_DIR="$pair_run_dir"
    fi
  fi

  require_file "${pair_run_dir}/train_pairs.jsonl" "bridge train_pairs"
  require_file "${pair_run_dir}/validation_pairs.jsonl" "bridge validation_pairs"
  require_file "${pair_run_dir}/summary.json" "bridge pair summary"

  local stage1_run_name="${RUN_PREFIX}_$(echo "$ACTIVE_PHASE_DBR_GROUP" | tr '[:upper:]' '[:lower:]')_s${seed}_stage1"
  local stage1_train_args=(
    -u scripts/phase_b_train_value.py
    --train-dir "$PHASE_C_STAGE1_TRAIN_DIR"
    --eval-dir "$PHASE_C_STAGE1_EVAL_DIR"
    --run-name "$stage1_run_name"
    --output-root "$STAGE1_OUTPUT_ROOT"
    --target-source q_mean_smoothed
    --target-source-missing-policy fail
    --require-cuda
    --dtype bfloat16
    --device-map auto
    --learning-rate "$STAGE1_LR"
    --num-train-epochs "$STAGE1_EPOCHS"
    --per-device-train-batch-size "$STAGE1_TRAIN_BATCH_SIZE"
    --per-device-eval-batch-size "$STAGE1_EVAL_BATCH_SIZE"
    --seed "$seed"
    --external-pair-jsonl "${pair_run_dir}/train_pairs.jsonl"
    --external-pair-weight 1.0
    --external-pair-source-balance uniform
    --external-pair-permutation-mode "$STAGE1_EXTERNAL_PAIR_PERM_MODE"
    --external-pair-min-confidence "$STAGE1_MIN_PAIR_CONFIDENCE"
    --external-pair-use-confidence-weights
    --train-mode ranking_only
    --calibration-loss mse
    --checkpoint-selection-metric ranking_score
    --posthoc-calibration none
    --feature-cache-root "$FEATURE_CACHE_ROOT"
    --feature-cache-mode "$FEATURE_CACHE_MODE"
    --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC"
    --use-contrastive-loss
    --external-pair-only
    --lambda-contrastive 1.0
    --contrastive-margin 0.2
    --dropout-prob 0.1
    --anti-saturation-weight 0.03
    --anti-saturation-logit-threshold 4.0
  )
  # 中文：smoke 组需要把 stage-1 也裁小，否则虽然 external pair 数少，
  # 但仍会读取整套 Phase C train/eval 特征，导致“烟测不烟”。
  if [[ -n "${STAGE1_MAX_TRAIN_SAMPLES:-}" ]]; then
    stage1_train_args+=(--max-train-samples "$STAGE1_MAX_TRAIN_SAMPLES")
  fi
  if [[ -n "${STAGE1_MAX_EVAL_SAMPLES:-}" ]]; then
    stage1_train_args+=(--max-eval-samples "$STAGE1_MAX_EVAL_SAMPLES")
  fi
  if [[ "$STRICT_DETERMINISM" -eq 1 ]]; then
    stage1_train_args+=(--strict-determinism)
  fi
  append_extra_args stage1_train_args "${STAGE1_TRAIN_EXTRA_ARGS:-}"

  CURRENT_STAGE="seed_${seed}_stage1_train"
  {
    log_line "[seed=${seed}] Stage-1 external ranking pretrain"
    log_line "[seed=${seed}] Command: $PYTHON_BIN ${stage1_train_args[*]}"
  } | tee -a "$SUITE_LOG_FILE" >/dev/null
  "$PYTHON_BIN" "${stage1_train_args[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

  local stage1_run_dir=""
  stage1_run_dir="$(latest_dir_for_prefix "${STAGE1_OUTPUT_ROOT}/${stage1_run_name}_*")"
  require_file "${stage1_run_dir}/best_value_head.pt" "stage-1 best checkpoint"

  local stage1_ext_eval_name="${RUN_PREFIX}_$(echo "$ACTIVE_PHASE_DBR_GROUP" | tr '[:upper:]' '[:lower:]')_s${seed}_stage1_ext_eval"
  local stage1_ext_eval_args=(
    -u scripts/phase_d_eval_external_pairs.py
    --value-run-dir "$stage1_run_dir"
    --external-pair-jsonl "${pair_run_dir}/validation_pairs.jsonl"
    --run-name "$stage1_ext_eval_name"
    --output-root "$STAGE1_EVAL_OUTPUT_ROOT"
    --checkpoint-name best
    --batch-size "$STAGE1_EVAL_BATCH_SIZE"
    --require-cuda
    --feature-cache-root "$FEATURE_CACHE_ROOT"
    --feature-cache-mode "$FEATURE_CACHE_MODE"
    --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC"
  )
  CURRENT_STAGE="seed_${seed}_stage1_eval"
  {
    log_line "[seed=${seed}] Stage-1 external held-out eval"
    log_line "[seed=${seed}] Command: $PYTHON_BIN ${stage1_ext_eval_args[*]}"
  } | tee -a "$SUITE_LOG_FILE" >/dev/null
  "$PYTHON_BIN" "${stage1_ext_eval_args[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

  local stage1_ext_eval_dir=""
  stage1_ext_eval_dir="$(latest_dir_for_prefix "${STAGE1_EVAL_OUTPUT_ROOT}/${stage1_ext_eval_name}_*")"
  require_file "${stage1_ext_eval_dir}/metrics.json" "stage-1 external eval metrics"

  local stage2_run_name="${RUN_PREFIX}_$(echo "$ACTIVE_PHASE_DBR_GROUP" | tr '[:upper:]' '[:lower:]')_s${seed}_stage2"
  local stage2_train_args=(
    -u scripts/phase_b_train_value.py
    --train-dir "$PHASE_C_STAGE2_TRAIN_DIR"
    --eval-dir "$PHASE_C_STAGE2_EVAL_DIR"
    --run-name "$stage2_run_name"
    --output-root "$STAGE2_OUTPUT_ROOT"
    --init-value-head-path "${stage1_run_dir}/best_value_head.pt"
    --target-source q_mean_smoothed
    --target-source-missing-policy fail
    --require-cuda
    --dtype bfloat16
    --device-map auto
    --learning-rate "$STAGE2_LR"
    --num-train-epochs "$STAGE2_EPOCHS"
    --per-device-train-batch-size "$STAGE2_TRAIN_BATCH_SIZE"
    --per-device-eval-batch-size "$STAGE2_EVAL_BATCH_SIZE"
    --seed "$seed"
    --train-mode "$STAGE2_TRAIN_MODE"
    --checkpoint-selection-metric "$STAGE2_CHECKPOINT_METRIC"
    --posthoc-calibration none
    --feature-cache-root "$FEATURE_CACHE_ROOT"
    --feature-cache-mode "$FEATURE_CACHE_MODE"
    --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC"
    --use-contrastive-loss
  )
  if [[ -n "${STAGE2_MAX_TRAIN_SAMPLES:-}" ]]; then
    stage2_train_args+=(--max-train-samples "$STAGE2_MAX_TRAIN_SAMPLES")
  fi
  if [[ -n "${STAGE2_MAX_EVAL_SAMPLES:-}" ]]; then
    stage2_train_args+=(--max-eval-samples "$STAGE2_MAX_EVAL_SAMPLES")
  fi
  if [[ "$STRICT_DETERMINISM" -eq 1 ]]; then
    stage2_train_args+=(--strict-determinism)
  fi
  append_extra_args stage2_train_args "$STAGE2_TRAIN_EXTRA_DEFAULT"
  append_extra_args stage2_train_args "${STAGE2_TRAIN_EXTRA_ARGS:-}"

  CURRENT_STAGE="seed_${seed}_stage2_train"
  {
    log_line "[seed=${seed}] Stage-2 in-domain bridge training"
    log_line "[seed=${seed}] Command: $PYTHON_BIN ${stage2_train_args[*]}"
  } | tee -a "$SUITE_LOG_FILE" >/dev/null
  "$PYTHON_BIN" "${stage2_train_args[@]}" 2>&1 | tee -a "$SUITE_LOG_FILE"

  local stage2_run_dir=""
  stage2_run_dir="$(latest_dir_for_prefix "${STAGE2_OUTPUT_ROOT}/${stage2_run_name}_*")"
  require_file "${stage2_run_dir}/eval_metrics.json" "stage-2 eval metrics"

  CURRENT_STAGE="seed_${seed}_append_result"
  local reference_path=""
  if [[ "$REFERENCE_COMPARE" == "1" && -f "$REFERENCE_C2_METRICS" ]]; then
    reference_path="$REFERENCE_C2_METRICS"
  fi
  append_seed_result "$seed" "$pair_run_dir" "$stage1_run_dir" "$stage1_ext_eval_dir" "$stage2_run_dir" "$reference_path"
}

main() {
  mkdir -p "$LOG_ROOT"
  : > "$SUITE_LOG_FILE"
  : > "$SEED_RESULTS_JSONL"

  resolve_group

  # 中文：PyTorch 仅开启 deterministic flag 还不够；对于 CUDA>=10.2 的
  # cuBLAS 路径，还需要在进程启动前设置工作区配置，否则只会收到 warning，
  # 但实际并不能保证最严格的确定性。
  if [[ "$STRICT_DETERMINISM" -eq 1 ]]; then
    export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  fi

  require_file "$MATH_SHEPHERD_PATH" "Math-Shepherd JSONL"
  require_dir "$PHASE_C_STAGE1_TRAIN_DIR" "Stage-1 Phase C train dir"
  require_dir "$PHASE_C_STAGE1_EVAL_DIR" "Stage-1 Phase C eval dir"
  require_dir "$PHASE_C_STAGE2_TRAIN_DIR" "Stage-2 Phase C train dir"
  require_dir "$PHASE_C_STAGE2_EVAL_DIR" "Stage-2 Phase C eval dir"
  require_file "${PHASE_C_STAGE1_TRAIN_DIR}/manifest.json" "Stage-1 train manifest"
  require_file "${PHASE_C_STAGE1_EVAL_DIR}/manifest.json" "Stage-1 eval manifest"
  require_file "${PHASE_C_STAGE2_TRAIN_DIR}/manifest.json" "Stage-2 train manifest"
  require_file "${PHASE_C_STAGE2_EVAL_DIR}/manifest.json" "Stage-2 eval manifest"

  {
    log_line "Phase D Bridge Suite"
    log_line "group_id=${ACTIVE_PHASE_DBR_GROUP}"
    log_line "group_title=${GROUP_TITLE}"
    log_line "group_intention=${GROUP_INTENTION}"
    log_line "group_observe=${GROUP_OBSERVE}"
    log_line "group_expect=${GROUP_EXPECT}"
    log_line "run_prefix=${RUN_PREFIX}"
    log_line "stage1_train_dir=${PHASE_C_STAGE1_TRAIN_DIR}"
    log_line "stage1_eval_dir=${PHASE_C_STAGE1_EVAL_DIR}"
    log_line "stage2_train_dir=${PHASE_C_STAGE2_TRAIN_DIR}"
    log_line "stage2_eval_dir=${PHASE_C_STAGE2_EVAL_DIR}"
    log_line "reference_c2_metrics=${REFERENCE_C2_METRICS}"
    log_line "stage1_max_train_samples=${STAGE1_MAX_TRAIN_SAMPLES:-<full>}"
    log_line "stage1_max_eval_samples=${STAGE1_MAX_EVAL_SAMPLES:-<full>}"
    log_line "stage2_max_train_samples=${STAGE2_MAX_TRAIN_SAMPLES:-<full>}"
    log_line "stage2_max_eval_samples=${STAGE2_MAX_EVAL_SAMPLES:-<full>}"
    log_line "seeds=${SEEDS[*]}"
    log_line "pair_split_mode=${DPAIR_SPLIT_MODE}"
    log_line "pair_split_seed=${DPAIR_SPLIT_SEED}"
    log_line "strict_determinism=${STRICT_DETERMINISM}"
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
