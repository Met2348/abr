#!/usr/bin/env bash
# Stable wrapper around the D6-T triplet-validation suite.
#
# Why this file exists:
# - The raw D6-T suite exposes many knobs. That is good for diagnosis, but bad
#   for routine execution when the immediate goal is "run one conservative suite
#   with minimal operator error".
# - This wrapper freezes the currently safest StrategyQA/Math-Shepherd recipe:
#   shared external-pair split, deterministic ordering, and conservative ranking
#   hyperparameters.
#
# What this file does:
# 1) Fill default env vars for the stable DT2 group.
# 2) Run lightweight preflight checks on required inputs.
# 3) Delegate the actual work to `scripts/run_phase_d_triplet_validation_suite.sh`.
#
# Example:
#   RUN_PREFIX=phase_d6t_stable_0306 \
#   CUDA_VISIBLE_DEVICES=1 \
#   bash scripts/run_phase_d6t_stable_suite.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ACTIVE_PHASE_D6T_GROUP="${ACTIVE_PHASE_D6T_GROUP:-DT2_MATH_SHEPHERD_SEED3_STABLE}"
RUN_PREFIX="${RUN_PREFIX:-phase_d6t_stable_suite}"
MATH_SHEPHERD_PATH="${MATH_SHEPHERD_PATH:-assets/external_datasets/peiyi_math_shepherd/math-shepherd.jsonl}"
#
# Prefer the original full k=8 StrategyQA rollout artifacts for this stable
# bundle. They are the cleanest "base Phase C" inputs and avoid accidentally
# inheriting extra transformations from later D-stage derived directories.
# 中文：stable 套件默认钉死这两个输入目录，避免“自动找最新目录”时捞到
# 半截 artifact 或别的实验链路产物，导致 seed 稳定性结论被污染。
PHASE_C_TRAIN_DIR="${PHASE_C_TRAIN_DIR:-assets/artifacts/phase_c_data/strategyqa/strategyqa_value_rollouts_k8_train_full__59aed6ac5f99}"
PHASE_C_EVAL_DIR="${PHASE_C_EVAL_DIR:-assets/artifacts/phase_c_data/strategyqa/strategyqa_value_rollouts_k8_val_full__c5cb0e294b2c}"

# Stable-by-default controls.
D6T_PAIR_SPLIT_MODE="${D6T_PAIR_SPLIT_MODE:-shared}"
D6T_PAIR_SPLIT_SEED="${D6T_PAIR_SPLIT_SEED:-42}"
D6T_EXTERNAL_PAIR_PERM_MODE="${D6T_EXTERNAL_PAIR_PERM_MODE:-stable_hash}"
D6T_STRICT_DETERMINISM="${D6T_STRICT_DETERMINISM:-1}"

# Export the resolved defaults so the delegated suite receives exactly the
# configuration shown in this wrapper's preflight logs.
# 中文：这里只赋值不 export 的话，子脚本读不到这些默认值，
# 会退回到它自己的自动解析逻辑。之前就因此出现过“日志看起来对，实际跑错目录”的问题。
export ACTIVE_PHASE_D6T_GROUP
export RUN_PREFIX
export MATH_SHEPHERD_PATH
export PHASE_C_TRAIN_DIR
export PHASE_C_EVAL_DIR
export D6T_PAIR_SPLIT_MODE
export D6T_PAIR_SPLIT_SEED
export D6T_EXTERNAL_PAIR_PERM_MODE
export D6T_STRICT_DETERMINISM

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

require_file "$MATH_SHEPHERD_PATH" "Math-Shepherd JSONL"
require_dir "assets/artifacts/phase_c_data/strategyqa" "Phase C StrategyQA artifact root"
require_dir "$PHASE_C_TRAIN_DIR" "Phase C train dir"
require_dir "$PHASE_C_EVAL_DIR" "Phase C eval dir"
require_file "$PHASE_C_TRAIN_DIR/manifest.json" "Phase C train manifest"
require_file "$PHASE_C_EVAL_DIR/manifest.json" "Phase C eval manifest"
require_file "$PHASE_C_TRAIN_DIR/prefixes.jsonl" "Phase C train prefixes"
require_file "$PHASE_C_EVAL_DIR/prefixes.jsonl" "Phase C eval prefixes"
require_file "$PHASE_C_TRAIN_DIR/rollout_targets.jsonl" "Phase C train rollout targets"
require_file "$PHASE_C_EVAL_DIR/rollout_targets.jsonl" "Phase C eval rollout targets"

echo "[stable-suite] group=${ACTIVE_PHASE_D6T_GROUP}"
echo "[stable-suite] run_prefix=${RUN_PREFIX}"
echo "[stable-suite] pair_split_mode=${D6T_PAIR_SPLIT_MODE}"
echo "[stable-suite] pair_split_seed=${D6T_PAIR_SPLIT_SEED}"
echo "[stable-suite] external_pair_perm_mode=${D6T_EXTERNAL_PAIR_PERM_MODE}"
echo "[stable-suite] strict_determinism=${D6T_STRICT_DETERMINISM}"
echo "[stable-suite] math_shepherd_path=${MATH_SHEPHERD_PATH}"
echo "[stable-suite] phase_c_train_dir=${PHASE_C_TRAIN_DIR}"
echo "[stable-suite] phase_c_eval_dir=${PHASE_C_EVAL_DIR}"

exec bash scripts/run_phase_d_triplet_validation_suite.sh
