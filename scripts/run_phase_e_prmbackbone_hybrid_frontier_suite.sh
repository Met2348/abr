#!/usr/bin/env bash
# Safe wrapper around the ProcessBench-hybrid Phase E suite using the
# Math-PRM backbone by default.
#
# 中文：
# 这个包装器不改原始 hybrid suite 的逻辑，只把当前更可信、更有价值的默认值固定下来：
# 1. 默认 backbone = Qwen2.5-Math-PRM-7B
# 2. 默认 full-benchmark eval（BENCH_MAX_SAMPLES=1000）
# 3. 默认较保守的 batch，减少 VRAM 波动

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUN_PREFIX="${RUN_PREFIX:-phase_e_prmbackbone_frontier_$(date +%m%d_%H%M)}"
ACTIVE_PHASE_E_PB_HYBRID_GROUP="${ACTIVE_PHASE_E_PB_HYBRID_GROUP:-PH3_PRM_LOCAL_TA15_MSGRID5_ARCH_SWEEP_SMOKE}"
MODEL_PATH="${MODEL_PATH:-assets/models/Qwen2.5-Math-PRM-7B}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-24}"
BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-1000}"

export RUN_PREFIX
export ACTIVE_PHASE_E_PB_HYBRID_GROUP
export MODEL_PATH
export TRAIN_BATCH_SIZE
export EVAL_BATCH_SIZE
export BENCH_MAX_SAMPLES

exec bash scripts/run_phase_e_processbench_hybrid_suite.sh
