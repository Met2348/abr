#!/usr/bin/env bash
# Changelog (prepend-only, newest first):
# 2026-03-13: Initial version. Launch a coordinated overnight Phase E frontier
#             program across multiple free GPUs, with explicit literature-
#             motivated experiment selection and persistent job manifests.
#
# Phase E Overnight Frontier Launcher
# ===================================
# English
# -------
# This launcher exists for one practical reason:
# the repo already has many focused Phase E suites, but the user often needs
# one "sleep-safe" command that:
# 1. chooses a small set of high-value directions,
# 2. maps them onto currently free GPUs conservatively,
# 3. persists PID / log / command metadata,
# 4. lets tomorrow-morning diagnosis start from a clean manifest instead of
#    terminal scrollback.
#
# The selected directions are intentionally orthogonal:
# - `F2_DUAL_HEAD_PBR10`:
#     factorize local vs terminal pressure, directly attacking the known
#     terminal blind spot.
# - `F3_LORA_PBR10`:
#     test whether the frozen-backbone representation is the remaining ceiling.
# - `PH2_PRM_LOCAL_TA10_MSGRID10_ARCH_SWEEP_SMOKE`:
#     build a benchmark-oriented hybrid artifact rather than trusting naive
#     source transfer.
# - `CR1_CURATED_CENTER_GATE_SMOKE`:
#     queue a curation + centering repair after the hybrid run finishes.
# - `terminal_ratio_sweep`:
#     queue a supervision-geometry sweep after the dual-head run finishes.
#
# 中文
# ----
# 这个 launcher 的目标很实际：
# 当前仓库已经有很多专门的 Phase E suite，但夜间跑实验时，需要一个
# “睡前一键启动、第二天早上能直接看 manifest”的统一入口。
#
# 它负责：
# 1. 选择少而硬、彼此正交的高价值方向；
# 2. 保守地映射到当前空闲 GPU；
# 3. 持久化 PID / log / command 信息；
# 4. 避免第二天早上只能翻终端历史记录。
#
# 当前选择的实验方向是：
# - `F2_DUAL_HEAD_PBR10`
#   直接针对已知 terminal blind spot，把 local 和 terminal 压力拆开。
# - `F3_LORA_PBR10`
#   检查 frozen backbone 表征是否已经成为真正瓶颈。
# - `PH2_PRM_LOCAL_TA10_MSGRID10_ARCH_SWEEP_SMOKE`
#   不再相信简单 source transfer，而是显式构造 benchmark-oriented hybrid。
# - `CR1_CURATED_CENTER_GATE_SMOKE`
#   在 hybrid 跑完后继续测试 curated + centering 修复。
# - `terminal_ratio_sweep`
#   在 dual-head 跑完后继续扫 supervision geometry。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUN_PREFIX="${RUN_PREFIX:-phase_e_overnight_$(date +%m%d_%H%M)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# 优先 GPU 顺序遵循当前项目约定：1 -> 2 -> 3 -> 0。
# 但考虑到本机当前 GPU2 常驻拥挤，这个 launcher 默认用 1/3/0，
# 明确避开 GPU2，减少夜间 OOM 和抢占风险。
GPU_F2="${GPU_F2:-1}"
GPU_F3="${GPU_F3:-0}"
GPU_HYBRID="${GPU_HYBRID:-3}"
GPU_CURATED="${GPU_CURATED:-3}"
GPU_TA="${GPU_TA:-1}"

BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-256}"
RECIPE_RISK_POLICY="${RECIPE_RISK_POLICY:-error}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
LAUNCH_LOG="${LOG_ROOT}/launch.log"
MANIFEST_JSONL="${LOG_ROOT}/launch_manifest.jsonl"
PLAN_MD="${LOG_ROOT}/overnight_plan.md"

mkdir -p "$LOG_ROOT"
: > "$LAUNCH_LOG"
: > "$MANIFEST_JSONL"

log_line() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1" | tee -a "$LAUNCH_LOG" >&2
}

gpu_status_snapshot() {
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
}

launch_job() {
  local job_name="$1"
  local gpu_id="$2"
  local command="$3"
  local log_file="${LOG_ROOT}/${job_name}.log"
  local pid
  log_line "LAUNCH ${job_name} on GPU${gpu_id}"
  log_line "CMD ${command}"
  pid="$(nohup bash -lc "cd '$REPO_ROOT' && export PYTORCH_CUDA_ALLOC_CONF='$PYTORCH_CUDA_ALLOC_CONF' && ${command}" >"$log_file" 2>&1 < /dev/null & echo \$!)"
  "$PYTHON_BIN" - "$job_name" "$gpu_id" "$pid" "$log_file" "$command" "$MANIFEST_JSONL" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

job_name, gpu_id, pid, log_file, command, out_path = sys.argv[1:]
row = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "job_name": job_name,
    "gpu_id": int(gpu_id),
    "pid": int(pid),
    "log_file": log_file,
    "command": command,
    "status": "launched",
}
with Path(out_path).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps(row, ensure_ascii=False))
PY
}

render_plan_md() {
  cat >"$PLAN_MD" <<EOF
# Phase E Overnight Frontier Plan

- run_prefix: \`${RUN_PREFIX}\`
- launch_log: \`${LAUNCH_LOG}\`
- manifest_jsonl: \`${MANIFEST_JSONL}\`

## GPU Snapshot At Launch

\`\`\`
$(gpu_status_snapshot)
\`\`\`

## Experiment Design

1. \`F2_DUAL_HEAD_PBR10\`
   - purpose: split local vs terminal scoring pressure
   - source: strong shared \`PBR10\` artifact
   - gpu: \`${GPU_F2}\`
2. \`F3_LORA_PBR10\`
   - purpose: probe frozen-backbone representation ceiling
   - source: same \`PBR10\` artifact for fair comparison
   - gpu: \`${GPU_F3}\`
3. \`PH2_PRM_LOCAL_TA10_MSGRID10_ARCH_SWEEP_SMOKE\`
   - purpose: benchmark-oriented hybrid curation instead of naive transfer
   - gpu: \`${GPU_HYBRID}\`
4. queued \`CR1_CURATED_CENTER_GATE_SMOKE\`
   - purpose: reward-centering + gated-MLP repair after hybrid
   - gpu: \`${GPU_CURATED}\`
5. queued terminal ratio sweep
   - purpose: supervision geometry sweep after dual-head
   - gpu: \`${GPU_TA}\`

## Morning Check

1. \`cat ${MANIFEST_JSONL}\`
2. \`ls -dt ${LOG_ROOT}/*.log\`
3. \`tail -n 80 ${LOG_ROOT}/*.log\`

EOF
}

render_plan_md

log_line "GPU snapshot before launch:"
gpu_status_snapshot | tee -a "$LAUNCH_LOG" >&2

CMD_F2="CUDA_VISIBLE_DEVICES=${GPU_F2} RUN_PREFIX=${RUN_PREFIX}_f2 ACTIVE_PHASE_E_FRONTIER_GROUP=F2_DUAL_HEAD_PBR10 BENCH_MAX_SAMPLES=${BENCH_MAX_SAMPLES} RECIPE_RISK_POLICY=${RECIPE_RISK_POLICY} bash scripts/run_phase_e_frontier_suite.sh"
CMD_F3="CUDA_VISIBLE_DEVICES=${GPU_F3} RUN_PREFIX=${RUN_PREFIX}_f3 ACTIVE_PHASE_E_FRONTIER_GROUP=F3_LORA_PBR10 BENCH_MAX_SAMPLES=${BENCH_MAX_SAMPLES} RECIPE_RISK_POLICY=${RECIPE_RISK_POLICY} bash scripts/run_phase_e_frontier_suite.sh"
CMD_HYBRID_CHAIN="CUDA_VISIBLE_DEVICES=${GPU_HYBRID} RUN_PREFIX=${RUN_PREFIX}_ph2 ACTIVE_PHASE_E_PB_HYBRID_GROUP=PH2_PRM_LOCAL_TA10_MSGRID10_ARCH_SWEEP_SMOKE BENCH_MAX_SAMPLES=${BENCH_MAX_SAMPLES} RECIPE_RISK_POLICY=${RECIPE_RISK_POLICY} bash scripts/run_phase_e_processbench_hybrid_suite.sh && CUDA_VISIBLE_DEVICES=${GPU_CURATED} RUN_PREFIX=${RUN_PREFIX}_cr1 ACTIVE_PHASE_E_CURATED_GROUP=CR1_CURATED_CENTER_GATE_SMOKE BENCH_MAX_SAMPLES=${BENCH_MAX_SAMPLES} RECIPE_RISK_POLICY=${RECIPE_RISK_POLICY} bash scripts/run_phase_e_curated_rlready_suite.sh"
CMD_F2_CHAIN="CUDA_VISIBLE_DEVICES=${GPU_TA} RUN_PREFIX=${RUN_PREFIX}_ta TRAIN_BATCH_SIZE=192 EVAL_BATCH_SIZE=192 RECIPE_RISK_POLICY=${RECIPE_RISK_POLICY} bash scripts/run_phase_e_terminal_ratio_sweep.sh"

launch_job "f2_dual_head" "$GPU_F2" "$CMD_F2"
launch_job "f3_lora_probe" "$GPU_F3" "$CMD_F3"
launch_job "ph2_then_cr1" "$GPU_HYBRID" "$CMD_HYBRID_CHAIN"
launch_job "ta_sweep_after_f2" "$GPU_TA" "until [[ -f assets/artifacts/phase_e_logs/${RUN_PREFIX}_f2/final_summary.md ]]; do sleep 120; done; ${CMD_F2_CHAIN}"

log_line "Overnight frontier launcher complete."
log_line "Plan file: ${PLAN_MD}"
log_line "Manifest:  ${MANIFEST_JSONL}"
