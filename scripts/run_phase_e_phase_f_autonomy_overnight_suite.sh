#!/usr/bin/env bash
# Changelog (prepend-only, newest first):
# 2026-03-12: Initial version. Queue two literature-guided overnight tracks:
#             (1) PRMBench selected-relabel benchmark-facing repair + Phase F preflight,
#             (2) ProcessBench research PBR6 LoRA backbone smoke.
#
# English
# -------
# This wrapper schedules two long jobs with `wait_for_gpu_idle_and_launch.py`.
# It is meant for busy shared servers: no force-run, only queued launch.
# It uses detached tmux sessions so the watchers survive the parent shell.
#
# 中文
# ----
# 这个 wrapper 用 `wait_for_gpu_idle_and_launch.py` 排两条长作业。
# 适用于共享服务器：不抢占，只排队等待空闲 GPU。
# 这里用独立 `tmux` session 托管 watcher，避免父 shell 退出后队列消失。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_phase_f_auto_$(date +%m%d_%H%M)}"
GPU_RELABEL="${GPU_RELABEL:-1}"
GPU_LORA="${GPU_LORA:-3}"
POLL_SECONDS="${POLL_SECONDS:-180}"
MAX_USED_MIB="${MAX_USED_MIB:-8192}"
MAX_UTIL="${MAX_UTIL:-15}"

LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
mkdir -p "$LOG_ROOT"
QUEUE_LOG="$LOG_ROOT/queue_summary.md"
RELABEL_JOB="$LOG_ROOT/job_prmbench_selected_relabel_and_preflight.sh"
PBR6_JOB="$LOG_ROOT/job_pbr6_lora_backbone.sh"
RELABEL_WATCH_LOG="$LOG_ROOT/watch_relabel.log"
PBR6_WATCH_LOG="$LOG_ROOT/watch_pbr6.log"
RELABEL_SESSION="${RUN_PREFIX}_relabel"
PBR6_SESSION="${RUN_PREFIX}_pbr6"

cat > "$RELABEL_JOB" <<'EOF_RELABEL'
#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

SELREL_PREFIX="${RUN_PREFIX_BASE}_selrel"
PREFLIGHT_PREFIX="${RUN_PREFIX_BASE}_selrel_preflight"

RUN_PREFIX="$SELREL_PREFIX" \
RECIPE_RISK_POLICY=error \
BENCH_MAX_SAMPLES=512 \
TRAIN_BATCH_SIZE=96 \
TRAIN_EVAL_BATCH_SIZE=128 \
BENCH_EVAL_BATCH_SIZE=128 \
CUDA_VISIBLE_DEVICES="${GPU_JOB}" \
bash scripts/run_phase_e_prmbench_selected_relabel_suite.sh

SELECTED_VALUE_DIR="$(python - <<'PY_RESOLVE1'
from pathlib import Path
import os
prefix = os.environ['RUN_PREFIX_BASE'] + '_selrel_value_'
matches = sorted(Path('assets/artifacts/phase_e_runs').glob(prefix + '*'), key=lambda p: p.stat().st_mtime)
if not matches:
    raise SystemExit('No selected-relabel value run found')
print(matches[-1])
PY_RESOLVE1
)"
SELECTED_GSM_EVAL_DIR="$(python - <<'PY_RESOLVE2'
from pathlib import Path
import os
prefix = os.environ['RUN_PREFIX_BASE'] + '_selrel_processbench_gsm8k_'
matches = sorted(Path('assets/artifacts/phase_e_eval').glob(prefix + '*'), key=lambda p: p.stat().st_mtime)
if not matches:
    raise SystemExit('No selected-relabel GSM eval found')
print(matches[-1])
PY_RESOLVE2
)"
SELECTED_MATH_EVAL_DIR="$(python - <<'PY_RESOLVE3'
from pathlib import Path
import os
prefix = os.environ['RUN_PREFIX_BASE'] + '_selrel_processbench_math_'
matches = sorted(Path('assets/artifacts/phase_e_eval').glob(prefix + '*'), key=lambda p: p.stat().st_mtime)
if not matches:
    raise SystemExit('No selected-relabel Math eval found')
print(matches[-1])
PY_RESOLVE3
)"

RUN_PREFIX="$PREFLIGHT_PREFIX" \
CAND_A_ID=prm_e46 \
CAND_B_ID=prm_selrel \
PBR26_RUN_DIR="assets/artifacts/phase_e_runs/phase_e_prmbench_acc90_full_0310_1914_e46_prmbench_acc90_mlp_joint_seed3_e46_prmbench_acc90_mlp_joint_seed3_s43_value_20260310T113737Z" \
PBR26_GSM_EVAL="assets/artifacts/phase_e_eval/phase_e_rltops_0311_1124_prm_e46_processbench_gsm8k_20260311T032704Z" \
PBR26_MATH_EVAL="assets/artifacts/phase_e_eval/phase_e_rltops_0311_1124_prm_e46_processbench_math_20260311T032713Z" \
PBR31_RUN_DIR="$SELECTED_VALUE_DIR" \
PBR31_GSM_EVAL="$SELECTED_GSM_EVAL_DIR" \
PBR31_MATH_EVAL="$SELECTED_MATH_EVAL_DIR" \
GPU_DEVICE="${GPU_JOB}" \
THRESHOLD_BATCH_SIZE=96 \
PROBE_MAX_ERROR=96 \
PROBE_MAX_CORRECT=96 \
bash scripts/run_phase_f_modern_preflight_suite.sh
EOF_RELABEL
chmod +x "$RELABEL_JOB"

cat > "$PBR6_JOB" <<'EOF_PBR6'
#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
RUN_PREFIX="${RUN_PREFIX_BASE}_pbr6" \
ACTIVE_PHASE_E_PB_RESEARCH_GROUP=PBR6_LORA_BACKBONE_SMOKE \
RECIPE_RISK_POLICY=error \
CUDA_VISIBLE_DEVICES="${GPU_JOB}" \
bash scripts/run_phase_e_processbench_research_suite.sh
EOF_PBR6
chmod +x "$PBR6_JOB"

cat > "$QUEUE_LOG" <<EOF_LOG
# Phase E/F Autonomy Overnight Queue

- generated_at: $(date '+%Y-%m-%d %H:%M:%S %z')
- run_prefix: ${RUN_PREFIX}
- gpu_relabel: ${GPU_RELABEL}
- gpu_lora: ${GPU_LORA}
- tmux_sessions:
  - ${RELABEL_SESSION}
  - ${PBR6_SESSION}

## queued_jobs

1. PRMBench selected-relabel + Phase F modern preflight
   - job_script: ${RELABEL_JOB}
   - watch_log: ${RELABEL_WATCH_LOG}
   - purpose:
     - benchmark-facing conservative repair for PRMBench_Preview
     - immediate controller-facing preflight against E46

2. PBR6 LoRA backbone smoke
   - job_script: ${PBR6_JOB}
   - watch_log: ${PBR6_WATCH_LOG}
   - purpose:
     - test representation-ceiling hypothesis under stronger backbone adaptation
EOF_LOG

log_launch() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1"
}

# Replace old sessions if the same prefix is reused.
tmux has-session -t "$RELABEL_SESSION" 2>/dev/null && tmux kill-session -t "$RELABEL_SESSION"
tmux has-session -t "$PBR6_SESSION" 2>/dev/null && tmux kill-session -t "$PBR6_SESSION"

log_launch "queue relabel/preflight on GPU ${GPU_RELABEL} via tmux ${RELABEL_SESSION}"
tmux new-session -d -s "$RELABEL_SESSION"   "cd '$REPO_ROOT' && '$PYTHON_BIN' -u scripts/wait_for_gpu_idle_and_launch.py --gpu-id '$GPU_RELABEL' --max-used-mib '$MAX_USED_MIB' --max-util '$MAX_UTIL' --poll-seconds '$POLL_SECONDS' --log-file '$RELABEL_WATCH_LOG' --workdir '$REPO_ROOT' --command 'RUN_PREFIX_BASE=${RUN_PREFIX} GPU_JOB=${GPU_RELABEL} bash ${RELABEL_JOB}'"

log_launch "queue PBR6 on GPU ${GPU_LORA} via tmux ${PBR6_SESSION}"
tmux new-session -d -s "$PBR6_SESSION"   "cd '$REPO_ROOT' && '$PYTHON_BIN' -u scripts/wait_for_gpu_idle_and_launch.py --gpu-id '$GPU_LORA' --max-used-mib '$MAX_USED_MIB' --max-util '$MAX_UTIL' --poll-seconds '$POLL_SECONDS' --log-file '$PBR6_WATCH_LOG' --workdir '$REPO_ROOT' --command 'RUN_PREFIX_BASE=${RUN_PREFIX} GPU_JOB=${GPU_LORA} bash ${PBR6_JOB}'"

log_launch "queue file written: ${QUEUE_LOG}"
log_launch "tmux sessions: ${RELABEL_SESSION}, ${PBR6_SESSION}"
