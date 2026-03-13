#!/usr/bin/env bash
# run_phase_f_overnight_suite.sh
#
# 用途 / Purpose:
#   Legacy overnight launcher for one Phase E track plus one Phase F live-validation track.
#
# 风险修复 / Safety fixes in this version:
# 1. enable `set -euo pipefail` so `python ... | tee ...` cannot silently hide failures,
# 2. resolve value-head artifacts with marker files instead of `ls | head`,
# 3. keep the GRPO launcher on the TRL-enabled `bcr` python by default.
#
# This is still a legacy GSM8K-heavy launcher. It should be read as an
# infrastructure canary, not as the final research mainline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON_BIN:-/home/zling/anaconda3/envs/bcr/bin/python3}"
LOGDIR="${LOGDIR:-assets/artifacts/phase_f_logs/phase_f_overnight_suite}"
SUITE_LOG="${LOGDIR}/overnight_suite.log"
mkdir -p "$LOGDIR"
: > "$SUITE_LOG"

log_line() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1" | tee -a "$SUITE_LOG"
}

latest_dir_by_prefix_with_marker() {
  local root="$1"
  local prefix="$2"
  local marker="$3"
  "$PY" - "$root" "$prefix" "$marker" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
prefix = sys.argv[2]
marker = sys.argv[3]
matches = sorted(
    [path for path in root.glob(f"{prefix}_*") if (path / marker).exists()],
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
print(matches[0] if matches else "")
PY
}

PBR35_GPU="${PBR35_GPU:-1}"
PHASE_F_GPU="${PHASE_F_GPU:-3}"
PHASE_F_GRPO_COMMON_ARGS="${PHASE_F_GRPO_COMMON_ARGS:---reward-shaping clip_delta --trl-loss-type dr_grpo --trl-scale-rewards batch --trl-beta 0.04 --trl-epsilon 0.2 --trl-mask-truncated-completions --trl-temperature 1.0 --save-only-model}"
PHASE_F_GRPO_PROCESS_ARGS="${PHASE_F_GRPO_PROCESS_ARGS:-}"
PHASE_F_GRPO_OUTCOME_ARGS="${PHASE_F_GRPO_OUTCOME_ARGS:-}"

log_line "Launching PBR35 contrastive chain on GPU ${PBR35_GPU}"
nohup bash -lc "
  set -euo pipefail
  cd '$REPO_ROOT'
  CUDA_DEVICE='${PBR35_GPU}' PYTHON_BIN='${PY}' bash scripts/run_pbr35_lora_contrastive.sh
  echo '[PBR35 chain] DONE' >> '${SUITE_LOG}'
" >> "$SUITE_LOG" 2>&1 &
PBR35_PID=$!
log_line "PBR35 PID: ${PBR35_PID}"

log_line "Resolving best available value head for live Phase F track"
VALUE_RUN_DIR="$(latest_dir_by_prefix_with_marker assets/artifacts/phase_e_runs phase_e_pbr32_lora_mathprm_alllayers_pbr12data_s42 best_value_head.pt)"
if [[ -z "$VALUE_RUN_DIR" ]]; then
  log_line "PBR32 not found with marker; falling back to PBR26 frozen"
  VALUE_RUN_DIR="$(latest_dir_by_prefix_with_marker assets/artifacts/phase_e_runs phase_e_pbr26_dpo_ms_full_s42_value best_value_head.pt)"
fi
if [[ -z "$VALUE_RUN_DIR" ]]; then
  echo "ERROR: could not resolve a completed Phase E value run for live Phase F track" >&2
  exit 1
fi
log_line "Using value head: ${VALUE_RUN_DIR}"

log_line "Launching Phase F live track on GPU ${PHASE_F_GPU}"
nohup bash -lc "
  set -euo pipefail
  cd '$REPO_ROOT'
  echo '[Phase F live] Starting BoN eval...' >> '${SUITE_LOG}'

  CUDA_VISIBLE_DEVICES='${PHASE_F_GPU}' '${PY}' -u scripts/phase_f_best_of_n_eval.py \
      --value-run-dir '${VALUE_RUN_DIR}' \
      --backbone-path assets/models/Qwen2.5-Math-PRM-7B \
      --generator-path assets/models/Qwen2.5-Math-7B-Instruct \
      --num-problems 200 \
      --k-samples 4 \
      --temperature 0.7 \
      --max-new-tokens 512 \
      --max-scoring-length 1024 \
      --generator-batch-size 4 \
      --run-name phase_f_legacy_bon4_gsm8k \
      --require-cuda \
      --seed 42 \
      2>&1 | tee '${LOGDIR}/phase_f_bon_eval.log'

  echo '[Phase F live] BoN done. Starting PRM-assisted GRPO canary...' >> '${SUITE_LOG}'
  CUDA_VISIBLE_DEVICES='${PHASE_F_GPU}' '${PY}' -u scripts/phase_f_grpo_lite.py \
      --value-run-dir '${VALUE_RUN_DIR}' \
      --backbone-path assets/models/Qwen2.5-Math-PRM-7B \
      --policy-path assets/models/Qwen2.5-Math-7B-Instruct \
      --num-train-problems 500 \
      --num-eval-problems 200 \
      --lambda-process 0.3 \
      --k-samples 4 \
      --num-train-epochs 1 \
      --learning-rate 1e-6 \
      --max-new-tokens 512 \
      --per-device-train-batch-size 2 \
      --gradient-accumulation-steps 8 \
      --run-name phase_f_legacy_grpo_process_gsm8k \
      --require-cuda \
      --seed 42 \
      ${PHASE_F_GRPO_COMMON_ARGS} \
      ${PHASE_F_GRPO_PROCESS_ARGS} \
      2>&1 | tee '${LOGDIR}/phase_f_grpo_lite.log'

  echo '[Phase F live] PRM-assisted GRPO done. Starting outcome-only GRPO canary...' >> '${SUITE_LOG}'
  CUDA_VISIBLE_DEVICES='${PHASE_F_GPU}' '${PY}' -u scripts/phase_f_grpo_lite.py \
      --value-run-dir '${VALUE_RUN_DIR}' \
      --backbone-path assets/models/Qwen2.5-Math-PRM-7B \
      --policy-path assets/models/Qwen2.5-Math-7B-Instruct \
      --num-train-problems 500 \
      --num-eval-problems 200 \
      --lambda-process 0.0 \
      --k-samples 4 \
      --num-train-epochs 1 \
      --learning-rate 1e-6 \
      --max-new-tokens 512 \
      --per-device-train-batch-size 2 \
      --gradient-accumulation-steps 8 \
      --run-name phase_f_legacy_grpo_outcome_only_gsm8k \
      --require-cuda \
      --seed 42 \
      ${PHASE_F_GRPO_COMMON_ARGS} \
      ${PHASE_F_GRPO_OUTCOME_ARGS} \
      2>&1 | tee '${LOGDIR}/phase_f_grpo_outcome_only.log'

  echo '[Phase F live] All live-track jobs done.' >> '${SUITE_LOG}'
" >> "$SUITE_LOG" 2>&1 &
GPU3_PID=$!
log_line "Phase F live-track PID: ${GPU3_PID}"

log_line "All overnight experiments launched"
log_line "  PBR35 PID=${PBR35_PID}"
log_line "  Phase F PID=${GPU3_PID}"
log_line "  suite_log=${SUITE_LOG}"
