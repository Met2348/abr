#!/usr/bin/env bash
# run_phase_f_overnight_suite.sh
#
# 用途 / Purpose:
#   Overnight Phase F + Phase E experiments (2026-03-12 overnight run).
#
# Experiments:
#   GPU 1 — Phase E PBR35: LoRA r=8 top-4 + PBR26 data + contrastive loss (w=0.10)
#   GPU 3 — Phase F BoN: PRM-guided Best-of-N on GSM8K (K=4, 200 problems)
#   GPU 3 — Phase F GRPO: RL training with process reward after BoN finishes
#
# Schedule (estimated, starting 02:00 +0800):
#   02:00 — PBR35 starts on GPU 1
#   02:00 — Phase F BoN starts on GPU 3
#   04:30 — Phase F BoN done → Phase F GRPO starts on GPU 3
#   07:30 — PBR35 done → eval on GPU 1
#   10:00 — GRPO done → all results ready
#
# Usage: bash run_phase_f_overnight_suite.sh
#   (runs both GPU 1 and GPU 3 tracks in parallel via nohup)

set -e

PY="${PYTHON_BIN:-/home/zling/anaconda3/envs/bcr/bin/python3}"
LOGDIR="assets/artifacts/phase_e_logs"
mkdir -p "${LOGDIR}"

# ── GPU 1 track: PBR35 contrastive ──────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching PBR35 (GPU 1)..."
nohup bash -c "
  CUDA_DEVICE=1 PYTHON_BIN='${PY}' bash scripts/run_pbr35_lora_contrastive.sh
  echo '[PBR35 chain] DONE' >> '${LOGDIR}/overnight_suite.log'
" >> "${LOGDIR}/overnight_suite.log" 2>&1 &
PBR35_PID=$!
echo "[$(date '+%Y-%m-%d %H:%M:%S')] PBR35 PID: ${PBR35_PID}"
echo "PBR35_PID=${PBR35_PID}" >> "${LOGDIR}/overnight_suite.log"

# ── GPU 3 track: Phase F BoN then GRPO ──────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching Phase F GPU-3 track..."

# Find the best available value run dir (PBR32 = best MATH F1 as of now)
VALUE_RUN_DIR=$(ls -td assets/artifacts/phase_e_runs/phase_e_pbr32_lora_mathprm_alllayers_pbr12data_s42_* 2>/dev/null | head -1)
if [ -z "${VALUE_RUN_DIR}" ]; then
    echo "ERROR: PBR32 run dir not found. Falling back to PBR26 frozen."
    VALUE_RUN_DIR=$(ls -td assets/artifacts/phase_e_runs/phase_e_pbr26_dpo_ms_full_s42_value_* 2>/dev/null | head -1)
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using value head: ${VALUE_RUN_DIR}"
echo "VALUE_RUN_DIR=${VALUE_RUN_DIR}" >> "${LOGDIR}/overnight_suite.log"

nohup bash -c "
  set -e
  echo '[Phase F GPU-3] Starting BoN eval...' >> '${LOGDIR}/overnight_suite.log'

  CUDA_VISIBLE_DEVICES=3 '${PY}' -u scripts/phase_f_best_of_n_eval.py \
      --value-run-dir '${VALUE_RUN_DIR}' \
      --backbone-path assets/models/Qwen2.5-Math-PRM-7B \
      --generator-path assets/models/Qwen2.5-Math-7B-Instruct \
      --num-problems 200 \
      --k-samples 4 \
      --temperature 0.7 \
      --max-new-tokens 512 \
      --max-scoring-length 1024 \
      --generator-batch-size 4 \
      --run-name pbr32_bon4_gsm8k \
      --require-cuda \
      --seed 42 \
      2>&1 | tee '${LOGDIR}/phase_f_bon_eval.log'

  echo '[Phase F GPU-3] BoN done. Starting GRPO...' >> '${LOGDIR}/overnight_suite.log'

  CUDA_VISIBLE_DEVICES=3 '${PY}' -u scripts/phase_f_grpo_lite.py \
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
      --run-name grpo_pbr32_process_gsm8k \
      --require-cuda \
      --seed 42 \
      2>&1 | tee '${LOGDIR}/phase_f_grpo_lite.log'

  echo '[Phase F GPU-3] GRPO done.' >> '${LOGDIR}/overnight_suite.log'

  # Outcome-only baseline GRPO for comparison
  echo '[Phase F GPU-3] Starting outcome-only GRPO baseline...' >> '${LOGDIR}/overnight_suite.log'
  CUDA_VISIBLE_DEVICES=3 '${PY}' -u scripts/phase_f_grpo_lite.py \
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
      --run-name grpo_outcome_only_gsm8k \
      --require-cuda \
      --seed 42 \
      2>&1 | tee '${LOGDIR}/phase_f_grpo_outcome_only.log'

  echo '[Phase F GPU-3] All F-track done.' >> '${LOGDIR}/overnight_suite.log'
" >> "${LOGDIR}/overnight_suite.log" 2>&1 &
GPU3_PID=$!
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU-3 track PID: ${GPU3_PID}"
echo "GPU3_TRACK_PID=${GPU3_PID}" >> "${LOGDIR}/overnight_suite.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All overnight experiments launched."
echo "  PBR35 (GPU 1): PID ${PBR35_PID}"
echo "  Phase F (GPU 3): PID ${GPU3_PID}"
echo "  Logs: ${LOGDIR}/pbr35_lora_contrastive_top4_pbr26data.log"
echo "        ${LOGDIR}/phase_f_bon_eval.log"
echo "        ${LOGDIR}/phase_f_grpo_lite.log"
echo "        ${LOGDIR}/phase_f_grpo_outcome_only.log"
echo "        ${LOGDIR}/overnight_suite.log"
