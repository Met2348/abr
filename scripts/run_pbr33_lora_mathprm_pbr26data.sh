#!/usr/bin/env bash
# run_pbr33_lora_mathprm_pbr26data.sh
#
# 用途 / Purpose:
#   PBR33: Math-PRM-7B + LoRA r=8 top-4 layers + PBR26 data (7366 pairs, DPO 2398 + MS_full 4968)
#   Tests the DATA effect: PBR31 used PBR12 data (5705 pairs); PBR33 uses PBR26's richer dataset.
#   Hypothesis: better/more data with same LoRA architecture → surpass PBR31 on ProcessBench MATH AUC.
#
# Compare with:
#   PBR31: Math-PRM-7B + LoRA top-4 + PBR12 data (5705 pairs) → val pair_acc=0.887 @ ep1
#   PBR26: Math-PRM-7B + frozen + PBR26 data             → ProcessBench MATH AUC=0.888 (SOTA)
#
# Usage: CUDA_DEVICE=<gpu> bash run_pbr33_lora_mathprm_pbr26data.sh

set -euo pipefail

CUDA_DEVICE="${CUDA_DEVICE:-0}"
PY="${PYTHON_BIN:-/home/zling/anaconda3/bin/python3}"
LOGDIR="assets/artifacts/phase_e_logs"
EVAL_ROOT="assets/artifacts/phase_e_eval"
RUN_ROOT="assets/artifacts/phase_e_runs"
PAIR_DIR="assets/artifacts/phase_e_pairs/phase_e_pbr26_dpo_plus_ms_full_pairs__b17437d10dfc"

mkdir -p "${LOGDIR}"

latest_complete_run_dir() {
    local prefix="$1"
    python - "${RUN_ROOT}" "${prefix}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
prefix = sys.argv[2]
matches = sorted(root.glob(f"{prefix}_*"), key=lambda path: path.stat().st_mtime, reverse=True)
for path in matches:
    if (path / "manifest.json").exists():
        print(path)
        raise SystemExit(0)
raise SystemExit(f"No completed run dir with manifest.json found for prefix={prefix!r}")
PY
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting PBR33 training on GPU ${CUDA_DEVICE}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PY}" -u scripts/phase_e_train_value_lora.py \
    --train-pairs-jsonl "${PAIR_DIR}/train_pairs.jsonl" \
    --eval-pairs-jsonl  "${PAIR_DIR}/validation_pairs.jsonl" \
    --model-path assets/models/Qwen2.5-Math-PRM-7B \
    --run-name phase_e_pbr33_lora_mathprm_top4_pbr26data_s42 \
    --objective-mode joint \
    --lambda-bce 0.5 \
    --lambda-ranking 1.0 \
    --terminal-bce-lambda 0.25 \
    --learning-rate 3e-5 \
    --num-train-epochs 5 \
    --per-device-train-batch-size 4 \
    --per-device-eval-batch-size 4 \
    --gradient-accumulation-steps 24 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-target-modules q_proj,v_proj \
    --lora-top-k-layers 4 \
    --ranking-target-space score \
    --pair-weight-mode none \
    --checkpoint-selection-metric pair_acc \
    --head-architecture mlp \
    --head-mlp-hidden-size 512 \
    --head-dropout-prob 0.05 \
    --anti-saturation-weight 5e-4 \
    --anti-saturation-logit-threshold 3.5 \
    --source-balance uniform \
    --seed 42 \
    --require-cuda \
    2>&1 | tee "${LOGDIR}/pbr33_lora_mathprm_top4_pbr26data.log"

# 只有完整写出 manifest 的 run dir 才算训练成功，避免半成品目录被误当成最新成功结果。
# Only accept a run dir once `manifest.json` exists, so an incomplete crashed run
# cannot be mistaken for the latest successful artifact.
RUN_DIR="$(latest_complete_run_dir "phase_e_pbr33_lora_mathprm_top4_pbr26data_s42")"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training done. Run dir: ${RUN_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ProcessBench MATH eval..."
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PY}" -u scripts/phase_e_eval_benchmark.py \
    --value-run-dir "${RUN_DIR}" \
    --benchmark-id processbench_math \
    --run-name "$(basename ${RUN_DIR})_pb_math" \
    --output-root "${EVAL_ROOT}" \
    --batch-size 16 \
    --feature-cache-mode off \
    --require-cuda \
    2>&1 | tee -a "${LOGDIR}/pbr33_lora_mathprm_top4_pbr26data.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ProcessBench GSM8K eval..."
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PY}" -u scripts/phase_e_eval_benchmark.py \
    --value-run-dir "${RUN_DIR}" \
    --benchmark-id processbench_gsm8k \
    --run-name "$(basename ${RUN_DIR})_pb_gsm" \
    --output-root "${EVAL_ROOT}" \
    --batch-size 16 \
    --feature-cache-mode off \
    --require-cuda \
    2>&1 | tee -a "${LOGDIR}/pbr33_lora_mathprm_top4_pbr26data.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] PBR33 complete."
