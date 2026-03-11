#!/usr/bin/env bash
# Changelog (prepend-only, newest first):
# 2026-03-12: Initial creation. LoRA smoke experiments for Phase E value head.
#             Tests whether backbone LoRA fine-tuning improves over frozen SOTA.
#             Frozen SOTA (fix_c_gated, s42): MATH AUC=0.749, GSM AUC=0.711.
#
# LoRA Smoke Suite — 骨干 LoRA 微调验证实验
#
# English
# -------
# Frozen-backbone Phase E ceiling: MATH AUC=0.749 (DPO 8K + gated_mlp, seed 42).
# Community evidence suggests LoRA adds +5-10 AUC points over frozen features.
# This smoke suite tests three LoRA configurations to find the viable range:
#
#   LORA_S1: Small LoRA (r=8, q+v only, all layers)     — lightweight baseline
#   LORA_S2: Medium LoRA (r=16, q+v+k+o, top-8 layers) — balanced
#   LORA_S3: Full attn LoRA (r=32, q+v+k+o, all layers) — high capacity
#   LORA_S4: Stacked (r=16, q+v, all layers, smaller LR) — safer recipe
#
# Design principles:
# - Use existing DPO 8K pairs (same data as frozen SOTA for fair comparison).
# - 5 epochs (LoRA converges faster than head-only due to joint gradient flow).
# - Smaller batch size (4) for backbone forward; grad_accum keeps effective batch = 32.
# - Distinct feature cache root per config (LoRA weights are NOT cached to disk).
# - Joint objective (ranking + BCE) matching the proven frozen SOTA recipe.
#
# 中文
# ----
# 冻结骨干上限：MATH AUC=0.749（DPO 8K + gated_mlp, seed 42）。
# 社区证据表明 LoRA 在冻结特征基础上可以带来 +5-10 AUC 的提升。
# 本 smoke suite 测试三种 LoRA 配置，寻找可行区间：
#
#   LORA_S1: 小 LoRA（r=8, q+v, 全层）             — 轻量 baseline
#   LORA_S2: 中 LoRA（r=16, q+v+k+o, top-8 层）    — 均衡
#   LORA_S3: 全注意力 LoRA（r=32, q+v+k+o, 全层）   — 高容量
#   LORA_S4: 叠加（r=16, q+v, 全层, 更小 LR）       — 更保守
#
# GPU assignment: one single GPU (default GPU 0); sequential execution.
# Recommended: run on a free GPU with ≥40 GB VRAM (e.g. A100/H100).
# Memory breakdown:
#   Backbone (bfloat16): ~14 GB
#   LoRA adapters (r=16): ~280 MB
#   Tokenized cache: negligible (CPU)
#   Value head: ~10 MB
#   Effective working set: ~18-22 GB (safe on A100-40GB)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-conda run -n bcr python3}"
ACTIVE_LORA_GROUP="${ACTIVE_LORA_GROUP:-LORA_S1_RANK8_QV_ALL}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_lora_smoke_0312}"

MODEL_PATH="${MODEL_PATH:-assets/models/Qwen2.5-7B-Instruct}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-assets/artifacts/phase_e_feature_cache}"
PAIR_OUTPUT_ROOT="${PAIR_OUTPUT_ROOT:-assets/artifacts/phase_e_pairs}"
VALUE_OUTPUT_ROOT="${VALUE_OUTPUT_ROOT:-assets/artifacts/phase_e_runs}"
BENCH_OUTPUT_ROOT="${BENCH_OUTPUT_ROOT:-assets/artifacts/phase_e_eval}"
LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"

# DPO 8K pairs — the same source used for frozen SOTA (fix_c_gated).
DPO_8K_PAIR_DIR="${DPO_8K_PAIR_DIR:-assets/artifacts/phase_e_pairs/phase_e_flb_0311_fix_c_dpo_scale_8k_dpo_scale_v1_pairs__77e7755b95a8}"
DPO_8K_TRAIN_PAIRS="${DPO_8K_PAIR_DIR}/train_pairs.jsonl"
DPO_8K_EVAL_PAIRS="${DPO_8K_PAIR_DIR}/validation_pairs.jsonl"

# Benchmark pairs for ProcessBench eval.
BENCH_MATH_PAIRS="${BENCH_MATH_PAIRS:-assets/artifacts/phase_e_pairs/processbench_eval_math_pairs}"
BENCH_GSM_PAIRS="${BENCH_GSM_PAIRS:-assets/artifacts/phase_e_pairs/processbench_eval_gsm8k_pairs}"

# Hyperparameters — balanced for LoRA training.
# Smaller per-batch size + higher grad_accum to fit backbone in memory.
LORA_TRAIN_BATCH_SIZE="${LORA_TRAIN_BATCH_SIZE:-4}"
LORA_EVAL_BATCH_SIZE="${LORA_EVAL_BATCH_SIZE:-4}"
LORA_GRAD_ACCUM="${LORA_GRAD_ACCUM:-8}"  # effective batch = 4 × 8 = 32
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-1e-4}"  # LoRA typically needs higher LR
HEAD_MLP_HIDDEN_SIZE="${HEAD_MLP_HIDDEN_SIZE:-1024}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-256}"
RECIPE_RISK_POLICY="${RECIPE_RISK_POLICY:-error}"

# GPU device
CUDA_DEVICE="${CUDA_DEVICE:-0}"

CURRENT_STAGE="bootstrap"

GROUP_TITLE=""
GROUP_INTENTION=""
GROUP_CASES=()

# Frozen SOTA reference for comparison in summary.
FROZEN_SOTA_MATH_AUC="0.749"
FROZEN_SOTA_GSM_AUC="0.711"

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

log() { echo "[$(date '+%Y-%m-%dT%H:%M:%S')] $*" | tee -a "$SUITE_LOG_FILE"; }
die() { echo "[FATAL] $*" >&2; exit 1; }
stage() { CURRENT_STAGE="$1"; log "=== STAGE: $1 ==="; }

require_file() {
    local path="$1"
    local label="${2:-$1}"
    [[ -f "$path" ]] || die "Required file not found: $label -> $path"
}

check_gpu() {
    local gpu_id="$1"
    if command -v nvidia-smi &>/dev/null; then
        local used_mib
        used_mib=$(nvidia-smi --id="$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        if [[ -n "$used_mib" && "$used_mib" -gt 8192 ]]; then
            log "[WARNING] GPU $gpu_id has ${used_mib} MiB already in use — LoRA training needs ~20 GB free"
        else
            log "[OK] GPU $gpu_id: ${used_mib} MiB in use"
        fi
    fi
}

mkdir -p "$LOG_ROOT"
log "LoRA Smoke Suite started: group=${ACTIVE_LORA_GROUP} prefix=${RUN_PREFIX}"
log "Model: $MODEL_PATH"
log "DPO 8K train pairs: $DPO_8K_TRAIN_PAIRS"
check_gpu "$CUDA_DEVICE"

# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

run_lora_train() {
    local case_id="$1"
    local lora_rank="$2"
    local lora_alpha="$3"
    local lora_target_modules="$4"
    local lora_num_top_layers="$5"        # "" = all layers
    local learning_rate="${6:-$BASE_LEARNING_RATE}"
    local head_arch="${7:-gated_mlp}"
    local seed="${8:-42}"

    local run_name="${RUN_PREFIX}_${case_id}"
    local log_file="${LOG_ROOT}/${case_id}.log"
    local top_layers_arg=""
    if [[ -n "$lora_num_top_layers" ]]; then
        top_layers_arg="--lora-num-top-layers $lora_num_top_layers"
    fi

    log "--- TRAIN: $case_id (r=$lora_rank, alpha=$lora_alpha, modules=$lora_target_modules, top_layers=${lora_num_top_layers:-all}, lr=$learning_rate, head=$head_arch, seed=$seed) ---"

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
    $PYTHON_BIN scripts/phase_e_train_value.py \
        --train-pairs-jsonl "$DPO_8K_TRAIN_PAIRS" \
        --eval-pairs-jsonl  "$DPO_8K_EVAL_PAIRS" \
        --model-path        "$MODEL_PATH" \
        --run-name          "$run_name" \
        --output-root       "$VALUE_OUTPUT_ROOT" \
        --objective-mode    joint \
        --learning-rate     "$learning_rate" \
        --num-train-epochs  "$TRAIN_EPOCHS" \
        --per-device-train-batch-size "$LORA_TRAIN_BATCH_SIZE" \
        --per-device-eval-batch-size  "$LORA_EVAL_BATCH_SIZE" \
        --gradient-accumulation-steps "$LORA_GRAD_ACCUM" \
        --max-length        "$MAX_LENGTH" \
        --lambda-ranking    1.0 \
        --lambda-bce        1.0 \
        --ranking-target-space score \
        --pair-weight-mode  none \
        --checkpoint-selection-metric pair_acc \
        --head-architecture "$head_arch" \
        --head-mlp-hidden-size "$HEAD_MLP_HIDDEN_SIZE" \
        --anti-saturation-weight 5e-4 \
        --anti-saturation-logit-threshold 3.5 \
        --head-dropout-prob 0.05 \
        --seed              "$seed" \
        --dtype             bfloat16 \
        --device-map        auto \
        --feature-cache-mode off \
        --recipe-risk-policy "$RECIPE_RISK_POLICY" \
        --lora-rank         "$lora_rank" \
        --lora-alpha        "$lora_alpha" \
        --lora-target-modules "$lora_target_modules" \
        $top_layers_arg \
        2>&1 | tee "$log_file"
}

run_bench_eval() {
    local case_id="$1"
    local bench_name="$2"
    local bench_pairs="$3"
    local run_name="${RUN_PREFIX}_${case_id}"
    local log_file="${LOG_ROOT}/${case_id}_bench_${bench_name}.log"

    log "--- BENCH: $case_id / $bench_name ---"

    # Find the latest run directory matching the run_name.
    local run_dir
    run_dir=$(find "$VALUE_OUTPUT_ROOT" -maxdepth 1 -type d -name "${run_name}_*" | sort | tail -1)
    if [[ -z "$run_dir" ]]; then
        log "[ERROR] No run directory found for $run_name"
        return 1
    fi

    local best_ckpt="$run_dir/best_value_head.pt"
    local value_head_config="$run_dir/value_head_config.json"
    if [[ ! -f "$best_ckpt" ]]; then
        log "[ERROR] No best_value_head.pt in $run_dir"
        return 1
    fi

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
    $PYTHON_BIN scripts/phase_e_eval_benchmark.py \
        --eval-pairs-jsonl  "$bench_pairs" \
        --model-path        "$MODEL_PATH" \
        --value-head-path   "$best_ckpt" \
        --value-head-config "$value_head_config" \
        --benchmark-name    "$bench_name" \
        --output-root       "$BENCH_OUTPUT_ROOT" \
        --run-name          "${run_name}_bench_${bench_name}" \
        --max-eval-samples  "$BENCH_MAX_SAMPLES" \
        --per-device-eval-batch-size 4 \
        --max-length        "$MAX_LENGTH" \
        --dtype             bfloat16 \
        --device-map        auto \
        2>&1 | tee "$log_file"
}

# ---------------------------------------------------------------------------
# Summary collection
# ---------------------------------------------------------------------------

collect_summary() {
    local case_id="$1"
    local run_name="${RUN_PREFIX}_${case_id}"
    local run_dir
    run_dir=$(find "$VALUE_OUTPUT_ROOT" -maxdepth 1 -type d -name "${run_name}_*" | sort | tail -1)
    if [[ -z "$run_dir" ]]; then
        echo "| $case_id | NOT FOUND | — | — | — | — |"
        return
    fi
    local pair_acc auc
    pair_acc=$(python3 -c "import json; d=json.load(open('$run_dir/summary.json')); print(f\"{d['eval_pairs']['pair_accuracy']:.4f}\")" 2>/dev/null || echo "?")
    auc=$(python3 -c "import json; d=json.load(open('$run_dir/summary.json')); print(f\"{d['eval_pairs']['auc']:.4f}\")" 2>/dev/null || echo "?")

    local math_auc="—" gsm_auc="—"
    local bench_dir
    bench_dir=$(find "$BENCH_OUTPUT_ROOT" -maxdepth 1 -type d -name "${run_name}_bench_processbench_math_*" | sort | tail -1)
    if [[ -n "$bench_dir" ]]; then
        math_auc=$(python3 -c "import json; d=json.load(open('$bench_dir/benchmark_metrics.json')); print(f\"{d.get('pair_auc_good_vs_bad', d.get('auc', '?')):.4f}\")" 2>/dev/null || echo "?")
    fi
    bench_dir=$(find "$BENCH_OUTPUT_ROOT" -maxdepth 1 -type d -name "${run_name}_bench_processbench_gsm8k_*" | sort | tail -1)
    if [[ -n "$bench_dir" ]]; then
        gsm_auc=$(python3 -c "import json; d=json.load(open('$bench_dir/benchmark_metrics.json')); print(f\"{d.get('pair_auc_good_vs_bad', d.get('auc', '?')):.4f}\")" 2>/dev/null || echo "?")
    fi

    echo "| $case_id | $pair_acc | $auc | $math_auc | $gsm_auc | vs frozen SOTA: MATH ${FROZEN_SOTA_MATH_AUC} |"
}

write_summary() {
    {
        echo "# LoRA Smoke Suite Summary"
        echo ""
        echo "Generated: $(date '+%Y-%m-%d %H:%M:%S %z')"
        echo "Group: ${ACTIVE_LORA_GROUP}"
        echo ""
        echo "**Frozen SOTA reference (fix_c_gated, s42):** MATH AUC=${FROZEN_SOTA_MATH_AUC}, GSM AUC=${FROZEN_SOTA_GSM_AUC}"
        echo ""
        echo "| case_id | pair_acc | train_auc | MATH AUC | GSM AUC | notes |"
        echo "|---|---:|---:|---:|---:|---|"
        for case_id in "${GROUP_CASES[@]}"; do
            collect_summary "$case_id"
        done
        echo ""
        echo "## Notes"
        echo "- LoRA training uses live backbone forward per mini-batch (no feature caching for train)"
        echo "- Eval cache is rebuilt from current backbone at end of each epoch"
        echo "- Batch size: ${LORA_TRAIN_BATCH_SIZE} × grad_accum ${LORA_GRAD_ACCUM} = effective ${LORA_TRAIN_BATCH_SIZE}×${LORA_GRAD_ACCUM}=32"
        echo "- Backbone: $MODEL_PATH"
    } > "$SUMMARY_FILE"
    log "Summary written to $SUMMARY_FILE"
    cat "$SUMMARY_FILE"
}

# ---------------------------------------------------------------------------
# Experiment groups
# ---------------------------------------------------------------------------

case "$ACTIVE_LORA_GROUP" in

# ---------------------------------------------------------------------------
# LORA_S1: Lightweight LoRA baseline (r=8, q+v only, all layers)
# Memory: ~14 GB backbone + ~140 MB LoRA = ~15 GB total
# ---------------------------------------------------------------------------
LORA_S1_RANK8_QV_ALL)
    GROUP_TITLE="LORA_S1: r=8, q_proj+v_proj, all layers, LR=1e-4"
    GROUP_INTENTION="Lightweight LoRA baseline — minimal parameter count, safe recipe"
    GROUP_CASES=(lora_s1_r8_qv_all)
    stage "LORA_S1_TRAIN"
    run_lora_train \
        lora_s1_r8_qv_all \
        8    \
        16   \
        "q_proj,v_proj" \
        ""   \
        "1e-4" \
        gated_mlp \
        42
    stage "LORA_S1_BENCH"
    if [[ -f "$BENCH_MATH_PAIRS/validation_pairs.jsonl" ]]; then
        run_bench_eval lora_s1_r8_qv_all processbench_math "$BENCH_MATH_PAIRS/validation_pairs.jsonl"
    fi
    if [[ -f "$BENCH_GSM_PAIRS/validation_pairs.jsonl" ]]; then
        run_bench_eval lora_s1_r8_qv_all processbench_gsm8k "$BENCH_GSM_PAIRS/validation_pairs.jsonl"
    fi
    write_summary
    ;;

# ---------------------------------------------------------------------------
# LORA_S2: Medium LoRA (r=16, q+v+k+o, top-8 layers)
# Memory: ~14 GB backbone + ~430 MB LoRA (top-8 of 28 layers) = ~15.5 GB
# ---------------------------------------------------------------------------
LORA_S2_RANK16_QVKO_TOP8)
    GROUP_TITLE="LORA_S2: r=16, q+v+k+o projections, top-8 layers, LR=1e-4"
    GROUP_INTENTION="Balanced LoRA — full attention coverage on near-output layers"
    GROUP_CASES=(lora_s2_r16_qvko_top8)
    stage "LORA_S2_TRAIN"
    run_lora_train \
        lora_s2_r16_qvko_top8 \
        16   \
        32   \
        "q_proj,v_proj,k_proj,o_proj" \
        8    \
        "1e-4" \
        gated_mlp \
        42
    stage "LORA_S2_BENCH"
    if [[ -f "$BENCH_MATH_PAIRS/validation_pairs.jsonl" ]]; then
        run_bench_eval lora_s2_r16_qvko_top8 processbench_math "$BENCH_MATH_PAIRS/validation_pairs.jsonl"
    fi
    if [[ -f "$BENCH_GSM_PAIRS/validation_pairs.jsonl" ]]; then
        run_bench_eval lora_s2_r16_qvko_top8 processbench_gsm8k "$BENCH_GSM_PAIRS/validation_pairs.jsonl"
    fi
    write_summary
    ;;

# ---------------------------------------------------------------------------
# LORA_S3: High-capacity LoRA (r=32, q+v+k+o, all layers)
# Memory: ~14 GB backbone + ~1.1 GB LoRA (all 28 layers) = ~16 GB
# ---------------------------------------------------------------------------
LORA_S3_RANK32_QVKO_ALL)
    GROUP_TITLE="LORA_S3: r=32, q+v+k+o projections, all layers, LR=5e-5"
    GROUP_INTENTION="High-capacity LoRA — max parameter count, conservative LR"
    GROUP_CASES=(lora_s3_r32_qvko_all)
    stage "LORA_S3_TRAIN"
    run_lora_train \
        lora_s3_r32_qvko_all \
        32   \
        64   \
        "q_proj,v_proj,k_proj,o_proj" \
        ""   \
        "5e-5" \
        gated_mlp \
        42
    stage "LORA_S3_BENCH"
    if [[ -f "$BENCH_MATH_PAIRS/validation_pairs.jsonl" ]]; then
        run_bench_eval lora_s3_r32_qvko_all processbench_math "$BENCH_MATH_PAIRS/validation_pairs.jsonl"
    fi
    if [[ -f "$BENCH_GSM_PAIRS/validation_pairs.jsonl" ]]; then
        run_bench_eval lora_s3_r32_qvko_all processbench_gsm8k "$BENCH_GSM_PAIRS/validation_pairs.jsonl"
    fi
    write_summary
    ;;

# ---------------------------------------------------------------------------
# LORA_S4: Conservative LoRA (r=16, q+v, all layers, LR=3e-5)
# Matches frozen SOTA learning rate — safest comparison point.
# ---------------------------------------------------------------------------
LORA_S4_RANK16_QV_CONSERVATIVE)
    GROUP_TITLE="LORA_S4: r=16, q+v only, all layers, LR=3e-5 (conservative)"
    GROUP_INTENTION="Conservative LoRA matching frozen SOTA LR — cleanest ablation"
    GROUP_CASES=(lora_s4_r16_qv_conservative)
    stage "LORA_S4_TRAIN"
    run_lora_train \
        lora_s4_r16_qv_conservative \
        16   \
        32   \
        "q_proj,v_proj" \
        ""   \
        "3e-5" \
        gated_mlp \
        42
    stage "LORA_S4_BENCH"
    if [[ -f "$BENCH_MATH_PAIRS/validation_pairs.jsonl" ]]; then
        run_bench_eval lora_s4_r16_qv_conservative processbench_math "$BENCH_MATH_PAIRS/validation_pairs.jsonl"
    fi
    if [[ -f "$BENCH_GSM_PAIRS/validation_pairs.jsonl" ]]; then
        run_bench_eval lora_s4_r16_qv_conservative processbench_gsm8k "$BENCH_GSM_PAIRS/validation_pairs.jsonl"
    fi
    write_summary
    ;;

# ---------------------------------------------------------------------------
# ALL: Run all groups sequentially on one GPU
# ---------------------------------------------------------------------------
ALL_SEQUENTIAL)
    GROUP_TITLE="ALL LoRA Smoke Groups (sequential on GPU ${CUDA_DEVICE})"
    GROUP_INTENTION="Full LoRA smoke test: 4 configs, DPO 8K, gated_mlp, 5 epochs each"
    GROUP_CASES=(lora_s1_r8_qv_all lora_s2_r16_qvko_top8 lora_s3_r32_qvko_all lora_s4_r16_qv_conservative)

    ACTIVE_LORA_GROUP=LORA_S1_RANK8_QV_ALL \
        bash "$0" && \
    ACTIVE_LORA_GROUP=LORA_S4_RANK16_QV_CONSERVATIVE \
        bash "$0" && \
    ACTIVE_LORA_GROUP=LORA_S2_RANK16_QVKO_TOP8 \
        bash "$0" && \
    ACTIVE_LORA_GROUP=LORA_S3_RANK32_QVKO_ALL \
        bash "$0" && \
    write_summary
    ;;

*)
    echo "Unknown ACTIVE_LORA_GROUP: $ACTIVE_LORA_GROUP"
    echo ""
    echo "Available groups:"
    echo "  LORA_S1_RANK8_QV_ALL          — r=8, q+v, all layers (lightweight)"
    echo "  LORA_S2_RANK16_QVKO_TOP8      — r=16, q+v+k+o, top-8 layers (balanced)"
    echo "  LORA_S3_RANK32_QVKO_ALL       — r=32, q+v+k+o, all layers (high capacity)"
    echo "  LORA_S4_RANK16_QV_CONSERVATIVE — r=16, q+v, all layers, LR=3e-5 (conservative)"
    echo "  ALL_SEQUENTIAL                 — run all 4 sequentially"
    echo ""
    echo "Usage examples:"
    echo "  ACTIVE_LORA_GROUP=LORA_S1_RANK8_QV_ALL CUDA_DEVICE=0 bash scripts/run_phase_e_lora_smoke.sh"
    echo "  ACTIVE_LORA_GROUP=LORA_S4_RANK16_QV_CONSERVATIVE CUDA_DEVICE=1 bash scripts/run_phase_e_lora_smoke.sh"
    exit 1
    ;;
esac

log "LoRA Smoke Suite finished: group=${ACTIVE_LORA_GROUP}"
