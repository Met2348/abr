#!/usr/bin/env bash
# Changelog (prepend-only, newest first):
# 2026-03-12: Initial creation. FLB2 — Frozen Backbone Level-2 Pushthrough Suite.
#             Based on FLB1 findings: pure DPO data + gated_mlp head is the winning combo.
#             FLB2 objectives:
#               FLB2A: DPO full-scale (all 10.8K rows, ~9.7K train) + gated_mlp, 3 seeds
#                      — push frozen-head ceiling from 0.749 (FIX_C gated, s42) toward 0.75+
#               FLB2B: DPO 8K + gated_mlp, seeds 1 and 7 (seed 42 already done as fix_c_gated)
#                      — confirm inter-seed stability at 8K scale
#               FLB2C: DPO 8K + gated_mlp + Qwen2.5-Math-7B-Instruct backbone
#                      — math-specialized LLM features vs. general instruction LLM
#               FLB2D: DPO 8K + MLP + Qwen2.5-Math-PRM-7B backbone
#                      — dedicated PRM backbone; Qwen2.5-Math-PRM-7B-ProcessRewardModel
#               FLB2E: DPO 8K + gated_mlp + Qwen2.5-Math-PRM-7B backbone
#                      — best head + best PRM backbone combined
#
# FLB2 Suite — 冻结骨干第二轮突破实验
#
# English
# -------
# FLB1 established: DPO sibling_branch pairs + gated_mlp head = MATH AUC=0.749 (new SOTA).
# Score ordering: all_correct_last > good_prefix > bad_prefix (healthy terminal calibration).
# FLB2 now pushes on three independent axes:
#   1. Data scale: max out Math-Step-DPO-10K (~9.7K train pairs)
#   2. Seed stability: 3-seed variance check for gated_mlp + DPO 8K
#   3. Backbone: Qwen2.5-Math-7B-Instruct and Qwen2.5-Math-PRM-7B vs. current Qwen2.5-7B-Instruct
#
# 中文
# ----
# FLB1 已确认：DPO sibling_branch pair + gated_mlp head = 冻结骨干 SOTA（MATH AUC=0.749）。
# FLB2 在三个独立维度上继续推进：
#   1. 数据规模：用满 Math-Step-DPO-10K 全量（~9.7K 训练 pair）
#   2. 种子稳定性：gated_mlp + DPO 8K 跑 3 seed，确认方差
#   3. 骨干：Qwen2.5-Math-7B-Instruct 和 Qwen2.5-Math-PRM-7B 对比
#
# GPU assignment:
#   FLB2A (3 seeds, full-scale DPO): GPU 1 (sequential)
#   FLB2B (seeds 1+7, DPO 8K): GPU 2 (sequential)
#   FLB2C (Math-7B-Instruct backbone): GPU 3 (independent)
#   FLB2D+E (Math-PRM-7B backbone): GPU 1 (after FLB2A) or GPU 3 (parallel w/ FLB2C)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACTIVE_FLB2_GROUP="${ACTIVE_FLB2_GROUP:-FLB2A_DPO_FULLSCALE_GATED_SEED3}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_flb2_0312}"

MODEL_PATH="${MODEL_PATH:-assets/models/Qwen2.5-7B-Instruct}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-assets/artifacts/phase_e_feature_cache}"
PAIR_OUTPUT_ROOT="${PAIR_OUTPUT_ROOT:-assets/artifacts/phase_e_pairs}"
VALUE_OUTPUT_ROOT="${VALUE_OUTPUT_ROOT:-assets/artifacts/phase_e_runs}"
BENCH_OUTPUT_ROOT="${BENCH_OUTPUT_ROOT:-assets/artifacts/phase_e_eval}"
LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"

# Default hyperparameters — proven best config from FLB1: score+none+pair_acc
TARGET_TOTAL_PAIRS="${TARGET_TOTAL_PAIRS:-8192}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
HEAD_MLP_HIDDEN_SIZE="${HEAD_MLP_HIDDEN_SIZE:-1024}"
BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-256}"
RECIPE_RISK_POLICY="${RECIPE_RISK_POLICY:-error}"
HEAD_ARCHITECTURE="${HEAD_ARCHITECTURE:-gated_mlp}"

# GPU device
CUDA_DEVICE="${CUDA_DEVICE:-1}"

CURRENT_STAGE="bootstrap"

GROUP_TITLE=""
GROUP_INTENTION=""
GROUP_CASES=()

log_line() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1"
}

on_exit() {
  local exit_code=$?
  mkdir -p "$LOG_ROOT"
  if [[ $exit_code -ne 0 ]]; then
    {
      echo "# Phase E FLB2 Suite Summary"
      echo "- group_id: ${ACTIVE_FLB2_GROUP}"
      echo "- status: FAILED at stage: ${CURRENT_STAGE}"
    } > "$SUMMARY_FILE"
  fi
}
trap on_exit EXIT

resolve_group() {
  case "$ACTIVE_FLB2_GROUP" in
    FLB2A_DPO_FULLSCALE_GATED_SEED3)
      # 中文: DPO 全量（~9.7K）+ gated_mlp + 3 seeds。
      # Math-Step-DPO-10K 共 10795 行，16384 pair 目标实际会 cap 到 ~9.7K train pairs。
      # FIX_C gated (s42) 在 8K DPO + gated_mlp 上拿到 MATH AUC=0.749。
      # 这里用全量 DPO 测试数据上限，并跑 3 seed 确认稳定性。
      #
      # English: max out Math-Step-DPO-10K (~9.7K train pairs) with gated_mlp head, 3 seeds.
      # The 16384 target will naturally cap at available DPO data.
      GROUP_TITLE="FLB2A — DPO Full-Scale + GatedMLP, 3 Seeds"
      GROUP_INTENTION="FIX_C (8K DPO, gated_mlp, s42) = MATH AUC=0.749 (frozen-head SOTA). Scale to ~9.7K pairs (max available DPO data) and run 3 seeds. Target: median MATH AUC >= 0.75, all seeds >= 0.72. Also primary stability check for gated_mlp+DPO."
      GROUP_CASES=(
        "flb2a_dpo_full_gated_s42|dpo_scale_v1|16384|gated_mlp|42"
        "flb2a_dpo_full_gated_s1|dpo_scale_v1|16384|gated_mlp|1"
        "flb2a_dpo_full_gated_s7|dpo_scale_v1|16384|gated_mlp|7"
      )
      ;;
    FLB2B_DPO_8K_GATED_SEEDS17)
      # 中文: DPO 8K + gated_mlp，seeds 1 和 7（seed 42 已有 fix_c_gated: MATH=0.749）。
      # 复用已有的 fix_c_dpo_scale_8k pair artifact，避免重复 curate。
      # English: seeds 1 and 7 for DPO 8K + gated_mlp (seed 42 already exists as fix_c_gated).
      GROUP_TITLE="FLB2B — DPO 8K + GatedMLP, Seeds 1 and 7"
      GROUP_INTENTION="Seed 42 (fix_c_gated) = MATH AUC=0.749. Seed 7 MLP = 0.737. Run gated_mlp with seeds 1 and 7 to check if gated_mlp advantage is consistent across seeds."
      GROUP_CASES=(
        "flb2b_dpo_8k_gated_s1|dpo_scale_v1|8192|gated_mlp|1"
        "flb2b_dpo_8k_gated_s7|dpo_scale_v1|8192|gated_mlp|7"
      )
      ;;
    FLB2C_MATH_INSTRUCT_BACKBONE)
      # 中文: Qwen2.5-Math-7B-Instruct 骨干 + DPO 8K + gated_mlp。
      # Math-7B-Instruct 是数学特化的 instruction LLM（同 hidden_size=3584，与 7B-Instruct 兼容）。
      # 数学特化的预训练特征是否能进一步提升 step-level 区分能力？
      # 注意: 需要独立特征缓存（模型不同 → 特征不同）。
      #
      # English: Qwen2.5-Math-7B-Instruct backbone + DPO 8K + gated_mlp.
      # Tests whether math-domain backbone features improve step-quality discrimination.
      # Requires separate feature cache (different model = different features).
      GROUP_TITLE="FLB2C — Qwen2.5-Math-7B-Instruct Backbone + DPO 8K + GatedMLP"
      GROUP_INTENTION="Current SOTA uses Qwen2.5-7B-Instruct. Math-7B-Instruct has math-specific pretraining that may encode step-quality signals not present in general instruction LLM. Same hidden_size=3584, same feature caching infrastructure. Target: MATH AUC > 0.749."
      GROUP_CASES=(
        "flb2c_math_instruct_gated_s42|dpo_scale_v1|8192|gated_mlp|42"
      )
      ;;
    FLB2D_PRM_BACKBONE_MLP)
      # 中文: Qwen2.5-Math-PRM-7B 骨干 + DPO 8K + MLP 头。
      # Qwen2.5-Math-PRM-7B 是专门为过程奖励建模训练的骨干（已发布 ProcessBench F1=73.5%）。
      # 用 MLP 头而不是 gated_mlp，避免两个变量同时改变。
      # 注意: 需要设置 MODEL_PATH 到 Math-PRM-7B，以及独立特征缓存。
      #
      # English: Qwen2.5-Math-PRM-7B backbone + DPO 8K + MLP head.
      # Published ProcessBench F1=73.5% for the full model. Our lightweight head on top of
      # its frozen features should capture significantly better step representations.
      GROUP_TITLE="FLB2D — Qwen2.5-Math-PRM-7B Backbone + DPO 8K + MLP"
      GROUP_INTENTION="Qwen2.5-Math-PRM-7B published ProcessBench F1=73.5%. Our frozen-head on Qwen2.5-7B-Instruct gives F1~0.40. Using Math-PRM backbone as feature extractor should dramatically improve step discrimination. MLP head (not gated_mlp) to isolate backbone effect."
      GROUP_CASES=(
        "flb2d_prm_backbone_mlp_s42|dpo_scale_v1|8192|mlp|42"
      )
      ;;
    FLB2E_PRM_BACKBONE_GATED)
      # 中文: Qwen2.5-Math-PRM-7B 骨干 + DPO 8K + gated_mlp 头。
      # 最强骨干 + 最强头，预期上限测试。
      #
      # English: best backbone + best head. Ceiling test for frozen-backbone regime.
      GROUP_TITLE="FLB2E — Qwen2.5-Math-PRM-7B Backbone + DPO 8K + GatedMLP"
      GROUP_INTENTION="Combine best backbone (Math-PRM-7B) with best head (gated_mlp) and best data (DPO 8K). Ceiling test for frozen-backbone approach before committing to LoRA implementation."
      GROUP_CASES=(
        "flb2e_prm_backbone_gated_s42|dpo_scale_v1|8192|gated_mlp|42"
      )
      ;;
    FLB2_BACKBONE_SMOKE)
      # 中文: 骨干对比快速验证：3 骨干 × 1 seed × DPO 8K × gated_mlp。
      # 一次性运行，发现最优骨干后再跑 3 seed 稳定性实验。
      # English: quick smoke of all 3 backbones with gated_mlp head.
      GROUP_TITLE="FLB2 Backbone Smoke — 3 Backbones × DPO 8K × GatedMLP"
      GROUP_INTENTION="Quickly compare Qwen2.5-7B-Instruct (AUC=0.749, baseline), Math-7B-Instruct, and Math-PRM-7B on DPO 8K + gated_mlp. Pick winner for FLB2C/D/E full runs."
      GROUP_CASES=(
        "flb2_math_instruct_gated_s42|dpo_scale_v1|8192|gated_mlp|42"
        "flb2_prm_backbone_mlp_s42|dpo_scale_v1|8192|mlp|42"
        "flb2_prm_backbone_gated_s42|dpo_scale_v1|8192|gated_mlp|42"
      )
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_FLB2_GROUP=$ACTIVE_FLB2_GROUP" >&2
      echo "  Valid groups:" >&2
      echo "    FLB2A_DPO_FULLSCALE_GATED_SEED3   — DPO full-scale (~9.7K) + gated_mlp, 3 seeds" >&2
      echo "    FLB2B_DPO_8K_GATED_SEEDS17        — DPO 8K + gated_mlp, seeds 1+7" >&2
      echo "    FLB2C_MATH_INSTRUCT_BACKBONE       — Qwen2.5-Math-7B-Instruct + DPO 8K + gated_mlp" >&2
      echo "    FLB2D_PRM_BACKBONE_MLP             — Qwen2.5-Math-PRM-7B + DPO 8K + MLP" >&2
      echo "    FLB2E_PRM_BACKBONE_GATED           — Qwen2.5-Math-PRM-7B + DPO 8K + gated_mlp" >&2
      echo "    FLB2_BACKBONE_SMOKE                — Quick smoke: 3 backbones × DPO 8K × gated_mlp" >&2
      exit 1
      ;;
  esac
}

# ──────────────────────────────────────────────────────────────────────────────
# Helper: resolve model path and cache prefix from case_id
# ──────────────────────────────────────────────────────────────────────────────
resolve_model_and_cache() {
  local case_id="$1"
  local _model_path="$MODEL_PATH"
  local _cache_prefix="$FEATURE_CACHE_ROOT"

  # 根据 case_id 前缀决定骨干路径和特征缓存根目录。
  # Determine backbone path from case_id prefix.
  if [[ "$case_id" == *"math_instruct"* ]]; then
    _model_path="assets/models/Qwen2.5-Math-7B-Instruct"
    _cache_prefix="${FEATURE_CACHE_ROOT}/math_instruct"
  elif [[ "$case_id" == *"prm_backbone"* ]] || [[ "$case_id" == *"prm7b"* ]]; then
    _model_path="assets/models/Qwen2.5-Math-PRM-7B"
    _cache_prefix="${FEATURE_CACHE_ROOT}/math_prm"
  fi

  printf '%s\n%s\n' "$_model_path" "$_cache_prefix"
}

run_curate() {
  local case_id="$1"
  local profile="$2"
  local case_pairs="$3"
  local seed="$4"
  local run_name="${RUN_PREFIX}_${case_id}_${profile}_pairs"
  local cmd=(
    "$PYTHON_BIN" -u scripts/phase_e_curate_processbench_transfer_pairs.py
    --profile "$profile"
    --run-name "$run_name"
    --output-root "$PAIR_OUTPUT_ROOT"
    --seed "$seed"
    --validation-ratio 0.1
    --split-granularity source_sample
    --target-total-pairs "$case_pairs"
    --min-pair-confidence 0.55
  )
  CURRENT_STAGE="curate_${case_id}"
  log_line "RUN: ${cmd[*]}" | tee -a "$SUITE_LOG_FILE" >&2
  "${cmd[@]}" | tee -a "$SUITE_LOG_FILE" >&2
  "$PYTHON_BIN" - "$PAIR_OUTPUT_ROOT" "$run_name" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
prefix = sys.argv[2]
matches = sorted(root.glob(f"{prefix}__*"))
if not matches:
    raise SystemExit(f"No artifact directory matches prefix: {prefix}")
print(matches[-1])
PY
}

run_train() {
  local case_id="$1"
  local train_jsonl="$2"
  local eval_jsonl="$3"
  local head_arch="$4"
  local seed="$5"
  local case_model_path="$6"
  local case_cache_root="$7"
  local run_name="${RUN_PREFIX}_${case_id}_value"
  local cmd=(
    "$PYTHON_BIN" -u scripts/phase_e_train_value.py
    --train-pairs-jsonl "$train_jsonl"
    --eval-pairs-jsonl "$eval_jsonl"
    --model-path "$case_model_path"
    --run-name "$run_name"
    --output-root "$VALUE_OUTPUT_ROOT"
    --objective-mode joint
    --learning-rate "$LEARNING_RATE"
    --num-train-epochs "$TRAIN_EPOCHS"
    --per-device-train-batch-size "$TRAIN_BATCH_SIZE"
    --per-device-eval-batch-size "$EVAL_BATCH_SIZE"
    --max-length "$MAX_LENGTH"
    --lambda-ranking 1.0
    --lambda-bce 1.0
    --ranking-margin 0.02
    --ranking-target-space score
    --pair-weight-mode none
    --source-balance none
    --permutation-mode stable_hash
    --checkpoint-selection-metric pair_acc
    --recipe-risk-policy "$RECIPE_RISK_POLICY"
    --seed "$seed"
    --feature-cache-root "$case_cache_root"
    --feature-cache-mode read_write
    --feature-cache-lock-timeout-sec 600
    --head-architecture "$head_arch"
    --head-mlp-hidden-size "$HEAD_MLP_HIDDEN_SIZE"
    --head-dropout-prob 0.05
    --head-init-std 0.02
    --head-activation gelu
    --anti-saturation-weight 5e-4
    --anti-saturation-logit-threshold 3.5
    --nonfinite-feature-policy drop
    --require-cuda
  )
  CURRENT_STAGE="train_${case_id}"
  log_line "RUN (GPU ${CUDA_DEVICE}, model=${case_model_path##*/}): ${cmd[*]}" | tee -a "$SUITE_LOG_FILE" >&2
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${cmd[@]}" | tee -a "$SUITE_LOG_FILE" >&2
  "$PYTHON_BIN" - "$VALUE_OUTPUT_ROOT" "$run_name" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
prefix = sys.argv[2]
matches = sorted(root.glob(f"{prefix}_*"))
if not matches:
    raise SystemExit(f"No value run dir matches prefix: {prefix}")
print(matches[-1])
PY
}

run_bench() {
  local case_id="$1"
  local value_run_dir="$2"
  local benchmark_id="$3"
  local case_model_path="$4"
  local case_cache_root="$5"
  local run_name="${RUN_PREFIX}_${case_id}_${benchmark_id}"
  local cmd=(
    "$PYTHON_BIN" -u scripts/phase_e_eval_benchmark.py
    --value-run-dir "$value_run_dir"
    --benchmark-id "$benchmark_id"
    --run-name "$run_name"
    --output-root "$BENCH_OUTPUT_ROOT"
    --max-samples "$BENCH_MAX_SAMPLES"
    --batch-size "$EVAL_BATCH_SIZE"
    --feature-cache-root "$case_cache_root"
    --feature-cache-mode read_write
    --feature-cache-lock-timeout-sec 600
    --require-cuda
  )
  CURRENT_STAGE="bench_${case_id}_${benchmark_id}"
  log_line "RUN (GPU ${CUDA_DEVICE}): ${cmd[*]}" | tee -a "$SUITE_LOG_FILE" >&2
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${cmd[@]}" | tee -a "$SUITE_LOG_FILE" >&2
  "$PYTHON_BIN" - "$BENCH_OUTPUT_ROOT" "$run_name" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
prefix = sys.argv[2]
matches = sorted(root.glob(f"{prefix}_*"))
if not matches:
    raise SystemExit(f"No bench dir matches prefix: {prefix}")
print(matches[-1])
PY
}

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p "$LOG_ROOT"
resolve_group

log_line "Suite start: ${ACTIVE_FLB2_GROUP} — ${GROUP_TITLE}" | tee -a "$SUITE_LOG_FILE" >&2
log_line "Intention: ${GROUP_INTENTION}" | tee -a "$SUITE_LOG_FILE" >&2
log_line "Using CUDA_DEVICE=${CUDA_DEVICE}" | tee -a "$SUITE_LOG_FILE" >&2
log_line "HParams: score+none+pair_acc | ${TRAIN_EPOCHS} epochs | batch=${TRAIN_BATCH_SIZE} | head_default=${HEAD_ARCHITECTURE}" | tee -a "$SUITE_LOG_FILE" >&2

declare -A CASE_BENCH_GSM=()
declare -A CASE_BENCH_MATH=()

for case_spec in "${GROUP_CASES[@]}"; do
  IFS='|' read -r case_id profile case_pairs head_arch seed <<< "$case_spec"

  # Resolve model path and cache root for this case
  resolve_output=$(resolve_model_and_cache "$case_id")
  case_model_path=$(echo "$resolve_output" | head -1)
  case_cache_root=$(echo "$resolve_output" | tail -1)
  mkdir -p "$case_cache_root"

  log_line "=== Case: ${case_id} (profile: ${profile}, pairs: ${case_pairs}, head: ${head_arch}, seed: ${seed}, model: ${case_model_path##*/}) ===" | tee -a "$SUITE_LOG_FILE" >&2

  # 1. Curate pairs (skip if artifact already exists for same profile+pairs+seed)
  CURRENT_STAGE="curate_${case_id}"
  pair_dir=$(run_curate "$case_id" "$profile" "$case_pairs" "$seed")
  train_jsonl="${pair_dir}/train_pairs.jsonl"
  eval_jsonl="${pair_dir}/validation_pairs.jsonl"

  # 2. Train value head
  CURRENT_STAGE="train_${case_id}"
  value_dir=$(run_train "$case_id" "$train_jsonl" "$eval_jsonl" "$head_arch" "$seed" "$case_model_path" "$case_cache_root")

  # 3. Eval on ProcessBench (GSM8K + MATH)
  CURRENT_STAGE="bench_${case_id}_gsm8k"
  bench_gsm_dir=$(run_bench "$case_id" "$value_dir" "processbench_gsm8k" "$case_model_path" "$case_cache_root")
  CASE_BENCH_GSM[$case_id]="$bench_gsm_dir"

  CURRENT_STAGE="bench_${case_id}_math"
  bench_math_dir=$(run_bench "$case_id" "$value_dir" "processbench_math" "$case_model_path" "$case_cache_root")
  CASE_BENCH_MATH[$case_id]="$bench_math_dir"

  log_line "Case ${case_id} done." | tee -a "$SUITE_LOG_FILE" >&2

  # Print intermediate result
  "$PYTHON_BIN" - "$case_id" "$bench_gsm_dir" "$bench_math_dir" "$SUITE_LOG_FILE" <<'PY'
from pathlib import Path
import json, sys
case_id, gsm_dir, math_dir, log_file = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
def load(d, key, default=0.0):
    fp = d / "metrics.json"
    if not fp.exists(): return default
    return json.loads(fp.read_text()).get(key, default)
line = (
    f"=== {case_id} === "
    f"GSM pair_acc={load(gsm_dir,'pair_accuracy_good_vs_bad'):.4f} "
    f"auc={load(gsm_dir,'pair_auc_good_vs_bad'):.4f} | "
    f"MATH pair_acc={load(math_dir,'pair_accuracy_good_vs_bad'):.4f} "
    f"auc={load(math_dir,'pair_auc_good_vs_bad'):.4f} "
    f"f1={load(math_dir,'processbench_f1'):.4f} "
    f"first_edge={load(math_dir,'first_error_edge_accuracy'):.4f} "
    f"good={load(math_dir,'mean_good_prefix_score'):.4f} "
    f"bad={load(math_dir,'mean_bad_prefix_score'):.4f} "
    f"last={load(math_dir,'mean_all_correct_last_score'):.4f}"
)
print(line)
with open(log_file, "a") as f:
    f.write(line + "\n")
PY
done

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

{
  echo "# Phase E FLB2 (Frozen Backbone Level-2 Pushthrough) Suite Summary"
  echo ""
  echo "- group_id: ${ACTIVE_FLB2_GROUP}"
  echo "- group_title: ${GROUP_TITLE}"
  echo "- run_prefix: ${RUN_PREFIX}"
  echo "- status: completed"
  echo "- suite_log_file: ${SUITE_LOG_FILE}"
  echo "- cuda_device: ${CUDA_DEVICE}"
  echo "- hparams: score+none+pair_acc | ${TRAIN_EPOCHS} epochs | batch=${TRAIN_BATCH_SIZE}"
  echo ""
  echo "## Baseline (FLB1)"
  echo ""
  echo "- fix_c_dpo_scale_8k (MLP, s42): GSM AUC=0.659, MATH AUC=0.721"
  echo "- fix_c_gated (gated_mlp, s42):   GSM AUC=0.711, MATH AUC=0.749  ← frozen-head SOTA"
  echo "- fix_c_s7 (MLP, s7):             GSM AUC=0.669, MATH AUC=0.737"
  echo ""
  echo "## FLB2 Results"
  echo ""
  echo "| case_id | backbone | head | seed | pairs | pb_gsm_auc | pb_math_auc | pb_math_f1 | math_good | math_bad | math_last | good>bad | last>good |"
  echo "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
} > "$SUMMARY_FILE"

for case_spec in "${GROUP_CASES[@]}"; do
  IFS='|' read -r case_id profile case_pairs head_arch seed <<< "$case_spec"
  gsm_dir="${CASE_BENCH_GSM[$case_id]:-}"
  math_dir="${CASE_BENCH_MATH[$case_id]:-}"
  resolve_output=$(resolve_model_and_cache "$case_id")
  case_model_name=$(echo "$resolve_output" | head -1 | xargs basename)
  "$PYTHON_BIN" - "$case_id" "$case_model_name" "$head_arch" "$seed" "$case_pairs" "$gsm_dir" "$math_dir" "$SUMMARY_FILE" <<'PY'
from pathlib import Path
import json, sys
case_id, model, head, seed, pairs, gsm_dir, math_dir, summary = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
    Path(sys.argv[6]), Path(sys.argv[7]), sys.argv[8]
)
def load(d, key, default=0.0):
    fp = d / "metrics.json"
    if not fp.exists(): return default
    v = json.loads(fp.read_text()).get(key, default)
    return float(v) if v is not None else default
g_auc = load(gsm_dir, 'pair_auc_good_vs_bad')
m_auc = load(math_dir, 'pair_auc_good_vs_bad')
m_f1 = load(math_dir, 'processbench_f1')
good = load(math_dir, 'mean_good_prefix_score')
bad = load(math_dir, 'mean_bad_prefix_score')
last = load(math_dir, 'mean_all_correct_last_score')
row = (
    f"| {case_id} | {model} | {head} | {seed} | {pairs} "
    f"| {g_auc:.4f} | {m_auc:.4f} | {m_f1:.4f} "
    f"| {good:.4f} | {bad:.4f} | {last:.4f} "
    f"| {'✅' if good > bad else '❌'} | {'✅' if last > good else '❌'} |"
)
print(row)
with open(summary, "a") as f:
    f.write(row + "\n")
PY
done

log_line "Suite complete: ${SUMMARY_FILE}" | tee -a "$SUITE_LOG_FILE" >&2
