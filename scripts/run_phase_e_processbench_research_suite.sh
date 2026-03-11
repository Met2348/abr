#!/usr/bin/env bash
# Changelog (prepend-only, newest first):
# 2026-03-11: Added PBR2_FULL_MIXED_MLP_SEED3 and PBR3_LATER_BAD_BRANCH_SEED3 groups.
#             Extended case spec to 6-field format (case_id|mode|profile|head|balance|seed).
#             Updated run_curate / run_train to accept per-case seed.
#             Added GROUP_TARGET_PAIRS to support per-group artifact scale override.
#
# Phase E ProcessBench redesign research suite.
#
# English
# -------
# This suite is the new research harness for the current bottleneck:
# 1. curate ProcessBench-aligned supervision profiles,
# 2. train redesigned value-head recipes,
# 3. run same-family trust eval,
# 4. run ProcessBench eval,
# 5. summarize transfer and RL-promotion diagnostics.
#
# 中文
# ----
# 这个 suite 是给当前瓶颈问题准备的新研究入口：
# 1. 先构造 ProcessBench-aligned 的监督 profile，
# 2. 再训练重设计后的 value head recipe，
# 3. 跑 same-family trust 评测，
# 4. 跑 ProcessBench 评测，
# 5. 最后汇总 transfer 和 RL-promotion 诊断。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACTIVE_PHASE_E_PB_RESEARCH_GROUP="${ACTIVE_PHASE_E_PB_RESEARCH_GROUP:-PBR1_PROCESSBENCH_REDESIGN_SMOKE}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_processbench_research}"
MODEL_PATH="${MODEL_PATH:-assets/models/Qwen2.5-7B-Instruct}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-assets/artifacts/phase_e_feature_cache}"
PAIR_OUTPUT_ROOT="${PAIR_OUTPUT_ROOT:-assets/artifacts/phase_e_pairs}"
VALUE_OUTPUT_ROOT="${VALUE_OUTPUT_ROOT:-assets/artifacts/phase_e_runs}"
SAMEFAMILY_OUTPUT_ROOT="${SAMEFAMILY_OUTPUT_ROOT:-assets/artifacts/phase_e_samefamily_eval}"
BENCH_OUTPUT_ROOT="${BENCH_OUTPUT_ROOT:-assets/artifacts/phase_e_eval}"
PROMOTION_OUTPUT_ROOT="${PROMOTION_OUTPUT_ROOT:-assets/artifacts/phase_e_rl_promotion_diag}"
TRANSFER_COMPARE_ROOT="${TRANSFER_COMPARE_ROOT:-assets/artifacts/phase_e_transfer_compare}"

TARGET_TOTAL_PAIRS="${TARGET_TOTAL_PAIRS:-4096}"
BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-96}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-96}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-4}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-2}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
STAGE2_LEARNING_RATE="${STAGE2_LEARNING_RATE:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
HEAD_MLP_HIDDEN_SIZE="${HEAD_MLP_HIDDEN_SIZE:-1024}"
HEAD_DROPOUT_PROB="${HEAD_DROPOUT_PROB:-0.05}"
ANTI_SATURATION_WEIGHT="${ANTI_SATURATION_WEIGHT:-5e-4}"
ANTI_SATURATION_LOGIT_THRESHOLD="${ANTI_SATURATION_LOGIT_THRESHOLD:-3.5}"

REFERENCE_E87_VALUE_RUN_DIR="${REFERENCE_E87_VALUE_RUN_DIR:-assets/artifacts/phase_e_runs/phase_e_processbench_rlrepair_0311_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_s42_value_20260311T032957Z}"
REFERENCE_E87_SAMEFAMILY_DIR="${REFERENCE_E87_SAMEFAMILY_DIR:-assets/artifacts/phase_e_samefamily_eval/phase_e_rlready_e87_samefamily_0311_20260311T040021Z}"
REFERENCE_E87_PB_GSM_DIR="${REFERENCE_E87_PB_GSM_DIR:-assets/artifacts/phase_e_eval/phase_e_processbench_rlrepair_0311_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_s42_processbench_gsm8k_20260311T033946Z}"
REFERENCE_E87_PB_MATH_DIR="${REFERENCE_E87_PB_MATH_DIR:-assets/artifacts/phase_e_eval/phase_e_processbench_rlrepair_0311_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_s42_processbench_math_20260311T033955Z}"

LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"
RESULTS_JSONL="${LOG_ROOT}/research_results.jsonl"
CURRENT_STAGE="bootstrap"

GROUP_TITLE=""
GROUP_INTENTION=""
GROUP_OBSERVE=""
GROUP_EXPECT=""
GROUP_TARGET_PAIRS=""  # empty = use global TARGET_TOTAL_PAIRS; set per-group in resolve_group
CASES=()

log_line() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1"
}

on_exit() {
  local exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    return
  fi
  mkdir -p "$LOG_ROOT"
  {
    echo "# Phase E ProcessBench Research Suite Summary"
    echo
    echo "- group_id: ${ACTIVE_PHASE_E_PB_RESEARCH_GROUP}"
    echo "- group_title: ${GROUP_TITLE}"
    echo "- run_prefix: ${RUN_PREFIX}"
    echo "- status: failed"
    echo "- failed_stage: ${CURRENT_STAGE}"
    echo "- suite_log_file: ${SUITE_LOG_FILE}"
  } > "$SUMMARY_FILE"
}
trap on_exit EXIT

resolve_group() {
  case "$ACTIVE_PHASE_E_PB_RESEARCH_GROUP" in
    PBR1_PROCESSBENCH_REDESIGN_SMOKE)
      GROUP_TARGET_PAIRS=""  # use global TARGET_TOTAL_PAIRS (default 4096 smoke scale)
      GROUP_TITLE="PBR1 ProcessBench Redesign Smoke"
      GROUP_INTENTION="Test whether processbench-aligned curate profiles, semantic weighting, and alternative heads improve transfer beyond the current E87 baseline."
      GROUP_OBSERVE="The primary readouts are ProcessBench AUC / first-edge / terminal slices together with same-family prompt-level utility."
      GROUP_EXPECT="At least one redesigned case should beat the E87 reference on either benchmark AUC or terminal behavior without collapsing same-family utility."
      CASES=(
        "pbr1_ms_align_mlp|one_shot|ms_align_v1|mlp|none|42"
        "pbr2_ms_align_gated|one_shot|ms_align_v1|gated_mlp|none|42"
        "pbr3_ms_prm_align_gated|one_shot|ms_prm_align_v1|gated_mlp|uniform|42"
        "pbr4_ms_curriculum_gated|curriculum|ms_align_v1|gated_mlp|none|42"
      )
      ;;
    PBR2_FULL_MIXED_MLP_SEED3)
      # 中文: 把 smoke 阶段验证最优的 ms_align_v1+mlp 配方扩展到完整数据规模，跑 3 个 seed 评估稳定性。
      GROUP_TARGET_PAIRS=16384
      GROUP_TITLE="PBR2 Full-Scale Mixed MLP, 3 Seeds"
      GROUP_INTENTION="Scale up the proven ms_align_v1+mlp recipe from smoke (4096 pairs) to full artifact (16384 pairs); run 3 seeds to assess stability of the current best data geometry before introducing any architecture or profile change."
      GROUP_OBSERVE="Readouts: heldout pair_acc/AUC, ProcessBench GSM8K/Math AUC and first_edge slice, same-family top1 and local_first_bad. Compare against PBR1 pbr1_ms_align_mlp (smoke scale) and ref_e87 baseline."
      GROUP_EXPECT="All 3 seeds should beat PBR1 smoke on both pb_gsm_auc and pb_math_auc. Median run target: pb_math_auc >= 0.58, pb_gsm_auc >= 0.52. If inter-seed variance is high (auc range > 0.05), the recipe is not yet stable enough for PBR3."
      CASES=(
        "pbr2a_ms_align_mlp_s42|one_shot|ms_align_v1|mlp|none|42"
        "pbr2b_ms_align_mlp_s1|one_shot|ms_align_v1|mlp|none|1"
        "pbr2c_ms_align_mlp_s7|one_shot|ms_align_v1|mlp|none|7"
      )
      ;;
    PBR3_LATER_BAD_BRANCH_SEED3)
      # 中文: 用新 ms_laterbad_v1 profile（30% 定向 later-bad，用 pair_type_allowlist 过滤到
      #        lastsafe_vs_laterbad + earlygood_vs_laterbad 类型）攻克 later-bad 泛化 canary。
      GROUP_TARGET_PAIRS=16384
      GROUP_TITLE="PBR3 Later-Bad Branch Ablation, 3 Seeds"
      GROUP_INTENTION="Test whether the new ms_laterbad_v1 profile (strict 40% + fanout 20% + targeted_laterbad 30% + terminal 10%) improves the later-bad generalization canary vs ms_align_v1. The key change: laterbad component is filtered to only lastsafe_vs_laterbad and earlygood_vs_laterbad pair types, not generic all_good_vs_all_bad grid."
      GROUP_OBSERVE="Primary canaries: ProcessBench first_edge_accuracy and pb_gsm/pb_math AUC. Secondary: same-family local_first_bad_accuracy. A successful result shows AUC improvement with first_edge regression <= 0.05 absolute vs PBR2 median."
      GROUP_EXPECT="ms_laterbad_v1 should improve pb_gsm_auc and pb_math_auc vs PBR2. Acceptable: first_edge regression up to 0.05. Failure signal: first_edge collapses below 0.45 (targeted later-bad still too disruptive) or later-bad fraction too sparse (need more data)."
      CASES=(
        "pbr3a_ms_laterbad_mlp_s42|one_shot|ms_laterbad_v1|mlp|none|42"
        "pbr3b_ms_laterbad_mlp_s1|one_shot|ms_laterbad_v1|mlp|none|1"
        "pbr3c_ms_laterbad_mlp_s7|one_shot|ms_laterbad_v1|mlp|none|7"
      )
      ;;
    PBR4_PRM_AND_LATERBAD_FOLLOWUP_SMOKE)
      # 中文: 这是基于互联网调研后的 follow-up smoke：
      # 1. 用 `ms_prm_align_v1` 检验“显式 PRMBench 局部错误监督”是否真能修本地 benchmark 迁移，
      # 2. 同时直接比较 `mlp` vs `gated_mlp`，避免把 profile 改动和 head 改动混在一起，
      # 3. 再用 `ms_laterbad_v1` 检验 targeted later-bad 支持是否比 generic grid 更有效。
      #
      # English: this follow-up smoke is derived from the latest literature read:
      # 1. `ms_prm_align_v1` tests whether explicit PRMBench local supervision fixes local benchmark transfer,
      # 2. `mlp` vs `gated_mlp` is compared on the same profile so architecture and data geometry are not conflated,
      # 3. `ms_laterbad_v1` tests whether targeted later-bad support helps more than generic grid coverage.
      GROUP_TARGET_PAIRS=4096
      GROUP_TITLE="PBR4 PRMBench Auxiliary + Later-Bad Follow-Up Smoke"
      GROUP_INTENTION="Test the two strongest remaining hypotheses after PBR1 and the curated RL-ready smoke: explicit PRMBench local auxiliary may help benchmark-local transfer, and targeted later-bad support may help without broad grid noise."
      GROUP_OBSERVE="Readouts: ProcessBench GSM8K/Math AUC and first-edge are primary. Same-family top1/local stay as safety canaries. The architecture comparison should be read only inside the same profile."
      GROUP_EXPECT="At least one follow-up case should beat the earlier ms_align_v1+gated_mlp smoke on either pb_gsm_auc or pb_math_auc, without collapsing same-family local utility below 0.85."
      CASES=(
        "pbr4a_ms_prm_align_mlp|one_shot|ms_prm_align_v1|mlp|uniform|42"
        "pbr4b_ms_prm_align_gated|one_shot|ms_prm_align_v1|gated_mlp|uniform|42"
        "pbr4c_ms_laterbad_gated|one_shot|ms_laterbad_v1|gated_mlp|none|42"
      )
      ;;
    PBR5_DUAL_HEAD_ROUTING_SEED3)
      # 中文: 为 dual_head 架构增加 pair_semantics 路由后的第一批全量验证实验。
      # local_proj 接受 first_bad_edge / fanout / laterbad pair 的梯度；
      # terminal_proj 仅接受 terminal_completion_anchor pair 的梯度；
      # 推理时用 inference_alpha=0.5 混合两个头的 logit。
      #
      # !! 路由逻辑已在 training.py 中实现 (2026-03-11):
      #    - _resolve_pair_training_route_weights() 按 pair_semantics 分配 local/terminal 权重
      #    - terminal_completion_anchor → terminal_proj only
      #    - first_bad_edge / fanout → local_proj only
      #    - good_bad_prefix_grid → both (equal weight)
      # !! 当前状态: RUNNABLE — 路由已实现。先用 run_phase_e_dual_head_smoke.sh 做 smoke 验证。
      #
      # English: dual_head with pair_semantics routing implemented on 2026-03-11.
      # Run run_phase_e_dual_head_smoke.sh first to validate, then run this full suite.
      GROUP_TARGET_PAIRS=16384
      GROUP_TITLE="PBR5 dual_head with pair_semantics routing, 3 Seeds"
      GROUP_INTENTION="Validate whether routing local-pair gradients to local_proj and terminal-pair gradients to terminal_proj (with alpha-blend at inference) resolves the local-vs-terminal tradeoff that gated_mlp and mlp cannot solve with joint supervision."
      GROUP_OBSERVE="Primary: compare local_proj slice acc vs terminal_proj slice acc vs blended acc (alpha sweep). Secondary: ProcessBench AUC and first_edge accuracy vs PBR3 best."
      GROUP_EXPECT="dual_head with routing should match or exceed PBR3 on ProcessBench AUC while preserving terminal_top1 >= 0.50. If local_proj and terminal_proj specialize correctly, the alpha sweep should show a sharp transition (not flat) indicating the heads learned different objectives."
      CASES=(
        "pbr5a_ms_laterbad_dualhead_s42|one_shot|ms_laterbad_v1|dual_head|none|42"
        "pbr5b_ms_laterbad_dualhead_s1|one_shot|ms_laterbad_v1|dual_head|none|1"
        "pbr5c_ms_laterbad_dualhead_s7|one_shot|ms_laterbad_v1|dual_head|none|7"
      )
      ;;
    PBR6_LORA_BACKBONE_SMOKE)
      # 中文: 解冻 Qwen2.5-7B-Instruct 最后 4 层（LoRA rank=4, target=q_proj+v_proj+mlp），
      # 同步训练 value head（mlp 架构）。这是当前 frozen-backbone 路线的突破性尝试。
      #
      # !! 前置条件 (BLOCKING):
      #    1. 训练脚本必须新增 LoRA 模式（--lora-rank, --lora-target-modules, --lora-top-k-layers）
      #    2. feature cache 必须在 LoRA 路线下禁用（LoRA 改变 backbone，cache 会 stale）
      #    3. 需要一个 on-the-fly mini-batch encoding path 替代 cache-based encoding
      # !! 当前状态: NOT RUNNABLE — 等待 training 脚本 LoRA 模式实现。
      # !! 预期收益: 文献数据显示 frozen→LoRA 可把 ProcessBench AUC 从 ~0.62 推向 ~0.70+
      #
      # English: unfreeze last 4 Qwen layers via LoRA (rank=4), jointly train with MLP head.
      # This is the path to crossing the ~0.65 ProcessBench AUC ceiling of frozen-backbone.
      # BLOCKING prerequisites: training script LoRA mode + cache invalidation path.
      GROUP_TARGET_PAIRS=8192  # smoke scale for first LoRA run
      GROUP_TITLE="PBR6 LoRA Backbone Smoke [BLOCKED - needs LoRA infra in training script]"
      GROUP_INTENTION="Validate whether partial backbone unfreezing (LoRA on last 4 layers) breaks the ~0.62 AUC ceiling of frozen-backbone PRMs. Literature benchmark: Skywork-o1-PRM (~75%), RLHFlow/Mistral-PRM (~70%), both use backbone LoRA."
      GROUP_OBSERVE="Primary: ProcessBench GSM8K/Math AUC. Compare directly against best PBR3 frozen-backbone run. If LoRA breaks ceiling, AUC should improve by >= 0.05 absolute."
      GROUP_EXPECT="If frozen-backbone ceiling is confirmed by PBR2/PBR3 at ~0.50-0.55 AUC, LoRA should push toward 0.60+. If frozen baseline already reached 0.65+, LoRA improvement may be smaller."
      CASES=(
        "pbr6a_ms_laterbad_lora_mlp_s42|one_shot|ms_laterbad_v1|mlp|none|42"
        "pbr6b_ms_laterbad_lora_mlp_s1|one_shot|ms_laterbad_v1|mlp|none|1"
      )
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_E_PB_RESEARCH_GROUP=$ACTIVE_PHASE_E_PB_RESEARCH_GROUP" >&2
      exit 1
      ;;
  esac
  if [[ -n "${PHASE_E_PB_RESEARCH_CASES_OVERRIDE:-}" ]]; then
    # shellcheck disable=SC2206
    CASES=(${PHASE_E_PB_RESEARCH_CASES_OVERRIDE})
  fi
}

latest_run_dir() {
  local prefix="$1"
  "$PYTHON_BIN" - "$prefix" <<'PY'
from pathlib import Path
import sys
prefix = sys.argv[1]
matches = sorted(Path(prefix).parent.glob(Path(prefix).name + "__*"))
if not matches:
    raise SystemExit(f"No artifact directory matches prefix: {prefix}")
print(matches[-1])
PY
}

latest_timestamped_dir() {
  local prefix="$1"
  "$PYTHON_BIN" - "$prefix" <<'PY'
from pathlib import Path
import sys
prefix = sys.argv[1]
matches = sorted(Path(prefix).parent.glob(Path(prefix).name + "_*"))
if not matches:
    raise SystemExit(f"No artifact directory matches prefix: {prefix}")
print(matches[-1])
PY
}

run_curate() {
  local case_id="$1"
  local profile="$2"
  local seed="${3:-42}"
  local _target_pairs="${GROUP_TARGET_PAIRS:-$TARGET_TOTAL_PAIRS}"
  local run_name="${RUN_PREFIX}_${case_id}_${profile}_pairs"
  local cmd=(
    "$PYTHON_BIN" -u scripts/phase_e_curate_processbench_transfer_pairs.py
    --profile "$profile"
    --run-name "$run_name"
    --output-root "$PAIR_OUTPUT_ROOT"
    --seed "$seed"
    --validation-ratio 0.1
    --split-granularity source_sample
    --target-total-pairs "$_target_pairs"
    --min-pair-confidence 0.55
  )
  CURRENT_STAGE="curate_${case_id}"
  log_line "RUN: ${cmd[*]}" | tee -a "$SUITE_LOG_FILE" >&2
  "${cmd[@]}" | tee -a "$SUITE_LOG_FILE" >&2
  latest_run_dir "${PAIR_OUTPUT_ROOT}/${run_name}"
}

run_train() {
  local case_id="$1"
  local train_pairs_jsonl="$2"
  local eval_pairs_jsonl="$3"
  local head_architecture="$4"
  local source_balance="$5"
  local learning_rate="$6"
  local num_train_epochs="$7"
  local init_value_head_path="${8:-}"
  local seed="${9:-42}"
  local run_name="${RUN_PREFIX}_${case_id}_value"
  local cmd=(
    "$PYTHON_BIN" -u scripts/phase_e_train_value.py
    --train-pairs-jsonl "$train_pairs_jsonl"
    --eval-pairs-jsonl "$eval_pairs_jsonl"
    --model-path "$MODEL_PATH"
    --run-name "$run_name"
    --output-root "$VALUE_OUTPUT_ROOT"
    --objective-mode joint
    --learning-rate "$learning_rate"
    --num-train-epochs "$num_train_epochs"
    --per-device-train-batch-size "$TRAIN_BATCH_SIZE"
    --per-device-eval-batch-size "$EVAL_BATCH_SIZE"
    --max-length "$MAX_LENGTH"
    --lambda-ranking 1.0
    --lambda-bce 1.0
    --ranking-margin 0.02
    --ranking-target-space logit
    --pair-weight-mode confidence_semantic
    --source-balance "$source_balance"
    --permutation-mode stable_hash
    --checkpoint-selection-metric ranking_score
    --seed "$seed"
    --feature-cache-root "$FEATURE_CACHE_ROOT"
    --feature-cache-mode read_write
    --feature-cache-lock-timeout-sec 600
    --head-architecture "$head_architecture"
    --head-mlp-hidden-size "$HEAD_MLP_HIDDEN_SIZE"
    --head-dropout-prob "$HEAD_DROPOUT_PROB"
    --head-init-std 0.02
    --head-activation gelu
    --anti-saturation-weight "$ANTI_SATURATION_WEIGHT"
    --anti-saturation-logit-threshold "$ANTI_SATURATION_LOGIT_THRESHOLD"
    --require-cuda
  )
  if [[ -n "$init_value_head_path" ]]; then
    cmd+=(--init-value-head-path "$init_value_head_path")
  fi
  CURRENT_STAGE="train_${case_id}"
  log_line "RUN: ${cmd[*]}" | tee -a "$SUITE_LOG_FILE" >&2
  "${cmd[@]}" | tee -a "$SUITE_LOG_FILE" >&2
  latest_timestamped_dir "${VALUE_OUTPUT_ROOT}/${run_name}"
}

run_samefamily_eval() {
  local case_id="$1"
  local value_run_dir="$2"
  local eval_pairs_jsonl="$3"
  local run_name="${RUN_PREFIX}_${case_id}_samefamily"
  local cmd=(
    "$PYTHON_BIN" -u scripts/phase_e_eval_samefamily_trust.py
    --value-run-dir "$value_run_dir"
    --eval-pairs-jsonl "$eval_pairs_jsonl"
    --run-name "$run_name"
    --output-root "$SAMEFAMILY_OUTPUT_ROOT"
    --checkpoint-name best
    --batch-size "$EVAL_BATCH_SIZE"
    --max-length "$MAX_LENGTH"
    --feature-cache-root "$FEATURE_CACHE_ROOT"
    --feature-cache-mode read_write
    --feature-cache-lock-timeout-sec 600
    --edge-weight-mode confidence
    --require-cuda
  )
  CURRENT_STAGE="samefamily_${case_id}"
  log_line "RUN: ${cmd[*]}" | tee -a "$SUITE_LOG_FILE" >&2
  "${cmd[@]}" | tee -a "$SUITE_LOG_FILE" >&2
  latest_timestamped_dir "${SAMEFAMILY_OUTPUT_ROOT}/${run_name}"
}

run_benchmark_eval() {
  local case_id="$1"
  local value_run_dir="$2"
  local benchmark_id="$3"
  local run_name="${RUN_PREFIX}_${case_id}_${benchmark_id}"
  local cmd=(
    "$PYTHON_BIN" -u scripts/phase_e_eval_benchmark.py
    --value-run-dir "$value_run_dir"
    --benchmark-id "$benchmark_id"
    --run-name "$run_name"
    --output-root "$BENCH_OUTPUT_ROOT"
    --checkpoint-name best
    --max-samples "$BENCH_MAX_SAMPLES"
    --batch-size "$EVAL_BATCH_SIZE"
    --max-length "$MAX_LENGTH"
    --feature-cache-root "$FEATURE_CACHE_ROOT"
    --feature-cache-mode read_write
    --feature-cache-lock-timeout-sec 600
    --require-cuda
  )
  CURRENT_STAGE="benchmark_${case_id}_${benchmark_id}"
  log_line "RUN: ${cmd[*]}" | tee -a "$SUITE_LOG_FILE" >&2
  "${cmd[@]}" | tee -a "$SUITE_LOG_FILE" >&2
  latest_timestamped_dir "${BENCH_OUTPUT_ROOT}/${run_name}"
}

append_result() {
  local case_id="$1"
  local mode="$2"
  local profile="$3"
  local head_architecture="$4"
  local source_balance="$5"
  local curated_dir="$6"
  local value_run_dir="$7"
  local samefamily_dir="$8"
  local pb_gsm_dir="$9"
  local pb_math_dir="${10}"
  "$PYTHON_BIN" - "$case_id" "$mode" "$profile" "$head_architecture" "$source_balance" "$curated_dir" "$value_run_dir" "$samefamily_dir" "$pb_gsm_dir" "$pb_math_dir" "$RESULTS_JSONL" <<'PY'
import json
import sys
from pathlib import Path

case_id = sys.argv[1]
mode = sys.argv[2]
profile = sys.argv[3]
head_architecture = sys.argv[4]
source_balance = sys.argv[5]
curated_dir = Path(sys.argv[6])
value_run_dir = Path(sys.argv[7])
samefamily_dir = Path(sys.argv[8])
pb_gsm_dir = Path(sys.argv[9])
pb_math_dir = Path(sys.argv[10])
out_path = Path(sys.argv[11])

samefamily_metrics = json.loads((samefamily_dir / "metrics.json").read_text(encoding="utf-8"))
pb_gsm_summary = json.loads((pb_gsm_dir / "summary.json").read_text(encoding="utf-8"))
pb_math_summary = json.loads((pb_math_dir / "summary.json").read_text(encoding="utf-8"))
curated_summary = json.loads((curated_dir / "summary.json").read_text(encoding="utf-8"))
value_metrics = json.loads((value_run_dir / "eval_metrics.json").read_text(encoding="utf-8"))
eval_pairs = dict(value_metrics.get("eval_pairs", {}))

row = {
    "case_id": case_id,
    "mode": mode,
    "profile": profile,
    "head_architecture": head_architecture,
    "source_balance": source_balance,
    "curated_dir": str(curated_dir),
    "value_run_dir": str(value_run_dir),
    "samefamily_dir": str(samefamily_dir),
    "pb_gsm_dir": str(pb_gsm_dir),
    "pb_math_dir": str(pb_math_dir),
    "heldout_pair_acc": float(eval_pairs.get("pair_accuracy", 0.0)),
    "heldout_auc": float(eval_pairs.get("auc", 0.0)),
    "samefamily_top1": float(samefamily_metrics.get("prompt_pool_top1_accuracy", 0.0)),
    "samefamily_local_first_bad": float(
        samefamily_metrics.get("local_first_bad_edge_accuracy")
        or samefamily_metrics.get("local_safe_vs_bad_pair_accuracy")
        or 0.0
    ),
    "pb_gsm_pair_acc": float(pb_gsm_summary["metrics"].get("pair_accuracy_good_vs_bad", 0.0)),
    "pb_gsm_auc": float(pb_gsm_summary["metrics"].get("pair_auc_good_vs_bad", 0.0)),
    "pb_gsm_first_edge": float(pb_gsm_summary["metrics"].get("first_error_edge_accuracy", 0.0)),
    "pb_math_pair_acc": float(pb_math_summary["metrics"].get("pair_accuracy_good_vs_bad", 0.0)),
    "pb_math_auc": float(pb_math_summary["metrics"].get("pair_auc_good_vs_bad", 0.0)),
    "pb_math_first_edge": float(pb_math_summary["metrics"].get("first_error_edge_accuracy", 0.0)),
    "curated_train_summary": curated_summary.get("train_summary", {}),
}
with out_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

build_compare_case_args() {
  local benchmark_key="$1"
  local args=()
  if [[ -f "$RESULTS_JSONL" ]]; then
    while IFS= read -r raw; do
      [[ -z "$raw" ]] && continue
      local case_id
      local value_run_dir
      local benchmark_dir
      case_id="$($PYTHON_BIN - "$raw" <<'PY'
import json, sys
row = json.loads(sys.argv[1])
print(row["case_id"])
PY
)"
      value_run_dir="$($PYTHON_BIN - "$raw" <<'PY'
import json, sys
row = json.loads(sys.argv[1])
print(row["value_run_dir"])
PY
)"
      benchmark_dir="$($PYTHON_BIN - "$raw" "$benchmark_key" <<'PY'
import json, sys
row = json.loads(sys.argv[1])
benchmark_key = sys.argv[2]
print(row[benchmark_key])
PY
)"
      args+=(--case "${case_id}=${value_run_dir}::${benchmark_dir}")
    done < "$RESULTS_JSONL"
  fi
  if [[ -d "$REFERENCE_E87_VALUE_RUN_DIR" ]]; then
    if [[ "$benchmark_key" == "pb_gsm_dir" && -d "$REFERENCE_E87_PB_GSM_DIR" ]]; then
      args+=(--case "ref_e87=${REFERENCE_E87_VALUE_RUN_DIR}::${REFERENCE_E87_PB_GSM_DIR}")
    fi
    if [[ "$benchmark_key" == "pb_math_dir" && -d "$REFERENCE_E87_PB_MATH_DIR" ]]; then
      args+=(--case "ref_e87=${REFERENCE_E87_VALUE_RUN_DIR}::${REFERENCE_E87_PB_MATH_DIR}")
    fi
  fi
  printf '%s\n' "${args[@]}"
}

run_compare_and_promotion() {
  local gsm_args=()
  local math_args=()
  mapfile -t gsm_args < <(build_compare_case_args pb_gsm_dir)
  mapfile -t math_args < <(build_compare_case_args pb_math_dir)

  CURRENT_STAGE="compare_transfer"
  if [[ ${#gsm_args[@]} -gt 0 ]]; then
    local gsm_cmd=("$PYTHON_BIN" -u scripts/phase_e_compare_processbench_transfer.py --run-name "${RUN_PREFIX}_gsm_compare" --output-root "$TRANSFER_COMPARE_ROOT" "${gsm_args[@]}")
    log_line "RUN: ${gsm_cmd[*]}" | tee -a "$SUITE_LOG_FILE"
    "${gsm_cmd[@]}" | tee -a "$SUITE_LOG_FILE"
  fi
  if [[ ${#math_args[@]} -gt 0 ]]; then
    local math_cmd=("$PYTHON_BIN" -u scripts/phase_e_compare_processbench_transfer.py --run-name "${RUN_PREFIX}_math_compare" --output-root "$TRANSFER_COMPARE_ROOT" "${math_args[@]}")
    log_line "RUN: ${math_cmd[*]}" | tee -a "$SUITE_LOG_FILE"
    "${math_cmd[@]}" | tee -a "$SUITE_LOG_FILE"
  fi

  local promotion_cmd=(
    "$PYTHON_BIN" -u scripts/phase_e_diagnose_rl_promotion.py
    --run-name "${RUN_PREFIX}_rl_promotion"
    --output-root "$PROMOTION_OUTPUT_ROOT"
  )
  while IFS= read -r raw; do
    [[ -z "$raw" ]] && continue
    local audit_spec
    audit_spec="$($PYTHON_BIN - "$raw" <<'PY'
import json, sys
row = json.loads(sys.argv[1])
print(f"{row['case_id']}|math_shepherd|{row['samefamily_dir']}|{row['pb_gsm_dir']}|{row['pb_math_dir']}")
PY
)"
    promotion_cmd+=(--audit-spec "$audit_spec")
  done < "$RESULTS_JSONL"
  if [[ -d "$REFERENCE_E87_SAMEFAMILY_DIR" && -d "$REFERENCE_E87_PB_GSM_DIR" && -d "$REFERENCE_E87_PB_MATH_DIR" ]]; then
    promotion_cmd+=(--audit-spec "ref_e87|math_shepherd|${REFERENCE_E87_SAMEFAMILY_DIR}|${REFERENCE_E87_PB_GSM_DIR}|${REFERENCE_E87_PB_MATH_DIR}")
  fi
  CURRENT_STAGE="rl_promotion"
  log_line "RUN: ${promotion_cmd[*]}" | tee -a "$SUITE_LOG_FILE"
  "${promotion_cmd[@]}" | tee -a "$SUITE_LOG_FILE"
}

render_final_summary() {
  "$PYTHON_BIN" - "$RESULTS_JSONL" "$SUMMARY_FILE" "$RUN_PREFIX" "$ACTIVE_PHASE_E_PB_RESEARCH_GROUP" "$GROUP_TITLE" "$GROUP_INTENTION" "$GROUP_OBSERVE" "$GROUP_EXPECT" <<'PY'
import json
import sys
from pathlib import Path

results_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
run_prefix = sys.argv[3]
group_id = sys.argv[4]
group_title = sys.argv[5]
group_intention = sys.argv[6]
group_observe = sys.argv[7]
group_expect = sys.argv[8]
rows = []
if results_path.exists():
    rows = [json.loads(raw) for raw in results_path.read_text(encoding="utf-8").splitlines() if raw.strip()]
rows.sort(key=lambda row: row["case_id"])
lines = [
    "# Phase E ProcessBench Research Suite Summary",
    "",
    f"- group_id: `{group_id}`",
    f"- group_title: {group_title}",
    f"- run_prefix: `{run_prefix}`",
    f"- group_intention: {group_intention}",
    f"- group_observe: {group_observe}",
    f"- group_expect: {group_expect}",
    "",
    "| case | mode | profile | head | heldout_pair | heldout_auc | samefamily_top1 | samefamily_local | pb_gsm_auc | pb_math_auc | pb_gsm_first_edge | pb_math_first_edge |",
    "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['case_id']} | {row['mode']} | {row['profile']} | {row['head_architecture']} | "
        f"{row['heldout_pair_acc']:.4f} | {row['heldout_auc']:.4f} | {row['samefamily_top1']:.4f} | "
        f"{row['samefamily_local_first_bad']:.4f} | {row['pb_gsm_auc']:.4f} | {row['pb_math_auc']:.4f} | "
        f"{row['pb_gsm_first_edge']:.4f} | {row['pb_math_first_edge']:.4f} |"
    )
summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

resolve_group
mkdir -p "$LOG_ROOT"
: > "$RESULTS_JSONL"
{
  echo "# Phase E ProcessBench Research Suite"
  echo
  echo "- group_id: ${ACTIVE_PHASE_E_PB_RESEARCH_GROUP}"
  echo "- group_title: ${GROUP_TITLE}"
} > "$SUMMARY_FILE"

log_line "Suite start: ${ACTIVE_PHASE_E_PB_RESEARCH_GROUP}" | tee -a "$SUITE_LOG_FILE"

for case_spec in "${CASES[@]}"; do
  IFS='|' read -r case_id mode profile head_architecture source_balance case_seed <<< "$case_spec"
  case_seed="${case_seed:-42}"
  if [[ "$mode" == "one_shot" ]]; then
    curated_dir="$(run_curate "$case_id" "$profile" "$case_seed")"
    value_run_dir="$(run_train "$case_id" "$curated_dir/train_pairs.jsonl" "$curated_dir/validation_pairs.jsonl" "$head_architecture" "$source_balance" "$LEARNING_RATE" "$TRAIN_EPOCHS" "" "$case_seed")"
  else
    core_dir="$(run_curate "${case_id}_core" "ms_core_v1" "$case_seed")"
    stage1_run_dir="$(run_train "${case_id}_stage1" "$core_dir/train_pairs.jsonl" "$core_dir/validation_pairs.jsonl" "$head_architecture" "$source_balance" "$LEARNING_RATE" "$STAGE1_EPOCHS" "" "$case_seed")"
    curated_dir="$(run_curate "$case_id" "$profile" "$case_seed")"
    value_run_dir="$(run_train "$case_id" "$curated_dir/train_pairs.jsonl" "$curated_dir/validation_pairs.jsonl" "$head_architecture" "$source_balance" "$STAGE2_LEARNING_RATE" "$STAGE2_EPOCHS" "$stage1_run_dir/best_value_head.pt" "$case_seed")"
  fi
  samefamily_dir="$(run_samefamily_eval "$case_id" "$value_run_dir" "$curated_dir/validation_pairs.jsonl")"
  pb_gsm_dir="$(run_benchmark_eval "$case_id" "$value_run_dir" processbench_gsm8k)"
  pb_math_dir="$(run_benchmark_eval "$case_id" "$value_run_dir" processbench_math)"
  append_result "$case_id" "$mode" "$profile" "$head_architecture" "$source_balance" "$curated_dir" "$value_run_dir" "$samefamily_dir" "$pb_gsm_dir" "$pb_math_dir"
done

run_compare_and_promotion
render_final_summary

log_line "Suite complete: ${SUMMARY_FILE}" | tee -a "$SUITE_LOG_FILE"
