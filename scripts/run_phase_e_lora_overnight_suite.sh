#!/usr/bin/env bash
# Phase E LoRA overnight frontier suite.
#
# 这个脚本的职责很聚焦：
# 1. 不再把夜间算力花在旧的 `PBR10` smoke 上，
# 2. 而是围绕当前最强的 `PBR26` 数据池，连续跑几组更像真正 frontier 的
#    `LoRA + architecture + objective` 组合，
# 3. 并在每个训练结束后自动补 same-family / ProcessBench 双路评测，
# 4. 最终生成一份可直接晨读的汇总表。
#
# This suite is intentionally narrow:
# 1. stop spending overnight budget on older `PBR10` smoke directions,
# 2. move the budget onto the stronger `PBR26` pair pool,
# 3. test a small number of orthogonal `LoRA + architecture + objective`
#    combinations that could plausibly improve Phase E,
# 4. and write one morning-readable summary instead of leaving only raw logs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-3}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_lora_overnight_$(date +%m%d_%H%M)}"
ACTIVE_PHASE_E_LORA_OVERNIGHT_GROUP="${ACTIVE_PHASE_E_LORA_OVERNIGHT_GROUP:-LALL_CURATED_LORA_FRONTIER}"

MODEL_PATH="${MODEL_PATH:-assets/models/Qwen2.5-Math-PRM-7B}"
PAIR_DIR="${PAIR_DIR:-assets/artifacts/phase_e_pairs/phase_e_pbr26_dpo_plus_ms_full_pairs__b17437d10dfc}"
VALUE_OUTPUT_ROOT="${VALUE_OUTPUT_ROOT:-assets/artifacts/phase_e_runs}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-assets/artifacts/phase_e_eval}"
SAMEFAMILY_OUTPUT_ROOT="${SAMEFAMILY_OUTPUT_ROOT:-assets/artifacts/phase_e_samefamily_eval}"
RECIPE_RISK_POLICY="${RECIPE_RISK_POLICY:-error}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TRAIN_EVAL_BATCH_SIZE="${TRAIN_EVAL_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-24}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
SEED="${SEED:-42}"

# LoRA-backed eval keeps the full 7B backbone + adapter resident on GPU.
# 这里的评测 batch size 故意低于 96，因为 LoRA 评测会常驻整套 7B backbone，
# 1024-token 的 ProcessBench / same-family 在 80GB 上更稳妥的范围是 32-48。
# These eval batch sizes intentionally stay below 96 because LoRA evaluation
# keeps the full 7B backbone resident; 1024-token ProcessBench / same-family
# passes have been more stable in the 32-48 range on 80GB.
SAMEFAMILY_BATCH_SIZE="${SAMEFAMILY_BATCH_SIZE:-48}"
BENCHMARK_BATCH_SIZE="${BENCHMARK_BATCH_SIZE:-32}"
WAIT_FOR_GPU_FREE="${WAIT_FOR_GPU_FREE:-1}"
GPU_FREE_MAX_USED_MIB="${GPU_FREE_MAX_USED_MIB:-2048}"
GPU_FREE_POLL_SEC="${GPU_FREE_POLL_SEC:-120}"

LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
RESULTS_JSONL="${LOG_ROOT}/results.jsonl"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"

mkdir -p "$LOG_ROOT"
: > "$RESULTS_JSONL"

log_line() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1" | tee -a "$SUITE_LOG_FILE"
}

# 把 case 级日志写成 best-effort side log，避免再用 `tee` 把日志问题放大成实验失败。
# Write case-level logs as best-effort side logs so logging issues do not flip a completed run into a suite failure.
append_case_log_line() {
  local case_log="$1"
  local message="$2"
  mkdir -p "$(dirname "$case_log")"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$message" >> "$case_log" 2>/dev/null || true
}

# suite log 继续保留全局可见性，case log 只做逐 case 深挖。
# Keep suite log globally visible while case log remains the per-case deep-debug trail.
log_case_line() {
  local case_log="$1"
  local message="$2"
  log_line "$message"
  append_case_log_line "$case_log" "$message"
}

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    exit 1
  fi
}

latest_dir_by_prefix() {
  local root="$1"
  local prefix="$2"
  python - "$root" "$prefix" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
prefix = sys.argv[2]
matches = sorted(root.glob(f"{prefix}_*"), key=lambda path: path.stat().st_mtime, reverse=True)
if not matches:
    raise SystemExit(f"No artifact directory matches prefix={prefix!r} under {root}")
print(matches[0])
PY
}

latest_dir_by_prefix_with_marker() {
  local root="$1"
  local prefix="$2"
  local marker_path="$3"
  python - "$root" "$prefix" "$marker_path" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
prefix = sys.argv[2]
marker_path = sys.argv[3]
matches = [
    path
    for path in sorted(root.glob(f"{prefix}_*"), key=lambda item: item.stat().st_mtime, reverse=True)
    if (path / marker_path).exists()
]
if not matches:
    raise SystemExit(
        f"No completed artifact directory matches prefix={prefix!r} under {root} "
        f"with marker={marker_path!r}"
    )
print(matches[0])
PY
}

current_gpu_used_mib() {
  local gpu_id="$1"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' -v gpu_id="$gpu_id" '
      {
        gsub(/ /, "", $1)
        gsub(/ /, "", $2)
        if ($1 == gpu_id) {
          print $2
          found = 1
          exit
        }
      }
      END {
        if (!found) {
          exit 1
        }
      }
    '
}

wait_for_gpu_if_needed() {
  if [[ "$WAIT_FOR_GPU_FREE" != "1" ]]; then
    return
  fi
  while true; do
    local used_mib
    used_mib="$(current_gpu_used_mib "$CUDA_DEVICE")"
    if [[ "$used_mib" -le "$GPU_FREE_MAX_USED_MIB" ]]; then
      log_line "GPU${CUDA_DEVICE} appears free enough: used_mib=${used_mib}"
      return
    fi
    log_line "GPU${CUDA_DEVICE} still busy: used_mib=${used_mib}; sleeping ${GPU_FREE_POLL_SEC}s before retry"
    sleep "$GPU_FREE_POLL_SEC"
  done
}

append_result_row() {
  local config_id="$1"
  local status="$2"
  local head_arch="$3"
  local lora_scope="$4"
  local run_dir="${5:-}"
  local samefamily_dir="${6:-}"
  local math_dir="${7:-}"
  local gsm_dir="${8:-}"
  python - "$RESULTS_JSONL" "$config_id" "$status" "$head_arch" "$lora_scope" "$run_dir" "$samefamily_dir" "$math_dir" "$gsm_dir" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    out_path,
    config_id,
    status,
    head_arch,
    lora_scope,
    run_dir,
    samefamily_dir,
    math_dir,
    gsm_dir,
) = sys.argv[1:]


def _load_json(path_text: str, file_name: str) -> dict:
    if not path_text:
        return {}
    path = Path(path_text) / file_name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


samefamily_metrics = _load_json(samefamily_dir, "metrics.json")
math_metrics = _load_json(math_dir, "metrics.json")
gsm_metrics = _load_json(gsm_dir, "metrics.json")
run_summary = _load_json(run_dir, "summary.json")

row = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "config_id": config_id,
    "status": status,
    "head_architecture": head_arch,
    "lora_scope": lora_scope,
    "run_dir": run_dir or None,
    "samefamily_dir": samefamily_dir or None,
    "math_dir": math_dir or None,
    "gsm_dir": gsm_dir or None,
    "train_selection_value": run_summary.get("selection_value"),
    "samefamily_prompt_top1": samefamily_metrics.get("prompt_pool_top1_accuracy"),
    "samefamily_local_first_bad": samefamily_metrics.get("local_first_bad_accuracy"),
    "samefamily_reject_at_80": samefamily_metrics.get("rejection_curve", {}).get("0.8", {}).get("accuracy"),
    "math_auc": math_metrics.get("pair_auc_good_vs_bad"),
    "math_f1": math_metrics.get("processbench_f1"),
    "math_first_edge": math_metrics.get("first_error_edge_accuracy"),
    "math_all_correct_last": math_metrics.get("mean_all_correct_last_score"),
    "gsm_auc": gsm_metrics.get("pair_auc_good_vs_bad"),
    "gsm_f1": gsm_metrics.get("processbench_f1"),
    "gsm_first_edge": gsm_metrics.get("first_error_edge_accuracy"),
    "gsm_all_correct_last": gsm_metrics.get("mean_all_correct_last_score"),
}
with Path(out_path).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps(row, ensure_ascii=False))
PY
}

run_samefamily_eval() {
  local config_id="$1"
  local run_dir="$2"
  local case_log="$3"
  local eval_run_name="${RUN_PREFIX}_${config_id}_samefamily"
  log_case_line "$case_log" "SAMEFAMILY ${config_id}: ${eval_run_name}"
  {
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "$PYTHON_BIN" -u scripts/phase_e_eval_samefamily_trust.py \
      --value-run-dir "$run_dir" \
      --run-name "$eval_run_name" \
      --output-root "$SAMEFAMILY_OUTPUT_ROOT" \
      --batch-size "$SAMEFAMILY_BATCH_SIZE" \
      --feature-cache-mode off \
      --require-cuda
  } >> "$case_log" 2>&1
  latest_dir_by_prefix_with_marker "$SAMEFAMILY_OUTPUT_ROOT" "$eval_run_name" "metrics.json"
}

run_benchmark_eval() {
  local config_id="$1"
  local benchmark_id="$2"
  local run_dir="$3"
  local case_log="$4"
  local eval_run_name="${RUN_PREFIX}_${config_id}_${benchmark_id}"
  log_case_line "$case_log" "BENCH ${config_id} ${benchmark_id}: ${eval_run_name}"
  {
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "$PYTHON_BIN" -u scripts/phase_e_eval_benchmark.py \
      --value-run-dir "$run_dir" \
      --benchmark-id "$benchmark_id" \
      --run-name "$eval_run_name" \
      --output-root "$EVAL_OUTPUT_ROOT" \
      --batch-size "$BENCHMARK_BATCH_SIZE" \
      --feature-cache-mode off \
      --require-cuda
  } >> "$case_log" 2>&1
  latest_dir_by_prefix_with_marker "$EVAL_OUTPUT_ROOT" "$eval_run_name" "metrics.json"
}

run_case() {
  local config_id="$1"
  local case_title="$2"
  local head_arch="$3"
  local lora_top_k="$4"
  local lora_rank="$5"
  local lora_alpha="$6"
  local terminal_bce_lambda="$7"
  local reward_centering_weight="$8"
  local contrastive_loss_weight="$9"
  local contrastive_margin="${10}"
  local head_hidden_size="${11}"
  local head_inference_alpha="${12}"

  local lora_scope="all_layers"
  if [[ "$lora_top_k" != "0" ]]; then
    lora_scope="top_${lora_top_k}"
  fi

  local run_name="${RUN_PREFIX}_${config_id}"
  local case_log="${LOG_ROOT}/${config_id}.log"
  local run_dir=""
  local samefamily_dir=""
  local math_dir=""
  local gsm_dir=""

  if (
    log_case_line "$case_log" "CASE ${config_id}: ${case_title}"
    log_case_line "$case_log" "  case_log=${case_log}"
    log_case_line "$case_log" "  head=${head_arch} lora_scope=${lora_scope} rank=${lora_rank} contrastive=${contrastive_loss_weight} center=${reward_centering_weight}"
    wait_for_gpu_if_needed
    {
      CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "$PYTHON_BIN" -u scripts/phase_e_train_value_lora.py         --train-pairs-jsonl "${PAIR_DIR}/train_pairs.jsonl"         --eval-pairs-jsonl "${PAIR_DIR}/validation_pairs.jsonl"         --model-path "$MODEL_PATH"         --run-name "$run_name"         --output-root "$VALUE_OUTPUT_ROOT"         --objective-mode joint         --lambda-bce 0.5         --lambda-ranking 1.0         --terminal-bce-lambda "$terminal_bce_lambda"         --learning-rate "$LEARNING_RATE"         --num-train-epochs "$TRAIN_EPOCHS"         --per-device-train-batch-size "$TRAIN_BATCH_SIZE"         --per-device-eval-batch-size "$TRAIN_EVAL_BATCH_SIZE"         --gradient-accumulation-steps "$GRAD_ACCUM"         --max-length "$MAX_LENGTH"         --lora-rank "$lora_rank"         --lora-alpha "$lora_alpha"         --lora-target-modules q_proj,v_proj         --lora-top-k-layers "$lora_top_k"         --ranking-target-space score         --pair-weight-mode none         --checkpoint-selection-metric pair_acc         --head-architecture "$head_arch"         --head-mlp-hidden-size "$head_hidden_size"         --head-dropout-prob 0.05         --head-inference-alpha "$head_inference_alpha"         --anti-saturation-weight 5e-4         --anti-saturation-logit-threshold 3.5         --source-balance uniform         --reward-centering-weight "$reward_centering_weight"         --contrastive-loss-weight "$contrastive_loss_weight"         --contrastive-margin "$contrastive_margin"         --seed "$SEED"         --recipe-risk-policy "$RECIPE_RISK_POLICY"         --gradient-checkpointing         --require-cuda
    } >> "$case_log" 2>&1
    run_dir="$(latest_dir_by_prefix_with_marker "$VALUE_OUTPUT_ROOT" "$run_name" "manifest.json")"
    samefamily_dir="$(run_samefamily_eval "$config_id" "$run_dir" "$case_log")"
    math_dir="$(run_benchmark_eval "$config_id" "processbench_math" "$run_dir" "$case_log")"
    gsm_dir="$(run_benchmark_eval "$config_id" "processbench_gsm8k" "$run_dir" "$case_log")"
  ); then
    log_case_line "$case_log" "CASE ${config_id} OK: run_dir=${run_dir}"
    append_result_row "$config_id" "ok" "$head_arch" "$lora_scope" "$run_dir" "$samefamily_dir" "$math_dir" "$gsm_dir" >> "$case_log" 2>&1
  else
    log_case_line "$case_log" "CASE ${config_id} FAILED"
    append_result_row "$config_id" "failed" "$head_arch" "$lora_scope" "$run_dir" "$samefamily_dir" "$math_dir" "$gsm_dir" >> "$case_log" 2>&1
  fi
}

render_summary() {
  python - "$RESULTS_JSONL" "$SUMMARY_FILE" "$RUN_PREFIX" "$ACTIVE_PHASE_E_LORA_OVERNIGHT_GROUP" "$SUITE_LOG_FILE" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

results_path, summary_path, run_prefix, group_id, suite_log = sys.argv[1:]
rows = [
    json.loads(line)
    for line in Path(results_path).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
failed = [row for row in rows if row.get("status") != "ok"]
status = "ok" if rows and not failed else ("partial_failure" if rows else "no_results")
lines = [
    "# Phase E LoRA Overnight Frontier Suite",
    "",
    f"- run_prefix: `{run_prefix}`",
    f"- group_id: `{group_id}`",
    f"- status: `{status}`",
    f"- suite_log: `{suite_log}`",
    "",
    "| config_id | status | head | lora_scope | samefamily_top1 | samefamily_local | math_auc | math_f1 | gsm_auc | gsm_f1 |",
    "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    def fmt(value):
        return "N/A" if value is None else f"{float(value):.4f}"

    lines.append(
        "| {config_id} | {status} | {head} | {lora_scope} | {sf_top1} | {sf_local} | {math_auc} | {math_f1} | {gsm_auc} | {gsm_f1} |".format(
            config_id=row["config_id"],
            status=row["status"],
            head=row["head_architecture"],
            lora_scope=row["lora_scope"],
            sf_top1=fmt(row.get("samefamily_prompt_top1")),
            sf_local=fmt(row.get("samefamily_local_first_bad")),
            math_auc=fmt(row.get("math_auc")),
            math_f1=fmt(row.get("math_f1")),
            gsm_auc=fmt(row.get("gsm_auc")),
            gsm_f1=fmt(row.get("gsm_f1")),
        )
    )
lines.extend(
    [
        "",
        "## Reading Guide",
        "",
        "- `L1` is the conservative all-layer LoRA + mild contrastive + centering control.",
        "- `L2` keeps all-layer LoRA but swaps to `gated_mlp` and stronger contrastive pressure.",
        "- `L3` is a controlled dual-head retry only after moving to the stronger Math-PRM + PBR26 setting.",
        "",
    ]
)
Path(summary_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"summary_md: {summary_path}")
PY
}

require_path "$MODEL_PATH" "Math-PRM backbone"
require_path "${PAIR_DIR}/train_pairs.jsonl" "Phase E train pairs"
require_path "${PAIR_DIR}/validation_pairs.jsonl" "Phase E validation pairs"

case "$ACTIVE_PHASE_E_LORA_OVERNIGHT_GROUP" in
  L1_ALL28_CTR005_CENTER_MLP)
    # 目的：保守验证“all-layer LoRA + 轻 contrastive + 轻 centering”能否把 PBR26 数据优势和 PBR32 的 MATH 方向合并。
    # 观察：Math AUC / F1 是否同时不低于 PBR26 / PBR32；same-family 是否保持高 top1。
    # 预期：如果数据几何已经够好，轻 contrastive 和 centering 应该提升而不是打坏 calibration。
    run_case "L1_all28_ctr005_center_mlp" "all-layer LoRA + mild contrastive + centering" "mlp" "0" "8" "16" "0.25" "0.01" "0.05" "0.15" "512" "0.50"
    ;;
  L2_ALL28_CTR010_CENTER_GATED)
    # 目的：检验 gated head + 稍强 contrastive 是否能修复 all-correct / terminal 侧的结构性残差。
    # 观察：GSM F1、all-correct last score、same-family rejection curve 是否优于 L1。
    # 预期：如果 head capacity 仍是瓶颈，`gated_mlp` 应比 plain `mlp` 更稳。
    run_case "L2_all28_ctr010_center_gated" "all-layer LoRA + stronger contrastive + gated head" "gated_mlp" "0" "8" "16" "0.25" "0.01" "0.10" "0.15" "512" "0.50"
    ;;
  L3_TOP4_DUALHEAD_TERMINAL)
    # 目的：只在更强 backbone / 数据条件下重试 dual-head，判断之前的负结果究竟是 head 本身错，还是数据/表征太弱。
    # 观察：same-family local_first_bad 与 benchmark AUC 是否仍像 PBR10 F2 那样一起崩。
    # 预期：大概率仍有风险，但这是最有信息增益的 controlled retry。
    run_case "L3_top4_dualhead_terminal" "top-4 LoRA + dual-head terminal retry" "dual_head" "4" "8" "16" "0.35" "0.00" "0.00" "0.15" "384" "0.55"
    ;;
  LALL_CURATED_LORA_FRONTIER)
    # 目的：一夜串行跑完当前最值得做的三组 LoRA frontier 尝试。
    # 观察：
    # - L1 是否成为新的稳健 all-round baseline；
    # - L2 是否说明 head / contrastive 还值得继续加码；
    # - L3 是否再次证明 dual-head 不适合主线。
    # 预期：
    # - L1 / L2 至少有一组应在 Math 或 GSM 上超过当前 LoRA 公开最好线；
    # - L3 更像是“高信息负结果探针”，不是默认 promotion candidate。
    run_case "L1_all28_ctr005_center_mlp" "all-layer LoRA + mild contrastive + centering" "mlp" "0" "8" "16" "0.25" "0.01" "0.05" "0.15" "512" "0.50"
    run_case "L2_all28_ctr010_center_gated" "all-layer LoRA + stronger contrastive + gated head" "gated_mlp" "0" "8" "16" "0.25" "0.01" "0.10" "0.15" "512" "0.50"
    run_case "L3_top4_dualhead_terminal" "top-4 LoRA + dual-head terminal retry" "dual_head" "4" "8" "16" "0.35" "0.00" "0.00" "0.15" "384" "0.55"
    ;;
  *)
    echo "ERROR: unknown ACTIVE_PHASE_E_LORA_OVERNIGHT_GROUP=${ACTIVE_PHASE_E_LORA_OVERNIGHT_GROUP}" >&2
    exit 1
    ;;
esac

render_summary | tee -a "$SUITE_LOG_FILE"
log_line "Suite complete. Summary -> ${SUMMARY_FILE}"
