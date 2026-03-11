#!/usr/bin/env bash
# Phase E curated transfer-repair smoke suite.
#
# 这个脚本的目标不是再做一轮大而散的 sweep，而是回答一个更窄的问题：
# 1. 如果我们先把监督池按语义桶显式 curate，
# 2. 再加上 reward centering 这种来自 reward-model 社区的低风险校准项，
# 3. 并比较单路 MLP 与 feature-gated MLP，
# 是否能让 ProcessBench 迁移行为更接近 RL-ready 所需的水平。
#
# This script is not another broad sweep.  It asks a narrower question:
# 1. if we explicitly curate the supervision pool by semantic buckets,
# 2. then add low-risk reward-centering regularization,
# 3. and compare a plain MLP against a feature-gated MLP,
# do ProcessBench transfer metrics move closer to RL-ready territory?
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACTIVE_PHASE_E_CURATED_GROUP="${ACTIVE_PHASE_E_CURATED_GROUP:-CR1_CURATED_CENTER_GATE_SMOKE}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_curated_rlready}"

MODEL_PATH="${MODEL_PATH:-assets/models/Qwen2.5-7B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
MAX_GPU_MEMORY_GIB="${MAX_GPU_MEMORY_GIB:-48}"
MAX_CPU_MEMORY_GIB="${MAX_CPU_MEMORY_GIB:-96}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-assets/artifacts/phase_e_feature_cache}"
FEATURE_CACHE_MODE="${FEATURE_CACHE_MODE:-read_write}"
PAIR_OUTPUT_ROOT="${PAIR_OUTPUT_ROOT:-assets/artifacts/phase_e_pairs}"
VALUE_OUTPUT_ROOT="${VALUE_OUTPUT_ROOT:-assets/artifacts/phase_e_runs}"
SAMEFAMILY_OUTPUT_ROOT="${SAMEFAMILY_OUTPUT_ROOT:-assets/artifacts/phase_e_samefamily_eval}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-assets/artifacts/phase_e_eval}"
DIAG_OUTPUT_ROOT="${DIAG_OUTPUT_ROOT:-assets/artifacts/phase_e_transfer_diag}"

LOCAL_MIX_ARTIFACT="${LOCAL_MIX_ARTIFACT:-assets/artifacts/phase_e_pairs/phase_e_pb_transfer_mix_0311_1722_e82_ms_prmbench_transfer_mix_seed42_e82_ms_prmbench_transfer_mix_seed42_sharedsplit_s42_pairs__ae568fa2f36e}"
PRMBENCH_TERMINAL_ARTIFACT="${PRMBENCH_TERMINAL_ARTIFACT:-assets/artifacts/phase_e_pairs/phase_e_prmbench_terminal_anchor_full_0311__192ca71fd301}"

LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"
RESULTS_JSONL="${LOG_ROOT}/results.jsonl"
CURRENT_STAGE="bootstrap"

GROUP_TITLE=""
GROUP_INTENTION=""
GROUP_OBSERVE=""
GROUP_EXPECT=""
CURATE_SLICES=()
CONFIG_SPECS=()
BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
LEARNING_RATE="${LEARNING_RATE:-}"

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
    echo "# Phase E Curated RL-Ready Suite Summary"
    echo
    echo "- group_id: ${ACTIVE_PHASE_E_CURATED_GROUP}"
    echo "- group_title: ${GROUP_TITLE}"
    echo "- run_prefix: ${RUN_PREFIX}"
    echo "- status: failed"
    echo "- failed_stage: ${CURRENT_STAGE}"
    echo "- suite_log_file: ${SUITE_LOG_FILE}"
  } > "$SUMMARY_FILE"
}
trap on_exit EXIT

resolve_group() {
  case "$ACTIVE_PHASE_E_CURATED_GROUP" in
    CR1_CURATED_CENTER_GATE_SMOKE)
      GROUP_TITLE="CR1 Curated Semantic Mix + Reward Centering Smoke"
      GROUP_INTENTION="Build one small but explicitly bucket-balanced supervision pool, then compare whether reward centering and feature-gated heads improve transfer over a plain curated MLP baseline."
      GROUP_OBSERVE="The main readout is whether ProcessBench Math AUC / first-edge improve without destroying held-out prompt-pool utility."
      GROUP_EXPECT="Reward centering should stabilize score-space drift; if architecture capacity matters after curation, gated_mlp should beat plain mlp on at least one benchmark slice."
      # 这里用三条固定切片，把监督池分成：
      # 1. Math-Shepherd 的 fanout local，
      # 2. PRMBench 的 same-step local，
      # 3. PRMBench 的轻量 terminal anchors。
      # This supervision pool is explicitly split into:
      # 1. Math-Shepherd fanout local pairs,
      # 2. PRMBench same-step local pairs,
      # 3. lightweight PRMBench terminal anchors.
      CURATE_SLICES=(
        "ms_local=${LOCAL_MIX_ARTIFACT}|first_bad_fanout_prefix_ranking|math_shepherd|1600|160"
        "prm_local=${LOCAL_MIX_ARTIFACT}|local_modified_process_error_step|prmbench_preview|1600|160"
        "prm_terminal=${PRMBENCH_TERMINAL_ARTIFACT}|terminal_completion_anchor|prmbench_preview|320|32"
      )
      # 这三个配置是本轮的结构化对比：
      # 1. C1 保留 curated 数据，但不加 centering，作为最小对照；
      # 2. C2 只加 reward centering，检验“校准是否已经足够”；
      # 3. C3 在 C2 基础上换成 gated_mlp，检验“单路 head 是否仍然不足”。
      # These three configs form the controlled comparison:
      # 1. C1 keeps the curated data but no centering, as the minimal control;
      # 2. C2 adds reward centering only, testing whether calibration is enough;
      # 3. C3 swaps in `gated_mlp` on top of C2, testing whether a single-path head is still the bottleneck.
      CONFIG_SPECS=(
        "C1_CURATED_MLP_BASE|mlp|0.0|confidence_group_balance|0.0005"
        "C2_CURATED_MLP_CENTER|mlp|0.01|confidence_group_balance|0.0005"
        "C3_CURATED_GATED_CENTER|gated_mlp|0.01|confidence_group_balance|0.0005"
      )
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-128}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-4}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-96}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-96}"
      LEARNING_RATE="${LEARNING_RATE:-3e-5}"
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_E_CURATED_GROUP=$ACTIVE_PHASE_E_CURATED_GROUP" >&2
      exit 1
      ;;
  esac
}

latest_dir_for_prefix() {
  local root="$1"
  local prefix="$2"
  python - "$root" "$prefix" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
prefix = sys.argv[2]
matches = sorted(root.glob(f"{prefix}_*"))
if not matches:
    raise SystemExit(f"No artifact directory matches {prefix!r} under {root}")
matches = sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)
print(matches[0])
PY
}

append_result_row() {
  local config_id="$1"
  local curated_dir="$2"
  local value_run_dir="$3"
  local samefamily_dir="$4"
  local pb_gsm_dir="$5"
  local pb_math_dir="$6"
  python - "$config_id" "$curated_dir" "$value_run_dir" "$samefamily_dir" "$pb_gsm_dir" "$pb_math_dir" "$RESULTS_JSONL" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

config_id = sys.argv[1]
curated_dir = Path(sys.argv[2])
value_run_dir = Path(sys.argv[3])
samefamily_dir = Path(sys.argv[4])
pb_gsm_dir = Path(sys.argv[5])
pb_math_dir = Path(sys.argv[6])
results_path = Path(sys.argv[7])

curated_summary = json.loads((curated_dir / "summary.json").read_text(encoding="utf-8"))
value_summary = json.loads((value_run_dir / "summary.json").read_text(encoding="utf-8"))
samefamily_metrics = json.loads((samefamily_dir / "metrics.json").read_text(encoding="utf-8"))
pb_gsm_summary = json.loads((pb_gsm_dir / "summary.json").read_text(encoding="utf-8"))
pb_math_summary = json.loads((pb_math_dir / "summary.json").read_text(encoding="utf-8"))


def compute_all_correct_terminal_top1(eval_dir: Path) -> float:
    """按 example 聚合 scored rows，计算 all-correct 题上的 final-prefix top1。

    中文
    ----
    对于一条 all-correct 样本，如果最终 prefix 的分数严格高于该题之前所有 prefix，
    就记为命中。

    English
    -------
    For an all-correct example, count a hit when the final prefix scores above
    every earlier prefix from the same example.
    """
    scored_rows_path = eval_dir / "scored_rows.jsonl"
    rows = [
        json.loads(raw)
        for raw in scored_rows_path.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row["example_id"]), []).append(row)
    hits = 0
    total = 0
    for bucket in grouped.values():
        ordered = sorted(bucket, key=lambda item: int(item["prefix_step_index"]))
        if not ordered:
            continue
        if not all(bool(item.get("is_good_prefix")) for item in ordered):
            continue
        if len(ordered) < 2:
            continue
        final_score = float(ordered[-1]["score"])
        prev_max = max(float(item["score"]) for item in ordered[:-1])
        total += 1
        if final_score > prev_max:
            hits += 1
    return float(hits / total) if total > 0 else 0.0

row = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "config_id": config_id,
    "curated_dir": str(curated_dir),
    "value_run_dir": str(value_run_dir),
    "samefamily_dir": str(samefamily_dir),
    "processbench_gsm8k_dir": str(pb_gsm_dir),
    "processbench_math_dir": str(pb_math_dir),
    "train_pair_semantics": dict(curated_summary.get("train_pair_semantics") or {}),
    "heldout_pair_acc": float((value_summary.get("eval_pairs") or {}).get("pair_accuracy", 0.0)),
    "heldout_auc": float((value_summary.get("eval_pairs") or {}).get("auc", 0.0)),
    "samefamily_top1": float(samefamily_metrics.get("prompt_pool_top1_accuracy", 0.0)),
    "samefamily_local_first_bad": samefamily_metrics.get("local_first_bad_edge_accuracy"),
    "pb_gsm_auc": float((pb_gsm_summary.get("metrics") or {}).get("pair_auc_good_vs_bad", 0.0)),
    "pb_gsm_first_edge": float((pb_gsm_summary.get("metrics") or {}).get("first_error_edge_accuracy", 0.0)),
    "pb_math_auc": float((pb_math_summary.get("metrics") or {}).get("pair_auc_good_vs_bad", 0.0)),
    "pb_math_first_edge": float((pb_math_summary.get("metrics") or {}).get("first_error_edge_accuracy", 0.0)),
    "pb_math_terminal_top1": compute_all_correct_terminal_top1(pb_math_dir),
}
with results_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

render_final_summary() {
  python - "$RESULTS_JSONL" "$SUMMARY_FILE" "$ACTIVE_PHASE_E_CURATED_GROUP" "$GROUP_TITLE" "$RUN_PREFIX" "$SUITE_LOG_FILE" "$GROUP_INTENTION" "$GROUP_OBSERVE" "$GROUP_EXPECT" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

rows_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
group_id = sys.argv[3]
group_title = sys.argv[4]
run_prefix = sys.argv[5]
suite_log_file = sys.argv[6]
group_intention = sys.argv[7]
group_observe = sys.argv[8]
group_expect = sys.argv[9]

rows = []
if rows_path.exists():
    rows = [
        json.loads(raw)
        for raw in rows_path.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]

lines = [
    "# Phase E Curated RL-Ready Suite Summary",
    "",
    f"- generated_at: {datetime.now(timezone.utc).isoformat()}",
    f"- group_id: {group_id}",
    f"- group_title: {group_title}",
    f"- run_prefix: {run_prefix}",
    f"- status: {'ok' if rows else 'empty'}",
    f"- suite_log_file: {suite_log_file}",
    f"- group_intention: {group_intention}",
    f"- observe: {group_observe}",
    f"- expect: {group_expect}",
    "",
    "## Comparison",
    "",
    "| config | held_pair | held_auc | samefamily_top1 | pb_gsm_auc | pb_gsm_first | pb_math_auc | pb_math_first | pb_math_terminal_top1 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['config_id']} | {row['heldout_pair_acc']:.4f} | {row['heldout_auc']:.4f} | "
        f"{row['samefamily_top1']:.4f} | {row['pb_gsm_auc']:.4f} | {row['pb_gsm_first_edge']:.4f} | "
        f"{row['pb_math_auc']:.4f} | {row['pb_math_first_edge']:.4f} | {row['pb_math_terminal_top1']:.4f} |"
    )
if rows:
    best_math_auc = max(rows, key=lambda row: row["pb_math_auc"])
    best_math_first = max(rows, key=lambda row: row["pb_math_first_edge"])
    best_top1 = max(rows, key=lambda row: row["samefamily_top1"])
    lines.extend(
        [
            "",
            "## Highlights",
            "",
            f"- best_processbench_math_auc: `{best_math_auc['config_id']}` -> `{best_math_auc['pb_math_auc']:.4f}`",
            f"- best_processbench_math_first_edge: `{best_math_first['config_id']}` -> `{best_math_first['pb_math_first_edge']:.4f}`",
            f"- best_samefamily_top1: `{best_top1['config_id']}` -> `{best_top1['samefamily_top1']:.4f}`",
        ]
    )
summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

mkdir -p "$LOG_ROOT"
: > "$SUITE_LOG_FILE"
: > "$RESULTS_JSONL"

resolve_group

CURATED_RUN_NAME="${RUN_PREFIX}_curated_pairs"
CURRENT_STAGE="curate_pairs"
curate_cmd=(
  "$PYTHON_BIN" -u scripts/phase_e_curate_semantic_pairs.py
  --run-name "$CURATED_RUN_NAME"
  --output-root "$PAIR_OUTPUT_ROOT"
  --seed 42
)
for slice_text in "${CURATE_SLICES[@]}"; do
  curate_cmd+=(--slice "$slice_text")
done
log_line "RUN: ${curate_cmd[*]}" | tee -a "$SUITE_LOG_FILE"
"${curate_cmd[@]}" | tee -a "$SUITE_LOG_FILE"
CURATED_DIR="$(latest_dir_for_prefix "$PAIR_OUTPUT_ROOT" "$CURATED_RUN_NAME")"
TRAIN_PAIRS_JSONL="${CURATED_DIR}/train_pairs.jsonl"
EVAL_PAIRS_JSONL="${CURATED_DIR}/validation_pairs.jsonl"

for spec in "${CONFIG_SPECS[@]}"; do
  IFS='|' read -r CONFIG_ID HEAD_ARCH REWARD_CENTER_WEIGHT PAIR_WEIGHT_MODE ANTI_SAT_WEIGHT <<< "$spec"
  VALUE_RUN_NAME="${RUN_PREFIX}_${CONFIG_ID,,}_value"
  SAMEFAMILY_RUN_NAME="${RUN_PREFIX}_${CONFIG_ID,,}_samefamily"
  PB_GSM_RUN_NAME="${RUN_PREFIX}_${CONFIG_ID,,}_pb_gsm8k"
  PB_MATH_RUN_NAME="${RUN_PREFIX}_${CONFIG_ID,,}_pb_math"

  train_cmd=(
    "$PYTHON_BIN" -u scripts/phase_e_train_value.py
    --train-pairs-jsonl "$TRAIN_PAIRS_JSONL"
    --eval-pairs-jsonl "$EVAL_PAIRS_JSONL"
    --model-path "$MODEL_PATH"
    --run-name "$VALUE_RUN_NAME"
    --output-root "$VALUE_OUTPUT_ROOT"
    --objective-mode joint
    --ranking-target-space logit
    --lambda-ranking 1.0
    --lambda-bce 1.0
    --ranking-margin 0.02
    --learning-rate "$LEARNING_RATE"
    --num-train-epochs "$TRAIN_EPOCHS"
    --per-device-train-batch-size "$TRAIN_BATCH_SIZE"
    --per-device-eval-batch-size "$EVAL_BATCH_SIZE"
    --pair-weight-mode "$PAIR_WEIGHT_MODE"
    --source-balance uniform
    --permutation-mode stable_hash
    --head-architecture "$HEAD_ARCH"
    --head-mlp-hidden-size 1024
    --head-dropout-prob 0.05
    --head-init-std 0.02
    --head-activation gelu
    --anti-saturation-weight "$ANTI_SAT_WEIGHT"
    --anti-saturation-logit-threshold 3.5
    --reward-centering-weight "$REWARD_CENTER_WEIGHT"
    --checkpoint-selection-metric ranking_score
    --feature-cache-root "$FEATURE_CACHE_ROOT"
    --feature-cache-mode "$FEATURE_CACHE_MODE"
    --max-gpu-memory-gib "$MAX_GPU_MEMORY_GIB"
    --max-cpu-memory-gib "$MAX_CPU_MEMORY_GIB"
  )
  if [[ -n "$ADAPTER_PATH" ]]; then
    train_cmd+=(--adapter-path "$ADAPTER_PATH")
  fi
  CURRENT_STAGE="train_${CONFIG_ID}"
  log_line "RUN: ${train_cmd[*]}" | tee -a "$SUITE_LOG_FILE"
  "${train_cmd[@]}" | tee -a "$SUITE_LOG_FILE"
  VALUE_RUN_DIR="$(latest_dir_for_prefix "$VALUE_OUTPUT_ROOT" "$VALUE_RUN_NAME")"

  samefamily_cmd=(
    "$PYTHON_BIN" -u scripts/phase_e_eval_samefamily_trust.py
    --value-run-dir "$VALUE_RUN_DIR"
    --eval-pairs-jsonl "$EVAL_PAIRS_JSONL"
    --run-name "$SAMEFAMILY_RUN_NAME"
    --output-root "$SAMEFAMILY_OUTPUT_ROOT"
    --batch-size "$EVAL_BATCH_SIZE"
    --feature-cache-root "$FEATURE_CACHE_ROOT"
    --feature-cache-mode "$FEATURE_CACHE_MODE"
    --max-gpu-memory-gib "$MAX_GPU_MEMORY_GIB"
    --max-cpu-memory-gib "$MAX_CPU_MEMORY_GIB"
  )
  CURRENT_STAGE="samefamily_${CONFIG_ID}"
  log_line "RUN: ${samefamily_cmd[*]}" | tee -a "$SUITE_LOG_FILE"
  "${samefamily_cmd[@]}" | tee -a "$SUITE_LOG_FILE"
  SAMEFAMILY_DIR="$(latest_dir_for_prefix "$SAMEFAMILY_OUTPUT_ROOT" "$SAMEFAMILY_RUN_NAME")"

  for benchmark_id in processbench_gsm8k processbench_math; do
    if [[ "$benchmark_id" == "processbench_gsm8k" ]]; then
      BENCH_RUN_NAME="$PB_GSM_RUN_NAME"
    else
      BENCH_RUN_NAME="$PB_MATH_RUN_NAME"
    fi
    bench_cmd=(
      "$PYTHON_BIN" -u scripts/phase_e_eval_benchmark.py
      --value-run-dir "$VALUE_RUN_DIR"
      --benchmark-id "$benchmark_id"
      --run-name "$BENCH_RUN_NAME"
      --output-root "$EVAL_OUTPUT_ROOT"
      --batch-size "$EVAL_BATCH_SIZE"
      --feature-cache-root "$FEATURE_CACHE_ROOT"
      --feature-cache-mode "$FEATURE_CACHE_MODE"
      --max-gpu-memory-gib "$MAX_GPU_MEMORY_GIB"
      --max-cpu-memory-gib "$MAX_CPU_MEMORY_GIB"
      --max-samples "$BENCH_MAX_SAMPLES"
    )
    CURRENT_STAGE="benchmark_${CONFIG_ID}_${benchmark_id}"
    log_line "RUN: ${bench_cmd[*]}" | tee -a "$SUITE_LOG_FILE"
    "${bench_cmd[@]}" | tee -a "$SUITE_LOG_FILE"
  done

  PB_GSM_DIR="$(latest_dir_for_prefix "$EVAL_OUTPUT_ROOT" "$PB_GSM_RUN_NAME")"
  PB_MATH_DIR="$(latest_dir_for_prefix "$EVAL_OUTPUT_ROOT" "$PB_MATH_RUN_NAME")"
  append_result_row "$CONFIG_ID" "$CURATED_DIR" "$VALUE_RUN_DIR" "$SAMEFAMILY_DIR" "$PB_GSM_DIR" "$PB_MATH_DIR"
done

CURRENT_STAGE="render_summary"
render_final_summary
log_line "Suite completed: ${SUMMARY_FILE}" | tee -a "$SUITE_LOG_FILE"
