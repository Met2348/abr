#!/usr/bin/env bash
# Phase E suite: high-quality pair learnability on benchmark-native datasets.
#
# English
# -------
# This shell script is the orchestration layer for one Phase E experiment group.
# It glues together three conceptually separate stages:
# 1. pair artifact preparation,
# 2. external-pair-only value-head training,
# 3. benchmark-native evaluation.
#
# The key reason to keep them in one suite is reproducibility:
# 1. one group id resolves to one full experiment contract,
# 2. the suite emits one stable log directory and one stable final summary,
# 3. smoke runs and seed-3 promotion runs share the same control flow.
#
# 中文
# ----
# 这个 shell 脚本是单个 Phase E 实验组的“编排层”。
# 它把三个概念上独立的阶段串起来：
# 1. pair artifact 构造，
# 2. 仅依赖 external pair 的 value head 训练，
# 3. benchmark-native 评测。
#
# 之所以要用一个 suite 统一管起来，核心原因是复现性：
# 1. 一个 group id 对应一份完整实验契约；
# 2. suite 会生成稳定的日志目录和最终 summary；
# 3. smoke run 和 3-seed 正式运行共用同一条执行流。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# These env vars are the operator-facing inputs most users will touch.
# 这些环境变量是操作员最常直接修改的入口。
PYTHON_BIN="${PYTHON_BIN:-python}"
ACTIVE_PHASE_E_GROUP="${ACTIVE_PHASE_E_GROUP:-E1_MATH_SHEPHERD_PAIR_LEARN_SMOKE}"
RUN_PREFIX="${RUN_PREFIX:-phase_e_suite}"

MODEL_PATH="${MODEL_PATH:-assets/models/Qwen2.5-7B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
PAIR_OUTPUT_ROOT="${PAIR_OUTPUT_ROOT:-assets/artifacts/phase_e_pairs}"
VALUE_OUTPUT_ROOT="${VALUE_OUTPUT_ROOT:-assets/artifacts/phase_e_runs}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-assets/artifacts/phase_e_eval}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-assets/artifacts/phase_e_feature_cache}"
FEATURE_CACHE_MODE="${FEATURE_CACHE_MODE:-read_write}"
FEATURE_CACHE_LOCK_TIMEOUT_SEC="${FEATURE_CACHE_LOCK_TIMEOUT_SEC:-600}"
PHASE_E_DEFAULT_BATCH_SIZE="${PHASE_E_DEFAULT_BATCH_SIZE:-192}"
STEP_LABEL_PAIR_MODE="${STEP_LABEL_PAIR_MODE:-first_bad_edge_strict}"

# `PAIR_SPLIT_MODE=shared` means all seeds reuse the exact same pair split,
# so cross-seed variance mainly reflects optimization randomness rather than
# data resampling.
# `PAIR_SPLIT_MODE=shared` 表示所有 seed 共用同一份 pair 切分，
# 这样 seed 方差主要反映优化随机性，而不是重新抽样。
PAIR_SPLIT_MODE="${PAIR_SPLIT_MODE:-shared}"
PAIR_SPLIT_SEED="${PAIR_SPLIT_SEED:-42}"

CURRENT_STAGE="bootstrap"
LOG_ROOT="assets/artifacts/phase_e_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="${LOG_ROOT}/suite.log"
SUMMARY_FILE="${LOG_ROOT}/final_summary.md"
SEED_RESULTS_JSONL="${LOG_ROOT}/seed_results.jsonl"

GROUP_TITLE=""
GROUP_INTENTION=""
GROUP_OBSERVE=""
GROUP_EXPECT=""
SOURCE_BUNDLE=""
BENCHMARK_IDS=()
SEEDS=()

log_line() {
  # Timestamp every log line so later debugging can reconstruct which stage ran
  # before a failure.
  # 给每行日志加时间戳，后续排查失败时更容易还原执行顺序。
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$1"
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: Missing ${label}: $path" >&2
    exit 1
  fi
}

latest_dir_for_name() {
  # Many subcommands create timestamped directories.  This helper resolves
  # "the newest directory matching a logical name pattern".
  # 很多子命令都会创建带时间戳的目录，这个 helper 负责找到“最新那一个”。
  local pattern="$1"
  local latest=""
  latest="$(compgen -G "$pattern" | sort | tail -n 1 || true)"
  printf '%s\n' "$latest"
}

append_extra_args() {
  local -n target_ref="$1"
  local raw="${2:-}"
  if [[ -z "$raw" ]]; then
    return 0
  fi
  local extra_arr=()
  # We intentionally rely on shell word-splitting here because callers pass
  # small CLI fragments such as "--foo bar --baz".
  # 这里故意依赖 shell 的拆词，因为调用方传入的是类似
  # "--foo bar --baz" 这种小片段。
  # shellcheck disable=SC2206
  extra_arr=($raw)
  if [[ ${#extra_arr[@]} -gt 0 ]]; then
    target_ref+=("${extra_arr[@]}")
  fi
}

on_exit() {
  local exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    return
  fi
  # Even when the suite crashes halfway, leave behind one minimal summary file.
  # Otherwise dashboards and later debugging code would have nothing stable to open.
  # 即使中途失败，也要留下一个最小 summary 文件；
  # 否则面板和后续排查脚本会连稳定入口都没有。
  mkdir -p "$LOG_ROOT"
  {
    echo "# Phase E Suite Summary"
    echo
    echo "- generated_at: $(python - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)"
    echo "- group_id: ${ACTIVE_PHASE_E_GROUP}"
    echo "- group_title: ${GROUP_TITLE}"
    echo "- run_prefix: ${RUN_PREFIX}"
    echo "- status: failed"
    echo "- failed_stage: ${CURRENT_STAGE}"
    echo "- suite_log_file: ${SUITE_LOG_FILE}"
  } > "$SUMMARY_FILE"
}
trap on_exit EXIT

resolve_group() {
  # This giant case statement is the experiment registry for the single-group
  # Phase E suite.  Each branch defines a full contract:
  # 1. which source bundle to use,
  # 2. which benchmarks to evaluate,
  # 3. what default training recipe to apply,
  # 4. how many seeds to run.
  #
  # 这个大 case 可以理解成单组 Phase E suite 的“实验注册表”。
  # 每个分支都在定义一整份实验契约：
  # 1. 用哪个 source bundle，
  # 2. 跑哪些 benchmark，
  # 3. 默认训练 recipe 是什么，
  # 4. 跑几个 seed。
  case "$ACTIVE_PHASE_E_GROUP" in
    E1_MATH_SHEPHERD_PAIR_LEARN_SMOKE)
      GROUP_TITLE="E1 Math-Shepherd Pair Learn Smoke"
      GROUP_INTENTION="Quick end-to-end validation that Phase E can learn from a high-quality step-converted pair source without any StrategyQA artifact dependency."
      GROUP_OBSERVE="Check held-out pair ranking and one ProcessBench split on a small, cheap run before scaling."
      GROUP_EXPECT="Held-out pair metrics should be clearly above random; ProcessBench should at least show non-trivial separation."
      SOURCE_BUNDLE="math_shepherd"
      BENCHMARK_IDS=("processbench_gsm8k")
      SEEDS=(42)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-2000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-2000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.55}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-5e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-128}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-confidence}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-ranking_score}"
      ;;
    E2_MATH_SHEPHERD_PAIR_LEARN_SEED3)
      GROUP_TITLE="E2 Math-Shepherd Pair Learn Seed3"
      GROUP_INTENTION="Main learnability test on the strongest currently known external ranking source."
      GROUP_OBSERVE="Use a fixed shared split and 3 seeds, then evaluate both held-out pairs and ProcessBench benchmark behavior."
      GROUP_EXPECT="Seed variance should remain small, and learnability should be visible both on held-out pairs and ProcessBench."
      SOURCE_BUNDLE="math_shepherd"
      BENCHMARK_IDS=("processbench_gsm8k" "processbench_math")
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-6000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-12000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.55}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-4}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-5e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-confidence}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-ranking_score}"
      ;;
    E12_MATH_SHEPHERD_TRUST_LOWLR_SEED3)
      GROUP_TITLE="E12 Math-Shepherd Trust LowLR Seed3"
      GROUP_INTENTION="Test whether lower learning rate reduces seed-collapse risk on Math-Shepherd while keeping benchmark-native evaluation enabled."
      GROUP_OBSERVE="Compare held-out strength, ProcessBench transfer, and seed variance against the baseline Math-Shepherd recipe."
      GROUP_EXPECT="Lower LR should trade a little peak speed for lower variance and cleaner candidate selection."
      SOURCE_BUNDLE="math_shepherd"
      BENCHMARK_IDS=("processbench_gsm8k" "processbench_math")
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-6000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-12000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.55}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-6}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-2e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-confidence}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-ranking_score}"
      ;;
    E13_MATH_SHEPHERD_TRUST_UNWEIGHTED_SEED3)
      GROUP_TITLE="E13 Math-Shepherd Trust Unweighted Seed3"
      GROUP_INTENTION="Check whether disabling confidence weighting improves stability when Math-Shepherd pair confidence is only heuristic."
      GROUP_OBSERVE="Use the same held-out and ProcessBench evaluations, but remove confidence weighting from the pair objective."
      GROUP_EXPECT="If confidence heuristics are harming optimization, unweighted training should reduce collapse and improve worst-seed behavior."
      SOURCE_BUNDLE="math_shepherd"
      BENCHMARK_IDS=("processbench_gsm8k" "processbench_math")
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-6000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-12000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.55}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-4}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-5e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-none}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-ranking_score}"
      ;;
    E14_MATH_SHEPHERD_TRUST_ANTISAT_SEED3)
      GROUP_TITLE="E14 Math-Shepherd Trust AntiSat Seed3"
      GROUP_INTENTION="Probe whether anti-saturation regularization keeps logits in a healthier range and preserves ranking geometry on ProcessBench."
      GROUP_OBSERVE="Reuse the baseline recipe, but penalize over-confident logits before they collapse margin structure."
      GROUP_EXPECT="If current failures are partly caused by saturation, this group should improve worst-seed held-out and benchmark metrics."
      SOURCE_BUNDLE="math_shepherd"
      BENCHMARK_IDS=("processbench_gsm8k" "processbench_math")
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-6000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-12000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.55}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-4}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-5e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-confidence}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      ANTI_SATURATION_WEIGHT="${ANTI_SATURATION_WEIGHT:-1e-3}"
      ANTI_SATURATION_LOGIT_THRESHOLD="${ANTI_SATURATION_LOGIT_THRESHOLD:-3.0}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-ranking_score}"
      ;;
    E15_MATH_SHEPHERD_TRUST_ROBUST_SEED3)
      GROUP_TITLE="E15 Math-Shepherd Trust Robust Seed3"
      GROUP_INTENTION="Assemble the most conservative Math-Shepherd recipe for selecting a checkpoint that we may later trust in RL-style downstream stages."
      GROUP_OBSERVE="Combine lower LR, unweighted pairs, anti-saturation, and AUC-based checkpoint choice, then judge held-out strength, ProcessBench behavior, and seed stability together."
      GROUP_EXPECT="This group should sacrifice some peak aggressiveness for cleaner cross-seed reproducibility and a more trustworthy best checkpoint."
      SOURCE_BUNDLE="math_shepherd"
      BENCHMARK_IDS=("processbench_gsm8k" "processbench_math")
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-6000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-12000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.55}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-6}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-2e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-none}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      ANTI_SATURATION_WEIGHT="${ANTI_SATURATION_WEIGHT:-1e-3}"
      ANTI_SATURATION_LOGIT_THRESHOLD="${ANTI_SATURATION_LOGIT_THRESHOLD:-3.0}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-auc}"
      ;;
    E40_MS_ACC90_LINEAR_ROBUST_SEED3)
      GROUP_TITLE="E40 Math-Shepherd ACC90 Linear Robust Seed3"
      GROUP_INTENTION="Single-dataset ACC90 push on Math-Shepherd using the strongest conservative linear-head recipe."
      GROUP_OBSERVE="Ignore transfer entirely and judge only held-out Math-Shepherd pair accuracy/AUC."
      GROUP_EXPECT="If linear separation is sufficient, held-out pair accuracy should approach or exceed 0.90 with acceptable seed stability."
      SOURCE_BUNDLE="math_shepherd"
      BENCHMARK_IDS=()
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-20000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-20000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.55}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-8}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-2e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-none}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      ANTI_SATURATION_WEIGHT="${ANTI_SATURATION_WEIGHT:-1e-3}"
      ANTI_SATURATION_LOGIT_THRESHOLD="${ANTI_SATURATION_LOGIT_THRESHOLD:-3.0}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-pair_acc}"
      HEAD_ARCHITECTURE="${HEAD_ARCHITECTURE:-linear}"
      HEAD_DROPOUT_PROB="${HEAD_DROPOUT_PROB:-0.0}"
      HEAD_INIT_STD="${HEAD_INIT_STD:-0.02}"
      HEAD_ACTIVATION="${HEAD_ACTIVATION:-gelu}"
      ;;
    E41_MS_ACC90_MLP_RANK_SEED3)
      GROUP_TITLE="E41 Math-Shepherd ACC90 MLP Rank Seed3"
      GROUP_INTENTION="Test whether a stronger nonlinear head is enough to push Math-Shepherd same-source accuracy above 0.90."
      GROUP_OBSERVE="Only held-out Math-Shepherd pair metrics matter; no benchmark eval is mixed in."
      GROUP_EXPECT="If the current bottleneck is head expressivity, this group should clearly beat the linear robust anchor."
      SOURCE_BUNDLE="math_shepherd"
      BENCHMARK_IDS=()
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-20000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-20000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.55}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-8}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-5e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-none}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      ANTI_SATURATION_WEIGHT="${ANTI_SATURATION_WEIGHT:-1e-3}"
      ANTI_SATURATION_LOGIT_THRESHOLD="${ANTI_SATURATION_LOGIT_THRESHOLD:-3.0}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-pair_acc}"
      HEAD_ARCHITECTURE="${HEAD_ARCHITECTURE:-mlp}"
      HEAD_MLP_HIDDEN_SIZE="${HEAD_MLP_HIDDEN_SIZE:-1024}"
      HEAD_DROPOUT_PROB="${HEAD_DROPOUT_PROB:-0.05}"
      HEAD_INIT_STD="${HEAD_INIT_STD:-0.02}"
      HEAD_ACTIVATION="${HEAD_ACTIVATION:-gelu}"
      ;;
    E42_MS_ACC90_MLP_JOINT_SEED3)
      GROUP_TITLE="E42 Math-Shepherd ACC90 MLP Joint Seed3"
      GROUP_INTENTION="Use the same-source chosen-vs-rejected objective directly on Math-Shepherd with an MLP head and BCE auxiliary."
      GROUP_OBSERVE="Judge only held-out Math-Shepherd pair accuracy and AUC."
      GROUP_EXPECT="If same-source fit is the main issue, the joint objective should outperform pure ranking on held-out ACC."
      SOURCE_BUNDLE="math_shepherd"
      BENCHMARK_IDS=()
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-20000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-20000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.55}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-8}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-3e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-joint}"
      LAMBDA_RANKING="${LAMBDA_RANKING:-1.0}"
      LAMBDA_BCE="${LAMBDA_BCE:-1.0}"
      RANKING_MARGIN="${RANKING_MARGIN:-0.02}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-none}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      ANTI_SATURATION_WEIGHT="${ANTI_SATURATION_WEIGHT:-5e-4}"
      ANTI_SATURATION_LOGIT_THRESHOLD="${ANTI_SATURATION_LOGIT_THRESHOLD:-3.5}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-pair_acc}"
      HEAD_ARCHITECTURE="${HEAD_ARCHITECTURE:-mlp}"
      HEAD_MLP_HIDDEN_SIZE="${HEAD_MLP_HIDDEN_SIZE:-1024}"
      HEAD_DROPOUT_PROB="${HEAD_DROPOUT_PROB:-0.05}"
      HEAD_INIT_STD="${HEAD_INIT_STD:-0.02}"
      HEAD_ACTIVATION="${HEAD_ACTIVATION:-gelu}"
      ;;
    E43_MS_ACC90_MLP_HIGHCONF_SEED3)
      GROUP_TITLE="E43 Math-Shepherd ACC90 MLP HighConf Seed3"
      GROUP_INTENTION="Aggressively denoise Math-Shepherd and test whether a smaller but cleaner pair pool can cross 0.90 held-out ACC."
      GROUP_OBSERVE="Use only same-source held-out metrics."
      GROUP_EXPECT="If source noise is the blocker, this high-confidence variant should be the sharpest Math-Shepherd recipe."
      SOURCE_BUNDLE="math_shepherd"
      BENCHMARK_IDS=()
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-16000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-16000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.70}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-3e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-joint}"
      LAMBDA_RANKING="${LAMBDA_RANKING:-1.0}"
      LAMBDA_BCE="${LAMBDA_BCE:-1.0}"
      RANKING_MARGIN="${RANKING_MARGIN:-0.02}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-none}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      ANTI_SATURATION_WEIGHT="${ANTI_SATURATION_WEIGHT:-5e-4}"
      ANTI_SATURATION_LOGIT_THRESHOLD="${ANTI_SATURATION_LOGIT_THRESHOLD:-3.5}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-pair_acc}"
      HEAD_ARCHITECTURE="${HEAD_ARCHITECTURE:-mlp}"
      HEAD_MLP_HIDDEN_SIZE="${HEAD_MLP_HIDDEN_SIZE:-1024}"
      HEAD_DROPOUT_PROB="${HEAD_DROPOUT_PROB:-0.05}"
      HEAD_INIT_STD="${HEAD_INIT_STD:-0.02}"
      HEAD_ACTIVATION="${HEAD_ACTIVATION:-gelu}"
      ;;
    E44_PRMBENCH_ACC90_LINEAR_SEED3)
      GROUP_TITLE="E44 PRMBench Preview ACC90 Linear Seed3"
      GROUP_INTENTION="Same-dataset ACC90 baseline on PRMBench Preview using the simplest linear value head."
      GROUP_OBSERVE="Judge only held-out PRMBench Preview pairs."
      GROUP_EXPECT="Direct high-quality process pairs should let even the linear head fit strongly on held-out splits."
      SOURCE_BUNDLE="prmbench_preview"
      BENCHMARK_IDS=()
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-6000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-6000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.80}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-8}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-3e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-none}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-pair_acc}"
      HEAD_ARCHITECTURE="${HEAD_ARCHITECTURE:-linear}"
      HEAD_DROPOUT_PROB="${HEAD_DROPOUT_PROB:-0.0}"
      HEAD_INIT_STD="${HEAD_INIT_STD:-0.02}"
      HEAD_ACTIVATION="${HEAD_ACTIVATION:-gelu}"
      ;;
    E45_PRMBENCH_ACC90_MLP_RANK_SEED3)
      GROUP_TITLE="E45 PRMBench Preview ACC90 MLP Rank Seed3"
      GROUP_INTENTION="Use a stronger MLP head to maximize same-source ranking accuracy on PRMBench Preview."
      GROUP_OBSERVE="Only held-out PRMBench Preview pair metrics matter."
      GROUP_EXPECT="This should be the strongest pure-ranking path to 0.90+ ACC on PRMBench Preview."
      SOURCE_BUNDLE="prmbench_preview"
      BENCHMARK_IDS=()
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-6000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-6000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.80}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-5e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-none}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-pair_acc}"
      HEAD_ARCHITECTURE="${HEAD_ARCHITECTURE:-mlp}"
      HEAD_MLP_HIDDEN_SIZE="${HEAD_MLP_HIDDEN_SIZE:-1024}"
      HEAD_DROPOUT_PROB="${HEAD_DROPOUT_PROB:-0.05}"
      HEAD_INIT_STD="${HEAD_INIT_STD:-0.02}"
      HEAD_ACTIVATION="${HEAD_ACTIVATION:-gelu}"
      ;;
    E46_PRMBENCH_ACC90_MLP_JOINT_SEED3)
      GROUP_TITLE="E46 PRMBench Preview ACC90 MLP Joint Seed3"
      GROUP_INTENTION="Fit PRMBench Preview as a same-source pair discriminator using both ranking and BCE objectives."
      GROUP_OBSERVE="Only held-out PRMBench Preview metrics matter."
      GROUP_EXPECT="If the source is internally consistent enough, this should be the cleanest route to >0.90 held-out ACC."
      SOURCE_BUNDLE="prmbench_preview"
      BENCHMARK_IDS=()
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-6000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-6000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.80}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-3e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-joint}"
      LAMBDA_RANKING="${LAMBDA_RANKING:-1.0}"
      LAMBDA_BCE="${LAMBDA_BCE:-1.0}"
      RANKING_MARGIN="${RANKING_MARGIN:-0.02}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-none}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-pair_acc}"
      HEAD_ARCHITECTURE="${HEAD_ARCHITECTURE:-mlp}"
      HEAD_MLP_HIDDEN_SIZE="${HEAD_MLP_HIDDEN_SIZE:-1024}"
      HEAD_DROPOUT_PROB="${HEAD_DROPOUT_PROB:-0.05}"
      HEAD_INIT_STD="${HEAD_INIT_STD:-0.02}"
      HEAD_ACTIVATION="${HEAD_ACTIVATION:-gelu}"
      ;;
    E47_RPRM_ACC90_LINEAR_SEED3)
      GROUP_TITLE="E47 R-PRM ACC90 Linear Seed3"
      GROUP_INTENTION="Same-dataset ACC90 baseline on R-PRM direct chosen/rejected pairs."
      GROUP_OBSERVE="Judge only held-out R-PRM pair accuracy and AUC."
      GROUP_EXPECT="If R-PRM aligns tightly with the pair objective, even a linear head should fit strongly."
      SOURCE_BUNDLE="r_prm_train"
      BENCHMARK_IDS=()
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-12000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-12000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.75}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-8}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-3e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-none}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-pair_acc}"
      HEAD_ARCHITECTURE="${HEAD_ARCHITECTURE:-linear}"
      HEAD_DROPOUT_PROB="${HEAD_DROPOUT_PROB:-0.0}"
      HEAD_INIT_STD="${HEAD_INIT_STD:-0.02}"
      HEAD_ACTIVATION="${HEAD_ACTIVATION:-gelu}"
      ;;
    E48_RPRM_ACC90_MLP_RANK_SEED3)
      GROUP_TITLE="E48 R-PRM ACC90 MLP Rank Seed3"
      GROUP_INTENTION="Use a stronger MLP head to maximize same-source ranking accuracy on R-PRM."
      GROUP_OBSERVE="Only held-out R-PRM metrics matter."
      GROUP_EXPECT="This should be one of the best candidates for crossing 0.90 same-source ACC."
      SOURCE_BUNDLE="r_prm_train"
      BENCHMARK_IDS=()
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-12000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-12000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.75}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-5e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-none}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-pair_acc}"
      HEAD_ARCHITECTURE="${HEAD_ARCHITECTURE:-mlp}"
      HEAD_MLP_HIDDEN_SIZE="${HEAD_MLP_HIDDEN_SIZE:-1024}"
      HEAD_DROPOUT_PROB="${HEAD_DROPOUT_PROB:-0.05}"
      HEAD_INIT_STD="${HEAD_INIT_STD:-0.02}"
      HEAD_ACTIVATION="${HEAD_ACTIVATION:-gelu}"
      ;;
    E49_RPRM_ACC90_MLP_JOINT_SEED3)
      GROUP_TITLE="E49 R-PRM ACC90 MLP Joint Seed3"
      GROUP_INTENTION="Fit R-PRM as a same-source pair discriminator with both ranking and BCE objectives."
      GROUP_OBSERVE="Only held-out R-PRM pair metrics matter."
      GROUP_EXPECT="If direct chosen/rejected preference supervision is internally consistent, this group should be the cleanest route to >0.90 ACC."
      SOURCE_BUNDLE="r_prm_train"
      BENCHMARK_IDS=()
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-12000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-12000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.75}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-3e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-joint}"
      LAMBDA_RANKING="${LAMBDA_RANKING:-1.0}"
      LAMBDA_BCE="${LAMBDA_BCE:-1.0}"
      RANKING_MARGIN="${RANKING_MARGIN:-0.02}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-none}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-pair_acc}"
      HEAD_ARCHITECTURE="${HEAD_ARCHITECTURE:-mlp}"
      HEAD_MLP_HIDDEN_SIZE="${HEAD_MLP_HIDDEN_SIZE:-1024}"
      HEAD_DROPOUT_PROB="${HEAD_DROPOUT_PROB:-0.05}"
      HEAD_INIT_STD="${HEAD_INIT_STD:-0.02}"
      HEAD_ACTIVATION="${HEAD_ACTIVATION:-gelu}"
      ;;
    E20_STAGEA_MS_ANCHOR_SEED3)
      GROUP_TITLE="E20 Stage A Math-Shepherd Anchor Seed3"
      GROUP_INTENTION="Establish the strongest single-source Math-Shepherd anchor under the robust ranking-first recipe."
      GROUP_OBSERVE="Judge same-family held-out strength, ProcessBench behavior, and PRMBench-preview behavior before any mixture claims."
      GROUP_EXPECT="This should remain the strongest single-source anchor and provide the warm-start checkpoint for later staged curriculum runs."
      SOURCE_BUNDLE="math_shepherd"
      BENCHMARK_IDS=("processbench_gsm8k" "processbench_math" "prmbench_preview")
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-6000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-6000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.55}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-6}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-2e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-none}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      ANTI_SATURATION_WEIGHT="${ANTI_SATURATION_WEIGHT:-1e-3}"
      ANTI_SATURATION_LOGIT_THRESHOLD="${ANTI_SATURATION_LOGIT_THRESHOLD:-3.0}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-auc}"
      ;;
    E21_STAGEA_RPRM_ANCHOR_SEED3)
      GROUP_TITLE="E21 Stage A R-PRM Anchor Seed3"
      GROUP_INTENTION="Measure a direct preference-style math source under the same robust ranking recipe used for the Math-Shepherd anchor."
      GROUP_OBSERVE="This isolates whether direct chosen/rejected supervision is a stronger same-family starting point than converted Math-Shepherd pairs."
      GROUP_EXPECT="If R-PRM is the better anchor, it should show stronger held-out ranking and less seed fragility than the Math-Shepherd anchor."
      SOURCE_BUNDLE="r_prm_train"
      BENCHMARK_IDS=("processbench_gsm8k" "processbench_math" "prmbench_preview")
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-6000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-6000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.75}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-6}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-2e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-none}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      ANTI_SATURATION_WEIGHT="${ANTI_SATURATION_WEIGHT:-1e-3}"
      ANTI_SATURATION_LOGIT_THRESHOLD="${ANTI_SATURATION_LOGIT_THRESHOLD:-3.0}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-auc}"
      ;;
    E22_STAGEA_PRMBENCH_PREVIEW_ANCHOR_SEED3)
      GROUP_TITLE="E22 Stage A PRMBench Preview Anchor Seed3"
      GROUP_INTENTION="Measure the benchmark-aligned preview source as a standalone anchor before mixing it with other math sources."
      GROUP_OBSERVE="This tells us whether the benchmark-aligned source is strong enough to stand alone or should only be treated as an auxiliary mixture component."
      GROUP_EXPECT="A good result here would justify giving PRMBench-preview more weight in later mixtures; a weak result means it should stay auxiliary."
      SOURCE_BUNDLE="prmbench_preview"
      BENCHMARK_IDS=("processbench_gsm8k" "processbench_math" "prmbench_preview")
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-6000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-6000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.80}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-6}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-2e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-none}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      ANTI_SATURATION_WEIGHT="${ANTI_SATURATION_WEIGHT:-1e-3}"
      ANTI_SATURATION_LOGIT_THRESHOLD="${ANTI_SATURATION_LOGIT_THRESHOLD:-3.0}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-auc}"
      ;;
    E23_STAGEA_PRM800K_CTRL_SEED3)
      GROUP_TITLE="E23 Stage A PRM800K Control Seed3"
      GROUP_INTENTION="Keep a weak-source control in the same matrix so later mixture gains are not misread as inevitable."
      GROUP_OBSERVE="PRM800K is expected to stay weaker under the current adapter, but we need a fresh anchor under the exact same robust recipe for fair comparison."
      GROUP_EXPECT="This should remain weaker than the stronger anchors and justify treating PRM800K only as a low-weight ablation source."
      SOURCE_BUNDLE="prm800k"
      BENCHMARK_IDS=("processbench_gsm8k" "processbench_math" "prmbench_preview")
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-6000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-6000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.60}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-6}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-2e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-none}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      ANTI_SATURATION_WEIGHT="${ANTI_SATURATION_WEIGHT:-1e-3}"
      ANTI_SATURATION_LOGIT_THRESHOLD="${ANTI_SATURATION_LOGIT_THRESHOLD:-3.0}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-auc}"
      ;;
    E24_STAGEB_MS_RPRM_MIX_SEED3)
      GROUP_TITLE="E24 Stage B Math-Shepherd + R-PRM Mix Seed3"
      GROUP_INTENTION="Test whether mixing the strongest converted source with the strongest direct-preference source improves benchmark-facing robustness."
      GROUP_OBSERVE="This is the first balanced two-source mixture and answers whether complementary supervision is better than either anchor alone."
      GROUP_EXPECT="If the two sources are complementary, held-out strength should stay positive and benchmark metrics should improve over single-source anchors."
      SOURCE_BUNDLE="math_shepherd_r_prm"
      BENCHMARK_IDS=("processbench_gsm8k" "processbench_math" "prmbench_preview")
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-6000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-12000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.55}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-6}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-2e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-none}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-uniform}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      ANTI_SATURATION_WEIGHT="${ANTI_SATURATION_WEIGHT:-1e-3}"
      ANTI_SATURATION_LOGIT_THRESHOLD="${ANTI_SATURATION_LOGIT_THRESHOLD:-3.0}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-auc}"
      ;;
    E25_STAGEB_MS_PRMBENCH_MIX_SEED3)
      GROUP_TITLE="E25 Stage B Math-Shepherd + PRMBench Preview Mix Seed3"
      GROUP_INTENTION="Test whether adding a benchmark-aligned auxiliary source helps more than adding another generic preference source."
      GROUP_OBSERVE="This mixture directly probes the value of PRMBench-aligned supervision when combined with the strongest same-source anchor."
      GROUP_EXPECT="If benchmark alignment matters, this group should outperform the pure Math-Shepherd anchor on PRMBench-preview and maybe ProcessBench."
      SOURCE_BUNDLE="math_shepherd_prmbench_preview"
      BENCHMARK_IDS=("processbench_gsm8k" "processbench_math" "prmbench_preview")
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-6000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-12000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.55}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-6}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-2e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-none}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-uniform}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      ANTI_SATURATION_WEIGHT="${ANTI_SATURATION_WEIGHT:-1e-3}"
      ANTI_SATURATION_LOGIT_THRESHOLD="${ANTI_SATURATION_LOGIT_THRESHOLD:-3.0}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-auc}"
      ;;
    E26_STAGEB_RPRM_PRMBENCH_MIX_SEED3)
      GROUP_TITLE="E26 Stage B R-PRM + PRMBench Preview Mix Seed3"
      GROUP_INTENTION="Compare a direct-preference plus benchmark-aligned two-source mixture without Math-Shepherd."
      GROUP_OBSERVE="This tells us whether Math-Shepherd is actually required, or whether a more direct preference family is enough on its own."
      GROUP_EXPECT="A strong result here would weaken the claim that Math-Shepherd must always be the anchor."
      SOURCE_BUNDLE="r_prm_prmbench_preview"
      BENCHMARK_IDS=("processbench_gsm8k" "processbench_math" "prmbench_preview")
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-6000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-12000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.55}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-6}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-2e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-none}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-uniform}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      ANTI_SATURATION_WEIGHT="${ANTI_SATURATION_WEIGHT:-1e-3}"
      ANTI_SATURATION_LOGIT_THRESHOLD="${ANTI_SATURATION_LOGIT_THRESHOLD:-3.0}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-auc}"
      ;;
    E27_STAGEC_MS_RPRM_PRMBENCH_MIX_SEED3)
      GROUP_TITLE="E27 Stage C Math-Shepherd + R-PRM + PRMBench Mix Seed3"
      GROUP_INTENTION="Build the main same-family three-source mixture that has the best chance of producing a benchmark-trustworthy math value head."
      GROUP_OBSERVE="This is the central Phase E mixture hypothesis: strong converted anchor + direct preference source + benchmark-aligned auxiliary source."
      GROUP_EXPECT="If same-family multi-source training is the right next step, this group should beat the single-source anchors on benchmark-native metrics without collapsing held-out source quality."
      SOURCE_BUNDLE="math_shepherd_r_prm_prmbench_preview"
      BENCHMARK_IDS=("processbench_gsm8k" "processbench_math" "prmbench_preview")
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-6000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-18000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.55}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-6}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-2e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-none}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-uniform}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      ANTI_SATURATION_WEIGHT="${ANTI_SATURATION_WEIGHT:-1e-3}"
      ANTI_SATURATION_LOGIT_THRESHOLD="${ANTI_SATURATION_LOGIT_THRESHOLD:-3.0}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-auc}"
      ;;
    E28_STAGED_TRIMIX_PLUS_PRM800K_LOWWT_SEED3)
      GROUP_TITLE="E28 Stage D Tri-Mix + Low-Weight PRM800K Seed3"
      GROUP_INTENTION="Test whether PRM800K adds useful diversity when it is explicitly prevented from dominating the stronger three-source mixture."
      GROUP_OBSERVE="This is a weak-source ablation, not a mainline candidate: PRM800K stays in the pool but is down-weighted at optimization time."
      GROUP_EXPECT="If PRM800K helps at all, it should do so only as a low-weight auxiliary source; if it hurts, the three-source mixture should remain the mainline."
      SOURCE_BUNDLE="math_shepherd_r_prm_prmbench_preview_prm800k"
      BENCHMARK_IDS=("processbench_gsm8k" "processbench_math" "prmbench_preview")
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-6000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-24000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.55}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-6}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-2e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-none}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-uniform}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      SOURCE_WEIGHT_OVERRIDES_JSON="${SOURCE_WEIGHT_OVERRIDES_JSON:-{\"prm800k\":0.25}}"
      ANTI_SATURATION_WEIGHT="${ANTI_SATURATION_WEIGHT:-1e-3}"
      ANTI_SATURATION_LOGIT_THRESHOLD="${ANTI_SATURATION_LOGIT_THRESHOLD:-3.0}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-auc}"
      ;;
    E29_STAGED_MS_PLUS_PRM800K_LOWWT_SEED3)
      GROUP_TITLE="E29 Stage D Math-Shepherd + Low-Weight PRM800K Seed3"
      GROUP_INTENTION="Isolate whether PRM800K helps or hurts the simplest strong anchor when added under explicit low-weight control."
      GROUP_OBSERVE="This group is the cleanest weak-source ablation because it compares directly against the Math-Shepherd anchor."
      GROUP_EXPECT="A positive delta here would justify keeping PRM800K as a tiny auxiliary source; a negative delta would justify dropping it from the mainline."
      SOURCE_BUNDLE="math_shepherd_prm800k"
      BENCHMARK_IDS=("processbench_gsm8k" "processbench_math" "prmbench_preview")
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-6000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-12000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.55}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-6}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-2e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-none}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-uniform}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      SOURCE_WEIGHT_OVERRIDES_JSON="${SOURCE_WEIGHT_OVERRIDES_JSON:-{\"prm800k\":0.25}}"
      ANTI_SATURATION_WEIGHT="${ANTI_SATURATION_WEIGHT:-1e-3}"
      ANTI_SATURATION_LOGIT_THRESHOLD="${ANTI_SATURATION_LOGIT_THRESHOLD:-3.0}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-auc}"
      ;;
    E3_RPRM_PRMBENCH_PREVIEW_SMOKE)
      GROUP_TITLE="E3 R-PRM + PRMBench Preview Smoke"
      GROUP_INTENTION="Check a more direct preference-style pair source family under the new Phase E stack."
      GROUP_OBSERVE="Use one seed and PRMBench preview eval to see whether direct pair supervision is easier to learn than converted math-process labels."
      GROUP_EXPECT="Held-out pairs and PRMBench preview should both show non-trivial ranking above random."
      SOURCE_BUNDLE="r_prm_prmbench_preview"
      BENCHMARK_IDS=("prmbench_preview")
      SEEDS=(42)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-2000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-3000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.7}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-5e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-256}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-confidence}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-uniform}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-ranking_score}"
      ;;
    E4_RPRM_PRMBENCH_PREVIEW_SEED3)
      GROUP_TITLE="E4 R-PRM + PRMBench Preview Seed3"
      GROUP_INTENTION="Main direct-pair learnability test on R-PRM plus PRMBench-preview style supervision."
      GROUP_OBSERVE="Use 3 seeds and shared split to judge whether direct pair supervision gives a more stable benchmark-native story."
      GROUP_EXPECT="If the benchmark fits the objective, held-out pair metrics and PRMBench preview metrics should both be stable across seeds."
      SOURCE_BUNDLE="r_prm_prmbench_preview"
      BENCHMARK_IDS=("prmbench_preview")
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-6000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-12000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.7}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-4}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-5e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-confidence}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-uniform}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-ranking_score}"
      ;;
    E5_PRM800K_PAIR_LEARN_SEED3)
      GROUP_TITLE="E5 PRM800K Pair Learn Seed3"
      GROUP_INTENTION="Re-test canonical PRM800K in a clean benchmark-native stack, separate from StrategyQA transfer assumptions."
      GROUP_OBSERVE="Judge whether PRM800K is weak because of old Phase D wiring, or because it is genuinely weak for our ranking objective."
      GROUP_EXPECT="If PRM800K still stays weak here, the issue is methodological/data-semantic rather than StrategyQA entanglement."
      SOURCE_BUNDLE="prm800k"
      BENCHMARK_IDS=("processbench_gsm8k")
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-6000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-6000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.6}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-4}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-5e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-confidence}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-ranking_score}"
      ;;
    E6_MATH_SHEPHERD_SAME_SOURCE_SMOKE)
      GROUP_TITLE="E6 Math-Shepherd Same-Source Smoke"
      GROUP_INTENTION="Validate same-source learnability on one high-quality converted pair dataset without introducing any cross-benchmark confound."
      GROUP_OBSERVE="Judge only the held-out pair split from Math-Shepherd itself."
      GROUP_EXPECT="Held-out pair accuracy and AUC should be clearly above random on a cheap smoke."
      SOURCE_BUNDLE="math_shepherd"
      # Same-source groups intentionally skip extra benchmark evals.
      # same-source 组故意不跑额外 benchmark eval。
      # The first question here is whether the dataset can train the value/ranking head on its own.
      # 目的就是先回答“这个数据集自己能不能把 value/ranking 训起来”。
      # Cross-dataset generalization should not be mixed into this first-principles check.
      # 避免把跨数据集泛化混进第一性结论里。
      BENCHMARK_IDS=()
      SEEDS=(42)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-2000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-2000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.55}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-5e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-confidence}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-ranking_score}"
      ;;
    E7_MATH_SHEPHERD_SAME_SOURCE_SEED3)
      GROUP_TITLE="E7 Math-Shepherd Same-Source Seed3"
      GROUP_INTENTION="Main same-source learnability test on Math-Shepherd."
      GROUP_OBSERVE="Use a fixed shared split and 3 seeds, but only judge held-out Math-Shepherd pairs."
      GROUP_EXPECT="If Math-Shepherd is a genuinely learnable source, held-out pair metrics should be positive and seed variance should stay low."
      SOURCE_BUNDLE="math_shepherd"
      BENCHMARK_IDS=()
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-6000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-6000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.55}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-4}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-5e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-confidence}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-ranking_score}"
      ;;
    E8_PRMBENCH_PREVIEW_SAME_SOURCE_SMOKE)
      GROUP_TITLE="E8 PRMBench Preview Same-Source Smoke"
      GROUP_INTENTION="Validate learnability on a direct high-quality process-pair source with minimal conversion noise."
      GROUP_OBSERVE="Judge only the held-out pair split from PRMBench_Preview itself."
      GROUP_EXPECT="Held-out pair ranking should be clearly above random if direct process-pair supervision fits the current head/objective."
      SOURCE_BUNDLE="prmbench_preview"
      BENCHMARK_IDS=()
      SEEDS=(42)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-2000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-2000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.80}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-5e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-confidence}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-ranking_score}"
      ;;
    E9_PRMBENCH_PREVIEW_SAME_SOURCE_SEED3)
      GROUP_TITLE="E9 PRMBench Preview Same-Source Seed3"
      GROUP_INTENTION="Main same-source learnability test on PRMBench_Preview direct process pairs."
      GROUP_OBSERVE="Use shared split and 3 seeds, but only judge held-out PRMBench_Preview pairs."
      GROUP_EXPECT="If direct process pairs are a better fit than converted labels, this group should show strong held-out ranking and lower variance."
      SOURCE_BUNDLE="prmbench_preview"
      BENCHMARK_IDS=()
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-6000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-6000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.80}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-4}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-5e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-confidence}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-ranking_score}"
      ;;
    E10_RPRM_SAME_SOURCE_SMOKE)
      GROUP_TITLE="E10 R-PRM Same-Source Smoke"
      GROUP_INTENTION="Validate learnability on one direct chosen/rejected preference dataset without benchmark-transfer pressure."
      GROUP_OBSERVE="Judge only the held-out pair split from R-PRM train."
      GROUP_EXPECT="Held-out pair ranking should be non-trivial if direct chosen/rejected supervision is compatible with the current value head."
      SOURCE_BUNDLE="r_prm_train"
      BENCHMARK_IDS=()
      SEEDS=(42)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-2000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-2000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.75}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-5e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-confidence}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-ranking_score}"
      ;;
    E11_RPRM_SAME_SOURCE_SEED3)
      GROUP_TITLE="E11 R-PRM Same-Source Seed3"
      GROUP_INTENTION="Main same-source learnability test on R-PRM direct chosen/rejected pairs."
      GROUP_OBSERVE="Use shared split and 3 seeds, but only judge held-out R-PRM pairs."
      GROUP_EXPECT="If R-PRM is a high-quality supervision source for our objective, it should show stable held-out ranking across seeds."
      SOURCE_BUNDLE="r_prm_train"
      BENCHMARK_IDS=()
      SEEDS=(42 43 44)
      MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-6000}"
      MAX_PAIRS_TOTAL="${MAX_PAIRS_TOTAL:-6000}"
      MIN_PAIR_CONFIDENCE="${MIN_PAIR_CONFIDENCE:-0.75}"
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-4}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$PHASE_E_DEFAULT_BATCH_SIZE}"
      LEARNING_RATE="${LEARNING_RATE:-5e-5}"
      MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
      MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
      BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"
      OBJECTIVE_MODE="${OBJECTIVE_MODE:-ranking_only}"
      PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-confidence}"
      SOURCE_BALANCE="${SOURCE_BALANCE:-none}"
      PERMUTATION_MODE="${PERMUTATION_MODE:-stable_hash}"
      CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-ranking_score}"
      ;;
    *)
      echo "ERROR: Unknown ACTIVE_PHASE_E_GROUP=$ACTIVE_PHASE_E_GROUP" >&2
      exit 1
      ;;
  esac
}

apply_resolved_overrides() {
  # Wrapper suites sometimes want to reuse the exact same group definition but
  # with fewer seeds.  Doing the override here avoids copying whole case blocks.
  # 某些 wrapper 想复用同一个 group 定义，但把 seed 数量临时改少。
  # 在这里统一覆盖，就不必复制整段 case。
  if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then
    # shellcheck disable=SC2206
    SEEDS=(${SEEDS_OVERRIDE})
  fi
}

prepare_pair_artifact() {
  local run_name="$1"
  # This command is the only place where raw source data becomes the canonical
  # train/validation pair artifact for this suite run.
  # 这里是原始 source 数据正式变成 canonical train/validation pair artifact 的地方。
  local prep_cmd=(
    "$PYTHON_BIN" -u scripts/phase_e_prepare_pairs.py
    --source-bundle "$SOURCE_BUNDLE"
    --run-name "$run_name"
    --output-root "$PAIR_OUTPUT_ROOT"
    --seed "$PAIR_SPLIT_SEED"
    --validation-ratio 0.1
    --max-pairs-total "$MAX_PAIRS_TOTAL"
    --max-pairs-per-source "$MAX_PAIRS_PER_SOURCE"
    --min-pair-confidence "$MIN_PAIR_CONFIDENCE"
    --step-label-pair-mode "$STEP_LABEL_PAIR_MODE"
  )
  if [[ -n "${SOURCE_WEIGHT_OVERRIDES_JSON:-}" ]]; then
    prep_cmd+=(--source-weight-overrides-json "$SOURCE_WEIGHT_OVERRIDES_JSON")
  fi
  append_extra_args prep_cmd "${PAIR_PREP_EXTRA_ARGS:-}"
  CURRENT_STAGE="prepare_pairs"
  log_line "RUN: ${prep_cmd[*]}" | tee -a "$SUITE_LOG_FILE"
  "${prep_cmd[@]}" | tee -a "$SUITE_LOG_FILE"
}

run_value_train() {
  local seed="$1"
  local train_pairs_jsonl="$2"
  local eval_pairs_jsonl="$3"
  local run_name="$4"
  # This stage consumes canonical pair JSONL and produces one timestamped value
  # run directory containing checkpoints, metrics, and manifests.
  # 这个阶段消费 canonical pair JSONL，并生成一个带 checkpoint/metrics/manifest 的
  # value run 目录。
  local cmd=(
    "$PYTHON_BIN" -u scripts/phase_e_train_value.py
    --train-pairs-jsonl "$train_pairs_jsonl"
    --eval-pairs-jsonl "$eval_pairs_jsonl"
    --model-path "$MODEL_PATH"
    --run-name "$run_name"
    --output-root "$VALUE_OUTPUT_ROOT"
    --objective-mode "$OBJECTIVE_MODE"
    --learning-rate "$LEARNING_RATE"
    --num-train-epochs "$TRAIN_EPOCHS"
    --per-device-train-batch-size "$TRAIN_BATCH_SIZE"
    --per-device-eval-batch-size "$EVAL_BATCH_SIZE"
    --pair-weight-mode "$PAIR_WEIGHT_MODE"
    --source-balance "$SOURCE_BALANCE"
    --permutation-mode "$PERMUTATION_MODE"
    --checkpoint-selection-metric "$CHECKPOINT_SELECTION_METRIC"
    --seed "$seed"
    --dtype "${DTYPE:-bfloat16}"
    --device-map "${DEVICE_MAP:-auto}"
    --feature-cache-root "$FEATURE_CACHE_ROOT"
    --feature-cache-mode "$FEATURE_CACHE_MODE"
    --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC"
  )
  if [[ -n "${WEIGHT_DECAY:-}" ]]; then
    cmd+=(--weight-decay "$WEIGHT_DECAY")
  fi
  if [[ -n "${MAX_LENGTH:-}" ]]; then
    cmd+=(--max-length "$MAX_LENGTH")
  fi
  if [[ -n "${LAMBDA_RANKING:-}" ]]; then
    cmd+=(--lambda-ranking "$LAMBDA_RANKING")
  fi
  if [[ -n "${LAMBDA_BCE:-}" ]]; then
    cmd+=(--lambda-bce "$LAMBDA_BCE")
  fi
  if [[ -n "${RANKING_MARGIN:-}" ]]; then
    cmd+=(--ranking-margin "$RANKING_MARGIN")
  fi
  if [[ -n "${ANTI_SATURATION_WEIGHT:-}" ]]; then
    cmd+=(--anti-saturation-weight "$ANTI_SATURATION_WEIGHT")
  fi
  if [[ -n "${ANTI_SATURATION_LOGIT_THRESHOLD:-}" ]]; then
    cmd+=(--anti-saturation-logit-threshold "$ANTI_SATURATION_LOGIT_THRESHOLD")
  fi
  if [[ -n "$ADAPTER_PATH" ]]; then
    cmd+=(--adapter-path "$ADAPTER_PATH")
  fi
  if [[ -n "${INIT_VALUE_HEAD_PATH:-}" ]]; then
    cmd+=(--init-value-head-path "$INIT_VALUE_HEAD_PATH")
  fi
  if [[ -n "${HEAD_ARCHITECTURE:-}" ]]; then
    cmd+=(--head-architecture "$HEAD_ARCHITECTURE")
  fi
  if [[ -n "${HEAD_DROPOUT_PROB:-}" ]]; then
    cmd+=(--head-dropout-prob "$HEAD_DROPOUT_PROB")
  fi
  if [[ -n "${HEAD_INIT_STD:-}" ]]; then
    cmd+=(--head-init-std "$HEAD_INIT_STD")
  fi
  if [[ -n "${HEAD_MLP_HIDDEN_SIZE:-}" ]]; then
    cmd+=(--head-mlp-hidden-size "$HEAD_MLP_HIDDEN_SIZE")
  fi
  if [[ -n "${HEAD_ACTIVATION:-}" ]]; then
    cmd+=(--head-activation "$HEAD_ACTIVATION")
  fi
  if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
    cmd+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
  fi
  if [[ -n "${MAX_EVAL_SAMPLES:-}" ]]; then
    cmd+=(--max-eval-samples "$MAX_EVAL_SAMPLES")
  fi
  if [[ "${REQUIRE_CUDA:-1}" == "1" ]]; then
    cmd+=(--require-cuda)
  fi
  if [[ "${STRICT_DETERMINISM:-1}" == "1" ]]; then
    cmd+=(--strict-determinism)
  fi
  append_extra_args cmd "${TRAIN_EXTRA_ARGS:-}"
  CURRENT_STAGE="train_value_s${seed}"
  log_line "RUN: ${cmd[*]}" | tee -a "$SUITE_LOG_FILE"
  "${cmd[@]}" | tee -a "$SUITE_LOG_FILE"
}

run_benchmark_eval() {
  local seed="$1"
  local value_run_dir="$2"
  local benchmark_id="$3"
  local run_name="$4"
  # Benchmark evaluation is kept as a separate stage on purpose:
  # a model can look strong on its held-out source split but still be weak on
  # benchmark-native process discrimination.
  # benchmark 评测故意保持为单独阶段，因为模型可能在 source held-out 上很强，
  # 但在 benchmark-native 的过程辨别上仍然很弱。
  local cmd=(
    "$PYTHON_BIN" -u scripts/phase_e_eval_benchmark.py
    --value-run-dir "$value_run_dir"
    --benchmark-id "$benchmark_id"
    --run-name "$run_name"
    --output-root "$EVAL_OUTPUT_ROOT"
    --checkpoint-name best
    --batch-size "$EVAL_BATCH_SIZE"
    --dtype "${DTYPE:-bfloat16}"
    --device-map "${DEVICE_MAP:-auto}"
    --feature-cache-root "$FEATURE_CACHE_ROOT"
    --feature-cache-mode "$FEATURE_CACHE_MODE"
    --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC"
  )
  if [[ -n "${BENCH_MAX_SAMPLES:-}" ]]; then
    cmd+=(--max-samples "$BENCH_MAX_SAMPLES")
  fi
  if [[ "${REQUIRE_CUDA:-1}" == "1" ]]; then
    cmd+=(--require-cuda)
  fi
  append_extra_args cmd "${EVAL_EXTRA_ARGS:-}"
  CURRENT_STAGE="eval_${benchmark_id}_s${seed}"
  log_line "RUN: ${cmd[*]}" | tee -a "$SUITE_LOG_FILE"
  "${cmd[@]}" | tee -a "$SUITE_LOG_FILE"
}

append_seed_result() {
  local seed="$1"
  local pair_dir="$2"
  local value_run_dir="$3"
  shift 3
  # Merge the artifacts from:
  # 1. pair construction,
  # 2. value-head evaluation,
  # 3. benchmark evaluation,
  # into one compact per-seed JSONL row.
  #
  # 把
  # 1. pair 构造结果，
  # 2. value head 评测结果，
  # 3. benchmark 评测结果，
  # 合并成一条紧凑的 per-seed JSONL 记录。
  "$PYTHON_BIN" - "$seed" "$pair_dir" "$value_run_dir" "$SEED_RESULTS_JSONL" "$@" <<'PY'
import json
import sys
from pathlib import Path

seed = int(sys.argv[1])
pair_dir = Path(sys.argv[2])
value_run_dir = Path(sys.argv[3])
out_path = Path(sys.argv[4])
eval_dirs = [Path(item) for item in sys.argv[5:]]

pair_summary = json.loads((pair_dir / "summary.json").read_text(encoding="utf-8"))
value_eval = json.loads((value_run_dir / "eval_metrics.json").read_text(encoding="utf-8"))
row = {
    "seed": seed,
    "pair_dir": str(pair_dir),
    "value_run_dir": str(value_run_dir),
    "train_pairs": int(pair_summary.get("num_train_rows", 0)),
    "val_pairs": int(pair_summary.get("num_validation_rows", 0)),
    "heldout_pair_acc": float(value_eval.get("eval_pairs", {}).get("pair_accuracy", 0.0)),
    "heldout_auc": float(value_eval.get("eval_pairs", {}).get("auc", 0.0)),
    "heldout_ranking_score": float(value_eval.get("eval_pairs", {}).get("ranking_score", 0.0)),
    "benchmarks": {},
}
for eval_dir in eval_dirs:
    metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    benchmark_id = str(summary.get("benchmark_id"))
    if "pair_accuracy_good_vs_bad" in metrics:
        pair_acc = float(metrics["pair_accuracy_good_vs_bad"])
        auc = float(metrics["pair_auc_good_vs_bad"])
    else:
        pair_acc = float(metrics.get("pair_accuracy", 0.0))
        auc = float(metrics.get("auc", 0.0))
    row["benchmarks"][benchmark_id] = {
        "pair_acc": pair_acc,
        "auc": auc,
        "metrics_path": str(eval_dir / "metrics.json"),
    }
with out_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

render_final_summary() {
  # Convert all per-seed JSONL rows into one operator-friendly Markdown report.
  # 把所有 per-seed JSONL 结果渲染成一个适合人工浏览的 Markdown 总结。
  "$PYTHON_BIN" - "$SEED_RESULTS_JSONL" "$SUMMARY_FILE" "$ACTIVE_PHASE_E_GROUP" "$GROUP_TITLE" "$RUN_PREFIX" "$SUITE_LOG_FILE" "$GROUP_INTENTION" "$GROUP_OBSERVE" "$GROUP_EXPECT" <<'PY'
import json
import statistics
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
    for raw in rows_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            rows.append(json.loads(raw))

benchmark_ids = []
for row in rows:
    for benchmark_id in row.get("benchmarks", {}):
        if benchmark_id not in benchmark_ids:
            benchmark_ids.append(benchmark_id)
benchmark_ids.sort()

lines = [
    "# Phase E Suite Summary",
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
    "## Per-Seed Metrics",
    "",
]
header = ["seed", "train_pairs", "val_pairs", "heldout_pair_acc", "heldout_auc", "heldout_ranking_score"]
for benchmark_id in benchmark_ids:
    header.append(f"{benchmark_id}_pair_acc")
    header.append(f"{benchmark_id}_auc")
lines.append("| " + " | ".join(header) + " |")
lines.append("|" + "|".join(["---"] * len(header)) + "|")
for row in rows:
    values = [
        str(row["seed"]),
        str(row["train_pairs"]),
        str(row["val_pairs"]),
        f"{float(row['heldout_pair_acc']):.4f}",
        f"{float(row['heldout_auc']):.4f}",
        f"{float(row['heldout_ranking_score']):.4f}",
    ]
    benchmarks = row.get("benchmarks", {})
    for benchmark_id in benchmark_ids:
        item = benchmarks.get(benchmark_id, {})
        values.append(f"{float(item.get('pair_acc', 0.0)):.4f}")
        values.append(f"{float(item.get('auc', 0.0)):.4f}")
    lines.append("| " + " | ".join(values) + " |")

if rows:
    lines.extend(["", "## Aggregated", ""])
    heldout_pair_accs = [float(row["heldout_pair_acc"]) for row in rows]
    heldout_aucs = [float(row["heldout_auc"]) for row in rows]
    heldout_ranking_scores = [float(row["heldout_ranking_score"]) for row in rows]
    lines.append(f"- mean_heldout_pair_acc: `{statistics.mean(heldout_pair_accs):.6f}`")
    lines.append(f"- mean_heldout_auc: `{statistics.mean(heldout_aucs):.6f}`")
    lines.append(f"- mean_heldout_ranking_score: `{statistics.mean(heldout_ranking_scores):.6f}`")
    if len(rows) > 1:
        lines.append(f"- std_heldout_pair_acc: `{statistics.pstdev(heldout_pair_accs):.6f}`")
        lines.append(f"- std_heldout_auc: `{statistics.pstdev(heldout_aucs):.6f}`")
        lines.append(f"- std_heldout_ranking_score: `{statistics.pstdev(heldout_ranking_scores):.6f}`")
    for benchmark_id in benchmark_ids:
        pair_accs = [
            float(row.get("benchmarks", {}).get(benchmark_id, {}).get("pair_acc", 0.0))
            for row in rows
        ]
        aucs = [
            float(row.get("benchmarks", {}).get(benchmark_id, {}).get("auc", 0.0))
            for row in rows
        ]
        lines.append(f"- mean_{benchmark_id}_pair_acc: `{statistics.mean(pair_accs):.6f}`")
        lines.append(f"- mean_{benchmark_id}_auc: `{statistics.mean(aucs):.6f}`")
        if len(rows) > 1:
            lines.append(f"- std_{benchmark_id}_pair_acc: `{statistics.pstdev(pair_accs):.6f}`")
            lines.append(f"- std_{benchmark_id}_auc: `{statistics.pstdev(aucs):.6f}`")
lines.append("")
summary_path.write_text("\n".join(lines), encoding="utf-8")
PY
}

resolve_group
apply_resolved_overrides
mkdir -p "$LOG_ROOT"
: > "$SUITE_LOG_FILE"
: > "$SEED_RESULTS_JSONL"

{
  log_line "Phase E Suite"
  log_line "group_id=${ACTIVE_PHASE_E_GROUP}"
  log_line "group_title=${GROUP_TITLE}"
  log_line "group_intention=${GROUP_INTENTION}"
  log_line "group_observe=${GROUP_OBSERVE}"
  log_line "group_expect=${GROUP_EXPECT}"
  log_line "run_prefix=${RUN_PREFIX}"
  log_line "model_path=${MODEL_PATH}"
  log_line "source_bundle=${SOURCE_BUNDLE}"
  log_line "source_weight_overrides=${SOURCE_WEIGHT_OVERRIDES_JSON:-<none>}"
  log_line "step_label_pair_mode=${STEP_LABEL_PAIR_MODE}"
  log_line "init_value_head_path=${INIT_VALUE_HEAD_PATH:-<none>}"
  log_line "benchmarks=${BENCHMARK_IDS[*]}"
  log_line "pair_split_mode=${PAIR_SPLIT_MODE}"
  log_line "pair_split_seed=${PAIR_SPLIT_SEED}"
} | tee -a "$SUITE_LOG_FILE"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${STRICT_DETERMINISM:-1}" == "1" ]]; then
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
fi

SHARED_PAIR_DIR=""
if [[ "$PAIR_SPLIT_MODE" == "shared" ]]; then
  # Build pair artifacts exactly once and let all seeds share them.
  # 这样不同 seed 之间比较时，差异主要来自训练随机性，而不是数据重新切分。
  SHARED_PAIR_RUN_NAME="${RUN_PREFIX}_${ACTIVE_PHASE_E_GROUP,,}_sharedsplit_s${PAIR_SPLIT_SEED}_pairs"
  prepare_pair_artifact "$SHARED_PAIR_RUN_NAME"
  SHARED_PAIR_DIR="$(latest_dir_for_name "${PAIR_OUTPUT_ROOT}/${SHARED_PAIR_RUN_NAME}__*")"
  require_file "${SHARED_PAIR_DIR}/train_pairs.jsonl" "shared train pairs"
  require_file "${SHARED_PAIR_DIR}/validation_pairs.jsonl" "shared validation pairs"
fi

for seed in "${SEEDS[@]}"; do
  CURRENT_STAGE="seed_${seed}_bootstrap"
  # In non-shared mode, each seed gets its own freshly prepared pair split.
  # 在非 shared 模式下，每个 seed 都会单独准备自己的 pair 切分。
  PAIR_DIR="$SHARED_PAIR_DIR"
  if [[ "$PAIR_SPLIT_MODE" != "shared" ]]; then
    PAIR_RUN_NAME="${RUN_PREFIX}_${ACTIVE_PHASE_E_GROUP,,}_s${seed}_pairs"
    prepare_pair_artifact "$PAIR_RUN_NAME"
    PAIR_DIR="$(latest_dir_for_name "${PAIR_OUTPUT_ROOT}/${PAIR_RUN_NAME}__*")"
  fi
  TRAIN_PAIRS_JSONL="${PAIR_DIR}/train_pairs.jsonl"
  EVAL_PAIRS_JSONL="${PAIR_DIR}/validation_pairs.jsonl"
  require_file "$TRAIN_PAIRS_JSONL" "train pairs"
  require_file "$EVAL_PAIRS_JSONL" "eval pairs"

  VALUE_RUN_NAME="${RUN_PREFIX}_${ACTIVE_PHASE_E_GROUP,,}_s${seed}_value"
  run_value_train "$seed" "$TRAIN_PAIRS_JSONL" "$EVAL_PAIRS_JSONL" "$VALUE_RUN_NAME"
  VALUE_RUN_DIR="$(latest_dir_for_name "${VALUE_OUTPUT_ROOT}/${VALUE_RUN_NAME}_*")"
  require_file "${VALUE_RUN_DIR}/eval_metrics.json" "Phase E eval metrics"

  # One value-head run can be evaluated on zero, one, or many benchmarks.
  # 一个 value-head run 后面可以接 0 个、1 个或多个 benchmark 评测目录。
  BENCH_EVAL_DIRS=()
  for benchmark_id in "${BENCHMARK_IDS[@]}"; do
    BENCH_RUN_NAME="${RUN_PREFIX}_${ACTIVE_PHASE_E_GROUP,,}_s${seed}_${benchmark_id}"
    run_benchmark_eval "$seed" "$VALUE_RUN_DIR" "$benchmark_id" "$BENCH_RUN_NAME"
    BENCH_DIR="$(latest_dir_for_name "${EVAL_OUTPUT_ROOT}/${BENCH_RUN_NAME}_*")"
    require_file "${BENCH_DIR}/metrics.json" "benchmark metrics ${benchmark_id}"
    BENCH_EVAL_DIRS+=("$BENCH_DIR")
  done

  append_seed_result "$seed" "$PAIR_DIR" "$VALUE_RUN_DIR" "${BENCH_EVAL_DIRS[@]}"
done

render_final_summary
log_line "Summary file   : ${SUMMARY_FILE}" | tee -a "$SUITE_LOG_FILE"
log_line "Group complete" | tee -a "$SUITE_LOG_FILE"
