#!/usr/bin/env bash
# One-click healthcheck for persistent feature-cache integration.
#
# Why this file exists:
# - Feature cache is cross-cutting infra for Phase C/D speed and stability.
# - We need a repeatable "ops-style" check that validates:
#   1) static script integrity,
#   2) cache module correctness,
#   3) optional runtime cache-hit behavior (C2/PIK/D4 smoke).
#
# Quick start (static + cache-module checks):
#   RUN_PREFIX=cache_hc_fast \
#   CHECK_PROFILE=fast \
#   bash scripts/run_feature_cache_healthcheck.sh
#
# Full check (auto runtime smoke when prerequisites are present):
#   RUN_PREFIX=cache_hc_full \
#   CHECK_PROFILE=full \
#   CUDA_VISIBLE_DEVICES=0 \
#   bash scripts/run_feature_cache_healthcheck.sh
#
# Optional toggles:
# - RUN_RUNTIME_SMOKE={auto,0,1}      # C2 runtime smoke
# - RUN_PIK_SMOKE={auto,0,1}          # PIK runtime smoke
# - RUN_D4_SMOKE={0,1}                # D4A runtime smoke (heavier)
# - FEATURE_CACHE_MODE={off,read,write,read_write}
# - FEATURE_CACHE_ROOT=assets/artifacts/phase_c_feature_cache
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECK_PROFILE="${CHECK_PROFILE:-full}" # fast | full
RUN_PREFIX="${RUN_PREFIX:-feature_cache_healthcheck}"
CURRENT_STAGE="init"

RUN_RUNTIME_SMOKE="${RUN_RUNTIME_SMOKE:-auto}"
RUN_PIK_SMOKE="${RUN_PIK_SMOKE:-auto}"
RUN_D4_SMOKE="${RUN_D4_SMOKE:-0}"
DATASET="${DATASET:-strategyqa}"

FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-assets/artifacts/phase_c_feature_cache}"
FEATURE_CACHE_MODE="${FEATURE_CACHE_MODE:-read_write}"
FEATURE_CACHE_LOCK_TIMEOUT_SEC="${FEATURE_CACHE_LOCK_TIMEOUT_SEC:-600}"

C2_TRAIN_DIR="${C2_TRAIN_DIR:-}"
C2_EVAL_DIR="${C2_EVAL_DIR:-}"
PIK_TRAIN_DIR="${PIK_TRAIN_DIR:-}"
PIK_EVAL_DIR="${PIK_EVAL_DIR:-}"

C2_MAX_STEPS="${C2_MAX_STEPS:-1}"
PIK_MAX_STEPS="${PIK_MAX_STEPS:-1}"

R_PRM_ROOT="${R_PRM_ROOT:-assets/external_datasets/kevinpro_r_prm}"
PRMBENCH_PREVIEW_PATH="${PRMBENCH_PREVIEW_PATH:-assets/external_datasets/hitsmy_prmbench_preview/prmbench_preview.jsonl}"

LOG_ROOT="assets/artifacts/healthcheck_logs/${RUN_PREFIX}"
SUITE_LOG_FILE="$LOG_ROOT/suite.log"
SUMMARY_FILE="$LOG_ROOT/final_summary.md"
RESULTS_FILE="$LOG_ROOT/stage_results.md"

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
FINAL_FAILED=0

timestamp() {
  date "+%Y-%m-%d %H:%M:%S %z"
}

log_line() {
  local msg="$1"
  echo "[$(timestamp)] $msg"
}

append_stage_result() {
  local stage="$1"
  local status="$2" # pass | fail | skip
  local note="$3"
  local safe_note="${note//|//}"
  printf '| `%s` | `%s` | %s |\n' "$stage" "$status" "$safe_note" >> "$RESULTS_FILE"
  case "$status" in
    pass) PASS_COUNT=$((PASS_COUNT + 1)) ;;
    fail) FAIL_COUNT=$((FAIL_COUNT + 1)) ;;
    skip) SKIP_COUNT=$((SKIP_COUNT + 1)) ;;
  esac
}

mark_failure() {
  local stage="$1"
  local note="$2"
  FINAL_FAILED=1
  append_stage_result "$stage" "fail" "$note"
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing $label: $path" >&2
    return 1
  fi
}

latest_matching_dir() {
  local pattern="$1"
  local latest=""
  latest="$(compgen -G "$pattern" | sort | tail -n 1 || true)"
  printf '%s\n' "$latest"
}

has_cuda() {
  "$PYTHON_BIN" - <<'PY'
try:
    import torch
    print("1" if torch.cuda.is_available() else "0")
except Exception:
    print("0")
PY
}

resolve_c2_dirs_if_needed() {
  if [[ -z "$C2_TRAIN_DIR" ]]; then
    C2_TRAIN_DIR="$(latest_matching_dir "assets/artifacts/phase_c_data/${DATASET}/*c1_train__*")"
    if [[ -z "$C2_TRAIN_DIR" ]]; then
      C2_TRAIN_DIR="$(latest_matching_dir "assets/artifacts/phase_c_data/${DATASET}/*train*__*")"
    fi
  fi
  if [[ -z "$C2_EVAL_DIR" ]]; then
    C2_EVAL_DIR="$(latest_matching_dir "assets/artifacts/phase_c_data/${DATASET}/*c1_eval__*")"
    if [[ -z "$C2_EVAL_DIR" ]]; then
      C2_EVAL_DIR="$(latest_matching_dir "assets/artifacts/phase_c_data/${DATASET}/*eval*__*")"
    fi
  fi
}

resolve_pik_dirs_if_needed() {
  if [[ -z "$PIK_TRAIN_DIR" ]]; then
    PIK_TRAIN_DIR="$(latest_matching_dir "assets/artifacts/phase_c_pik_data/${DATASET}/*c1_train__*")"
  fi
  if [[ -z "$PIK_EVAL_DIR" ]]; then
    PIK_EVAL_DIR="$(latest_matching_dir "assets/artifacts/phase_c_pik_data/${DATASET}/*c1_eval__*")"
  fi
}

write_failure_summary() {
  local exit_code="$1"
  cat > "$SUMMARY_FILE" <<EOF
# Feature Cache Healthcheck Summary

- generated_at: $(date --iso-8601=seconds)
- status: failed
- exit_code: ${exit_code}
- failed_stage: ${CURRENT_STAGE:-unknown}
- run_prefix: ${RUN_PREFIX}
- suite_log_file: ${SUITE_LOG_FILE}
- results_file: ${RESULTS_FILE}
EOF
}

on_exit() {
  local exit_code="$1"
  if [[ "$exit_code" -ne 0 ]]; then
    {
      log_line "Healthcheck terminated unexpectedly"
      log_line "Failed stage: ${CURRENT_STAGE:-unknown}"
      log_line "Exit code: $exit_code"
    } | tee -a "$SUITE_LOG_FILE"
    write_failure_summary "$exit_code"
  fi
}

trap 'on_exit $?' EXIT

mkdir -p "$LOG_ROOT" "$FEATURE_CACHE_ROOT"
: > "$SUITE_LOG_FILE"
cat > "$RESULTS_FILE" <<'EOF'
| Stage | Status | Note |
| --- | --- | --- |
EOF

{
  log_line "Feature Cache Healthcheck"
  log_line "repo_root=$REPO_ROOT"
  log_line "python=$PYTHON_BIN"
  log_line "check_profile=$CHECK_PROFILE"
  log_line "run_prefix=$RUN_PREFIX"
  log_line "dataset=$DATASET"
  log_line "feature_cache_root=$FEATURE_CACHE_ROOT"
  log_line "feature_cache_mode=$FEATURE_CACHE_MODE"
  log_line "feature_cache_lock_timeout_sec=$FEATURE_CACHE_LOCK_TIMEOUT_SEC"
  log_line "run_runtime_smoke=$RUN_RUNTIME_SMOKE"
  log_line "run_pik_smoke=$RUN_PIK_SMOKE"
  log_line "run_d4_smoke=$RUN_D4_SMOKE"
  log_line "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
} | tee -a "$SUITE_LOG_FILE"

CURRENT_STAGE="static_py_compile"
if "$PYTHON_BIN" -m py_compile \
  src/ours/phase_b/feature_cache.py \
  scripts/phase_b_train_value.py \
  scripts/phase_b_eval_faithfulness.py \
  scripts/phase_c_train_pik.py \
  scripts/phase_c_eval_pik.py \
  2>&1 | tee -a "$SUITE_LOG_FILE"; then
  append_stage_result "$CURRENT_STAGE" "pass" "py_compile passed for cache-integrated scripts"
else
  mark_failure "$CURRENT_STAGE" "py_compile failed"
fi

CURRENT_STAGE="static_shell_syntax"
if bash -n \
  scripts/run_phase_c_value_suite.sh \
  scripts/run_phase_c_pik_suite.sh \
  scripts/run_phase_d_external_pair_suite.sh \
  scripts/run_feature_cache_healthcheck.sh \
  2>&1 | tee -a "$SUITE_LOG_FILE"; then
  append_stage_result "$CURRENT_STAGE" "pass" "bash -n passed"
else
  mark_failure "$CURRENT_STAGE" "bash -n failed"
fi

CURRENT_STAGE="cli_feature_cache_flags"
if (
  "$PYTHON_BIN" scripts/phase_b_train_value.py --help | rg -q "feature-cache" &&
  "$PYTHON_BIN" scripts/phase_b_eval_faithfulness.py --help | rg -q "feature-cache" &&
  "$PYTHON_BIN" scripts/phase_c_train_pik.py --help | rg -q "feature-cache" &&
  "$PYTHON_BIN" scripts/phase_c_eval_pik.py --help | rg -q "feature-cache"
) 2>&1 | tee -a "$SUITE_LOG_FILE"; then
  append_stage_result "$CURRENT_STAGE" "pass" "all target CLIs expose feature-cache args"
else
  mark_failure "$CURRENT_STAGE" "missing feature-cache args in one or more CLIs"
fi

CURRENT_STAGE="feature_cache_module_selftest"
if PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" "$PYTHON_BIN" - <<'PY' 2>&1 | tee -a "$SUITE_LOG_FILE"
import shutil
import tempfile
from pathlib import Path

import torch

from ours.phase_b.feature_cache import (
    build_cache_key,
    feature_cache_can_read,
    feature_cache_can_write,
    move_tensors_to_device,
    save_feature_cache,
    try_load_feature_cache,
    validate_feature_cache_mode,
)

mode = validate_feature_cache_mode("READ_WRITE")
assert mode == "read_write"
assert feature_cache_can_read(mode) is True
assert feature_cache_can_write(mode) is True
assert feature_cache_can_read("off") is False
assert feature_cache_can_write("off") is False

tmp_root = Path(tempfile.mkdtemp(prefix="feature_cache_selftest_"))
try:
    payload = {"x": torch.arange(12, dtype=torch.float32).view(3, 4)}
    signature = {"kind": "selftest", "n": 3}
    cache_key, signature_hash = build_cache_key("diag_cache", signature)
    save_feature_cache(
        cache_root=tmp_root,
        cache_key=cache_key,
        signature_hash=signature_hash,
        payload=payload,
        torch_module=torch,
        producer="healthcheck:selftest",
        lock_timeout_sec=30.0,
        extra_metadata={"rows": 3},
    )
    loaded, meta, cache_dir = try_load_feature_cache(
        cache_root=tmp_root,
        cache_key=cache_key,
        expected_signature_hash=signature_hash,
        torch_module=torch,
    )
    assert loaded is not None
    assert isinstance(meta, dict)
    assert cache_dir.exists()
    assert torch.allclose(loaded["x"], payload["x"])

    miss, _, _ = try_load_feature_cache(
        cache_root=tmp_root,
        cache_key=cache_key,
        expected_signature_hash="0" * 64,
        torch_module=torch,
    )
    assert miss is None

    moved = move_tensors_to_device(loaded, torch.device("cpu"), torch)
    assert moved["x"].device.type == "cpu"

    print("selftest_ok=1")
finally:
    shutil.rmtree(tmp_root, ignore_errors=True)
PY
then
  append_stage_result "$CURRENT_STAGE" "pass" "cache save/load/signature/device selftest passed"
else
  mark_failure "$CURRENT_STAGE" "feature_cache module selftest failed"
fi

should_run_runtime=0
if [[ "$CHECK_PROFILE" == "full" ]]; then
  if [[ "$RUN_RUNTIME_SMOKE" == "1" ]]; then
    should_run_runtime=1
  elif [[ "$RUN_RUNTIME_SMOKE" == "auto" ]]; then
    should_run_runtime=1
  fi
fi

CURRENT_STAGE="runtime_smoke_c2"
if [[ "$should_run_runtime" != "1" ]]; then
  append_stage_result "$CURRENT_STAGE" "skip" "runtime smoke disabled (CHECK_PROFILE=$CHECK_PROFILE, RUN_RUNTIME_SMOKE=$RUN_RUNTIME_SMOKE)"
else
  resolve_c2_dirs_if_needed
  CUDA_AVAILABLE="$(has_cuda)"
  if [[ "$CUDA_AVAILABLE" != "1" ]]; then
    append_stage_result "$CURRENT_STAGE" "skip" "CUDA unavailable"
  elif [[ -z "$C2_TRAIN_DIR" || -z "$C2_EVAL_DIR" ]]; then
    append_stage_result "$CURRENT_STAGE" "skip" "C2 train/eval dirs not found (set C2_TRAIN_DIR/C2_EVAL_DIR)"
  elif ! require_file "$C2_TRAIN_DIR/prefixes.jsonl" "C2 train prefixes" || ! require_file "$C2_EVAL_DIR/prefixes.jsonl" "C2 eval prefixes"; then
    append_stage_result "$CURRENT_STAGE" "skip" "C2 dirs found but required files missing"
  else
    C2_PASS1_NAME="${RUN_PREFIX}_c2_cache_diag_pass1"
    C2_PASS2_NAME="${RUN_PREFIX}_c2_cache_diag_pass2"
    C2_TRAIN_PASS1_LOG="$LOG_ROOT/runtime_c2_train_pass1.log"
    C2_TRAIN_PASS2_LOG="$LOG_ROOT/runtime_c2_train_pass2.log"
    C2_EVAL_PASS1_LOG="$LOG_ROOT/runtime_c2_eval_pass1.log"
    C2_EVAL_PASS2_LOG="$LOG_ROOT/runtime_c2_eval_pass2.log"

    {
      log_line "Resolved C2 train dir: $C2_TRAIN_DIR"
      log_line "Resolved C2 eval dir : $C2_EVAL_DIR"
      log_line "C2 smoke max_steps   : $C2_MAX_STEPS"
    } | tee -a "$SUITE_LOG_FILE"

    if "$PYTHON_BIN" -u scripts/phase_b_train_value.py \
      --train-dir "$C2_TRAIN_DIR" \
      --eval-dir "$C2_EVAL_DIR" \
      --run-name "$C2_PASS1_NAME" \
      --require-cuda \
      --dtype bfloat16 \
      --device-map auto \
      --target-source q_mean_smoothed \
      --target-source-missing-policy fail \
      --posthoc-calibration none \
      --checkpoint-selection-metric raw_brier \
      --max-steps "$C2_MAX_STEPS" \
      --num-train-epochs 1 \
      --per-device-train-batch-size 16 \
      --per-device-eval-batch-size 64 \
      --feature-cache-root "$FEATURE_CACHE_ROOT" \
      --feature-cache-mode read_write \
      --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC" \
      2>&1 | tee "$C2_TRAIN_PASS1_LOG" | tee -a "$SUITE_LOG_FILE"; then
      :
    else
      mark_failure "$CURRENT_STAGE" "C2 train pass1 failed"
    fi

    if "$PYTHON_BIN" -u scripts/phase_b_train_value.py \
      --train-dir "$C2_TRAIN_DIR" \
      --eval-dir "$C2_EVAL_DIR" \
      --run-name "$C2_PASS2_NAME" \
      --require-cuda \
      --dtype bfloat16 \
      --device-map auto \
      --target-source q_mean_smoothed \
      --target-source-missing-policy fail \
      --posthoc-calibration none \
      --checkpoint-selection-metric raw_brier \
      --max-steps "$C2_MAX_STEPS" \
      --num-train-epochs 1 \
      --per-device-train-batch-size 16 \
      --per-device-eval-batch-size 64 \
      --feature-cache-root "$FEATURE_CACHE_ROOT" \
      --feature-cache-mode read \
      --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC" \
      2>&1 | tee "$C2_TRAIN_PASS2_LOG" | tee -a "$SUITE_LOG_FILE"; then
      :
    else
      mark_failure "$CURRENT_STAGE" "C2 train pass2 failed"
    fi

    C2_RUN_DIR="$(latest_matching_dir "assets/artifacts/phase_c_runs/${C2_PASS2_NAME}_*")"
    if [[ -z "$C2_RUN_DIR" ]]; then
      mark_failure "$CURRENT_STAGE" "unable to resolve C2 run dir for pass2"
    else
      if "$PYTHON_BIN" -u scripts/phase_b_eval_faithfulness.py \
        --value-run-dir "$C2_RUN_DIR" \
        --eval-dir "$C2_EVAL_DIR" \
        --checkpoint-name best \
        --target-source from_run \
        --target-source-missing-policy from_run \
        --posthoc-calibration none \
        --run-name "${RUN_PREFIX}_c2_eval_cache_pass1" \
        --feature-cache-root "$FEATURE_CACHE_ROOT" \
        --feature-cache-mode read_write \
        --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC" \
        2>&1 | tee "$C2_EVAL_PASS1_LOG" | tee -a "$SUITE_LOG_FILE"; then
        :
      else
        mark_failure "$CURRENT_STAGE" "C2 eval pass1 failed"
      fi

      if "$PYTHON_BIN" -u scripts/phase_b_eval_faithfulness.py \
        --value-run-dir "$C2_RUN_DIR" \
        --eval-dir "$C2_EVAL_DIR" \
        --checkpoint-name best \
        --target-source from_run \
        --target-source-missing-policy from_run \
        --posthoc-calibration none \
        --run-name "${RUN_PREFIX}_c2_eval_cache_pass2" \
        --feature-cache-root "$FEATURE_CACHE_ROOT" \
        --feature-cache-mode read \
        --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC" \
        2>&1 | tee "$C2_EVAL_PASS2_LOG" | tee -a "$SUITE_LOG_FILE"; then
        :
      else
        mark_failure "$CURRENT_STAGE" "C2 eval pass2 failed"
      fi
    fi

    if [[ "$FINAL_FAILED" -eq 0 ]]; then
      if rg -q "feature_cache\\s+: .* hit" "$C2_TRAIN_PASS2_LOG" && rg -q "feature_cache\\s+: .* hit" "$C2_EVAL_PASS2_LOG"; then
        append_stage_result "$CURRENT_STAGE" "pass" "C2 runtime smoke passed with cache-hit evidence in pass2"
      else
        mark_failure "$CURRENT_STAGE" "C2 runtime smoke finished but no cache-hit evidence in pass2 logs"
      fi
    fi
  fi
fi

CURRENT_STAGE="runtime_smoke_pik"
should_run_pik=0
if [[ "$CHECK_PROFILE" == "full" ]]; then
  if [[ "$RUN_PIK_SMOKE" == "1" ]]; then
    should_run_pik=1
  elif [[ "$RUN_PIK_SMOKE" == "auto" ]]; then
    should_run_pik=1
  fi
fi

if [[ "$should_run_pik" != "1" ]]; then
  append_stage_result "$CURRENT_STAGE" "skip" "PIK smoke disabled (CHECK_PROFILE=$CHECK_PROFILE, RUN_PIK_SMOKE=$RUN_PIK_SMOKE)"
else
  resolve_pik_dirs_if_needed
  CUDA_AVAILABLE="$(has_cuda)"
  if [[ "$CUDA_AVAILABLE" != "1" ]]; then
    append_stage_result "$CURRENT_STAGE" "skip" "CUDA unavailable"
  elif [[ -z "$PIK_TRAIN_DIR" || -z "$PIK_EVAL_DIR" ]]; then
    append_stage_result "$CURRENT_STAGE" "skip" "PIK train/eval dirs not found (set PIK_TRAIN_DIR/PIK_EVAL_DIR)"
  elif ! require_file "$PIK_TRAIN_DIR/pik_targets.jsonl" "PIK train targets" || ! require_file "$PIK_EVAL_DIR/pik_targets.jsonl" "PIK eval targets"; then
    append_stage_result "$CURRENT_STAGE" "skip" "PIK dirs found but required files missing"
  else
    PIK_PASS1_NAME="${RUN_PREFIX}_pik_cache_diag_pass1"
    PIK_PASS2_NAME="${RUN_PREFIX}_pik_cache_diag_pass2"
    PIK_TRAIN_PASS2_LOG="$LOG_ROOT/runtime_pik_train_pass2.log"
    PIK_EVAL_PASS2_LOG="$LOG_ROOT/runtime_pik_eval_pass2.log"

    if "$PYTHON_BIN" -u scripts/phase_c_train_pik.py \
      --train-dir "$PIK_TRAIN_DIR" \
      --eval-dir "$PIK_EVAL_DIR" \
      --run-name "$PIK_PASS1_NAME" \
      --require-cuda \
      --dtype bfloat16 \
      --device-map auto \
      --max-steps "$PIK_MAX_STEPS" \
      --num-train-epochs 1 \
      --per-device-train-batch-size 32 \
      --per-device-eval-batch-size 64 \
      --posthoc-calibration none \
      --checkpoint-selection-metric raw_brier \
      --feature-cache-root "$FEATURE_CACHE_ROOT" \
      --feature-cache-mode read_write \
      --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC" \
      2>&1 | tee -a "$SUITE_LOG_FILE"; then
      :
    else
      mark_failure "$CURRENT_STAGE" "PIK train pass1 failed"
    fi

    if "$PYTHON_BIN" -u scripts/phase_c_train_pik.py \
      --train-dir "$PIK_TRAIN_DIR" \
      --eval-dir "$PIK_EVAL_DIR" \
      --run-name "$PIK_PASS2_NAME" \
      --require-cuda \
      --dtype bfloat16 \
      --device-map auto \
      --max-steps "$PIK_MAX_STEPS" \
      --num-train-epochs 1 \
      --per-device-train-batch-size 32 \
      --per-device-eval-batch-size 64 \
      --posthoc-calibration none \
      --checkpoint-selection-metric raw_brier \
      --feature-cache-root "$FEATURE_CACHE_ROOT" \
      --feature-cache-mode read \
      --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC" \
      2>&1 | tee "$PIK_TRAIN_PASS2_LOG" | tee -a "$SUITE_LOG_FILE"; then
      :
    else
      mark_failure "$CURRENT_STAGE" "PIK train pass2 failed"
    fi

    PIK_RUN_DIR="$(latest_matching_dir "assets/artifacts/phase_c_pik_runs/${PIK_PASS2_NAME}_*")"
    if [[ -z "$PIK_RUN_DIR" ]]; then
      mark_failure "$CURRENT_STAGE" "unable to resolve PIK run dir for pass2"
    else
      if "$PYTHON_BIN" -u scripts/phase_c_eval_pik.py \
        --value-run-dir "$PIK_RUN_DIR" \
        --eval-dir "$PIK_EVAL_DIR" \
        --checkpoint-name best \
        --posthoc-calibration none \
        --run-name "${RUN_PREFIX}_pik_eval_cache_pass2" \
        --feature-cache-root "$FEATURE_CACHE_ROOT" \
        --feature-cache-mode read \
        --feature-cache-lock-timeout-sec "$FEATURE_CACHE_LOCK_TIMEOUT_SEC" \
        2>&1 | tee "$PIK_EVAL_PASS2_LOG" | tee -a "$SUITE_LOG_FILE"; then
        :
      else
        mark_failure "$CURRENT_STAGE" "PIK eval pass2 failed"
      fi
    fi

    if [[ "$FINAL_FAILED" -eq 0 ]]; then
      if rg -q "feature_cache\\s+: .* hit" "$PIK_TRAIN_PASS2_LOG" && rg -q "feature_cache\\s+: .* hit" "$PIK_EVAL_PASS2_LOG"; then
        append_stage_result "$CURRENT_STAGE" "pass" "PIK runtime smoke passed with cache-hit evidence in pass2"
      else
        mark_failure "$CURRENT_STAGE" "PIK runtime smoke finished but no cache-hit evidence in pass2 logs"
      fi
    fi
  fi
fi

CURRENT_STAGE="runtime_smoke_d4"
if [[ "$RUN_D4_SMOKE" != "1" ]]; then
  append_stage_result "$CURRENT_STAGE" "skip" "D4 smoke disabled (set RUN_D4_SMOKE=1 to enable)"
else
  if [[ -z "$C2_TRAIN_DIR" || -z "$C2_EVAL_DIR" ]]; then
    resolve_c2_dirs_if_needed
  fi
  if [[ -z "$C2_TRAIN_DIR" || -z "$C2_EVAL_DIR" ]]; then
    append_stage_result "$CURRENT_STAGE" "skip" "D4 smoke needs C2 train/eval dirs"
  elif [[ ! -d "$R_PRM_ROOT" || ! -f "$PRMBENCH_PREVIEW_PATH" ]]; then
    append_stage_result "$CURRENT_STAGE" "skip" "D4 smoke needs external datasets (R-PRM + PRMBench preview)"
  else
    if ACTIVE_PHASE_D4_GROUP=D4A_STRATEGYQA_SMOKE \
      RUN_PREFIX="${RUN_PREFIX}_d4a_smoke" \
      PHASE_C_TRAIN_DIR="$C2_TRAIN_DIR" \
      PHASE_C_EVAL_DIR="$C2_EVAL_DIR" \
      FEATURE_CACHE_ROOT="$FEATURE_CACHE_ROOT" \
      FEATURE_CACHE_MODE=read \
      FEATURE_CACHE_LOCK_TIMEOUT_SEC="$FEATURE_CACHE_LOCK_TIMEOUT_SEC" \
      D4_EVAL_POSTHOC_MODE=auto \
      D4A_PAIR_PREP_EXTRA_ARGS="--max-pairs-per-source 64 --max-pairs-total 96" \
      D4A_C2_TRAIN_EXTRA_ARGS="--max-steps 1 --num-train-epochs 1 --posthoc-calibration none --checkpoint-selection-metric raw_brier" \
      bash scripts/run_phase_d_external_pair_suite.sh \
      2>&1 | tee -a "$SUITE_LOG_FILE"; then
      append_stage_result "$CURRENT_STAGE" "pass" "D4A smoke completed"
    else
      mark_failure "$CURRENT_STAGE" "D4A smoke failed"
    fi
  fi
fi

CURRENT_STAGE="final_summary"
cat > "$SUMMARY_FILE" <<EOF
# Feature Cache Healthcheck Summary

- generated_at: $(date --iso-8601=seconds)
- run_prefix: $RUN_PREFIX
- check_profile: $CHECK_PROFILE
- dataset: $DATASET
- feature_cache_root: $FEATURE_CACHE_ROOT
- feature_cache_mode: $FEATURE_CACHE_MODE
- feature_cache_lock_timeout_sec: $FEATURE_CACHE_LOCK_TIMEOUT_SEC
- run_runtime_smoke: $RUN_RUNTIME_SMOKE
- run_pik_smoke: $RUN_PIK_SMOKE
- run_d4_smoke: $RUN_D4_SMOKE
- pass_count: $PASS_COUNT
- fail_count: $FAIL_COUNT
- skip_count: $SKIP_COUNT
- status: $([[ "$FINAL_FAILED" -eq 0 ]] && echo "success" || echo "failed")
- suite_log_file: $SUITE_LOG_FILE

## Stage Results

$(cat "$RESULTS_FILE")
EOF

{
  log_line "Summary written: $SUMMARY_FILE"
  log_line "Suite log: $SUITE_LOG_FILE"
} | tee -a "$SUITE_LOG_FILE"

if [[ "$FINAL_FAILED" -ne 0 ]]; then
  exit 1
fi
exit 0
