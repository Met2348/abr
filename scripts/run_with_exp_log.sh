#!/usr/bin/env bash
# Thin wrapper to run one command with experiment-command auto logging.
#
# Why this file exists:
# - Keep usage simple for daily experiments.
# - Let users opt in/out of markdown auto-maintenance via env switches.
#
# Basic usage:
#   bash scripts/run_with_exp_log.sh python -u scripts/phase_a_generate_and_eval.py ...
#
# Optional toggles:
# - EXP_LOG_ENABLE=0                 # bypass logger and run command directly
# - EXP_LOG_TAGS="phase_c,smoke"     # comma-separated tags
# - EXP_LOG_NOTE="short note"        # one-line note saved in run record
# - EXP_LOG_UPDATE_DOCS=0            # only write JSONL records, skip markdown refresh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXP_LOG_ENABLE="${EXP_LOG_ENABLE:-1}"
EXP_LOG_UPDATE_DOCS="${EXP_LOG_UPDATE_DOCS:-1}"
EXP_LOG_TAGS="${EXP_LOG_TAGS:-}"
EXP_LOG_NOTE="${EXP_LOG_NOTE:-}"

if [[ "$#" -eq 0 ]]; then
  echo "Usage: bash scripts/run_with_exp_log.sh <command> [args...]" >&2
  exit 2
fi

if [[ "$EXP_LOG_ENABLE" == "0" ]]; then
  exec "$@"
fi

LOGGER_ARGS=()
if [[ "$EXP_LOG_UPDATE_DOCS" == "0" ]]; then
  LOGGER_ARGS+=(--no-update-docs)
fi
if [[ -n "$EXP_LOG_NOTE" ]]; then
  LOGGER_ARGS+=(--note "$EXP_LOG_NOTE")
fi
if [[ -n "$EXP_LOG_TAGS" ]]; then
  IFS=',' read -r -a _tags <<< "$EXP_LOG_TAGS"
  for tag in "${_tags[@]}"; do
    tag_trimmed="$(echo "$tag" | xargs)"
    if [[ -n "$tag_trimmed" ]]; then
      LOGGER_ARGS+=(--tag "$tag_trimmed")
    fi
  done
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/experiment_command_logger.py" "${LOGGER_ARGS[@]}" -- "$@"
