#!/usr/bin/env bash
# Download the external datasets used by this project into a local folder.
#
# Why this file exists:
# - dataset repo IDs occasionally move or break,
# - the project wants one simple command for local dataset setup,
# - fallback repos should be kept in one place rather than repeated in notes.
#
# What this file does:
# 1. choose an output directory,
# 2. try primary and fallback Hugging Face dataset repos,
# 3. write each downloaded dataset under a stable local folder name.
#
# Example:
#   bash download_datasets.sh
#   bash download_datasets.sh assets/datasets
set -euo pipefail

# Usage:
#   bash download_datasets.sh
#   bash download_datasets.sh assets/datasets
#
# This script downloads datasets discussed in the project notes using
# currently available Hugging Face dataset repo IDs, with fallbacks when
# possible.

OUT_DIR="${1:-assets/datasets}"
MAX_WORKERS="${HF_MAX_WORKERS:-8}"

mkdir -p "${OUT_DIR}"

download_one() {
  # Download one dataset repo into one stable local directory.
  local repo_id="$1"
  local local_name="$2"
  echo "==> Trying ${repo_id} -> ${OUT_DIR}/${local_name}"
  hf download "${repo_id}" \
    --repo-type dataset \
    --local-dir "${OUT_DIR}/${local_name}" \
    --max-workers "${MAX_WORKERS}"
}

download_with_fallbacks() {
  # Try several candidate repos until one download succeeds.
  #
  # Example:
  #   download_with_fallbacks "strategyqa" "tasksource/strategy-qa" "wics/strategy-qa"
  local local_name="$1"
  shift
  local repos=("$@")

  for repo in "${repos[@]}"; do
    if download_one "${repo}" "${local_name}"; then
      echo "OK: ${local_name} (from ${repo})"
      return 0
    fi
    echo "WARN: failed from ${repo}, trying next fallback..."
  done

  echo "ERROR: all candidates failed for ${local_name}" >&2
  return 1
}

echo "Output dir: ${OUT_DIR}"
echo

# Already known to work in your environment:
download_with_fallbacks "gsm8k" "gsm8k"
download_with_fallbacks "drop" "drop"

# Previously failing IDs replaced with working namespaced IDs:
download_with_fallbacks "logiqa" \
  "lucasmccabe/logiqa" \
  "EleutherAI/logiqa"

download_with_fallbacks "strategyqa" \
  "tasksource/strategy-qa" \
  "wics/strategy-qa"

download_with_fallbacks "proofwriter" \
  "tasksource/proofwriter" \
  "wentingzhao/proofwriter"

download_with_fallbacks "bigbench_hard" \
  "lukaemon/bbh" \
  "maveriq/bigbenchhard"

# hendrycks/competition_math appears disabled. Use maintained alternatives.
download_with_fallbacks "hendrycks_math" \
  "EleutherAI/hendrycks_math" \
  "baber/hendrycks_math"

echo
echo "Done. Downloaded datasets are under: ${OUT_DIR}"
