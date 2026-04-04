#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zeus/content/upgradeguard}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results/h100_priority_external}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/h100_priority_external}"

export HF_HOME="${HF_HOME:-/home/zeus/content/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HUGGINGFACE_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

mkdir -p "$RESULTS_ROOT" "$LOG_ROOT" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE"
cd "$ROOT"

"$PYTHON_BIN" -u -m upgradeguard.main \
  --model qwen \
  --task translation \
  --method lora \
  --run-external-validation \
  --include-strongreject \
  --output-dir "$RESULTS_ROOT" \
  2>&1 | tee "$LOG_ROOT/qwen_translation_lora_external.log"
