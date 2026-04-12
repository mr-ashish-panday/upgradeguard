#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zeus/content/upgradeguard}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/h100_qwen_translation_dense_optimized}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/h100_qwen_translation_dense_optimized}"

export HF_HOME="${HF_HOME:-/home/zeus/content/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
cd "$ROOT"

LOG_FILE="$LOG_DIR/qwen_translation_full_ft_optimized.log"
echo "[$(date -Is)] START qwen translation optimized dense baseline" | tee -a "$LOG_FILE"
"$PYTHON_BIN" -u -m upgradeguard.main \
  --model qwen \
  --task translation \
  --method full_ft \
  --output-dir "$OUTPUT_DIR" \
  --learning-rate 1e-5 \
  --epochs 2 \
  --batch-size 4 \
  --generation-batch-size 4 \
  --run-external-validation \
  --include-strongreject \
  --harmbench-samples 320 \
  --xstest-samples 450 \
  --strongreject-samples 60 \
  2>&1 | tee -a "$LOG_FILE"
echo "[$(date -Is)] DONE qwen translation optimized dense baseline" | tee -a "$LOG_FILE"
