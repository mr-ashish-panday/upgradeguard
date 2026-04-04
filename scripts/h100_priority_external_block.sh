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

LLAMA_LORA_RUN_DIR="${LLAMA_LORA_RUN_DIR:-$RESULTS_ROOT/meta-llama_Meta-Llama-3_1-8B-Instruct_summarization_lora}"
LLAMA_QLORA_RUN_DIR="${LLAMA_QLORA_RUN_DIR:-$RESULTS_ROOT/meta-llama_Meta-Llama-3_1-8B-Instruct_summarization_qlora}"
QWEN_TRANSLATION_QLORA_RUN_DIR="${QWEN_TRANSLATION_QLORA_RUN_DIR:-$RESULTS_ROOT/Qwen_Qwen2_5-7B-Instruct_translation_qlora}"

mkdir -p "$RESULTS_ROOT" "$LOG_ROOT" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE"
cd "$ROOT"

run_step() {
  local name="$1"
  shift
  echo "[$(date -Iseconds)] Starting ${name}"
  "$@" 2>&1 | tee "$LOG_ROOT/${name}.log"
  echo "[$(date -Iseconds)] Finished ${name}"
}

run_step "llama_base_external" \
  "$PYTHON_BIN" -u -m upgradeguard.main \
    --model llama \
    --skip-finetune \
    --skip-posthoc \
    --evaluate-base-external \
    --include-strongreject \
    --output-dir "$RESULTS_ROOT"

run_step "llama_replication_external" \
  "$PYTHON_BIN" -u -m upgradeguard.main \
    --model llama \
    --skip-finetune \
    --skip-posthoc \
    --backfill-external-validation \
    --include-strongreject \
    --selected-run-dirs "$LLAMA_LORA_RUN_DIR" "$LLAMA_QLORA_RUN_DIR" \
    --output-dir "$RESULTS_ROOT"

run_step "qwen_translation_qlora_external" \
  "$PYTHON_BIN" -u -m upgradeguard.main \
    --model qwen \
    --skip-finetune \
    --skip-posthoc \
    --backfill-external-validation \
    --include-strongreject \
    --selected-run-dirs "$QWEN_TRANSLATION_QLORA_RUN_DIR" \
    --output-dir "$RESULTS_ROOT"
