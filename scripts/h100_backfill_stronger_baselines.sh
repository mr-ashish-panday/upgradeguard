#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zeus/content/upgradeguard}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/h100_conditioned_method_panels}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/h100_stronger_baselines}"

mkdir -p "$LOG_ROOT"
cd "$ROOT"

run_backfill() {
  local run_dir="$1"
  local alias="$2"
  local log_file="$LOG_ROOT/${alias}.log"
  if [[ ! -d "$run_dir" ]]; then
    echo "[$(date -Is)] SKIP missing run dir $run_dir" | tee -a "$log_file"
    return 0
  fi
  echo "[$(date -Is)] START stronger baseline backfill $run_dir" | tee -a "$log_file"
  "$PYTHON_BIN" -u scripts/run_stronger_baselines_saved_run.py \
    --run-dir "$run_dir" \
    --cache-root "$RUN_ROOT" \
    --device cuda \
    2>&1 | tee -a "$log_file"
  echo "[$(date -Is)] DONE stronger baseline backfill $run_dir" | tee -a "$log_file"
}

run_backfill "${QWEN_TRANSLATION_LORA_RUN_DIR:-$RUN_ROOT/Qwen_Qwen2_5-7B-Instruct_translation_lora}" "qwen_translation_lora"
run_backfill "${QWEN_TRANSLATION_QLORA_RUN_DIR:-$RUN_ROOT/Qwen_Qwen2_5-7B-Instruct_translation_qlora}" "qwen_translation_qlora"
run_backfill "${QWEN_TRANSLATION_FULL_FT_RUN_DIR:-$RUN_ROOT/Qwen_Qwen2_5-7B-Instruct_translation_full_ft}" "qwen_translation_full_ft"
run_backfill "${LLAMA_SUMMARIZATION_LORA_RUN_DIR:-$RUN_ROOT/meta-llama_Meta-Llama-3_1-8B-Instruct_summarization_lora}" "llama_summarization_lora"
run_backfill "${LLAMA_SUMMARIZATION_QLORA_RUN_DIR:-$RUN_ROOT/meta-llama_Meta-Llama-3_1-8B-Instruct_summarization_qlora}" "llama_summarization_qlora"
