#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zeus/content/upgradeguard}"
PYTHON_BIN="${PYTHON_BIN:-/home/zeus/miniconda3/envs/cloudspace/bin/python}"
RUN_ROOT="${RUN_ROOT:-/home/zeus/content/seed_runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/h100_parallel_followup_remaining}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/h100_parallel_followup_remaining}"
SHARED_CACHE_ROOT="${SHARED_CACHE_ROOT:-$OUTPUT_ROOT/shared_cache_root}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$SHARED_CACHE_ROOT"
cd "$ROOT"

run_decode() {
  local run_dir="$1"
  local alias="$2"
  local log_file="$LOG_ROOT/${alias}_sampling.log"
  echo "[$(date -Is)] START sampling external eval $run_dir" | tee -a "$log_file"
  "$PYTHON_BIN" -u scripts/run_external_decode_regime.py \
    --run-dir "$run_dir" \
    --output-dir "$OUTPUT_ROOT/${alias}/sampling_external" \
    --cache-root "$SHARED_CACHE_ROOT" \
    --decode-regime sampling \
    --include-strongreject \
    --device cuda \
    2>&1 | tee -a "$log_file"
  echo "[$(date -Is)] DONE sampling external eval $run_dir" | tee -a "$log_file"
}

run_token_form() {
  local run_dir="$1"
  local alias="$2"
  local log_file="$LOG_ROOT/${alias}_token_form.log"
  echo "[$(date -Is)] START token-form drift eval $run_dir" | tee -a "$log_file"
  "$PYTHON_BIN" -u scripts/run_token_form_drift_eval.py \
    --run-dir "$run_dir" \
    --output-dir "$OUTPUT_ROOT/${alias}/token_form_drift" \
    --prompt-bank canaries \
    --device cuda \
    2>&1 | tee -a "$log_file"
  echo "[$(date -Is)] DONE token-form drift eval $run_dir" | tee -a "$log_file"
}

run_backfill() {
  local run_dir="$1"
  local alias="$2"
  local log_file="$LOG_ROOT/${alias}_stronger_baselines.log"
  echo "[$(date -Is)] START stronger baseline backfill $run_dir" | tee -a "$log_file"
  "$PYTHON_BIN" -u scripts/run_stronger_baselines_saved_run.py \
    --run-dir "$run_dir" \
    --cache-root "$SHARED_CACHE_ROOT" \
    --device cuda \
    2>&1 | tee -a "$log_file"
  echo "[$(date -Is)] DONE stronger baseline backfill $run_dir" | tee -a "$log_file"
}

QWEN_SUMM_LORA="${QWEN_SUMM_LORA:-$RUN_ROOT/Qwen_Qwen2_5-7B-Instruct_summarization_lora}"
QWEN_SUMM_QLORA="${QWEN_SUMM_QLORA:-$RUN_ROOT/Qwen_Qwen2_5-7B-Instruct_summarization_qlora}"
LLAMA_SUMM_LORA="${LLAMA_SUMM_LORA:-$RUN_ROOT/meta-llama_Meta-Llama-3_1-8B-Instruct_summarization_lora}"
INCLUDE_LLAMA_LORA="${INCLUDE_LLAMA_LORA:-0}"

# Phase 1: finish the remaining Qwen summarization sampling robustness in parallel.
run_decode "$QWEN_SUMM_LORA" "qwen_summarization_lora" &
pid_a=$!
run_decode "$QWEN_SUMM_QLORA" "qwen_summarization_qlora" &
pid_b=$!
wait "$pid_a" "$pid_b"

# Phase 2: close Qwen token-form drift, and optionally add one Llama lane if gated access is available.
if [[ "$INCLUDE_LLAMA_LORA" == "1" ]]; then
  run_decode "$LLAMA_SUMM_LORA" "llama_summarization_lora" &
  pid_c=$!
  run_token_form "$QWEN_SUMM_LORA" "qwen_summarization_lora" &
  pid_d=$!
  wait "$pid_c" "$pid_d"
else
  run_token_form "$QWEN_SUMM_LORA" "qwen_summarization_lora"
fi

# Phase 3: finish remaining token-form drift lanes.
if [[ "$INCLUDE_LLAMA_LORA" == "1" ]]; then
  run_token_form "$QWEN_SUMM_QLORA" "qwen_summarization_qlora" &
  pid_e=$!
  run_token_form "$LLAMA_SUMM_LORA" "llama_summarization_lora" &
  pid_f=$!
  wait "$pid_e" "$pid_f"
else
  run_token_form "$QWEN_SUMM_QLORA" "qwen_summarization_qlora"
fi

# Phase 4: close the remaining stronger-baseline coverage for Qwen summarization.
run_backfill "$QWEN_SUMM_LORA" "qwen_summarization_lora" &
pid_g=$!
run_backfill "$QWEN_SUMM_QLORA" "qwen_summarization_qlora" &
pid_h=$!
wait "$pid_g" "$pid_h"

echo "[$(date -Is)] H100 FOLLOWUP COMPLETE" | tee -a "$LOG_ROOT/queue.log"
