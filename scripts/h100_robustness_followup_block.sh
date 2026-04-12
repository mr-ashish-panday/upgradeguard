#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zeus/content/upgradeguard}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/h100_conditioned_method_panels}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/h100_robustness_followup}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/h100_robustness_followup}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$ROOT"

run_decode_regime() {
  local run_dir="$1"
  local alias="$2"
  local log_file="$LOG_ROOT/${alias}_sampling.log"
  if [[ ! -d "$run_dir" ]]; then
    echo "[$(date -Is)] SKIP missing run dir $run_dir" | tee -a "$log_file"
    return 0
  fi
  echo "[$(date -Is)] START sampling external eval $run_dir" | tee -a "$log_file"
  "$PYTHON_BIN" -u scripts/run_external_decode_regime.py \
    --run-dir "$run_dir" \
    --output-dir "$OUTPUT_ROOT/${alias}/sampling_external" \
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
  if [[ ! -d "$run_dir" ]]; then
    echo "[$(date -Is)] SKIP missing run dir $run_dir" | tee -a "$log_file"
    return 0
  fi
  echo "[$(date -Is)] START token-form drift eval $run_dir" | tee -a "$log_file"
  "$PYTHON_BIN" -u scripts/run_token_form_drift_eval.py \
    --run-dir "$run_dir" \
    --output-dir "$OUTPUT_ROOT/${alias}/token_form_drift" \
    --prompt-bank canaries \
    --device cuda \
    2>&1 | tee -a "$log_file"
  echo "[$(date -Is)] DONE token-form drift eval $run_dir" | tee -a "$log_file"
}

run_decode_regime "${QWEN_TRANSLATION_FULL_FT_RUN_DIR:-$RUN_ROOT/Qwen_Qwen2_5-7B-Instruct_translation_full_ft}" "qwen_translation_full_ft"
run_decode_regime "${QWEN_TRANSLATION_LORA_RUN_DIR:-$RUN_ROOT/Qwen_Qwen2_5-7B-Instruct_translation_lora}" "qwen_translation_lora"
run_decode_regime "${QWEN_TRANSLATION_QLORA_RUN_DIR:-$RUN_ROOT/Qwen_Qwen2_5-7B-Instruct_translation_qlora}" "qwen_translation_qlora"
run_decode_regime "${LLAMA_SUMMARIZATION_LORA_RUN_DIR:-$RUN_ROOT/meta-llama_Meta-Llama-3_1-8B-Instruct_summarization_lora}" "llama_summarization_lora"
run_decode_regime "${LLAMA_SUMMARIZATION_QLORA_RUN_DIR:-$RUN_ROOT/meta-llama_Meta-Llama-3_1-8B-Instruct_summarization_qlora}" "llama_summarization_qlora"

run_token_form "${QWEN_TRANSLATION_FULL_FT_RUN_DIR:-$RUN_ROOT/Qwen_Qwen2_5-7B-Instruct_translation_full_ft}" "qwen_translation_full_ft"
run_token_form "${QWEN_TRANSLATION_LORA_RUN_DIR:-$RUN_ROOT/Qwen_Qwen2_5-7B-Instruct_translation_lora}" "qwen_translation_lora"
run_token_form "${QWEN_TRANSLATION_QLORA_RUN_DIR:-$RUN_ROOT/Qwen_Qwen2_5-7B-Instruct_translation_qlora}" "qwen_translation_qlora"
run_token_form "${LLAMA_SUMMARIZATION_LORA_RUN_DIR:-$RUN_ROOT/meta-llama_Meta-Llama-3_1-8B-Instruct_summarization_lora}" "llama_summarization_lora"
run_token_form "${LLAMA_SUMMARIZATION_QLORA_RUN_DIR:-$RUN_ROOT/meta-llama_Meta-Llama-3_1-8B-Instruct_summarization_qlora}" "llama_summarization_qlora"
