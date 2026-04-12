#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zeus/content/upgradeguard}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ROOT="${RUN_ROOT:-/home/zeus/content/seed_runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/l4_qwen_remaining_followup}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/l4_qwen_remaining_followup}"
PY_USER_SITE="${PY_USER_SITE:-/teamspace/studios/this_studio/.local/lib/python3.12/site-packages}"
PY_INCLUDE_ROOT="${PY_INCLUDE_ROOT:-/home/zeus/pyheaders/extracted/usr/include}"
PY_INCLUDE_MAIN="${PY_INCLUDE_MAIN:-/home/zeus/pyheaders/extracted/usr/include/python3.12}"
PY_INCLUDE_ARCH="${PY_INCLUDE_ARCH:-/home/zeus/pyheaders/extracted/usr/include/x86_64-linux-gnu/python3.12}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$ROOT"

export PYTHONPATH="$PY_USER_SITE${PYTHONPATH:+:$PYTHONPATH}"
export CPATH="$PY_INCLUDE_ROOT:$PY_INCLUDE_MAIN:$PY_INCLUDE_ARCH${CPATH:+:$CPATH}"

run_sampling_eval() {
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

run_token_form_eval() {
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

QWEN_SUMMARIZATION_LORA_RUN_DIR="${QWEN_SUMMARIZATION_LORA_RUN_DIR:-$RUN_ROOT/Qwen_Qwen2_5-7B-Instruct_summarization_lora}"
QWEN_SUMMARIZATION_QLORA_RUN_DIR="${QWEN_SUMMARIZATION_QLORA_RUN_DIR:-$RUN_ROOT/Qwen_Qwen2_5-7B-Instruct_summarization_qlora}"

# Broaden robustness coverage to the Qwen summarization method panel.
run_sampling_eval "$QWEN_SUMMARIZATION_LORA_RUN_DIR" "qwen_summarization_lora"
run_sampling_eval "$QWEN_SUMMARIZATION_QLORA_RUN_DIR" "qwen_summarization_qlora"
run_token_form_eval "$QWEN_SUMMARIZATION_LORA_RUN_DIR" "qwen_summarization_lora"
run_token_form_eval "$QWEN_SUMMARIZATION_QLORA_RUN_DIR" "qwen_summarization_qlora"

echo "[$(date -Is)] L4 QWEN FOLLOWUP COMPLETE" | tee -a "$LOG_ROOT/queue.log"
