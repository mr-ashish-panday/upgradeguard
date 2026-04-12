#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zeus/content/upgradeguard}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/h100_partial_unfreeze_repairs}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/h100_partial_unfreeze_repairs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
cd "$ROOT_DIR"

export HF_HOME="${HF_HOME:-/home/zeus/content/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

resolve_run_dir() {
  local model="$1"
  local task="$2"
  "$PYTHON_BIN" - "$OUTPUT_DIR" "$model" "$task" <<'PY'
from pathlib import Path
import sys
from upgradeguard import config
from upgradeguard.main import run_dir_for

output_dir, model_arg, task_name = sys.argv[1:]
model_name = config.resolve_model_name(model_arg)
print(run_dir_for(Path(output_dir), model_name, task_name, "partial_unfreeze"))
PY
}

wait_for_upgradeguard_idle() {
  while pgrep -af "python.*-m upgradeguard.main" >/dev/null || pgrep -af "h100_conditioned_method_panels.sh" >/dev/null; do
    echo "[$(date -Is)] Waiting for conditioned-method queue to finish before partial_unfreeze repair..."
    sleep 30
  done
}

validate_run_artifacts() {
  local run_dir="$1"
  local log_file="$2"
  local error_path="$run_dir/error.json"
  if [[ -f "$error_path" ]]; then
    echo "[$(date -Is)] ERROR artifact found at $error_path" | tee -a "$log_file"
    cat "$error_path" | tee -a "$log_file"
    return 1
  fi

  local required=(
    "training_summary.json"
    "utility_metrics.json"
    "safety_metrics.json"
    "audit_scores.json"
    "layer_drift.json"
    "audit_vs_baselines.json"
    "run_manifest.json"
    "external_benchmarks.json"
  )
  local missing=0
  for filename in "${required[@]}"; do
    if [[ ! -f "$run_dir/$filename" ]]; then
      echo "[$(date -Is)] MISSING $run_dir/$filename" | tee -a "$log_file"
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    return 1
  fi
}

run_is_complete() {
  local run_dir="$1"
  [[ ! -f "$run_dir/error.json" ]] || return 1
  local required=(
    "training_summary.json"
    "utility_metrics.json"
    "safety_metrics.json"
    "audit_scores.json"
    "layer_drift.json"
    "audit_vs_baselines.json"
    "run_manifest.json"
    "external_benchmarks.json"
  )
  for filename in "${required[@]}"; do
    [[ -f "$run_dir/$filename" ]] || return 1
  done
}

run_condition() {
  local model="$1"
  local task="$2"
  local alias
  alias="$(echo "${model}_${task}_partial_unfreeze" | tr '/.' '__')"
  local log_file="$LOG_DIR/${alias}.log"
  local run_dir
  run_dir="$(resolve_run_dir "$model" "$task")"
  if run_is_complete "$run_dir"; then
    echo "[$(date -Is)] SKIP $model $task partial_unfreeze (artifacts already complete)" | tee -a "$log_file"
    return 0
  fi
  echo "[$(date -Is)] START $model $task partial_unfreeze" | tee -a "$log_file"
  "$PYTHON_BIN" -u -m upgradeguard.main \
    --model "$model" \
    --task "$task" \
    --method partial_unfreeze \
    --output-dir "$OUTPUT_DIR" \
    --run-external-validation \
    --skip-posthoc \
    --include-strongreject \
    --harmbench-samples 320 \
    --xstest-samples 450 \
    --strongreject-samples 60 \
    --fail-on-condition-error \
    2>&1 | tee -a "$log_file"
  validate_run_artifacts "$run_dir" "$log_file"
  echo "[$(date -Is)] DONE $model $task partial_unfreeze" | tee -a "$log_file"
}

wait_for_upgradeguard_idle
run_condition "qwen" "translation"
run_condition "llama" "summarization"
