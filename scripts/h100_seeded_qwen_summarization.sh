#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zeus/content/upgradeguard}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/h100_qwen_summarization_seeded}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/h100_qwen_summarization_seeded}"
PLAN="${PLAN:-focused}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$ROOT"

export HF_HOME="${HF_HOME:-/home/zeus/content/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

artifacts_ready() {
  local run_dir="$1"
  [[ -f "$run_dir/training_summary.json" ]] && \
  [[ -f "$run_dir/utility_metrics.json" ]] && \
  [[ -f "$run_dir/safety_metrics.json" ]] && \
  [[ -f "$run_dir/audit_scores.json" ]] && \
  [[ -f "$run_dir/layer_drift.json" ]] && \
  [[ -f "$run_dir/audit_vs_baselines.json" ]] && \
  [[ -f "$run_dir/run_manifest.json" ]] && \
  [[ -f "$run_dir/external_benchmarks.json" ]]
}

resolve_run_dir() {
  local seed_output="$1"
  local method="$2"
  "$PYTHON_BIN" - "$seed_output" "$method" <<'PY'
from pathlib import Path
import sys
from upgradeguard import config
from upgradeguard.main import run_dir_for

seed_output, method = sys.argv[1:]
model_name = config.resolve_model_name("qwen")
print(run_dir_for(Path(seed_output), model_name, "summarization", method))
PY
}

run_seeded_condition() {
  local method="$1"
  local seed="$2"
  local batch_size="$3"
  local generation_batch_size="$4"
  local seed_output="$OUTPUT_ROOT/seed_${seed}"
  local run_dir
  run_dir="$(resolve_run_dir "$seed_output" "$method")"
  local alias="qwen_summarization_${method}_seed_${seed}"
  local log_file="$LOG_ROOT/${alias}.log"

  mkdir -p "$seed_output"
  if artifacts_ready "$run_dir"; then
    echo "[$(date -Is)] SKIP $alias (artifacts already complete)" | tee -a "$log_file"
    return 0
  fi

  echo "[$(date -Is)] START $alias" | tee -a "$log_file"
  "$PYTHON_BIN" -u -m upgradeguard.main \
    --model qwen \
    --task summarization \
    --method "$method" \
    --output-dir "$seed_output" \
    --seed "$seed" \
    --batch-size "$batch_size" \
    --generation-batch-size "$generation_batch_size" \
    --run-external-validation \
    --include-strongreject \
    --harmbench-samples 320 \
    --xstest-samples 450 \
    --strongreject-samples 60 \
    --skip-posthoc \
    --fail-on-condition-error \
    2>&1 | tee -a "$log_file"

  if artifacts_ready "$run_dir"; then
    echo "[$(date -Is)] DONE $alias" | tee -a "$log_file"
  else
    echo "[$(date -Is)] INCOMPLETE $alias" | tee -a "$log_file"
    exit 1
  fi
}

declare -a PLAN_RUNS
case "$PLAN" in
  focused)
    PLAN_RUNS=(
      "full_ft:43:2:2"
      "full_ft:44:2:2"
      "full_ft:45:2:2"
      "full_ft:46:2:2"
      "qlora:43:4:4"
      "qlora:44:4:4"
      "qlora:45:4:4"
      "qlora:46:4:4"
      "lora:43:4:4"
      "lora:44:4:4"
      "partial_unfreeze:43:4:4"
      "partial_unfreeze:44:4:4"
    )
    ;;
  uniform3)
    PLAN_RUNS=(
      "full_ft:43:2:2"
      "full_ft:44:2:2"
      "qlora:43:4:4"
      "qlora:44:4:4"
      "lora:43:4:4"
      "lora:44:4:4"
      "partial_unfreeze:43:4:4"
      "partial_unfreeze:44:4:4"
    )
    ;;
  *)
    echo "Unknown PLAN=$PLAN. Supported plans: focused, uniform3" >&2
    exit 1
    ;;
esac

for spec in "${PLAN_RUNS[@]}"; do
  IFS=":" read -r method seed batch_size generation_batch_size <<< "$spec"
  run_seeded_condition "$method" "$seed" "$batch_size" "$generation_batch_size"
done

echo "[$(date -Is)] H100 SEEDED QWEN SUMMARIZATION COMPLETE (plan=$PLAN)" | tee -a "$LOG_ROOT/orchestrator.log"
