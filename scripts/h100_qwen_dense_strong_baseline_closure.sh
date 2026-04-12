#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zeus/content/upgradeguard}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/h100_qwen_dense_strong_baseline_closure}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/h100_qwen_dense_strong_baseline_closure}"
SEED_RUNS_ROOT="${SEED_RUNS_ROOT:-/home/zeus/content/seed_runs}"

export HF_HOME="${HF_HOME:-/home/zeus/content/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
cd "$ROOT"

artifact_ready() {
  local run_dir="$1"
  [[ -f "$run_dir/utility_metrics.json" ]] && \
  [[ -f "$run_dir/safety_metrics.json" ]] && \
  [[ -f "$run_dir/audit_scores.json" ]] && \
  [[ -f "$run_dir/audit_vs_baselines.json" ]] && \
  [[ -f "$run_dir/external_benchmarks.json" ]] && \
  [[ -f "$run_dir/stronger_baselines.json" ]] && \
  [[ -d "$run_dir/model_artifacts" ]]
}

run_dense_recovery() {
  local task_name="$1"
  local alias="$2"
  local batch_size="$3"
  local generation_batch_size="$4"
  local log_file="$LOG_DIR/${alias}.log"
  local run_dir="$OUTPUT_DIR/Qwen_Qwen2_5-7B-Instruct_${task_name}_full_ft"
  if artifact_ready "$run_dir"; then
    echo "[$(date -Is)] SKIP qwen ${task_name} full_ft dense closure; artifacts already complete" | tee -a "$log_file"
    return 0
  fi
  echo "[$(date -Is)] START qwen ${task_name} full_ft dense closure" | tee -a "$log_file"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PYTHON_BIN" -u -m upgradeguard.main \
    --model qwen \
    --task "$task_name" \
    --method full_ft \
    --output-dir "$OUTPUT_DIR" \
    --save-model-artifacts \
    --batch-size "$batch_size" \
    --generation-batch-size "$generation_batch_size" \
    --run-external-validation \
    --include-strongreject \
    --harmbench-samples 320 \
    --xstest-samples 450 \
    --strongreject-samples 60 \
    --fail-on-condition-error \
    2>&1 | tee -a "$log_file"
  echo "[$(date -Is)] DONE qwen ${task_name} full_ft dense closure" | tee -a "$log_file"
}

run_dense_recovery "translation" "qwen_translation_full_ft_dense_closure" "4" "4"
run_dense_recovery "summarization" "qwen_summarization_full_ft_dense_closure" "1" "2"

run_stronger_backfill() {
  local run_dir="$1"
  local alias="$2"
  local log_file="$LOG_DIR/${alias}.log"
  if [[ ! -d "$run_dir" ]]; then
    echo "[$(date -Is)] SKIP missing backfill run $run_dir" | tee -a "$log_file"
    return 0
  fi
  echo "[$(date -Is)] START stronger baseline backfill $run_dir" | tee -a "$log_file"
  "$PYTHON_BIN" -u scripts/run_stronger_baselines_saved_run.py \
    --run-dir "$run_dir" \
    --cache-root "$OUTPUT_DIR" \
    --device cuda \
    2>&1 | tee -a "$log_file"
  echo "[$(date -Is)] DONE stronger baseline backfill $run_dir" | tee -a "$log_file"
}

run_stronger_backfill "${QWEN_TRANSLATION_LORA_RUN_DIR:-$SEED_RUNS_ROOT/Qwen_Qwen2_5-7B-Instruct_translation_lora}" "qwen_translation_lora_backfill"
run_stronger_backfill "${LLAMA_SUMMARIZATION_QLORA_RUN_DIR:-$SEED_RUNS_ROOT/meta-llama_Meta-Llama-3_1-8B-Instruct_summarization_qlora}" "llama_summarization_qlora_backfill"
run_stronger_backfill "${LLAMA_SUMMARIZATION_LORA_RUN_DIR:-$SEED_RUNS_ROOT/meta-llama_Meta-Llama-3_1-8B-Instruct_summarization_lora}" "llama_summarization_lora_backfill"

echo "[$(date -Is)] H100 QWEN DENSE STRONG-BASELINE CLOSURE COMPLETE" | tee -a "$LOG_DIR/orchestrator.log"
