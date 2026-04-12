#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zeus/content/upgradeguard}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/h100_partial_unfreeze_repairs}"
REPAIR_SCRIPT="${REPAIR_SCRIPT:-$ROOT_DIR/scripts/h100_partial_unfreeze_repairs.sh}"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"
WATCH_LOG="$LOG_DIR/watcher.log"

echo "[$(date -Is)] Waiting for active upgradeguard jobs before launching partial_unfreeze repairs..." | tee -a "$WATCH_LOG"
while pgrep -af "python.*-m upgradeguard.main" >/dev/null || pgrep -af "h100_conditioned_method_panels.sh" >/dev/null; do
  sleep 30
done

echo "[$(date -Is)] Main queue is idle. Launching partial_unfreeze repairs." | tee -a "$WATCH_LOG"
bash "$REPAIR_SCRIPT" 2>&1 | tee -a "$WATCH_LOG"
