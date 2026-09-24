#!/bin/bash
# Safe test runner: one process, timeout, memory guard
set -e
cd "$(dirname "$0")"

CKPT="models/LFM2.5-1.2B-Instruct/evagi_live.pt"
# Warn if checkpoint looks like old dense format
if [ -f "$CKPT" ]; then
  SZ=$(stat -c%s "$CKPT")
  if [ "$SZ" -gt 10485760 ]; then
    echo "WARNING: $CKPT is $((SZ/1024/1024))MB — likely old dense format, deleting"
    rm -f "$CKPT"
  fi
fi

# Kill any leftover python from previous runs
pkill -f "live_interactive_lfm2" 2>/dev/null || true
sleep 1

echo "=== Memory before ==="
free -h | head -2
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo "=== Running (timeout ${TIMEOUT:-300}s) ==="
timeout "${TIMEOUT:-300}" .venv/bin/python src/live_interactive_lfm2.py
EXIT=$?

echo "=== Exit: $EXIT ==="
free -h | head -2
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

# Clean GPU memory
sleep 2
nvidia-smi --query-gpu=memory.used --format=csv,noheader
exit $EXIT
