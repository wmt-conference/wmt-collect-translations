#!/usr/bin/env bash
# Run the WMT26 collection. Just run it:
#
#   mtoc/run.sh
#
# Starts the job in a tmux session and attaches you to it. If the session is
# already running (e.g. you detached with Ctrl-b d), it just re-attaches.
# Each model runs in its own subprocess (GPU freed between models), resume is
# enabled, and all output is teed to outputs/wmt26-release/launch.<stamp>.log.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

SESSION="mtoc-release"
INPUT="/mnt/tg/data/projects/wmt26/genmt-org/evals/data/release.jsonl"
OUTPUT_DIR="$HERE/outputs/wmt26-release"
GPUS="0,1,2,3,4,5,6,7"

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  mkdir -p "$OUTPUT_DIR"
  LOG_FILE="$OUTPUT_DIR/launch.$(date +%Y%m%d_%H%M%S).log"
  read -r -d '' CMD <<EOF || true
cd "$HERE"
export HF_HUB_CACHE="$HERE/models/hf-hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING PYTHONUNBUFFERED=1
{
  echo "input=$INPUT"; echo "output_dir=$OUTPUT_DIR"; echo "gpus=$GPUS"; echo "started=\$(date -Is)"
  /opt/venv/bin/python mtoc/collect.py launch --input "$INPUT" --models all --gpus "$GPUS" --output-dir "$OUTPUT_DIR" --log-dir "$OUTPUT_DIR" --gpu-memory-utilization 0.85
  echo "finished=\$(date -Is)"
} 2>&1 | tee "$LOG_FILE"
EOF
  tmux new-session -d -s "$SESSION" "$CMD"
fi

exec tmux attach -t "$SESSION"
