#!/usr/bin/env bash
# WMT26 blindset collection - heavy/multi-GPU offline models.
# Run after launch-blindset.sh (the single-GPU models) has finished.
#
# GPU plan (8x H100 80GB):
#   Phase 1: qwen3_6_27b (tp=2) and gemma4_31b (tp=2) concurrently
#            launcher assigns GPUs 0-1 and 2-3; GPUs 4-7 idle.
#   Phase 2: mistral_medium_3_5 (tp=8) - full node.
#   Phase 3: gpt_oss_120b (tp=8) - full node.
#
# Durable: outputs/logs live on /mnt/tg blob so a node reset does not lose work.
# Resumable: just re-run this script; resume=True skips rows already written.
set -uo pipefail

# Repo root — derived from this script's own location (mtoc/ inside the repo).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Single exposed external root — everything else is relative to it.
ROOT=/mnt/tg/data/projects/wmt26/genmt-org/evals
RUN_DIR="$ROOT/outputs/blindset-260624"
INPUT="$ROOT/data/wmt26_genmt_blindset.jsonl"

export HF_HUB_CACHE=/mnt/tg/data/cache/huggingface/hub/hf-hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING

cd "$REPO"
mkdir -p "$RUN_DIR/outputs" "$RUN_DIR/logs"

echo "===== blindset-heavy launch at $(date '+%y%m%d %H:%M:%S') (all 8 GPUs) ====="

# --- Phase 1: two tp=2 models concurrently ---
echo "--- Phase 1: qwen3_6_27b + gemma4_31b (tp=2 each, GPUs 0-3) ---"
/opt/venv/bin/python -u mtoc/collect.py launch \
  --input "$INPUT" \
  --models qwen3_6_27b gemma4_31b \
  --variants all --backend auto \
  --gpus 0,1,2,3 --gpu-memory-utilization 0.85 \
  --output-dir "$RUN_DIR/outputs" --log-dir "$RUN_DIR/logs"
status=$?
echo "--- Phase 1 exited (status=$status) at $(date '+%y%m%d %H:%M:%S') ---"
[ $status -ne 0 ] && exit $status

# --- Phase 2: mistral_medium_3_5 (tp=8, full node) ---
echo "--- Phase 2: mistral_medium_3_5 (tp=8, all 8 GPUs) ---"
/opt/venv/bin/python -u mtoc/collect.py launch \
  --input "$INPUT" \
  --models mistral_medium_3_5 \
  --variants all --backend auto \
  --gpus 0,1,2,3,4,5,6,7 --gpu-memory-utilization 0.90 \
  --output-dir "$RUN_DIR/outputs" --log-dir "$RUN_DIR/logs"
status=$?
echo "--- Phase 2 exited (status=$status) at $(date '+%y%m%d %H:%M:%S') ---"
[ $status -ne 0 ] && exit $status

# --- Phase 3: gpt_oss_120b (tp=8, full node) ---
echo "--- Phase 3: gpt_oss_120b (tp=8, all 8 GPUs) ---"
/opt/venv/bin/python -u mtoc/collect.py launch \
  --input "$INPUT" \
  --models gpt_oss_120b \
  --variants all --backend auto \
  --gpus 0,1,2,3,4,5,6,7 --gpu-memory-utilization 0.90 \
  --output-dir "$RUN_DIR/outputs" --log-dir "$RUN_DIR/logs"
status=$?
echo "--- Phase 3 exited (status=$status) at $(date '+%y%m%d %H:%M:%S') ---"

echo "===== blindset-heavy launcher exited (status=$status) at $(date '+%y%m%d %H:%M:%S') ====="
exit "$status"
