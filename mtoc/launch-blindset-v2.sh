#!/usr/bin/env bash
# WMT26 blindset collection v2 — all 10 offline models.
#
# GPU plan (8x H100 80GB):
#   Step 1: 6 single-GPU models (tp=1) run CONCURRENTLY across all 8 GPUs.
#           Launcher assigns one GPU per model; 2 GPUs left idle.
#           Expected time: ~40 min (was 3.5h sequential).
#   Step 2: qwen3_6_27b + gemma4_31b (tp=2 each) concurrently on GPUs 0-3.
#   Step 3: mistral_medium_3_5 (tp=8) — full node.
#   Step 4: gpt_oss_120b (tp=8, max_num_seqs=64) — full node.
#
# Durable: outputs/logs live on /mnt/tg blob — safe across node resets.
# Resumable: re-running this script resumes from where each model left off.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROOT=/mnt/tg/data/projects/wmt26/genmt-org/evals
RUN_DIR="$ROOT/outputs/blindset-260625"
INPUT="$ROOT/data/wmt26_genmt_blindset.jsonl"

export HF_HUB_CACHE=/mnt/tg/data/cache/huggingface/hub/hf-hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING

cd "$REPO"
mkdir -p "$RUN_DIR/outputs" "$RUN_DIR/logs"

PYTHON=/opt/venv/bin/python

echo "===== blindset-v2 launch at $(date '+%y%m%d %H:%M:%S') ====="

# --- Step 1: 6 tp=1 models concurrently across all 8 GPUs ---
echo "--- Step 1: 6 single-GPU models (tp=1) concurrent on GPUs 0-7 ---"
$PYTHON -u mtoc/collect.py launch \
  --input "$INPUT" \
  --models qwen3_5_9b gemma4_e4b ministral3_14b gpt_oss_20b tower_9b glm4_9b \
  --variants all --backend auto \
  --gpus 0,1,2,3,4,5,6,7 --gpu-memory-utilization 0.85 \
  --output-dir "$RUN_DIR/outputs" --log-dir "$RUN_DIR/logs"
status=$?
echo "--- Step 1 exited (status=$status) at $(date '+%y%m%d %H:%M:%S') ---"
[ $status -ne 0 ] && exit $status

# --- Step 2: two tp=2 models concurrently ---
echo "--- Step 2: qwen3_6_27b + gemma4_31b (tp=2 each, GPUs 0-3) ---"
$PYTHON -u mtoc/collect.py launch \
  --input "$INPUT" \
  --models qwen3_6_27b gemma4_31b \
  --variants all --backend auto \
  --gpus 0,1,2,3 --gpu-memory-utilization 0.85 \
  --output-dir "$RUN_DIR/outputs" --log-dir "$RUN_DIR/logs"
status=$?
echo "--- Step 2 exited (status=$status) at $(date '+%y%m%d %H:%M:%S') ---"
[ $status -ne 0 ] && exit $status

# --- Step 3: mistral_medium_3_5 (tp=8, full node) ---
echo "--- Step 3: mistral_medium_3_5 (tp=8, all 8 GPUs) ---"
$PYTHON -u mtoc/collect.py launch \
  --input "$INPUT" \
  --models mistral_medium_3_5 \
  --variants all --backend auto \
  --gpus 0,1,2,3,4,5,6,7 --gpu-memory-utilization 0.90 \
  --output-dir "$RUN_DIR/outputs" --log-dir "$RUN_DIR/logs"
status=$?
echo "--- Step 3 exited (status=$status) at $(date '+%y%m%d %H:%M:%S') ---"
[ $status -ne 0 ] && exit $status

# --- Step 4: gpt_oss_120b (tp=8, full node, max_num_seqs=64) ---
echo "--- Step 4: gpt_oss_120b (tp=8, all 8 GPUs, batch=64) ---"
$PYTHON -u mtoc/collect.py launch \
  --input "$INPUT" \
  --models gpt_oss_120b \
  --variants all --backend auto \
  --gpus 0,1,2,3,4,5,6,7 --gpu-memory-utilization 0.90 \
  --output-dir "$RUN_DIR/outputs" --log-dir "$RUN_DIR/logs"
status=$?
echo "--- Step 4 exited (status=$status) at $(date '+%y%m%d %H:%M:%S') ---"

echo "===== blindset-v2 launcher exited (status=$status) at $(date '+%y%m%d %H:%M:%S') ====="
exit "$status"
