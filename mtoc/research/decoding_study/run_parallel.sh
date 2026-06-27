#!/usr/bin/env bash
# Decoding-strategy study launcher — GNU parallel edition.
#
# Generates one job per (model, variant) up front, then schedules them all through
# GNU parallel with one GPU per slot. As soon as a GPU slot frees, the next buffered
# job starts — so a slow beam variant only holds its own slot while fast sampling
# variants keep cycling through the other GPUs.
#
# Models: qwen3_5_9b, glm4_9b, ministral3_14b (all tp=1) x 14 decoding variants = 42 jobs.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"

ROOT=/mnt/tg/data/projects/wmt26/llm-mt-dec
INPUT="$ROOT/data/decoding-study.jsonl"
RUN_DIR="$ROOT/outputs/decoding-study"
REGISTRY="$HERE/registry.yaml"

# GPUs to use (one job per GPU). Override by exporting GPUS before calling.
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NGPU="${#GPU_ARR[@]}"

export HF_HUB_CACHE=/mnt/tg/data/cache/huggingface/hub/hf-hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING

cd "$REPO"
mkdir -p "$RUN_DIR/outputs" "$RUN_DIR/logs"

JOBFILE="$RUN_DIR/jobs.txt"
echo "===== decoding-study (gnu parallel) at $(date '+%y%m%d %H:%M:%S') ====="
echo "Generating jobs -> $JOBFILE"
/opt/venv/bin/python "$HERE/gen_jobs.py" \
  --registry "$REGISTRY" \
  --input "$INPUT" \
  --output-dir "$RUN_DIR/outputs" \
  --log-dir "$RUN_DIR/logs" \
  --models all --variants all \
  --gpu-memory-utilization 0.85 \
  > "$JOBFILE"

NJOBS=$(wc -l < "$JOBFILE")
echo "Scheduling $NJOBS jobs across $NGPU GPU slot(s): [$GPUS]"

# GNU parallel, one GPU per slot (idiom from notes/docs/unix.adoc):
#   CUDA_VISIBLE_DEVICES='$(({%} - 1))'  -> slot 1->GPU0, slot 2->GPU1, ...
#     single-quoted so the shell evaluates the arithmetic AFTER parallel
#     substitutes the slot number {%}.
#   bash -c "{}"  -> run each job line as a full command.
#   --joblog      -> durable record; rerun with `--resume` to continue.
#   --progress    -> live slot/eta summary.
# NOTE: assumes GPUs 0..NGPU-1. For a custom subset, set CUDA_VISIBLE_DEVICES in
# the environment before calling and reduce NGPU accordingly.
parallel --will-cite --progress -j "$NGPU" \
  --joblog "$RUN_DIR/parallel.joblog" --line-buffer \
  CUDA_VISIBLE_DEVICES='$(({%} - 1))' bash -c "{}" \
  < "$JOBFILE"
status=$?

echo "===== decoding-study parallel exited (status=$status) at $(date '+%y%m%d %H:%M:%S') ====="
exit "$status"
