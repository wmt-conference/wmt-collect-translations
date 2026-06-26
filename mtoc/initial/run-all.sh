#!/usr/bin/env bash
set -euo pipefail
# Copy to $HOME/work/local-llm-translation-collector where filesystem is faster (local).
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR="$HOME/work/local-llm-translation-collector"
mkdir -p "${DIR}"
rsync -a --exclude='__pycache__' --exclude='.history' --exclude='*_outputs' --exclude='*_logs' "${SRC_DIR}/" "${DIR}/"
cd "${DIR}"

PYTHON="${PYTHON_BIN:-$(which python)}"
INPUT="${INPUT_JSONL:-wmt26.test.sampling.jsonl}"
RUN="${RUN_NAME:-genmt}"
OUT="${OUTPUT_DIR:-$HOME/work/${RUN}_outputs}"
LOG="${LOG_DIR:-$HOME/work/${RUN}_logs}"
BS="${BATCH_SIZE:-8}"
MAXTOK="${MAX_NEW_TOKENS:-1024}"
NGPUS="${NGPUS:-8}"
NLLB_GPUS="${NLLB_GPUS:-0,1,2,3,4,5,6,7}"

TOWER_TARGETS="${TOWER_TARGETS:-zh_Hans zh_Hant_TW cs de it ja ko is}"
AYA_TARGETS="${AYA_TARGETS:-zh_Hans zh_Hant_TW cs de it ja ko id}"
NLLB_TARGETS="${NLLB_TARGETS:-zh_Hans zh_Hant_TW cs de it ja ko is id}"

LOGDIR="${LOG}/launcher"
mkdir -p "${LOGDIR}"

translate="PYTHONUNBUFFERED=1 ${PYTHON} translate.py run --input ${INPUT} --batch-size ${BS} --max-new-tokens ${MAXTOK} --output-dir ${OUT} --log-dir ${LOG}"

echo "Run: ${RUN} | Output: ${OUT} | GPUs: ${NGPUS}"

# --- Stage 1: Tower + Aya, 1 GPU per job ---
echo "=== Stage 1: Tower + Aya (${NGPUS} GPU slots) ==="

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    for t in $TOWER_TARGETS; do echo "  tower $t"; done
    for t in $AYA_TARGETS; do echo "  aya $t"; done
else
    (
        for t in $TOWER_TARGETS; do
            echo "${translate} --models tower --targets $t > ${LOGDIR}/tower.${t}.log 2>&1"
        done
        for t in $AYA_TARGETS; do
            echo "${translate} --models aya --targets $t > ${LOGDIR}/aya.${t}.log 2>&1"
        done
    ) | parallel --progress --halt soon,fail=1 -j"${NGPUS}" \
        --joblog "${LOGDIR}/stage1.joblog" \
        'CUDA_VISIBLE_DEVICES=$(({%} - 1))' 'HIP_VISIBLE_DEVICES=$(({%} - 1))' bash -c "{}"
fi

# --- Stage 2: NLLB, all GPUs, sequential ---
echo "=== Stage 2: NLLB (GPUs: ${NLLB_GPUS}, sequential) ==="

for t in $NLLB_TARGETS; do
    echo "[nllb] $t"
    cmd="${translate} --models nllb --targets $t"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "  CUDA_VISIBLE_DEVICES=${NLLB_GPUS} HIP_VISIBLE_DEVICES=${NLLB_GPUS} ${cmd}"
    else
        CUDA_VISIBLE_DEVICES="${NLLB_GPUS}" HIP_VISIBLE_DEVICES="${NLLB_GPUS}" bash -c "${cmd}" \
            > "${LOGDIR}/nllb.${t}.log" 2>&1
    fi
done

echo "=== Done ==="
