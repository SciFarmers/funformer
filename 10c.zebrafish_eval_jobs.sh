#!/bin/bash
# Within-species test-set evaluation for every zebrafish checkpoint:
# 3 baselines x 4 LRs + FunFormer at 20 kb and 1 kb x 4 LRs = 20 jobs.
# Idempotent: skips checkpoints already evaluated or with an eval in flight.
#
# Usage: bash 10c.zebrafish_eval_jobs.sh
set -euo pipefail

LRS=(1e-5 5e-5 1e-4 5e-4)
SUBMITTER="olivia_submiter/evaluate_test_zebrafish.sh"
mkdir -p logs

CKPTS=()
for M in DeepSEA DeepFormer DeepATT; do
    for LR in "${LRS[@]}"; do CKPTS+=("checkpoints/${M}_zebrafish_${LR}_1000"); done
done
for LEN in 20000 1000; do
    B=250; G=2; [ "$LEN" = "1000" ] && { B=500; G=1; }
    for LR in "${LRS[@]}"; do
        CKPTS+=("checkpoints/funformer_zebrafish_len${LEN}_lr${LR}_b${B}_gac${G}")
    done
done

n=0; skip=0
for CKPT in "${CKPTS[@]}"; do
    TAG="$(basename "${CKPT}")"
    if [[ ! -d "${CKPT}/best_model" ]]; then
        echo "SKIP (training not finished): ${TAG}" >&2; skip=$((skip+1)); continue
    fi
    if [[ -f "eval_results/test_set/${TAG}_summary.csv" ]]; then
        echo "SKIP (already evaluated): ${TAG}"; continue
    fi
    if squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qx "eval_zf_${TAG}"; then
        echo "SKIP (eval already queued): ${TAG}"; continue
    fi
    echo "Submitting eval: ${TAG}"
    sbatch --export=ALL,CHECKPOINT_DIR="${CKPT}" \
           --job-name="eval_zf_${TAG}" \
           --output="logs/eval_zebrafish_${TAG}_%j.out" \
           "${SUBMITTER}"
    n=$((n+1))
done
echo ""
echo "${n} zebrafish evaluation job(s) submitted; ${skip} skipped (training not finished)."
