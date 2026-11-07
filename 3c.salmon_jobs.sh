#!/bin/bash
# Submit salmon baseline training jobs: DeepSEA, DeepFormer, DeepATT x 4 LR = 12 total jobs.
# WandB project: salmon
# FunFormer jobs: see 4c.salmon_funformer_jobs.sh
#
# Usage: bash 3c.salmon_jobs.sh

set -euo pipefail

MODELS=(DeepSEA DeepFormer DeepATT)
LRS=(1e-5 5e-5 1e-4 5e-4)

BASELINE_SUBMITTER="olivia_submiter/baseline_salmon.sh"

# --- Baseline models ---
for MODEL in "${MODELS[@]}"; do
    for LR in "${LRS[@]}"; do
        echo "Submitting baseline: MODEL_CLASS=${MODEL}  LEARNING_RATE=${LR}"
        sbatch --export=ALL,MODEL_CLASS="${MODEL}",LEARNING_RATE="${LR}" \
               --job-name="${MODEL}_salmon_${LR}" \
               --output="slurm_%j_${MODEL}_salmon_${LR}_train.out" \
               "${BASELINE_SUBMITTER}"
    done
done

echo ""
echo "All $((${#MODELS[@]} * ${#LRS[@]})) baseline jobs submitted for salmon."
echo "  Baseline models : ${MODELS[*]}"
echo "  Learning rates  : ${LRS[*]}"
echo "  WandB project   : salmon"
