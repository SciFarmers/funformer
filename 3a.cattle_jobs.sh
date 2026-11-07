#!/bin/bash
# Submit cattle baseline training jobs: DeepSEA, DeepFormer, DeepATT x 4 LR = 12 total jobs.
# WandB project: cattle
# FunFormer jobs: see 4a.cattle_funformer_jobs.sh
#
# Usage: bash 3a.cattle_jobs.sh

set -euo pipefail

MODELS=(DeepSEA DeepFormer DeepATT)
LRS=(1e-5 5e-5 1e-4 5e-4)

BASELINE_SUBMITTER="olivia_submiter/baseline_cattle.sh"

# --- Baseline models ---
for MODEL in "${MODELS[@]}"; do
    for LR in "${LRS[@]}"; do
        echo "Submitting baseline: MODEL_CLASS=${MODEL}  LEARNING_RATE=${LR}"
        sbatch --export=ALL,MODEL_CLASS="${MODEL}",LEARNING_RATE="${LR}" \
               --job-name="${MODEL}_cattle_${LR}" \
               --output="slurm_%j_${MODEL}_cattle_${LR}_train.out" \
               "${BASELINE_SUBMITTER}"
    done
done

echo ""
echo "All $((${#MODELS[@]} * ${#LRS[@]})) baseline jobs submitted for cattle."
echo "  Baseline models : ${MODELS[*]}"
echo "  Learning rates  : ${LRS[*]}"
echo "  WandB project   : cattle"
