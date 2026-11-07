#!/bin/bash
# Submit pig baseline training jobs: DeepSEA, DeepFormer, DeepATT x 4 LR = 12 total jobs.
# WandB project: pig
# FunFormer jobs: see 4b.pig_funformer_jobs.sh
#
# Usage: bash 3b.pig_jobs.sh

set -euo pipefail

MODELS=(DeepSEA DeepFormer DeepATT)
LRS=(1e-5 5e-5 1e-4 5e-4)

BASELINE_SUBMITTER="olivia_submiter/baseline_pig.sh"

# --- Baseline models ---
for MODEL in "${MODELS[@]}"; do
    for LR in "${LRS[@]}"; do
        echo "Submitting baseline: MODEL_CLASS=${MODEL}  LEARNING_RATE=${LR}"
        sbatch --export=ALL,MODEL_CLASS="${MODEL}",LEARNING_RATE="${LR}" \
               --job-name="${MODEL}_pig_${LR}" \
               --output="slurm_%j_${MODEL}_pig_${LR}_train.out" \
               "${BASELINE_SUBMITTER}"
    done
done

echo ""
echo "All $((${#MODELS[@]} * ${#LRS[@]})) baseline jobs submitted for pig."
echo "  Baseline models : ${MODELS[*]}"
echo "  Learning rates  : ${LRS[*]}"
echo "  WandB project   : pig"
