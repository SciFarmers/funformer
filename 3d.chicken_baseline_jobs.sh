#!/bin/bash
# Submit chicken baseline training jobs: DeepSEA, DeepFormer, DeepATT x 4 LR = 12 jobs.
# WandB project: chicken
# FunFormer: see 4d.chicken_funformer.sh
#
# Usage: bash 3d.chicken_baseline_jobs.sh

set -euo pipefail
mkdir -p logs

MODELS=(DeepSEA DeepFormer DeepATT)
LRS=(1e-5 5e-5 1e-4 5e-4)

for MODEL in "${MODELS[@]}"; do
    for LR in "${LRS[@]}"; do
        echo "Submitting: MODEL_CLASS=${MODEL}  LR=${LR}"
        sbatch --export=ALL,MODEL_CLASS="${MODEL}",LEARNING_RATE="${LR}" \
               --job-name="${MODEL}_chicken_${LR}" \
               --output="logs/slurm_%j_${MODEL}_chicken_${LR}_train.out" \
               olivia_submiter/baseline_chicken.sh
    done
done

echo ""
echo "Submitted $((${#MODELS[@]} * ${#LRS[@]})) chicken baseline jobs."
