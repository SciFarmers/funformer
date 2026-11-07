#!/bin/bash
# Submit pig FunFormer 20k training jobs: 4 LR = 4 total jobs.
# WandB project: pig
# Baseline jobs: see 3b.pig_jobs.sh
#
# Usage: bash 4b.pig_funformer_jobs.sh

set -euo pipefail

LRS=(1e-5 5e-5 1e-4 5e-4)

FUNFORMER_SUBMITTER="olivia_submiter/funformer_pig.sh"

# --- FunFormer 20k ---
for LR in "${LRS[@]}"; do
    echo "Submitting FunFormer 20k: LEARNING_RATE=${LR}"
    sbatch --export=ALL,LEARNING_RATE="${LR}" \
           --job-name="funFormer_pig_20k_${LR}" \
           --output="slurm_%j_funformer_pig_len20000_lr${LR}.out" \
           "${FUNFORMER_SUBMITTER}"
done

echo ""
echo "All ${#LRS[@]} FunFormer 20k jobs submitted for pig."
echo "  Learning rates : ${LRS[*]}"
echo "  WandB project  : pig"
