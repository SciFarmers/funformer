#!/bin/bash
# Submit cattle FunFormer 20k training jobs: 4 LR = 4 total jobs.
# WandB project: cattle
# Baseline jobs: see 3a.cattle_jobs.sh
#
# Usage: bash 4a.cattle_funformer_jobs.sh

set -euo pipefail

LRS=(1e-5 5e-5 1e-4 5e-4)

FUNFORMER_SUBMITTER="olivia_submiter/funformer_cattle.sh"

# --- FunFormer 20k ---
for LR in "${LRS[@]}"; do
    echo "Submitting FunFormer 20k: LEARNING_RATE=${LR}"
    sbatch --export=ALL,LEARNING_RATE="${LR}" \
           --job-name="funFormer_cattle_20k_${LR}" \
           --output="slurm_%j_funformer_cattle_len20000_lr${LR}.out" \
           "${FUNFORMER_SUBMITTER}"
done

echo ""
echo "All ${#LRS[@]} FunFormer 20k jobs submitted for cattle."
echo "  Learning rates : ${LRS[*]}"
echo "  WandB project  : cattle"
