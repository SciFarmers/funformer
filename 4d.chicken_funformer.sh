#!/bin/bash
# Submit FunFormer 20k training jobs for chicken — 4 learning rates.
# Best LR checkpoint will be used for cross-species evaluation.
#
# Usage: bash 4d.chicken_funformer.sh

set -euo pipefail
mkdir -p logs

LRS=(1e-5 5e-5 1e-4 5e-4)

for LR in "${LRS[@]}"; do
    echo "Submitting FunFormer chicken LR=${LR}"
    sbatch --export=ALL,LEARNING_RATE="${LR}" \
           --job-name="funformer_chicken_${LR}" \
           --output="logs/slurm_%j_funformer_chicken_len20000_${LR}.out" \
           olivia_submiter/funformer_chicken.sh
done

echo ""
echo "Submitted ${#LRS[@]} FunFormer chicken jobs (lr: ${LRS[*]})."
