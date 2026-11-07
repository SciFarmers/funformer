#!/bin/bash
# Submit FunFormer 1 kb training jobs for all five species: 5 species x 4 LRs
# = 20 jobs.
#
# Why: comparing FunFormer-20k against the three 1 kb baselines confounds
# architecture with context length. FunFormer-1k is the architecture- and
# parameter-matched control at the baselines' own context length, so
#   FunFormer-1k  vs baselines  -> pure architecture effect
#   FunFormer-20k vs FunFormer-1k -> pure context-length effect
#
# Optimization is held identical to the rest of the FunFormer ablation family:
# same 4 LRs, effective batch 1000 (500 per-device x 2 GPUs x 1 GAC), 30 max
# epochs, patience 5, bf16, seed 42. Only sequence_length (and the derived CNN
# pool sizes, (2,1,1) -> 500 tokens) changes. Parameter count is identical to
# the 20 kb models because n_tokens is pinned at 500.
#
# Checkpoints: checkpoints/funformer_<species>_len1000_lr<LR>_b500_gac1/
#
# Usage: bash 9a.funformer1k_jobs.sh
set -euo pipefail

LRS=(1e-5 5e-5 1e-4 5e-4)
SEQ_LEN=1000

# species -> submitter wrapper (all now accept SEQ_LEN; default stays 20000)
declare -A SUBMITTER=(
    [human]="olivia_submiter/funformer_entex423.sh"
    [cattle]="olivia_submiter/funformer_cattle.sh"
    [pig]="olivia_submiter/funformer_pig.sh"
    [salmon]="olivia_submiter/funformer_salmon.sh"
    [chicken]="olivia_submiter/funformer_chicken.sh"
)

mkdir -p logs

n=0
for SP in human cattle pig salmon chicken; do
    WRAPPER="${SUBMITTER[$SP]}"
    for LR in "${LRS[@]}"; do
        echo "Submitting FunFormer 1k: species=${SP} lr=${LR}"
        sbatch --export=ALL,SEQ_LEN="${SEQ_LEN}",LEARNING_RATE="${LR}" \
               --job-name="ff1k_${SP}_${LR}" \
               --output="logs/slurm_%j_funformer_${SP}_len1000_lr${LR}.out" \
               "${WRAPPER}"
        n=$((n + 1))
    done
done

echo ""
echo "${n} FunFormer 1k training jobs submitted."
echo "  Species        : human cattle pig salmon chicken"
echo "  Learning rates : ${LRS[*]}"
