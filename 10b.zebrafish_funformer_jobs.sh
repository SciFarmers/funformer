#!/bin/bash
# Zebrafish FunFormer: 2 context lengths (20 kb, 1 kb) x 4 LRs = 8 jobs.
#
# 20 kb is the model of record; 1 kb is the architecture- and parameter-matched
# control (see 9a). Training both keeps zebrafish on exactly the same footing
# as the other five species for the architecture-vs-context decomposition.
#
# Identical optimization to the rest of the FunFormer family: 4 LRs, effective
# batch 1000, 30 max epochs, patience 5, bf16, seed 42.
#
# Usage: bash 10b.zebrafish_funformer_jobs.sh
set -euo pipefail

LRS=(1e-5 5e-5 1e-4 5e-4)
SEQ_LENS=(20000 1000)
SUBMITTER="olivia_submiter/funformer_zebrafish.sh"
mkdir -p logs

n=0
for SEQ_LEN in "${SEQ_LENS[@]}"; do
    for LR in "${LRS[@]}"; do
        echo "Submitting FunFormer zebrafish len=${SEQ_LEN} lr=${LR}"
        sbatch --export=ALL,SEQ_LEN="${SEQ_LEN}",LEARNING_RATE="${LR}" \
               --job-name="zf_ff${SEQ_LEN}_${LR}" \
               --output="logs/slurm_%j_funformer_zebrafish_len${SEQ_LEN}_lr${LR}.out" \
               "${SUBMITTER}"
        n=$((n + 1))
    done
done
echo ""
echo "${n} zebrafish FunFormer jobs submitted (lengths ${SEQ_LENS[*]}, LRs ${LRS[*]})."
