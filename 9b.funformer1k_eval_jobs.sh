#!/bin/bash
# Within-species test-set evaluation for the FunFormer 1 kb checkpoints
# (5 species x 4 LRs = 20 jobs).
#
# All four LR runs per species are evaluated so that the "best-AUROC
# learning-rate run" selection is made the same way it was for every other
# model in the paper.
#
# Outputs: eval_results/test_set/funformer_<tag>_len1000_lr<LR>_b500_gac1_{per_track,summary}.csv
#
# Usage: bash 9b.funformer1k_eval_jobs.sh
set -euo pipefail

LRS=(1e-5 5e-5 1e-4 5e-4)

# species -> (checkpoint dataset tag, eval submitter)
declare -A TAG=(
    [human]="entex_423" [cattle]="cattle" [pig]="pig" [salmon]="salmon" [chicken]="chicken"
)
declare -A SUBMITTER=(
    [human]="olivia_submiter/evaluate_test_human.sh"
    [cattle]="olivia_submiter/evaluate_test_cattle.sh"
    [pig]="olivia_submiter/evaluate_test_pig.sh"
    [salmon]="olivia_submiter/evaluate_test_salmon.sh"
    [chicken]="olivia_submiter/evaluate_test_chicken.sh"
)

mkdir -p logs

n=0; missing=0
for SP in human cattle pig salmon chicken; do
    for LR in "${LRS[@]}"; do
        CKPT="checkpoints/funformer_${TAG[$SP]}_len1000_lr${LR}_b500_gac1"
        if [[ ! -d "${CKPT}/best_model" ]]; then
            echo "SKIP (training not finished): $(basename "${CKPT}")" >&2
            missing=$((missing + 1))
            continue
        fi
        TAGNAME="$(basename "${CKPT}")"
        # idempotent: don't re-evaluate a checkpoint that already has a summary,
        # and don't double-submit one whose eval job is still in the queue (the
        # summary only appears when that job finishes, so the file check alone
        # races against an in-flight job).
        if [[ -f "eval_results/test_set/${TAGNAME}_summary.csv" ]]; then
            echo "SKIP (already evaluated): ${TAGNAME}"
            continue
        fi
        if squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qx "eval_ff1k_${SP}_${LR}"; then
            echo "SKIP (eval already queued/running): ${TAGNAME}"
            continue
        fi
        echo "Submitting eval: ${TAGNAME}"
        sbatch --export=ALL,CHECKPOINT_DIR="${CKPT}" \
               --job-name="eval_ff1k_${SP}_${LR}" \
               --output="logs/eval_${SP}_${TAGNAME}_%j.out" \
               "${SUBMITTER[$SP]}"
        n=$((n + 1))
    done
done

echo ""
echo "${n} evaluation job(s) submitted; ${missing} skipped (training not finished)."
echo "  Output: eval_results/test_set/"
