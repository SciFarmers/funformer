#!/usr/bin/env bash
# Test-set evaluation for the human context-length sweep, at 5, 10, 30 and
# 50 kb. The 1 and 20 kb rows come from 5a, so this completes the curve that
# Figure 2c draws and that the 20 kb choice was checked against, with every
# length measured the same way.
#
# Batch size falls as context grows, to stay inside one GPU.
#
#   bash 5f.human_length_ablation_eval.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OUTPUT_DIR="eval_results/test_set"
mkdir -p "$OUTPUT_DIR" logs

run() {                       # run <batch> <time> <checkpoint-tag>...
    local bs="$1" tl="$2"; shift 2
    for tag in "$@"; do
        [[ -d "checkpoints/$tag" ]] || { echo "MISSING checkpoints/$tag" >&2; continue; }
        sbatch --parsable --time="$tl" --job-name="eval_human_${tag}" \
               --export=ALL,CHECKPOINT_DIR="checkpoints/$tag",OUTPUT_DIR="$OUTPUT_DIR",BATCH_SIZE="$bs" \
               olivia_submiter/evaluate_test_human.sh \
            | xargs -I{} echo "  {}  bs=$bs  $tag"
    done
}

for lr in 1e-5 5e-5 1e-4 5e-4; do
    run 1000 4:00:00  "funformer_entex_423_len5000_lr${lr}_b500_gac1"
    run  500 4:00:00  "funformer_entex_423_len10000_lr${lr}_b500_gac1"
    run  250 6:00:00  "funformer_entex_423_len30000_lr${lr}_b250_gac2"
    run  150 8:00:00  "funformer_entex_423_len50000_lr${lr}_b250_gac2"
done
