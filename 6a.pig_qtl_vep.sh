#!/bin/bash
# =============================================================================
# 6a — VEP scoring on the pig QTL benchmark (run 6a0 first).
#
# Scores every model against both the eQTL and sQTL variant sets on the pig
# genome. Nine models: the pig-trained FunFormer, the three pig-trained 1 kb
# baselines, and the five other species' FunFormers applied zero-shot.
#
# The pig sets are in the same Enformer VCF format as the human GTEx sets and
# carry the same variant-id convention, so they are scored by the same
# submitter and the same module as human — only the genome, the variant
# directory and the output root differ.
#
# Output: bench_results/vep_pig_qtl/{eqtl,sqtl}/<MODEL_TAG>/<TISSUE>_{pos,neg}.csv.gz
# Then run 6b.pig_qtl_rf.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PIG_GENOME="/cluster/projects/nn9780k/dat/training_data/genomes/Sus_scrofa.Sscrofa11.1.dna.toplevel.fa"

submit() {
  GENOME_PATH="$PIG_GENOME" VCF_DIR="$VDIR" OUT_ROOT="$OROOT" \
  MODEL_CLASS="$1" WINDOW_SIZE="$2" MODEL_PATH="$3" \
    sbatch --export=ALL olivia_submiter/vep_eqtl_finemap.sh
}

for KIND in eqtl sqtl; do
  VDIR="bench_data/pig_gtex_fine_vcf/${KIND}"
  OROOT="bench_results/vep_pig_qtl/${KIND}"

  # pig-trained FunFormer
  submit FunFormer 20000 checkpoints/funformer_pig_len20000_lr5e-4_b250_gac2/best_model

  # pig-trained 1 kb baselines
  for M in DeepSEA:DeepSEA_pig_5e-4_1000 DeepATT:DeepATT_pig_5e-4_1000 \
           DeepFormer:DeepFormer_pig_5e-4_1000; do
    submit "${M%%:*}" 1000 "checkpoints/${M##*:}/best_model"
  done

  # other species, zero-shot
  for SP in entex_423 cattle chicken salmon zebrafish; do
    submit FunFormer 20000 "checkpoints/funformer_${SP}_len20000_lr5e-4_b250_gac2/best_model"
  done
done
