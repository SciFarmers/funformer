#!/bin/bash
# =============================================================================
# 6b — random-forest evaluation on the pig QTL benchmark.
# Run after every 6a job has completed.
#
# One job per model.
# Output: bench_results/rf_pig_qtl/{eqtl,sqtl}/<TAG>.csv
#
# Set DEP to a colon-separated list of 6a job ids to queue these behind the VEP
# jobs instead of running them now:
#   DEP=$(...):$(...) bash 6b.pig_qtl_rf.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PIG=funformer_pig_len20000_lr5e-4_b250_gac2
DEPFLAG=""
if [[ -n "${DEP:-}" ]]; then DEPFLAG="--dependency=afterok:${DEP}"; fi

for KIND in eqtl sqtl; do

  # single models
  for T in $PIG DeepSEA_pig_5e-4_1000 DeepATT_pig_5e-4_1000 DeepFormer_pig_5e-4_1000 \
           funformer_entex_423_len20000_lr5e-4_b250_gac2 \
           funformer_cattle_len20000_lr5e-4_b250_gac2 \
           funformer_chicken_len20000_lr5e-4_b250_gac2 \
           funformer_salmon_len20000_lr5e-4_b250_gac2 \
           funformer_zebrafish_len20000_lr5e-4_b250_gac2; do
    KIND=$KIND SCORE_TAGS="$T" TAG="$T" \
      sbatch ${DEPFLAG} --export=ALL olivia_submiter/evaluate_pig_rf.sh
  done
done
