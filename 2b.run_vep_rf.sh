# RF Evaluation — main pipeline (src_evaluate.evaluate_eqtl_rf)
# Reads: bench_results/vep_eqtl/<MODEL_TAG>/
# Output: bench_results/rf_eqtl/<MODEL_TAG>.csv
# Run after 2a.run_vep.sh jobs complete.
# --partition=accel: container is ARM-only, cannot run on x86 login node.

MODEL_TAG=DeepSEA_1e-4_1000 \
  sbatch --export=ALL --partition=accel olivia_submiter/evaluate_eqtl_rf.sh

# MODEL_TAG=DanQ_1e-4_1000 \
#   sbatch --export=ALL --partition=accel olivia_submiter/evaluate_eqtl_rf.sh

MODEL_TAG=DeepATT_5e-4_1000 \
  sbatch --export=ALL --partition=accel olivia_submiter/evaluate_eqtl_rf.sh

MODEL_TAG=DeepFormer_5e-4_1000 \
  sbatch --export=ALL --partition=accel olivia_submiter/evaluate_eqtl_rf.sh

MODEL_TAG=funformer_entex_423_len20000_lr5e-4_b250_gac2 \
  sbatch --export=ALL --partition=accel olivia_submiter/evaluate_eqtl_rf.sh
