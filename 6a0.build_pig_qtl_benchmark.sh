#!/bin/bash
# =============================================================================
# 6a0 — build the pig eQTL and sQTL fine-mapping benchmark sets
#
# Runs BEFORE 6a (VEP scoring) and 6b (random-forest evaluation).
# Login-node safe: pandas/numpy only, no GPU, no container. ~15 min total.
#
# ── THE RULE ─────────────────────────────────────────────────────────────────
# Follows Avsec et al. 2021 (Enformer), Methods, "Fine-mapped GTEx
# classification", verbatim:
#
#   positive   PIP > 0.9
#   negative   "When available, we chose a negative variant matched to each
#               causal variant from the set with PIP<0.01 but |zscore|>4 tested
#               for the same gene. When unavailable for the same gene, we chose
#               from the set with PIP<0.01 and |z-score|>6 genome-wide."
#
# Eligibility IS the matching there: a negative qualifies by gene, PIP and |z|,
# with no further pairing on effect size. The raised |z| bar on the genome-wide
# tier is what keeps a cross-gene negative hard once it is no longer an LD
# partner of the causal variant.
#
# Two invariants the prose leaves implicit are enforced here, matching what
# holds in the published Enformer VCFs:
#   * a variant is used at most once — negatives drawn without replacement;
#   * a variant causal for ANY gene in the tissue is never eligible as a
#     negative, so no variant can carry both labels.
# A positive that cannot be matched in either tier is dropped, so each tissue is
# exactly 1:1.
#
# ── |z| THRESHOLDS AND SMALL TISSUES ────────────────────────────────────────
# The 4 and 6 cutoffs were calibrated on GTEx sample sizes. Small pig tissues
# lack the power to produce |z|>6 at all, so the genome-wide tier is empty and
# most positives cannot be matched:
#
#     tissue     positives   same-gene pool   genome-wide pool (|z|>6)
#     Artery           182               59                          0
#     Spleen           520            1,766                         55
#     Colon            307            1,809                         67
#     Adipose        2,542          264,164                     71,785
#     Muscle        10,291        1,227,427                    536,914
#
# With --min-causal 200 this keeps 19 of 32 eQTL and 18 of 34 sQTL tissues.
# Lowering --z-genome would recover more tissues but would no longer be the
# published rule; prefer losing the tissue to silently changing the benchmark.
#
# ── OUTPUT ───────────────────────────────────────────────────────────────────
#   bench_data/pig_gtex_fine_vcf/eqtl/<Tissue>.tsv
#   bench_data/pig_gtex_fine_vcf/sqtl/<Tissue>.tsv
#   columns: variant_id  gene_id  zval  prob  causal
#
# For sQTL, gene_id is a splice cluster (chr:start:end:GENE), so "same gene" in
# tier 1 means "same intron" — a finer locus unit than for eQTL.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

RAW="${RAW:-/cluster/projects/nn9780k/dat/pig_gtex}"
OUT="${OUT:-bench_data/pig_gtex_fine_vcf}"
PY="${PYTHON:-python3.12}"          # login-node python with pandas/numpy
MIN_CAUSAL="${MIN_CAUSAL:-200}"     # stricter than Enformer, which kept a
                                    # tissue with 54 positives (Kidney_Cortex)

for KIND in eQTL sQTL; do
    lower=$(echo "$KIND" | tr '[:upper:]' '[:lower:]')
    echo "=============================================================="
    echo " Building ${KIND}  ->  ${OUT}/${lower}"
    echo "=============================================================="
    "$PY" -m src_evaluate.build_pig_qtl_benchmark \
        --raw-dir     "${RAW}/PigGTEx_v0.finemapped_${KIND}" \
        --out-dir     "${OUT}/${lower}" \
        --pip-causal  0.9 \
        --pip-control 0.01 \
        --z-same-gene 4 \
        --z-genome    6 \
        --min-causal  "${MIN_CAUSAL}" \
        --seed        0
    echo
done

echo "Done. Next: 6a.pig_qtl_vep.sh (VEP), then 6b.pig_qtl_rf.sh (random forest)."
