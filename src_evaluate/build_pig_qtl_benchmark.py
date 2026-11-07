"""Build eQTL / sQTL fine-mapping benchmark sets from raw PigGTEx SuSiE output.

Design follows the fine-mapping benchmarks used for human GTEx in the Enformer
study and for pig in our earlier work: a positive set of confidently fine-mapped
causal variants, and an equal number of negatives drawn from the same loci so
that the task is to tell the causal variant from its LD partners rather than to
tell an associated region from an unassociated one.

    positives   PIP > --pip-causal (default 0.90)
    negatives   same gene, PIP < --pip-control (0.01) and |z| > --z-same-gene
                (4); when the gene offers none, anywhere in the tissue with
                PIP < 0.01 and |z| > --z-genome (6)

Three properties are enforced by construction:

1. No variant appears twice. Negatives are drawn without replacement, so a
   variant cannot reach both a training and a test fold and be scored on
   memory rather than on sequence.
2. No variant is both positive and negative. A variant causal for any gene in
   the tissue is excluded from the negative pool, so no variant carries two
   labels.
3. The 1:1 ratio counts distinct variants, not rows.

Eligibility is the matching: a negative qualifies by gene, PIP and |z|, with no
further pairing on effect size. The raised |z| bar on the second tier is what
keeps a cross-gene negative hard once it is no longer an LD partner of the
causal variant. Per-tissue counts of same-gene versus genome-wide matches are
reported, and a positive matchable by neither tier is dropped.

Output matches the Enformer GTEx benchmark layout in
bench_data/enformer_gtex_fine_vcf, so the pig and human sets are read by the
same code: one VCF per tissue and class,

    <Tissue>_pos.vcf   causal variants
    <Tissue>_neg.vcf   matched controls

each a VCFv4.2 header followed by CHROM POS ID REF ALT QUAL FILTER INFO, with
QUAL and FILTER as "." and INFO empty. The label is carried by which file a
variant is in, exactly as in the human set.

The id follows the human convention CHROM_POS_REF_ALT_<build>, so both species'
files parse with one reader. Chromosomes stay bare Ensembl names (1, 2, ...) to
match Sscrofa11.1.

Usage:
    python -m src_evaluate.build_pig_qtl_benchmark \
        --raw-dir /cluster/projects/nn9780k/dat/pig_gtex/PigGTEx_v0.finemapped_eQTL \
        --out-dir bench_data/pig_gtex_fine_vcf/eqtl \
        --min-causal 200
"""
import argparse
import gzip
import os
import sys

import numpy as np
import pandas as pd

USECOLS = ["gene_id", "variant_id", "zval", "prob"]


def write_vcf(df: pd.DataFrame, path: str, build: str) -> None:
    """Write one class of variants in the Enformer benchmark's VCF layout.

    The id follows the human convention CHROM_POS_REF_ALT_<build>, e.g.
    5_3189713_T_C_ss11 against the human chr1_14677_G_A_b38, so both parse with
    the same reader. CHROM stays a bare Ensembl name to match Sscrofa11.1.
    """
    parts = df.variant_id.str.split("_", expand=True)
    with open(path, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        for chrom, pos, ref, alt in zip(parts[0], parts[1], parts[2], parts[3]):
            vid = f"{chrom}_{pos}_{ref}_{alt}_{build}"
            fh.write(f"{chrom}\t{pos}\t{vid}\t{ref}\t{alt}\t.\t.\t\n")


def is_snv(vid: pd.Series) -> pd.Series:
    """variant_id is chrom_pos_ref_alt; keep single-base ref and alt."""
    parts = vid.str.split("_", expand=True)
    if parts.shape[1] < 4:
        return pd.Series(False, index=vid.index)
    return (parts[2].str.len() == 1) & (parts[3].str.len() == 1)


def build_tissue(path, pip_causal, pip_control, z_same_gene, z_genome, rng):
    """Construct one tissue's positive/negative set.

    Follows the fine-mapped eQTL benchmark of Avsec et al. (Enformer), Methods,
    "Fine-mapped GTEx classification", verbatim:

        positives   PIP > pip_causal (default 0.9)
        negatives   "When available, we chose a negative variant matched to each
                     causal variant from the set with PIP<0.01 but |zscore|>4
                     tested for the same gene. When unavailable for the same
                     gene, we chose from the set with PIP<0.01 and |z-score|>6
                     genome-wide."

    Note what "matched" means there: eligibility *is* the matching. A negative
    qualifies by being in the same gene with PIP below threshold and |z| above
    it; there is no further pairing on effect size. The raised |z| bar on the
    genome-wide tier (6 rather than 4) is what keeps a cross-gene negative hard,
    since it is no longer an LD partner of the causal variant.

    Two properties the tier rules leave implicit are enforced here, matching
    what holds in the published Enformer VCFs (no repeated ids within a class,
    no variant appearing in both classes, ratio exactly 1.000 per tissue):

      * a variant is used at most once, so negatives are drawn without
        replacement;
      * a variant that is causal for any gene in the tissue is never eligible
        as a negative, so no variant carries both labels.

    A positive for which no negative can be found in either tier is dropped, so
    the returned set is exactly 1:1.
    """
    df = pd.read_csv(path, sep="\t", usecols=USECOLS, compression="gzip")
    df = df[is_snv(df.variant_id)]
    df["abs_z"] = df.zval.abs()

    pos_rows = df[df.prob > pip_causal]
    if pos_rows.empty:
        return None, None
    pos = (pos_rows.sort_values("prob", ascending=False)
                   .drop_duplicates("variant_id")
                   .reset_index(drop=True))
    causal_ids = set(pos.variant_id)

    elig = df[(df.prob < pip_control) & (~df.variant_id.isin(causal_ids))]
    elig = elig.drop_duplicates("variant_id").reset_index(drop=True)

    # tier 1: same gene, |z| > z_same_gene
    t1 = elig[elig.abs_z > z_same_gene]
    by_gene = {g: list(idx) for g, idx in t1.groupby("gene_id").indices.items()}
    for g in by_gene:
        rng.shuffle(by_gene[g])
    # tier 2: anywhere in the tissue, |z| > z_genome
    t2 = list(elig.index[elig.abs_z > z_genome])
    rng.shuffle(t2)
    t2_ptr = 0

    used, keep_pos, keep_neg, fallback = set(), [], [], 0
    for i in rng.permutation(len(pos)):
        p = pos.iloc[i]
        pick = None
        pool = by_gene.get(p.gene_id)
        while pool:                              # tier 1
            c = pool.pop()
            if c not in used:
                pick = c
                break
        if pick is None:                         # tier 2
            while t2_ptr < len(t2):
                c = t2[t2_ptr]; t2_ptr += 1
                if c not in used:
                    pick = c
                    fallback += 1
                    break
        if pick is None:
            continue                             # unmatchable: drop this positive
        used.add(pick)
        keep_pos.append(i)
        keep_neg.append(pick)

    cols = ["variant_id", "gene_id", "zval", "prob"]
    out = pd.concat([
        pos.iloc[keep_pos].assign(causal=1)[cols + ["causal"]],
        elig.loc[keep_neg].assign(causal=0)[cols + ["causal"]],
    ], ignore_index=True)
    stats = dict(n_causal=len(keep_pos), n_control=len(keep_neg),
                 dropped=len(pos) - len(keep_pos), fallback=fallback,
                 t1_pool=len(t1), t2_pool=len(t2))
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pip-causal", type=float, default=0.90)
    ap.add_argument("--pip-control", type=float, default=0.01)
    ap.add_argument("--z-same-gene", type=float, default=4.0,
                    help="|z| floor for a within-gene negative (Enformer: 4)")
    ap.add_argument("--z-genome", type=float, default=6.0,
                    help="|z| floor for a genome-wide negative (Enformer: 6)")
    ap.add_argument("--min-causal", type=int, default=200,
                    help="skip tissues with fewer confidently causal variants")
    ap.add_argument("--build", default="ss11",
                    help="genome build tag appended to the variant id")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    files = sorted(f for f in os.listdir(a.raw_dir) if f.endswith(".susieinf.gz"))
    print(f"{'tissue':<20}{'causal':>8}{'dropped':>9}{'same-gene':>11}{'genome-wide':>13}  status")
    print("-" * 68)
    kept = 0
    for f in files:
        t = f[: -len(".susieinf.gz")]
        out, st = build_tissue(os.path.join(a.raw_dir, f), a.pip_causal,
                               a.pip_control, a.z_same_gene, a.z_genome, rng)
        if out is None or st["n_causal"] < a.min_causal:
            n = 0 if out is None else st["n_causal"]
            print(f"{t:<20}{n:>8}{'':>9}{'':>9}{'':>10}  skipped (< {a.min_causal})")
            continue
        # invariants
        assert out.variant_id.is_unique, f"{t}: duplicate variant_id"
        assert out.groupby("variant_id").causal.nunique().max() == 1, f"{t}: label conflict"
        write_vcf(out[out.causal == 1], os.path.join(a.out_dir, f"{t}_pos.vcf"), a.build)
        write_vcf(out[out.causal == 0], os.path.join(a.out_dir, f"{t}_neg.vcf"), a.build)
        kept += 1
        print(f"{t:<20}{st['n_causal']:>8}{st['dropped']:>9}"
              f"{st['n_causal']-st['fallback']:>11}{st['fallback']:>13}  written")
    print(f"\n{kept} tissues written to {a.out_dir}")


if __name__ == "__main__":
    main()
