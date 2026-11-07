"""What the donor draw actually did, from the logged donor coordinates.

The implementation audit reported that donor chromosomes are chosen uniformly
by name rather than weighted by length. This measures the realised draw per
direction and condition, so the claim can be checked against what ran rather
than against a reading of the code.

Two conditions are restricted to a single chromosome by design and cannot be
affected: random_source (the source species' held-out chromosome, so the
donor sequence is unseen by the source model) and within-species random_same
(the target's validation chromosome, for the same reason). The length
weighting only bites on random_same in the transfer directions, where donors
are drawn from every target chromosome except the held-out one.
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CACHE = os.path.join(ROOT, "analysis", "flank_ablation", "data", "cache")
GEN = "/cluster/projects/nn9780k/dat/training_data/genomes"
FA = {"human": "Homo_sapiens.GRCh38.dna.primary_assembly.fa",
      "pig": "Sus_scrofa.Sscrofa11.1.dna.toplevel.fa",
      "cattle": "Bos_taurus.ARS-UCD1.2.dna.toplevel.fa",
      "chicken": "Gallus_gallus.GRCg6a.dna.toplevel.fa",
      "salmon": "Salmo_salar.Ssal_v3.1.dna.toplevel.fa",
      "zebrafish": "Danio_rerio.GRCz10.dna_sm.toplevel.fa"}
MIN_BP = 9500 + 1000          # the eligibility rule in _donor_flank


def sizes(sp):
    f = os.path.join(GEN, FA[sp] + ".fai")
    d = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1],
                    names=["chrom", "len"], dtype={"chrom": str})
    return dict(zip(d.chrom, d["len"]))


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(CACHE, "*_to_*", "donors__*.tsv.gz"))):
        direction = os.path.basename(os.path.dirname(f))
        src, tgt = direction.split("_to_")
        cond = os.path.basename(f)[len("donors__"):].split("_k1000")[0]
        if cond == "shuffled":
            continue                       # no donor locus; sequence is permuted
        donor_sp = src if cond == "random_source" else tgt
        d = pd.read_csv(f, sep="\t", dtype={"donor_chrom": str})
        n = d.donor_chrom.value_counts()
        S = sizes(donor_sp)
        elig = {c: L for c, L in S.items() if L > MIN_BP}
        used = set(n.index)
        # per-base sampling density, relative to a length-weighted draw
        dens = {c: n[c] / S[c] for c in used if S.get(c, 0) > 0}
        ratio = (max(dens.values()) / min(dens.values())) if len(dens) > 1 else 1.0
        rows.append({
            "direction": direction, "condition": cond, "donor_genome": donor_sp,
            "donors": int(n.sum()), "chroms_used": len(used),
            "chroms_eligible": len(elig),
            "max_min_density_ratio": round(ratio, 1),
            "single_chrom_by_design": len(used) == 1})
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("=== by condition ===")
    print(t.groupby("condition").agg(
        n=("direction", "size"),
        chroms_used_min=("chroms_used", "min"),
        chroms_used_max=("chroms_used", "max"),
        worst_density_ratio=("max_min_density_ratio", "max")).to_string())
    multi = t[t.chroms_used > 1]
    print(f"\n=== the {len(multi)} runs where chromosome choice actually varies ===")
    print(multi.sort_values("max_min_density_ratio", ascending=False)
          .head(12).to_string(index=False))
    out = os.path.join(ROOT, "analysis", "flank_ablation", "out",
                       "donor_sampling_audit.tsv")
    t.to_csv(out, sep="\t", index=False)
    print(f"\nwrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
