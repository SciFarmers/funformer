"""Source-flank rescue: AUROC(random_source) - AUROC(random_same).

Direction-level values and bin means with 95% CI.

The species-level permutation is computed here as a diagnostic but is NOT
reported in the manuscript, and should not be. That null asks whether a
quantity is aligned with the tree. Rescue is not claimed to be: it is claimed
to depend on which genome supplies the flank, a within-direction contrast
between two conditions scored on the same intervals with the same checkpoint.
The interval on the bin mean is the appropriate summary. Note also that the
contrast returned by species_label_perm is one-sided for mammal > fish, the
orientation the gradient hypothesis needs; rescue runs the other way, so that
p_d is the complement of anything one would quote here.

The fish bin is split two ways, because the hypothesis is specific: a source
flank should help where the model has something to say about that sequence, so
the rescue should be larger from a mammal source into a fish than the reverse,
and larger near TSS than far from them.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HERE = os.path.join(ROOT, "analysis", "flank_ablation")
sys.path.insert(0, os.path.join(ROOT, "analysis", "tss_strata", "src"))
sys.path.insert(0, os.path.join(ROOT, "manuscript", "analysis"))
import strata as S                                      # noqa: E402
import stats_gradient as SG                             # noqa: E402
from divergence_times import CLADE                      # noqa: E402

OUT = os.path.join(HERE, "out")
C_CACHE = os.path.join(HERE, "data", "cache")
SAME = "FunFormer_20k__random_same_k1000_s0"
SRC = "FunFormer_20k__random_source_k1000_s0"


import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(ROOT, "analysis", "common"))
from ci import mean_ci  # noqa: E402

def ci(v):
    return mean_ci(v)


def main():
    d = pd.read_csv(os.path.join(OUT, "directions_stage1.tsv"), sep="\t")
    d = d[d.bin != "within-species"]
    w = d.pivot_table(index=["source", "target", "bin", "divergence_mya"],
                      columns="condition", values="auroc_ablated").reset_index()
    w = w.dropna(subset=["random_source", "random_same"])
    w["rescue_pp"] = (w.random_source - w.random_same) * 100
    w.to_csv(os.path.join(OUT, "rescue_directions.tsv"), sep="\t", index=False)

    print("-- rescue per direction (pp) --")
    print(w[["source", "target", "bin", "rescue_pp"]]
          .sort_values(["bin", "rescue_pp"]).round(3).to_string(index=False))

    print("\n-- bin mean with 95% CI --")
    rows = []
    for b, g in w.groupby("bin"):
        m, lo, hi = ci(g.rescue_pp)
        rows.append(dict(bin=b, n=len(g), mean_pp=m, lo=lo, hi=hi))
    bt = pd.DataFrame(rows)
    print(bt.round(3).to_string(index=False))
    bt.to_csv(os.path.join(OUT, "rescue_bins.tsv"), sep="\t", index=False)

    print("\n-- species-level exact test, rescue vs divergence --")
    sp = SG.species_label_perm(w[["source", "target", "rescue_pp"]], "rescue_pp")
    rng = np.random.default_rng(0)
    r_d, p_d = SG.pearson_perm(w["divergence_mya"], w["rescue_pp"], rng)
    print(f"  Pearson r = {sp['r']:+.3f}   exact P (species) = {sp['p_r']:.4f}   "
          f"perm P (direction) = {p_d:.5f}")
    print(f"  Jonckheere-Terpstra J = {sp['J']:.1f}  exact P = {sp['p_j']:.4f}")
    print(f"  mammal-mammal minus tetrapod-fish = {sp['d']:+.3f} pp  "
          f"exact P = {sp['p_d']:.4f}")
    pd.DataFrame([sp]).to_csv(os.path.join(OUT, "rescue_stats.tsv"),
                              sep="\t", index=False)

    print("\n-- fish bin split by source clade --")
    f = w[w.bin == "tetrapod-fish"].copy()
    f["direction_kind"] = np.where(
        f.source.map(CLADE) == "fish", "fish source -> tetrapod",
        "tetrapod source -> fish")
    for k, g in f.groupby("direction_kind"):
        m, lo, hi = ci(g.rescue_pp)
        print(f"  {k:26s} n={len(g):2d}  {m:+.2f} pp  [{lo:+.2f}, {hi:+.2f}]")
    f.to_csv(os.path.join(OUT, "rescue_fish_by_clade.tsv"), sep="\t", index=False)

    print("\n-- fish bin split by TSS stratum --")
    rows = []
    for _, r in f.iterrows():
        for tag, model in (("random_same", SAME), ("random_source", SRC)):
            p = os.path.join(C_CACHE, f"{r.source}_to_{r.target}", f"{model}.npz")
            if not os.path.exists(p):
                continue
            for x in S.direction(r.source, r.target, model, cache_root=C_CACHE,
                                 drop_n_flanks=True):
                rows.append(dict(source=r.source, target=r.target,
                                 condition=tag, assay=x["assay"],
                                 tissue=x["tissue"], stratum=x["stratum"],
                                 auroc=x["auroc"]))
        print(f"    {r.source}->{r.target}", flush=True)
    ch = pd.DataFrame(rows)
    if not ch.empty:
        pv = ch.pivot_table(index=["source", "target", "assay", "tissue", "stratum"],
                            columns="condition", values="auroc").reset_index()
        pv["rescue_pp"] = (pv.random_source - pv.random_same) * 100
        pv.to_csv(os.path.join(OUT, "rescue_fish_by_stratum.tsv"), sep="\t", index=False)
        st = pv.groupby("stratum").agg(n=("rescue_pp", "size"),
                                       mean_pp=("rescue_pp", "mean")).reset_index()
        print(st.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
