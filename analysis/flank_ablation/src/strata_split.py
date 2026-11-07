"""Flank usefulness split by the Analysis A strata, random_same only.

intact comes from A's cached stratified channel table; the ablated AUROC is
computed here on the same strata, so the drop is a within-channel,
within-stratum difference.
"""
import os
import sys

import numpy as np
import pandas as pd

# Self-contained so it does not depend on HERE, which is defined below
# and may itself point at a code snapshot.
ROOT = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Outputs belong in the real analysis directory. Deriving this from
# __file__ puts them inside the code snapshot when one is in use.
HERE = os.path.join(ROOT, "analysis", "flank_ablation") if os.environ.get("REPO_ROOT") \
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis", "tss_strata", "src"))
import strata as S                                              # noqa: E402

C_CACHE = os.path.join(HERE, "data", "cache")
A_OUT = os.path.join(ROOT, "analysis", "tss_strata", "out")
OUT = os.path.join(HERE, "out")
TAG = "FunFormer_20k__random_same_k1000_s0"


def main():
    intact = pd.read_csv(os.path.join(A_OUT, "channels_primary.tsv"), sep="\t")
    intact = intact[["source", "target", "bin", "assay", "tissue", "stratum",
                     "auroc_ff20k"]]
    rows = []
    for (src, tgt), _ in intact.groupby(["source", "target"]):
        p = os.path.join(C_CACHE, f"{src}_to_{tgt}", f"{TAG}.npz")
        if not os.path.exists(p):
            continue
        for r in S.direction(src, tgt, TAG, cache_root=C_CACHE,
                             drop_n_flanks=True):
            rows.append(dict(source=src, target=tgt, assay=r["assay"],
                             tissue=r["tissue"], stratum=r["stratum"],
                             auroc_ablated=r["auroc"]))
        print(f"  {src}->{tgt}", flush=True)
    abl = pd.DataFrame(rows)
    j = intact.merge(abl, on=["source", "target", "assay", "tissue", "stratum"])
    j["usefulness_pp"] = (j.auroc_ff20k - j.auroc_ablated) * 100
    j.to_csv(os.path.join(OUT, "strata_split_channels.tsv"), sep="\t", index=False)

    b = j.pivot_table(index="bin", columns="stratum", values="usefulness_pp",
                      aggfunc="mean")
    n = j.pivot_table(index="bin", columns="stratum", values="usefulness_pp",
                      aggfunc="size")
    b["n_channels"] = n["proximal"]
    b["distal_minus_proximal"] = b["distal"] - b["proximal"]
    b.to_csv(os.path.join(OUT, "strata_split_bins.tsv"), sep="\t")
    print("\nflank usefulness (pp), random_same, by A strata:")
    print(b.round(2).to_string())


if __name__ == "__main__":
    main()
