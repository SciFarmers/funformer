"""Channel -> direction -> bin aggregates for the flank ablation.

intact and FunFormer-1k are read from Analysis A's cache; the ablated runs from
this analysis's own. Scoring goes through analysis/tss_strata/src/strata.py so
both analyses use one AUROC path.

Derived quantities, all in percentage points:
  flank_dependence  AUROC(intact) - AUROC(ablated), within-species
  flank_usefulness  AUROC(intact) - AUROC(ablated), in transfer
  source_rescue     AUROC(random_source) - AUROC(random_same), in transfer
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

A_CACHE = os.path.join(ROOT, "analysis", "tss_strata", "data", "cache")
C_CACHE = os.path.join(HERE, "data", "cache")
OUT = os.path.join(HERE, "out")
MARGINS = os.path.join(ROOT, "manuscript/analysis/out/cross_species_margins.csv")


def _auroc(source, target, model, cache_root, groups_for=None):
    # drop_n_flanks applies to intact as well as the ablated conditions, so
    # every comparison runs on the same interval set.
    rows = S.direction(source, target, model, unstratified=True,
                       cache_root=cache_root, groups_for=groups_for,
                       drop_n_flanks=True)
    return {(r["assay"], r["tissue"]): r["auroc"] for r in rows}


def directions():
    m = pd.read_csv(MARGINS)
    out = [(r.source, r.target, r["bin"], r.divergence_mya)
           for _, r in m[m.in_primary_set].iterrows()]
    out += [(sp, sp, "within-species", 0.0) for sp in sorted(S.TAG)]
    return out


def channel_table(conditions, keep=1000, seed=0):
    rows = []
    for source, target, bin_, div in directions():
        gf = None if source != target else (source, target)
        if source == target:            # self: use any pair this species is in
            m = pd.read_csv(MARGINS)
            m = m[(m.in_primary_set) & (m.target == target)]
            if m.empty:
                continue
            gf = (m.iloc[0].source, target)
        try:
            intact = _auroc(source, target, "FunFormer_20k", A_CACHE, gf)
            ff1k = _auroc(source, target, "FunFormer_1k", A_CACHE, gf) \
                if source != target else {}
        except FileNotFoundError:
            continue
        for cond in conditions:
            tag = f"FunFormer_20k__{cond}_k{keep}_s{seed}"
            try:
                abl = _auroc(source, target, tag, C_CACHE, gf)
            except FileNotFoundError:
                continue
            for k in set(intact) & set(abl):
                rows.append(dict(
                    source=source, target=target, bin=bin_, divergence_mya=div,
                    assay=k[0], tissue=k[1], condition=cond,
                    auroc_intact=intact[k], auroc_ablated=abl[k],
                    auroc_ff1k=ff1k.get(k, np.nan),
                    drop_pp=(intact[k] - abl[k]) * 100,
                    ablated_minus_ff1k_pp=(abl[k] - ff1k[k]) * 100 if k in ff1k else np.nan))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    conds = [c for c in ("shuffled", "random_same", "random_source")]
    ch = channel_table(conds)
    if ch.empty:
        print("no ablation caches found")
        return
    ch.to_csv(os.path.join(OUT, "channels_stage1.tsv"), sep="\t", index=False)
    d = ch.groupby(["source", "target", "bin", "divergence_mya", "condition"],
                   as_index=False).agg(n_channels=("assay", "size"),
                                       auroc_intact=("auroc_intact", "mean"),
                                       auroc_ablated=("auroc_ablated", "mean"),
                                       auroc_ff1k=("auroc_ff1k", "mean"),
                                       drop_pp=("drop_pp", "mean"),
                                       ablated_minus_ff1k_pp=("ablated_minus_ff1k_pp", "mean"))
    d.to_csv(os.path.join(OUT, "directions_stage1.tsv"), sep="\t", index=False)
    b = d.groupby(["bin", "condition"], as_index=False).agg(
        n_directions=("source", "size"), auroc_intact=("auroc_intact", "mean"),
        auroc_ablated=("auroc_ablated", "mean"), auroc_ff1k=("auroc_ff1k", "mean"),
        drop_pp=("drop_pp", "mean"),
        ablated_minus_ff1k_pp=("ablated_minus_ff1k_pp", "mean"))
    b.to_csv(os.path.join(OUT, "bins_stage1.tsv"), sep="\t", index=False)
    print(b.to_string(index=False))


if __name__ == "__main__":
    main()
