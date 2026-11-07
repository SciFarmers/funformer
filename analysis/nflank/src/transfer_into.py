"""Transfer margins with and without the N-flank exclusion.

Covers the five directions into human, the pooled bin margins over the primary
set, and the permutation P values, under both conventions.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HERE = os.path.join(ROOT, "analysis", "nflank")
sys.path.insert(0, os.path.join(ROOT, "analysis", "tss_strata", "src"))
sys.path.insert(0, os.path.join(ROOT, "manuscript", "analysis"))
import strata as S                                           # noqa: E402
import stats_gradient as SG                                  # noqa: E402

A_CACHE = os.path.join(ROOT, "analysis", "tss_strata", "data", "cache")
MARG = os.path.join(ROOT, "manuscript", "analysis", "out", "cross_species_margins.csv")


def mean_auroc(src, tgt, model, drop):
    rows = S.direction(src, tgt, model, unstratified=True, cache_root=A_CACHE,
                       drop_n_flanks=drop)
    return float(np.mean([r["auroc"] for r in rows])) if rows else np.nan


def main():
    M = pd.read_csv(MARG)
    rows = []
    for _, r in M.iterrows():
        rec = dict(source=r.source, target=r.target, bin=r["bin"],
                   divergence_mya=r.divergence_mya,
                   in_primary_set=bool(r.in_primary_set))
        ok = True
        for key, model in (("ff20k", "FunFormer_20k"), ("ff1k", "FunFormer_1k"),
                           ("base", r.best_baseline_model)):
            for drop, tag in ((False, "all"), (True, "kept")):
                p = os.path.join(A_CACHE, f"{r.source}_to_{r.target}", f"{model}.npz")
                if not os.path.exists(p):
                    ok = False
                    continue
                rec[f"{key}_{tag}"] = mean_auroc(r.source, r.target, model, drop)
        if not ok:
            continue
        for tag in ("all", "kept"):
            rec[f"margin_arch_{tag}"] = (rec[f"ff20k_{tag}"] - rec[f"ff1k_{tag}"]) * 100
            rec[f"margin_base_{tag}"] = (rec[f"ff20k_{tag}"] - rec[f"base_{tag}"]) * 100
        rows.append(rec)
        print(f"  {r.source}->{r.target}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(HERE, "out", "transfer_old_vs_new.tsv"), sep="\t", index=False)

    print("\n-- the five directions into human (pp) --")
    h = d[d.target == "human"][["source", "bin", "in_primary_set",
                                "margin_arch_all", "margin_arch_kept",
                                "margin_base_all", "margin_base_kept"]].copy()
    h["arch_delta"] = h.margin_arch_kept - h.margin_arch_all
    print(h.round(3).to_string(index=False))

    print("\n-- pooled bin margins, primary set (pp) --")
    p = d[d.in_primary_set]
    b = p.groupby("bin", as_index=False)[["margin_arch_all", "margin_arch_kept",
                                          "margin_base_all", "margin_base_kept"]].mean()
    b["arch_delta"] = b.margin_arch_kept - b.margin_arch_all
    print(b.round(3).to_string(index=False))

    print("\n-- permutation statistics --")
    rng = np.random.default_rng(0)
    out = []
    for col in ("margin_arch_all", "margin_arch_kept",
                "margin_base_all", "margin_base_kept"):
        sub = p[["source", "target", "divergence_mya", col]].dropna()
        sp = SG.species_label_perm(sub[["source", "target", col]], col)
        r_d, p_d = SG.pearson_perm(sub["divergence_mya"], sub[col], rng)
        out.append(dict(quantity=col, n=len(sub), r=sp["r"],
                        p_species=sp["p_r"], p_direction=p_d))
    st = pd.DataFrame(out)
    print(st.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    st.to_csv(os.path.join(HERE, "out", "transfer_stats.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
