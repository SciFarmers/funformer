"""Partial correlation of GC distance with rescue, controlling transfer direction.

Caveat this exists to test: random_source donors are GC-matched to the target
flank they replace, so a genome-wide GC distance measures something the
sampling already equalises. If it still correlates with rescue, it is acting
through something other than the composition of the donors actually used --
most plausibly how hard the match was to find.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, rankdata

ROOT = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HERE = os.path.join(ROOT, "analysis", "flank_ablation")
sys.path.insert(0, os.path.join(ROOT, "manuscript", "analysis"))
from divergence_times import CLADE                                # noqa: E402


def partial(x, y, z, rank=False):
    """Correlation of x and y after removing a binary control z."""
    x, y = (rankdata(x), rankdata(y)) if rank else (np.asarray(x), np.asarray(y))
    z = np.asarray(z, float)
    Z = np.column_stack([np.ones_like(z), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return pearsonr(rx, ry)


def main():
    d = pd.read_csv(os.path.join(HERE, "out", "gc_distance.tsv"), sep="\t")
    f = d[d.bin == "tetrapod-fish"].copy()
    f["fish_source"] = (f.source.map(CLADE) == "fish").astype(int)

    print("-- fish bin, GC distance vs rescue --")
    r, p = pearsonr(f.wasserstein, f.rescue_pp)
    print(f"  unadjusted              n={len(f)}  r = {r:+.3f}  P = {p:.4f}")
    pr, pp = partial(f.wasserstein, f.rescue_pp, f.fish_source)
    sr, sp = partial(f.wasserstein, f.rescue_pp, f.fish_source, rank=True)
    print(f"  partial, direction held n={len(f)}  r = {pr:+.3f}  P = {pp:.4f}")
    print(f"  partial Spearman                     rho = {sr:+.3f}  P = {sp:.4f}")
    for k, g in f.groupby("fish_source"):
        lab = "fish source -> tetrapod" if k else "tetrapod source -> fish"
        if len(g) > 3:
            rr, ppp = pearsonr(g.wasserstein, g.rescue_pp)
            print(f"  within {lab:24s} n={len(g)}  r = {rr:+.3f}  P = {ppp:.4f}")

    print("\n-- tetrapod -> fish six, by target --")
    t = f[f.fish_source == 0].copy()
    print(t[["source", "target", "rescue_pp", "wasserstein"]]
          .sort_values(["target", "source"]).round(4).to_string(index=False))
    for tgt, g in t.groupby("target"):
        print(f"  {tgt:10s} n={len(g)}  mean rescue {g.rescue_pp.mean():+.2f} pp  "
              f"range {g.rescue_pp.min():+.2f} to {g.rescue_pp.max():+.2f}")

    print("\n-- is GC distance acting through match difficulty? --")
    rows = []
    for _, r0 in f.iterrows():
        p0 = os.path.join(HERE, "data", "cache", f"{r0.source}_to_{r0.target}",
                          "donors__random_source_k1000_s0.tsv.gz")
        if not os.path.exists(p0):
            continue
        dd = pd.read_csv(p0, sep="\t", usecols=["gc_tier", "gc_orig", "gc_new"])
        tier = pd.to_numeric(dd.gc_tier, errors="coerce")
        rows.append(dict(source=r0.source, target=r0.target,
                         wasserstein=r0.wasserstein, rescue_pp=r0.rescue_pp,
                         within2=float((tier <= 0.02).mean()),
                         achieved_gc_gap=float((dd.gc_orig - dd.gc_new).abs().mean())))
    m = pd.DataFrame(rows)
    for col in ("within2", "achieved_gc_gap"):
        a, _ = pearsonr(m.wasserstein, m[col])
        b, bp = pearsonr(m[col], m.rescue_pp)
        print(f"  {col:16s}  vs GC distance r = {a:+.3f}   vs rescue r = {b:+.3f} (P = {bp:.4f})")
    m.to_csv(os.path.join(HERE, "out", "gc_partial.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
