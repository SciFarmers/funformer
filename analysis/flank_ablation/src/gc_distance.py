"""Does source-flank rescue track GC compatibility between the two genomes?

If a source flank substitutes for the target's own because it merely looks like
plausible DNA, rescue should be larger where the source genome's composition is
closer to the target flanks it replaces. Wasserstein distance between the two
GC distributions is the cheapest test of that.

Source windows are sampled by seeking to random offsets in the FASTA, excluding
the source's held-out and validation chromosomes, so no genome is loaded.
Target flank GC comes from the donor logs, which already record it per flank.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

ROOT = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HERE = os.path.join(ROOT, "analysis", "flank_ablation")
sys.path.insert(0, os.path.join(ROOT, "analysis", "common"))
from n_content import FASTA, GEN                                  # noqa: E402

FLANK, N_SAMPLE, SEED = 9500, 4000, 0
TEST_CHR = {"human": "21", "pig": "16", "cattle": "21", "chicken": "21",
            "salmon": "21", "zebrafish": "21"}
VAL_CHR = {"human": "22", "pig": "17", "cattle": "25", "chicken": "25",
           "salmon": "25", "zebrafish": "25"}


def source_gc(species, rng):
    fa = os.path.join(GEN, FASTA[species])
    recs = []
    for line in open(fa + ".fai"):
        n, ln, off, lb, lw = line.split("\t")[:5]
        if n in (TEST_CHR[species], VAL_CHR[species]):
            continue
        ln = int(ln)
        if ln > FLANK * 4:
            recs.append((n, ln, int(off), int(lb), int(lw)))
    if not recs:
        return np.array([])
    w = np.array([r[1] for r in recs], float); w /= w.sum()
    out = []
    with open(fa, "rb") as fh:
        for _ in range(N_SAMPLE):
            i = rng.choice(len(recs), p=w)
            n, ln, off, lb, lw = recs[i]
            pos = int(rng.integers(0, ln - FLANK))
            byte = off + (pos // lb) * lw + (pos % lb)
            fh.seek(byte)
            raw = fh.read(FLANK + FLANK // lb + 16).replace(b"\n", b"")[:FLANK]
            a = np.frombuffer(raw.upper(), dtype=np.uint8)
            gc = ((a == 71) | (a == 67)).sum()
            at = ((a == 65) | (a == 84)).sum()
            if gc + at > 0.9 * FLANK:
                out.append(gc / (gc + at))
    return np.array(out)


def main():
    rng = np.random.default_rng(SEED)
    src_gc = {sp: source_gc(sp, rng) for sp in sorted(FASTA)}
    for sp, v in src_gc.items():
        print(f"  {sp:10s} {len(v)} windows  GC median {np.median(v):.4f}")

    resc = pd.read_csv(os.path.join(HERE, "out", "rescue_directions.tsv"), sep="\t")
    rows = []
    for _, r in resc.iterrows():
        p = os.path.join(HERE, "data", "cache", f"{r.source}_to_{r.target}",
                         "donors__random_source_k1000_s0.tsv.gz")
        if not os.path.exists(p):
            continue
        tg = pd.read_csv(p, sep="\t", usecols=["gc_orig"]).gc_orig.dropna().to_numpy()
        if len(tg) > 50000:
            tg = rng.choice(tg, 50000, replace=False)
        rows.append(dict(source=r.source, target=r.target, bin=r.bin,
                         rescue_pp=r.rescue_pp,
                         wasserstein=wasserstein_distance(src_gc[r.source], tg)))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(HERE, "out", "gc_distance.tsv"), sep="\t", index=False)
    print("\n" + d.round(4).to_string(index=False))
    from scipy.stats import pearsonr, spearmanr
    pr, pp = pearsonr(d.wasserstein, d.rescue_pp)
    sr, sp_ = spearmanr(d.wasserstein, d.rescue_pp)
    print(f"\nn = {len(d)}")
    print(f"  Pearson  r = {pr:+.3f}  (P = {pp:.3f})")
    print(f"  Spearman r = {sr:+.3f}  (P = {sp_:.3f})")
    f = d[d.bin == "tetrapod-fish"]
    if len(f) > 3:
        fr, fp = pearsonr(f.wasserstein, f.rescue_pp)
        print(f"  within tetrapod-fish only (n={len(f)}): r = {fr:+.3f} (P = {fp:.3f})")


if __name__ == "__main__":
    main()
