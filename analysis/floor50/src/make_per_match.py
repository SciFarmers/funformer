"""Rebuild per-match AUROCs under the final evaluation convention.

Two filters, applied together and identically to every species:

  * intervals whose 20 kb flank is more than 50% N on either side are dropped
    (they are almost all label-free, and in human they are 14.8% of the set);
  * target tracks carrying fewer than 50 positives AMONG THE SURVIVING
    intervals are dropped. Counting positives on the full set would admit a
    track whose positives all sit in the excluded N regions.

The committed per-match CSVs store a channel's AUROC already averaged over its
(source replicate, target replicate) pairs, so neither filter can be applied to
them after the fact: dropping a target track changes which pairs enter that
average. Everything is therefore recomputed from the cached predictions.

Output mirrors eval_results/cross_species/ so the manuscript's aggregator reads
it unchanged through CS_DIR, plus a record of exactly what was dropped.

The directory name is historical: this began as the 50-positive floor alone.
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
HERE = os.path.join(ROOT, "analysis", "floor50") if os.environ.get("REPO_ROOT") \
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis", "tss_strata", "src"))
import strata as S                                              # noqa: E402

MIN_POS = 50
A_CACHE = os.path.join(ROOT, "analysis", "tss_strata", "data", "cache")
OUT_CS = os.path.join(ROOT, "eval_results", "cross_species_final")
OUT = os.path.join(HERE, "out")


def per_match(source, target, model):
    """(rows, dropped) for one direction and model."""
    probs, col_of = S.load_probs(source, target, model, A_CACHE)
    L = S.labels(target)
    usable = S.interval_keep(target)      # N-flank exclusion
    # Positives are counted on the kept intervals: counting them on the full
    # set would admit a track whose positives all sit in the excluded regions.
    pos = (L[usable] == 1).sum(0)
    rows, dropped = [], []
    for m in S.matches(source, target):
        src = [s for s in m[source] if s in col_of]
        keep = [t for t in m[target] if pos[t] >= MIN_POS]
        cut = [t for t in m[target] if pos[t] < MIN_POS]
        for t in cut:
            dropped.append(dict(source=source, target=target, model=model,
                                assay=m["assay"], tissue=m["tissue"],
                                target_track=int(t), positives=int(pos[t]),
                                channel_dropped=not keep))
        if not src or not keep:
            continue
        vals = []
        for si in src:
            sc = probs[:, col_of[si]].astype(np.float32)
            for ti in keep:
                y = (L[:, ti] == 1) & usable
                neg = (L[:, ti] == 0) & usable
                if not y.any() or not neg.any():
                    continue
                vals.append(S.auroc(sc, y, neg))
        if vals:
            rows.append(dict(model=model, source_species=source,
                             target_species=target, assay=m["assay"],
                             tissue=m["tissue"], n_pairs=len(vals),
                             auroc=float(np.mean(vals)), auprc=float("nan")))
    return rows, dropped


def main():
    os.makedirs(OUT, exist_ok=True)
    all_dropped, done, missing = [], 0, []
    for d in sorted(os.listdir(A_CACHE)):
        if "_to_" not in d:
            continue
        source, target = d.split("_to_")
        if source == target:
            continue
        outdir = os.path.join(OUT_CS, d)
        os.makedirs(outdir, exist_ok=True)
        for f in sorted(os.listdir(os.path.join(A_CACHE, d))):
            if not f.endswith(".npz"):
                continue
            model = f[:-4]
            rows, dropped = per_match(source, target, model)
            all_dropped += dropped
            if rows:
                pd.DataFrame(rows).to_csv(
                    os.path.join(outdir, f"{model}_per_match.csv"), index=False)
                done += 1
        print(f"  {d}", flush=True)
    dd = pd.DataFrame(all_dropped)
    if not dd.empty:
        dd.to_csv(os.path.join(OUT, "dropped_tracks.tsv"), sep="\t", index=False)
    print(f"\nwrote {done} per-match files to {OUT_CS}")
    if not dd.empty:
        u = dd.drop_duplicates(["source", "target", "assay", "tissue", "target_track"])
        print(f"target tracks below {MIN_POS} positives: {len(u)} "
              f"(across {u.groupby(['source','target','assay','tissue']).ngroups} channels)")
        ch = dd[dd.channel_dropped].drop_duplicates(["source", "target", "assay", "tissue"])
        print(f"channels dropped entirely: {len(ch)}")
        if len(ch):
            print(ch[["source", "target", "assay", "tissue"]].to_string(index=False))


if __name__ == "__main__":
    main()
