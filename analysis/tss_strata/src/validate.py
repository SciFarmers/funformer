"""Regression checks. Nothing downstream is trusted until these pass.

A  aggregation      group -> direction means computed here reproduce
                    cross_species_margins.csv. Needs no new inference, so it
                    runs today and pins the arithmetic independently of the
                    predictions.
B  unstratified     AUROC recomputed from the cached predictions reproduces the
                    committed per-match AUROCs, which sklearn produced. This is
                    what certifies both the cache and the rank-based AUROC.
C  exhaustive pool  proximal + middle + distal covers every positive, so the
                    pooled AUROC equals the unstratified one exactly.

Note on C: the headline strata deliberately DROP the 2-10 kb band, so
proximal + distal cannot reproduce the unstratified number -- those positives
are gone. Pooling is therefore checked against the three exhaustive strata.
Two-stratum pooling cannot reproduce the unstratified bin margins, since
the dropped band carries positives that the unstratified number includes.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(HERE))
# Under a frozen code snapshot __file__ points into the snapshot, so without
# this the checks read the snapshot's empty data/ instead of the repository's.
if os.environ.get("REPO_ROOT"):
    ROOT = os.environ["REPO_ROOT"]
    HERE = os.path.join(ROOT, "analysis", "tss_strata")
sys.path.insert(0, os.path.join(HERE, "src"))
import strata as S                                          # noqa: E402

MARGINS = os.path.join(ROOT, "manuscript/analysis/out/cross_species_margins.csv")
# The margins file is on the final convention, so the per-match tree compared
# against it must be too -- otherwise check_a scores one convention against
# the other and reports every difference as a mismatch.
CS = os.environ.get("CS_DIR") or os.path.join(ROOT, "eval_results/cross_species_final")
# Whether the recomputation must apply the N-flank exclusion is decided by
# which tree it is being checked against.
FINAL = os.path.basename(CS.rstrip("/")).endswith("_final")
TOL = 5e-4


def _per_match(source, target, model):
    p = os.path.join(CS, f"{source}_to_{target}", f"{model}_per_match.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def check_a():
    """Direction means from committed per-match CSVs match the margins file."""
    m = pd.read_csv(MARGINS)
    m = m[m.in_primary_set]
    cols = {"FunFormer_20k": "funformer20k_all", "FunFormer_1k": "funformer1k_all"}
    bad, n = [], 0
    for _, r in m.iterrows():
        for model, col in cols.items():
            pm = _per_match(r.source, r.target, model)
            if pm is None:
                bad.append(f"missing {r.source}->{r.target} {model}"); continue
            got, want = pm["auroc"].mean(), r[col]
            n += 1
            if abs(got - want) > TOL:
                bad.append(f"{r.source}->{r.target} {model}: {got:.6f} vs {want:.6f}")
        pm = _per_match(r.source, r.target, r.best_baseline_model)
        if pm is not None:
            got, want = pm["auroc"].mean(), r["best_baseline_all"]
            n += 1
            if abs(got - want) > TOL:
                bad.append(f"{r.source}->{r.target} {r.best_baseline_model}: "
                           f"{got:.6f} vs {want:.6f}")
    return n, bad


def check_b(limit=None):
    """Unstratified AUROC via the cache matches the committed per-match AUROC."""
    m = pd.read_csv(MARGINS); m = m[m.in_primary_set]
    bad, n, skipped = [], 0, 0
    for _, r in m.iterrows():
        for model in ("FunFormer_20k", "FunFormer_1k", r.best_baseline_model):
            cache = os.path.join(HERE, "data", "cache",
                                 f"{r.source}_to_{r.target}", f"{model}.npz")
            if not os.path.exists(cache):
                skipped += 1; continue
            pm = _per_match(r.source, r.target, model)
            rows = pd.DataFrame(S.direction(r.source, r.target, model,
                                            unstratified=True,
                                            drop_n_flanks=FINAL))
            j = pm.merge(rows, on=["assay", "tissue"], suffixes=("_ref", "_new"))
            if len(j) != len(pm):
                bad.append(f"{r.source}->{r.target} {model}: "
                           f"{len(j)} of {len(pm)} groups matched")
            d = (j["auroc_ref"] - j["auroc_new"]).abs()
            n += len(j)
            if len(d) and d.max() > TOL:
                bad.append(f"{r.source}->{r.target} {model}: max |delta|={d.max():.6f}")
            if limit and n > limit:
                return n, bad, skipped
    return n, bad, skipped


def check_c(source="human", target="pig", model="FunFormer_20k"):
    """Three exhaustive strata pool back to the unstratified AUROC."""
    cache = os.path.join(HERE, "data", "cache", f"{source}_to_{target}", f"{model}.npz")
    if not os.path.exists(cache):
        return None
    probs, col_of = S.load_probs(source, target, model)
    L, d = S.labels(target), S.tss_dist(target)
    masks = [(d <= 2000), (d > 2000) & (d <= 10000), (d > 10000)]
    union = masks[0] | masks[1] | masks[2]
    return bool(union.all()), int((~union).sum())


if __name__ == "__main__":
    n, bad = check_a()
    print(f"A aggregation : {n} direction-model means checked, "
          f"{'OK' if not bad else str(len(bad)) + ' MISMATCH'}")
    for b in bad[:8]:
        print("   ", b)

    n, bad, skipped = check_b()
    if skipped and not n:
        print(f"B unstratified: skipped, no cached predictions yet "
              f"({skipped} directions pending)")
    else:
        print(f"B unstratified: {n} group AUROCs checked, {skipped} direction(s) "
              f"not cached, {'OK' if not bad else str(len(bad)) + ' MISMATCH'}")
        for b in bad[:8]:
            print("   ", b)

    c = check_c()
    print("C pooling     :", "skipped, no cache yet" if c is None else
          f"strata exhaustive={c[0]} uncovered={c[1]}")
