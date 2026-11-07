"""Channel -> direction -> bin aggregates for the TSS strata, plus the stats.

Stratum sizes are written before any aggregate, because a margin computed over
very few positives is not comparable to one computed over thousands and the
reader needs to see which is which.

Margins, all per stratum:
  margin_arch   FunFormer-20k - FunFormer-1k        (primary)
  margin_base   FunFormer-20k - best published baseline, that baseline held
                FIXED at the direction's unstratified winner so proximal and
                distal share one reference
  retained      transfer / the target's own FunFormer-20k on the same channels
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
HERE = os.path.join(ROOT, "analysis", "tss_strata") if os.environ.get("REPO_ROOT") \
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(ROOT, "manuscript", "analysis"))
import strata as S                                              # noqa: E402

OUT = os.path.join(HERE, "out")
MARGINS = os.path.join(ROOT, "manuscript/analysis/out/cross_species_margins.csv")


def stratum_sizes(strata):
    rows = []
    for sp in S.TAG:
        L, d = S.labels(sp), S.tss_dist(sp)
        ok, masks = S.eligible_tracks(sp, strata)
        for ti in range(L.shape[1]):
            keep, counts = ok[ti]
            rows.append(dict(species=sp, track=ti, eligible=keep,
                             **{f"n_{k}": v for k, v in counts.items()}))
    return pd.DataFrame(rows)


def direction_survival(strata):
    """Matched channels per direction that clear the 50-positive floor.

    The floor is applied jointly across strata, so a group either enters every
    stratum or none and this is one number per direction rather than one per
    stratum. Directions keeping under half their channels are flagged: their
    stratified margins rest on a different, smaller channel set than the
    unstratified margin they will be read against.
    """
    m = pd.read_csv(MARGINS); m = m[m.in_primary_set]
    rows = []
    for _, r in m.iterrows():
        ok, _ = S.eligible_tracks(r.target, strata)
        tot = kept = 0
        for g in S.matches(r.source, r.target):
            tot += 1
            if g[r.source] and any(ok[t][0] for t in g[r.target]):
                kept += 1
        rows.append(dict(source=r.source, target=r.target, bin=r["bin"],
                         n_channels=tot, n_surviving=kept,
                         frac=kept / tot if tot else np.nan,
                         flag_under_half=bool(tot and kept / tot < 0.5)))
    return pd.DataFrame(rows)


def channel_table(strata, cache_dir="cache"):
    m = pd.read_csv(MARGINS); m = m[m.in_primary_set]
    rows = []
    for _, r in m.iterrows():
        models = {"ff20k": "FunFormer_20k", "ff1k": "FunFormer_1k",
                  "base": r.best_baseline_model}
        got = {}
        for key, model in models.items():
            p = os.path.join(HERE, "data", cache_dir,
                             f"{r.source}_to_{r.target}", f"{model}.npz")
            if not os.path.exists(p):
                got = None
                break
            got[key] = pd.DataFrame(S.direction(r.source, r.target, model, strata,
                                                drop_n_flanks=True))
        if got is None:
            continue
        self_p = os.path.join(HERE, "data", cache_dir,
                              f"{r.target}_to_{r.target}", "FunFormer_20k.npz")
        # Self-reference: the target's own model, on the channels THIS
        # direction is scored on, so transfer and self are measured alike.
        selfdf = (pd.DataFrame(S.direction(r.target, r.target, "FunFormer_20k",
                                           strata, groups_for=(r.source, r.target),
                                           drop_n_flanks=True))
                  if os.path.exists(self_p) else None)

        j = got["ff20k"].merge(got["ff1k"], on=["assay", "tissue", "stratum"],
                               suffixes=("_ff20k", "_ff1k"))
        j = j.merge(got["base"][["assay", "tissue", "stratum", "auroc"]],
                    on=["assay", "tissue", "stratum"]).rename(
                        columns={"auroc": "auroc_base"})
        if selfdf is not None:
            j = j.merge(selfdf[["assay", "tissue", "stratum", "auroc"]],
                        on=["assay", "tissue", "stratum"], how="left").rename(
                            columns={"auroc": "auroc_self"})
        else:
            j["auroc_self"] = np.nan
        j["source"], j["target"] = r.source, r.target
        j["bin"], j["divergence_mya"] = r["bin"], r.divergence_mya
        j["baseline_model"] = r.best_baseline_model
        j["margin_arch"] = (j.auroc_ff20k - j.auroc_ff1k) * 100
        j["margin_base"] = (j.auroc_ff20k - j.auroc_base) * 100
        j["retained"] = j.auroc_ff20k / j.auroc_self * 100
        rows.append(j)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def to_direction(ch):
    g = ch.groupby(["source", "target", "bin", "divergence_mya", "stratum"],
                   as_index=False).agg(
        n_channels=("assay", "size"),
        auroc_ff20k=("auroc_ff20k", "mean"), auroc_ff1k=("auroc_ff1k", "mean"),
        auroc_base=("auroc_base", "mean"), margin_arch=("margin_arch", "mean"),
        margin_base=("margin_base", "mean"), retained=("retained", "mean"))
    return g


def run_stats(direction_tbl):
    import stats_gradient as SG
    out = []
    wide = direction_tbl.pivot_table(
        index=["source", "target"], columns="stratum",
        values="margin_arch").reset_index()
    if {"proximal", "distal"} <= set(wide.columns):
        wide["difference"] = wide["distal"] - wide["proximal"]
    for col in [c for c in ("proximal", "distal", "difference") if c in wide]:
        df = wide[["source", "target", col]].dropna()
        if len(df) < 3:
            continue
        sp = SG.species_label_perm(df, col)
        out.append(dict(quantity=col, n=len(df), **{k: v for k, v in sp.items()}
                        if isinstance(sp, dict) else {"result": str(sp)}))
    return pd.DataFrame(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, st in (("primary", S.STRATA), ("sensitivity", S.STRATA_SENS)):
        sz = stratum_sizes(st)
        sz.to_csv(os.path.join(OUT, f"stratum_sizes_{name}.tsv"), sep="\t", index=False)
        print(f"[{name}] stratum sizes -> out/stratum_sizes_{name}.tsv  "
              f"({int(sz.eligible.sum())} of {len(sz)} tracks eligible)")

        sv = direction_survival(st)
        sv.to_csv(os.path.join(OUT, f"direction_survival_{name}.tsv"), sep="\t", index=False)
        flagged = sv[sv.flag_under_half]
        print(f"[{name}] channels surviving the floor: "
              f"{int(sv.n_surviving.sum())} of {int(sv.n_channels.sum())} "
              f"({100*sv.n_surviving.sum()/sv.n_channels.sum():.1f}%), "
              f"{len(flagged)} direction(s) under half")
        if len(flagged):
            print(flagged[["source", "target", "n_channels", "n_surviving"]]
                  .to_string(index=False))

        ch = channel_table(st)
        if ch.empty:
            print(f"[{name}] no cached predictions yet; aggregates skipped")
            continue
        ch.to_csv(os.path.join(OUT, f"channels_{name}.tsv"), sep="\t", index=False)
        d = to_direction(ch)
        d.to_csv(os.path.join(OUT, f"directions_{name}.tsv"), sep="\t", index=False)
        b = d.groupby(["bin", "stratum"], as_index=False)[
            ["margin_arch", "margin_base", "retained"]].mean()
        b.to_csv(os.path.join(OUT, f"bins_{name}.tsv"), sep="\t", index=False)
        print(b.to_string(index=False))
        try:
            st_tbl = run_stats(d)
            if not st_tbl.empty:
                st_tbl.to_csv(os.path.join(OUT, f"stats_{name}.tsv"), sep="\t", index=False)
        except Exception as e:
            print(f"[{name}] stats skipped: {e}")


if __name__ == "__main__":
    main()
