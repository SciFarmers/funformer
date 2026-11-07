"""Numbers-only summary for Analysis A. No interpretation."""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(ROOT, "manuscript", "analysis"))
import strata as S                                     # noqa: E402
import stats_gradient as SG                            # noqa: E402

OUT = os.path.join(HERE, "out")


def assay_of_track(species):
    mt = json.load(open(os.path.join(ROOT, "meta_info", "matched_tracks.json")))
    out = {}
    for k, groups in mt.items():
        if species not in k.split("_"):
            continue
        for g in groups:
            for i in g[species]:
                out[i] = g["assay"]
    return out


def n_per_stratum(strata):
    rows = []
    for sp in sorted(S.TAG):
        L, d = S.labels(sp), S.tss_dist(sp)
        a_of = assay_of_track(sp)
        masks = {k: (d >= lo) & (d <= hi) for k, (lo, hi) in strata.items()}
        for ti, assay in sorted(a_of.items()):
            y = L[:, ti] == 1
            rows.append(dict(species=sp, track=ti, assay=assay,
                             **{f"n_{k}": int((y & m).sum()) for k, m in masks.items()}))
    df = pd.DataFrame(rows)
    return df.groupby(["species", "assay"], as_index=False).agg(
        tracks=("track", "size"),
        n_proximal_median=("n_proximal", "median"),
        n_distal_median=("n_distal", "median"),
        n_proximal_min=("n_proximal", "min"),
        n_distal_min=("n_distal", "min"))


def both_schemes(dirs, col):
    """Species-level exact test and the direction-level shuffle, for one column."""
    rng = np.random.default_rng(0)
    sp = SG.species_label_perm(dirs[["source", "target", col]].dropna(), col)
    sub = dirs.dropna(subset=[col])
    r_dir, p_dir = SG.pearson_perm(sub["divergence_mya"], sub[col], rng)
    return dict(quantity=col, n=len(sub), r=sp["r"],
                p_species_level=sp["p_r"], p_direction_level=p_dir)


def main():
    for name, strata in (("primary", S.STRATA), ("sensitivity", S.STRATA_SENS)):
        dpath = os.path.join(OUT, f"directions_{name}.tsv")
        if not os.path.exists(dpath):
            print(f"[{name}] no directions table yet"); continue
        print(f"\n{'='*72}\nANALYSIS A — {name} strata "
              f"({strata['proximal'][1]/1000:g} kb / >{strata['distal'][0]/1000:g} kb)\n{'='*72}")

        print("\n-- 1. positives per stratum, per species per assay --")
        print(n_per_stratum(strata).to_string(index=False))

        d = pd.read_csv(dpath, sep="\t")
        w = d.pivot_table(index=["source", "target", "bin", "divergence_mya"],
                          columns="stratum",
                          values=["margin_arch", "margin_base", "retained"]).reset_index()
        w.columns = ["_".join(c).strip("_") for c in w.columns]
        for mcol in ("margin_arch", "margin_base"):
            w[f"{mcol}_diff"] = w[f"{mcol}_distal"] - w[f"{mcol}_proximal"]

        print("\n-- 2a. distal minus proximal margin, per direction (pp) --")
        print(w[["source", "target", "bin", "margin_arch_proximal", "margin_arch_distal",
                 "margin_arch_diff", "margin_base_proximal", "margin_base_distal",
                 "margin_base_diff"]].round(2).to_string(index=False))

        print("\n-- 2b. per bin (pp) --")
        print(w.groupby("bin", as_index=False)[
            ["margin_arch_proximal", "margin_arch_distal", "margin_arch_diff",
             "margin_base_proximal", "margin_base_distal", "margin_base_diff"]
        ].mean().round(2).to_string(index=False))

        print("\n-- 3. retained fraction per stratum per bin (%) --")
        print(w.groupby("bin", as_index=False)[["retained_proximal", "retained_distal"]]
              .mean().round(2).to_string(index=False))

        print("\n-- 4. permutation P values, both schemes --")
        stats = []
        for mcol in ("margin_arch", "margin_base"):
            for part in ("proximal", "distal", "diff"):
                col = f"{mcol}_{part}"
                try:
                    stats.append(both_schemes(w, col))
                except Exception as e:
                    stats.append(dict(quantity=col, n=0, r=np.nan,
                                      p_species_level=np.nan, p_direction_level=np.nan))
        st = pd.DataFrame(stats)
        print(st.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        st.to_csv(os.path.join(OUT, f"stats_{name}.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
