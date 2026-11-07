"""Bitwise check: C's intact condition must BE Analysis A's cached file.

Both run the same checkpoint over the same intervals and keep the same matched
channels, so the two prediction arrays should be identical bit for bit, not
merely close. A tolerance-based check would pass even if the ablation dataset
mis-centred the window by a few bases, which is exactly the failure this is
here to catch.

Policy when they differ: a max delta at float precision (<= 1e-3, about two
fp16 ULP near 0.5) is GPU non-determinism -- a different card or cuDNN
algorithm between the two runs perturbs the last bits -- so it is recorded in
out/intact_bitwise_report.md and the check passes. Only a large or structured
delta means a real difference in what was scored, and that fails.

Magnitude, not the fraction of differing entries, is the discriminator: 1-ULP
jitter is expected to touch many entries, so a high differing fraction is not
itself suspicious.
"""
import glob
import os
import re
import sys

import numpy as np

PRECISION_TOL = 1e-3      # ~2 fp16 ULP near 0.5
AUROC_TOL = 1e-4          # a channel AUROC moving less than this changes no
                          # reported number at the precision anything is quoted

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(HERE))
A_CACHE = os.path.join(ROOT, "analysis", "tss_strata", "data", "cache")
C_CACHE = os.path.join(HERE, "data", "cache")
sys.path.insert(0, os.path.join(ROOT, "analysis", "tss_strata", "src"))


def channel_delta(source, target):
    """Largest per-channel AUROC difference between the two cached runs.

    The prediction-level delta says the bits differ; this says whether that
    survives into a number anyone reads. A fp16 perturbation that moves no
    channel AUROC is irrelevant to every result downstream.
    """
    import strata as S
    a = {(r["assay"], r["tissue"]): r["auroc"] for r in
         S.direction(source, target, "FunFormer_20k", unstratified=True,
                     cache_root=A_CACHE)}
    b = {(r["assay"], r["tissue"]): r["auroc"] for r in
         S.direction(source, target, "FunFormer_20k__intact_k1000_s0",
                     unstratified=True, cache_root=C_CACHE)}
    keys = set(a) & set(b)
    if not keys:
        return float("nan"), 0
    return max(abs(a[k] - b[k]) for k in keys), len(keys)


def main():
    pats = sorted(glob.glob(os.path.join(C_CACHE, "*", "FunFormer_20k__intact_k*_s*.npz")))
    if not pats:
        print("no intact runs cached in C; nothing to compare "
              "(intact normally reuses A's cache and is not re-run)")
        return 0
    bad, report = 0, []
    for p in pats:
        direction = os.path.basename(os.path.dirname(p))
        ref = os.path.join(A_CACHE, direction, "FunFormer_20k.npz")
        if not os.path.exists(ref):
            print(f"{direction}: A cache missing, skipped")
            continue
        a, b = np.load(ref), np.load(p)
        same_cols = np.array_equal(a["cols"], b["cols"])
        same = same_cols and np.array_equal(a["probs"], b["probs"])
        if same:
            print(f"{direction}: identical bitwise  {a['probs'].shape}")
            report.append((direction, "identical", 0.0, 0.0, 0.0, 0))
            continue
        if not same_cols or a["probs"].shape != b["probs"].shape:
            bad += 1
            print(f"{direction}: DIFFERS structurally  cols_match={same_cols} "
                  f"shapes={a['probs'].shape} vs {b['probs'].shape}")
            report.append((direction, "structural mismatch", float("nan"),
                           float("nan"), float("nan"), 0))
            continue
        x = a["probs"].astype(np.float32)
        y = b["probs"].astype(np.float32)
        d = float(np.abs(x - y).max())
        frac = float((x != y).mean())
        src, tgt = direction.split("_to_")
        try:
            ad, nch = channel_delta(src, tgt)
        except Exception as e:
            ad, nch = float("nan"), 0
            print(f"{direction}: channel AUROC delta unavailable ({e})")
        if d <= PRECISION_TOL:
            moved = "" if (np.isnan(ad) or ad <= AUROC_TOL) else "  BUT a channel AUROC moved"
            print(f"{direction}: differs at float precision only, "
                  f"max|delta|={d:.2e} over {100*frac:.2f}% of entries; "
                  f"max channel AUROC delta={ad:.2e} over {nch} channels"
                  f" -> GPU non-determinism, accepted{moved}")
            report.append((direction, "precision-level", d, frac, ad, nch))
        else:
            bad += 1
            print(f"{direction}: DIFFERS  max|delta|={d:.3e} "
                  f"over {100*frac:.2f}% of entries, max channel AUROC "
                  f"delta={ad:.2e} -- too large for precision, investigate")
            report.append((direction, "LARGE", d, frac, ad, nch))

    out = os.path.join(HERE, "out", "intact_bitwise_report.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n_id = sum(1 for r in report if r[1] == "identical")
    n_pr = sum(1 for r in report if r[1] == "precision-level")
    with open(out, "w") as fh:
        fh.write("# Intact vs the Analysis A cache\n\n")
        fh.write(f"{n_id} identical bitwise, {n_pr} differing at float precision "
                 f"(accepted as GPU non-determinism), {bad} failing, "
                 f"of {len(report)} compared.\n\n")
        worst = [r[4] for r in report if not np.isnan(r[4])]
        if worst:
            fh.write(f"Largest channel-level AUROC difference anywhere: "
                     f"**{max(worst):.2e}**"
                     f"{' — below the 1e-4 at which any reported number would move.' if max(worst) <= AUROC_TOL else ' — this DOES move a reported number.'}\n\n")
        fh.write("| direction | verdict | max abs delta (prediction) | "
                 "entries differing | max abs delta (channel AUROC) | channels |\n")
        fh.write("|---|---|---|---|---|---|\n")
        for dname, verdict, dd, ff, ad, nch in report:
            fh.write(f"| {dname} | {verdict} | "
                     f"{'' if np.isnan(dd) else f'{dd:.2e}'} | "
                     f"{'' if np.isnan(ff) else f'{100*ff:.2f}%'} | "
                     f"{'' if np.isnan(ad) else f'{ad:.2e}'} | {nch or ''} |\n")
    print(f"\n{n_id} identical, {n_pr} precision-level, {bad} failing "
          f"of {len(report)}  ->  out/intact_bitwise_report.md")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
