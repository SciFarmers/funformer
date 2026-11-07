"""What the N-flank-excluded intervals actually are.

Two questions the exclusion raises: is the central 1 kb -- the part the model
always sees -- also unplaced, and do these intervals carry any positive labels
at all? If they are overwhelmingly all-negative, excluding them removes
background rather than signal, which changes what the AUROC means.
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc

ROOT = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.join(ROOT, "analysis", "common")
sys.path.insert(0, HERE)
from n_content import read_chrom, FASTA, HF, DATA, GEN, L_WIN, KEEP, MAX_N_FRAC  # noqa


def main():
    rows = []
    for sp in sorted(FASTA):
        path = os.path.join(DATA, HF[sp], "test", "data-00000-of-00001.arrow")
        with open(path, "rb") as fh:
            t = ipc.open_stream(fh).read_all()
        chrom = t["chrom"].to_pylist()[0]
        start = np.asarray(t["start"], dtype=np.int64)
        end = np.asarray(t["end"], dtype=np.int64)
        L = np.array(t["labels"].to_pylist(), dtype=np.int8)

        seq = read_chrom(os.path.join(GEN, FASTA[sp]), chrom)
        arr = np.frombuffer(seq, dtype=np.uint8)
        isn = ((arr == 78) | (arr == 110)).astype(np.int32)
        cum = np.concatenate(([0], np.cumsum(isn)))
        size = len(arr)

        def n_frac(a, b):
            a_c, b_c = np.clip(a, 0, size), np.clip(b, 0, size)
            return ((cum[b_c] - cum[a_c]) + (a_c - a) + (b - b_c)) / (b - a)

        keep = np.loadtxt(os.path.join(HERE, "data", "n_content", f"{sp}.tsv"),
                          skiprows=1, usecols=3, dtype=bool)
        mid = (start + end) // 2
        w0 = mid - L_WIN // 2
        lo = (L_WIN - KEEP) // 2
        centre = n_frac(w0 + lo, w0 + lo + KEEP)

        ex = ~keep
        pos_per_interval = (L == 1).sum(1)
        rows.append(dict(
            species=sp, excluded=int(ex.sum()), kept=int(keep.sum()),
            centre_n_mean=float(centre[ex].mean()) if ex.any() else np.nan,
            centre_n_median=float(np.median(centre[ex])) if ex.any() else np.nan,
            centre_gt50_pct=100 * float((centre[ex] > 0.5).mean()) if ex.any() else np.nan,
            centre_eq0_pct=100 * float((centre[ex] == 0).mean()) if ex.any() else np.nan,
            excl_all_negative_pct=100 * float((pos_per_interval[ex] == 0).mean()) if ex.any() else np.nan,
            kept_all_negative_pct=100 * float((pos_per_interval[keep] == 0).mean()),
            excl_positives_total=int(pos_per_interval[ex].sum()),
            all_positives_total=int(pos_per_interval.sum()),
            excl_share_of_positives_pct=100 * float(pos_per_interval[ex].sum() / max(pos_per_interval.sum(), 1)),
        ))
    d = pd.DataFrame(rows)
    os.makedirs(os.path.join(HERE, "data", "n_content"), exist_ok=True)
    d.to_csv(os.path.join(HERE, "data", "n_content", "excluded_profile.tsv"),
             sep="\t", index=False)
    pd.set_option("display.width", 200)
    print("Excluded intervals: central 1 kb N content\n")
    print(d[["species", "excluded", "centre_n_mean", "centre_n_median",
             "centre_gt50_pct", "centre_eq0_pct"]].round(3).to_string(index=False))
    print("\nLabel content of excluded vs kept intervals\n")
    print(d[["species", "excluded", "excl_all_negative_pct", "kept_all_negative_pct",
             "excl_positives_total", "all_positives_total",
             "excl_share_of_positives_pct"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
