"""Distance from every test interval to the nearest protein-coding TSS.

One table per species, written to data/tss_distance/<species>.tsv, with the
interval's row index in the HF test split so downstream code can align to the
cached predictions positionally.

Two things decide whether this is right at all:

  * The GTF must come from the same ASSEMBLY the model was trained on.
    fetch_gtf.sh pins each one; zebrafish is the trap, since GRCz10 only
    exists up to Ensembl 91 and GRCz11 coordinates would be silently wrong.
  * Sequence names must agree. The HF intervals use bare Ensembl names
    ("21", "16"), as do the GTFs, so no translation is needed -- but the
    overlap is asserted rather than assumed, because a mismatch produces an
    empty join and a plausible-looking table of "no nearby TSS".

The test split is a single held-out chromosome per species (21, or 16 for
pig), so these distances describe that chromosome's gene density, not the
genome's.
"""
import argparse
import gzip
import os

import numpy as np
import pyarrow.ipc as ipc

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = "/cluster/projects/nn9780k/dat/training_data/seq2fun/binary"
HF = {"human": "hf_ENTEX_423_beds", "pig": "hf_pig_beds", "cattle": "hf_cattle_beds",
      "chicken": "hf_chicken_beds", "salmon": "hf_salmon_beds",
      "zebrafish": "hf_zebrafish_beds"}


def read_intervals(species, split="test"):
    path = os.path.join(DATA, HF[species], split, "data-00000-of-00001.arrow")
    with open(path, "rb") as fh:
        t = ipc.open_stream(fh).read_all()
    return (np.array(t["chrom"].to_pylist()),
            np.asarray(t["start"], dtype=np.int64),
            np.asarray(t["end"], dtype=np.int64))


def read_tss(gtf_gz, want_chroms):
    """Protein-coding transcript TSS, as {chrom: sorted array of positions}.

    TSS is the 5' end: `start` on + strand, `end` on -. Ensembl GTF is
    1-based inclusive; we keep it in that frame and compare against interval
    midpoints converted the same way, so the two never differ by a base.
    """
    by_chrom = {c: [] for c in want_chroms}
    with gzip.open(gtf_gz, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split("\t", 9)
            if f[2] != "transcript" or f[0] not in by_chrom:
                continue
            if 'transcript_biotype "protein_coding"' not in f[8]:
                continue
            by_chrom[f[0]].append(int(f[3]) if f[6] == "+" else int(f[4]))
    return {c: np.unique(np.array(v, dtype=np.int64)) for c, v in by_chrom.items()}


def nearest_distance(mid, tss_sorted):
    """|midpoint - nearest TSS| by binary search; inf where a chrom has none."""
    if len(tss_sorted) == 0:
        return np.full(len(mid), np.inf)
    i = np.searchsorted(tss_sorted, mid)
    lo = tss_sorted[np.clip(i - 1, 0, len(tss_sorted) - 1)]
    hi = tss_sorted[np.clip(i, 0, len(tss_sorted) - 1)]
    return np.minimum(np.abs(mid - lo), np.abs(mid - hi)).astype(np.int64)


def build(species):
    chrom, start, end = read_intervals(species)
    mid = (start + end) // 2 + 1          # 0-based half-open -> 1-based
    chroms = sorted(set(chrom.tolist()))
    tss = read_tss(os.path.join(HERE, "data", "gtf", f"{species}.gtf.gz"), chroms)

    empty = [c for c in chroms if len(tss[c]) == 0]
    if empty:
        raise SystemExit(
            f"{species}: no protein-coding TSS on {empty}. The GTF sequence "
            f"names do not match the interval names -- check the assembly.")

    dist = np.empty(len(mid), dtype=np.int64)
    for c in chroms:
        m = chrom == c
        dist[m] = nearest_distance(mid[m], tss[c])

    out = os.path.join(HERE, "data", "tss_distance", f"{species}.tsv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("row\tchrom\tmid\ttss_dist\n")
        for i in range(len(mid)):
            fh.write(f"{i}\t{chrom[i]}\t{mid[i]}\t{dist[i]}\n")

    q = np.percentile(dist, [10, 25, 50, 75, 90]).astype(int)
    print(f"{species:10s} n={len(mid):7d} chrom={','.join(chroms)} "
          f"TSS={sum(len(v) for v in tss.values()):6d}  "
          f"dist p10/25/50/75/90={q[0]:>6d}/{q[1]:>6d}/{q[2]:>7d}/{q[3]:>7d}/{q[4]:>8d}  "
          f"<=2kb={100*np.mean(dist <= 2000):5.1f}%  >10kb={100*np.mean(dist > 10000):5.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", nargs="*", default=sorted(HF))
    for sp in ap.parse_args().species:
        build(sp)
