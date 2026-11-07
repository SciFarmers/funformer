"""Per-interval N content of the two 20 kb flanks, for every species.

Two uses:

  * Ablation exclusion. A flank that is mostly N cannot be GC-matched to a
    donor, and the dinucleotide shuffle preserves N, so such an interval is
    effectively never ablated -- it is identical to intact in all three
    conditions and dilutes the measured flank effect. Those intervals are
    dropped from the ablation comparisons and from the intact reference, so
    every condition runs on one interval set.
  * Methods. The same quantity describes the main evaluation: how much of each
    species' test set sits beside unplaced sequence.

The test split is a single chromosome per species, so only that chromosome is
read, straight from the FASTA using the .fai offsets. No pysam, no genome in
RAM.

Flanks follow the ablation geometry: a 20 kb window centred on the interval
midpoint, central 1 kb kept, so each flank is 9.5 kb.
"""
import os
import sys

import numpy as np
import pyarrow.ipc as ipc

ROOT = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.join(ROOT, "analysis", "common")
GEN = "/cluster/projects/nn9780k/dat/training_data/genomes"
DATA = "/cluster/projects/nn9780k/dat/training_data/seq2fun/binary"

FASTA = {"human": "Homo_sapiens.GRCh38.dna.primary_assembly.fa",
         "pig": "Sus_scrofa.Sscrofa11.1.dna.toplevel.fa",
         "cattle": "Bos_taurus.ARS-UCD1.2.dna.toplevel.fa",
         "chicken": "Gallus_gallus.GRCg6a.dna.toplevel.fa",
         "salmon": "Salmo_salar.Ssal_v3.1.dna.toplevel.fa",
         "zebrafish": "Danio_rerio.GRCz10.dna_sm.toplevel.fa"}
HF = {"human": "hf_ENTEX_423_beds", "pig": "hf_pig_beds", "cattle": "hf_cattle_beds",
      "chicken": "hf_chicken_beds", "salmon": "hf_salmon_beds",
      "zebrafish": "hf_zebrafish_beds"}

L_WIN, KEEP = 20000, 1000
MAX_N_FRAC = 0.50


def read_chrom(fasta, want):
    for line in open(fasta + ".fai"):
        n, ln, off, lb, _ = line.split("\t")[:5]
        if n == want:
            ln, off, lb = int(ln), int(off), int(lb)
            with open(fasta, "rb") as fh:
                fh.seek(off)
                raw = fh.read(ln + ln // lb + 16)
            return raw.replace(b"\n", b"")[:ln]
    raise SystemExit(f"{want} not found in {fasta}.fai")


def build(species):
    path = os.path.join(DATA, HF[species], "test", "data-00000-of-00001.arrow")
    with open(path, "rb") as fh:
        t = ipc.open_stream(fh).read_all()
    chrom = t["chrom"].to_pylist()
    start = np.asarray(t["start"], dtype=np.int64)
    end = np.asarray(t["end"], dtype=np.int64)
    uc = set(chrom)
    if len(uc) != 1:
        raise SystemExit(f"{species}: test split spans {len(uc)} chromosomes")
    c = uc.pop()

    seq = read_chrom(os.path.join(GEN, FASTA[species]), c)
    arr = np.frombuffer(seq, dtype=np.uint8)
    isn = ((arr == 78) | (arr == 110)).astype(np.int32)       # N or n
    cum = np.concatenate(([0], np.cumsum(isn)))               # prefix sums
    size = len(arr)

    def n_frac(a, b):
        """N fraction of [a, b), counting off-chromosome padding as N."""
        a_c, b_c = np.clip(a, 0, size), np.clip(b, 0, size)
        inside = cum[b_c] - cum[a_c]
        pad = (a_c - a) + (b - b_c)
        return (inside + pad) / (b - a)

    half = L_WIN // 2
    mid = (start + end) // 2
    w0 = mid - half
    lo = (L_WIN - KEEP) // 2
    left = n_frac(w0, w0 + lo)
    right = n_frac(w0 + lo + KEEP, w0 + L_WIN)
    keep = (left <= MAX_N_FRAC) & (right <= MAX_N_FRAC)

    out = os.path.join(HERE, "data", "n_content", f"{species}.tsv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("row\tn_frac_left\tn_frac_right\tkeep\n")
        for i in range(len(mid)):
            fh.write(f"{i}\t{left[i]:.4f}\t{right[i]:.4f}\t{int(keep[i])}\n")
    return dict(species=species, chrom=c, n_intervals=len(mid),
                excluded=int((~keep).sum()),
                excluded_pct=100 * float((~keep).mean()),
                either_side_gt50=100 * float((~keep).mean()),
                left_gt50=100 * float((left > MAX_N_FRAC).mean()),
                right_gt50=100 * float((right > MAX_N_FRAC).mean()),
                both_gt50=100 * float(((left > MAX_N_FRAC) & (right > MAX_N_FRAC)).mean()))


def main():
    import pandas as pd
    rows = [build(sp) for sp in sorted(FASTA)]
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(HERE, "data", "n_content", "summary.tsv"),
             sep="\t", index=False)
    print("Test intervals whose 20 kb window is more than 50% N on a side\n")
    print(d[["species", "chrom", "n_intervals", "left_gt50", "right_gt50",
             "both_gt50", "either_side_gt50", "excluded"]]
          .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
