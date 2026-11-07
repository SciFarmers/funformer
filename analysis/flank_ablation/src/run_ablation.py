"""Score one (direction, condition, k, seed) and cache the matched-channel probs.

Mirrors analysis/tss_strata/src/cache_predictions.py: same inference function,
same matched-channel selection, same npz layout, so both analyses aggregate
through one code path. The intact condition is NOT run here -- Analysis A
already cached exactly those predictions.
"""
import argparse
import os
import sys

import numpy as np
import torch

# REPO_ROOT lets a frozen snapshot of this code run against the real
# repository, where the checkpoints and datasets live.
ROOT = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
# Outputs belong in the real analysis directory. Deriving this from
# __file__ puts them inside the code snapshot when one is in use.
HERE = os.path.join(ROOT, "analysis", "flank_ablation") if os.environ.get("REPO_ROOT") \
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis", "tss_strata", "src"))

from src_evaluate.evaluate_cross_species import run_inference      # noqa: E402
from src_evaluate.evaluate_test_set import load_backbone           # noqa: E402
from src.seq2fun.trainer import DualStrandWrapper                  # noqa: E402
from manifest import DATA, GENOME                                  # noqa: E402
from cache_predictions import matched_columns                      # noqa: E402
from ablation_dataset import (FlankAblationDataset, assert_intact,  # noqa: E402
                              test_chromosome)
from shuffle import gc_fraction                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--keep-bp", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--check-intact", action="store_true",
                    help="assert the intact window matches Seq2FunDataset, then exit")
    a = ap.parse_args()

    tag = f"{a.condition}_k{a.keep_bp}_s{a.seed}"
    out_dir = os.path.join(HERE, "data", "cache", f"{a.source}_to_{a.target}")
    out = os.path.join(out_dir, f"FunFormer_20k__{tag}.npz")
    if os.path.exists(out) and not a.check_intact:
        print(f"exists, skipping: {out}")
        return
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, model_class, seq_len, n_src = load_backbone(os.path.join(ROOT, a.checkpoint))

    if a.check_intact:
        n = assert_intact(DATA[a.target], GENOME[a.target], seq_len)
        print(f"intact matches Seq2FunDataset on {n} rows of {a.target} "
              f"(seq_len={seq_len})")
        return

    cols = matched_columns(a.source, a.target)
    model = DualStrandWrapper(backbone).to(device).eval()

    # Donor pool. Every random donor in this analysis must be sequence the
    # scoring model has NOT seen, or the contrasts carry a seen-vs-unseen
    # difference on top of the one they exist to isolate.
    #
    #   random_same, transfer       target genome, its TEST chromosome barred
    #                               (every donor there would overlap an
    #                               evaluated interval's own window). The source
    #                               model has never seen the target genome.
    #   random_same, within-species target's VALIDATION chromosome. Its training
    #                               chromosomes are sequence the model was fitted
    #                               on, which would make flank dependence rest on
    #                               seen donors while flank usefulness rests on
    #                               unseen ones.
    #   random_source               source genome, restricted to its held-out
    #                               chromosome, for the same reason. Overlap with
    #                               the source's own test windows is irrelevant:
    #                               the evaluated intervals are the target's.
    donor_sp = a.source if a.condition == "random_source" else a.target
    excl = only = None
    if a.condition == "random_same":
        if a.source == a.target:
            only = test_chromosome(DATA[a.target], split="validation")
            if only is None:
                print(f"NOTE: {a.target} validation split is not a single "
                      f"chromosome, so within-species donors fall back to "
                      f"training chromosomes and are SEEN sequence. This is an "
                      f"asymmetry against the transfer directions; recorded as "
                      f"a limitation.")
                excl = test_chromosome(DATA[a.target])
        else:
            excl = test_chromosome(DATA[a.target])
    elif a.condition == "random_source":
        only = test_chromosome(DATA[a.source])
        if only is None:
            raise SystemExit(
                f"{a.source} test split is not a single chromosome; "
                f"random_source needs one held-out chromosome to draw from.")

    log_dir = os.path.join(out_dir, f"donorlog__{tag}")
    ds = FlankAblationDataset(
        DATA[a.target], GENOME[a.target], seq_len, a.condition,
        keep_bp=a.keep_bp, seed=a.seed,
        donor_genome_path=GENOME[a.source] if a.condition == "random_source" else None,
        donor_exclude_chrom=excl, donor_only_chrom=only, log_dir=log_dir)
    print(f"{a.source}->{a.target} {tag}: seq_len={seq_len} cols={len(cols)} "
          f"N={len(ds)} donor_genome={donor_sp} "
          f"{'restricted_to=' + str(only) if only else 'excluded_chrom=' + str(excl)}")

    probs, _ = run_inference(model, ds, a.batch_size, a.num_workers, device)
    np.savez_compressed(out,
                        probs=probs[:, cols].astype(np.float16),
                        cols=np.asarray(cols, dtype=np.int32),
                        snapshot=os.environ.get("SNAPSHOT_ID", ""))

    # Fold the per-worker donor logs into one table and report the match rates.
    import glob
    import pandas as pd
    parts = sorted(glob.glob(os.path.join(log_dir, "donors_w*.tsv")))
    if parts:
        frames = []
        for f in parts:
            if os.path.getsize(f) == 0:
                continue                      # worker that logged nothing
            frames.append(pd.read_csv(f, sep="\t", dtype={"donor_chrom": str}))
        df = (pd.concat(frames, ignore_index=True).sort_values(["row", "side"])
              if frames else pd.DataFrame())

        # Completeness. Every interval contributes one record per replaced
        # flank, so a short table means records were lost rather than that
        # fewer were produced -- the failure mode that made the first run's
        # logs 3.3% short without anything erroring.
        lo = (seq_len - a.keep_bp) // 2
        per_interval = (1 if lo > 0 else 0) + (1 if lo + a.keep_bp < seq_len else 0)
        want = len(ds) * per_interval
        if len(df) != want:
            raise SystemExit(
                f"DONOR LOG INCOMPLETE: {len(df)} records, expected {want} "
                f"({len(ds)} intervals x {per_interval} flanks). Records were "
                f"lost; do not trust this run's provenance.")
        tsv = os.path.join(out_dir, f"donors__{tag}.tsv.gz")
        df.to_csv(tsv, sep="\t", index=False, compression="gzip")
        for f in parts:
            os.remove(f)
        os.rmdir(log_dir)
        d = (df.gc_orig - df.gc_new).abs()
        t = df.gc_tier.astype(float)
        r2, r3 = 100 * (t <= 0.02).mean(), 100 * (t <= 0.03).mean()
        unm = 100 * (~np.isfinite(t)).mean()
        ndon = df.donor_start.nunique()
        # Read chromosome sizes from the .fai, not from ds.donor: the genome is
        # opened lazily inside dataloader workers, so in the parent process
        # ds.donor is still None and this silently reported a zero-size pool.
        def _fai_sizes(fasta):
            out = {}
            with open(fasta + ".fai") as fh:
                for line in fh:
                    f = line.split("\t")
                    out[f[0]] = int(f[1])
            return out
        cs = _fai_sizes(GENOME[donor_sp])
        pool_bp = (cs.get(only, 0) if only is not None
                   else sum(L for c, L in cs.items() if c != excl))
        reuse = (len(df) * (seq_len - a.keep_bp) / 2) / pool_bp if pool_bp else float("nan")
        print(f"flanks={len(df)}  GC |delta| mean={d.mean():.4f} max={d.max():.4f}  "
              f"matched within 2%={r2:.1f}%  within 3%={r3:.1f}%  "
              f"unmatched(closest kept)={unm:.1f}%  distinct donor loci={ndon}  "
              f"pool={pool_bp/1e6:.1f}Mb  sequence reuse={reuse:.0f}x")
        # A restricted donor pool can run out of GC-compatible sequence. Say so
        # loudly rather than letting the closest-kept fallback hide it.
        if only is not None and (r3 < 90.0 or unm > 10.0):
            print(f"WARNING: donor pool {a.source}:{only} did not GC-match "
                  f"{100 - r3:.1f}% of flanks within 3%. The held-out chromosome "
                  f"may be too narrow in GC for this target; reported, not "
                  f"silently widened.")
        if ndon < 1000:
            print(f"WARNING: only {ndon} distinct donor loci drawn -- pool may be "
                  f"too small for independent flanks.")
        # Re-derive the pool rule from the condition and check it against the
        # coordinates actually written. A rule expressed only where the sampling
        # happens cannot catch an argument that never got passed.
        from verify_donor_pools import check_file
        _, nbad, note = check_file(tsv, a.source, a.target, a.condition)
        print(f"donor pool check: {note}")
        print(f"wrote {tsv}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
