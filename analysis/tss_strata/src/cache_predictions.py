"""Cache matched-channel predictions for one (direction, model), on GPU.

The existing pipeline keeps only per-match AUROC, so nothing downstream can be
re-sliced by TSS distance. This runs the same inference and writes the raw
per-interval probabilities instead -- but only for the matched source channels,
which is 6 to 88 columns rather than the model's full 37-423, so the whole
analysis caches in ~2 GB.

Caching rather than scoring in place is what makes the stratum thresholds free
to change: the 1 kb / 20 kb sensitivity run re-reads these files instead of
re-running 75 GPU jobs.

Inference is imported from src_evaluate.evaluate_cross_species, not
reimplemented, so transfer numbers out of this path are the same function the
main analysis used. Nothing in src_evaluate is modified.

Writes data/cache/<source>_to_<target>/<model>.npz with
  probs  (N, C) float16   -- sigmoid outputs, strand-averaged by DualStrandWrapper
  cols   (C,)   int32     -- source track indices, matching `probs` column order
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

# REPO_ROOT lets a frozen snapshot of this code run against the real
# repository, where the checkpoints and datasets live.
ROOT = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
# Outputs belong in the real analysis directory. Deriving this from
# __file__ puts them inside the code snapshot when one is in use.
HERE = os.path.join(ROOT, "analysis", "tss_strata") if os.environ.get("REPO_ROOT") \
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from src_evaluate.evaluate_cross_species import run_inference, _pair_key  # noqa: E402
from src_evaluate.evaluate_test_set import load_backbone                  # noqa: E402
from src.data.seq2fun import Seq2FunDataset                               # noqa: E402
from src.seq2fun.trainer import DualStrandWrapper                         # noqa: E402

sys.path.insert(0, os.path.join(HERE, "src"))
from manifest import DATA, GENOME                                          # noqa: E402


def matched_columns(source, target):
    """Source-model track indices this direction is scored on.

    Transfer: the source indices of every matched group for the pair.
    Self (source == target): that species' own indices across every pair it
    takes part in, which is the channel set the retention reference averages
    over in s3.transfer_retention.py.
    """
    mt = json.load(open(os.path.join(ROOT, "meta_info", "matched_tracks.json")))
    if source != target:
        return sorted({i for g in mt[_pair_key(source, target)][:] for i in g[source]})
    return sorted({i for k, groups in mt.items() if source in k.split("_")
                   for g in groups for i in g[source]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--model", required=True, help="model tag, e.g. FunFormer_20k")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    a = ap.parse_args()

    out_dir = os.path.join(HERE, "data", "cache", f"{a.source}_to_{a.target}")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{a.model}.npz")
    if os.path.exists(out):
        print(f"exists, skipping: {out}")
        return

    cols = matched_columns(a.source, a.target)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, model_class, seq_len, n_src = load_backbone(
        os.path.join(ROOT, a.checkpoint))
    if max(cols) >= n_src:
        raise SystemExit(
            f"matched index {max(cols)} exceeds the {n_src} tracks of "
            f"{a.checkpoint}: matched_tracks.json and this checkpoint disagree.")
    model = DualStrandWrapper(backbone).to(device).eval()

    ds = Seq2FunDataset(split="test", data_dir=DATA[a.target],
                        genome_path=GENOME[a.target], sequence_length=seq_len)
    print(f"{a.source}->{a.target} {a.model}: {model_class} seq_len={seq_len} "
          f"n_src={n_src} cols={len(cols)} N={len(ds)}")

    probs, _ = run_inference(model, ds, a.batch_size, a.num_workers, device)
    np.savez_compressed(out,
                        probs=probs[:, cols].astype(np.float16),
                        cols=np.asarray(cols, dtype=np.int32),
                        snapshot=os.environ.get("SNAPSHOT_ID", ""))
    print(f"wrote {out}  {probs[:, cols].shape}")


if __name__ == "__main__":
    main()
