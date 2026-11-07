"""Within-species per-track AUROC/AUPRC with and without the N-flank exclusion.

Computes both in one pass so the pair is exactly comparable: same model, same
predictions, differing only in which intervals are scored.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HERE = os.path.join(ROOT, "analysis", "nflank")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis", "tss_strata", "src"))

from sklearn.metrics import average_precision_score, roc_auc_score   # noqa: E402
from src_evaluate.evaluate_cross_species import run_inference        # noqa: E402
from src_evaluate.evaluate_test_set import load_backbone             # noqa: E402
from src.data.seq2fun import Seq2FunDataset                          # noqa: E402
from src.seq2fun.trainer import DualStrandWrapper                    # noqa: E402
from manifest import DATA, GENOME                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    a = ap.parse_args()

    out = os.path.join(HERE, "out", f"{a.species}__{a.model}.csv")
    if os.path.exists(out):
        print(f"exists, skipping: {out}")
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)

    keep = np.loadtxt(os.path.join(ROOT, "analysis", "common", "data",
                                   "n_content", f"{a.species}.tsv"),
                      skiprows=1, usecols=3, dtype=bool)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, cls, seq_len, n_t = load_backbone(os.path.join(ROOT, a.checkpoint))
    model = DualStrandWrapper(backbone).to(device).eval()
    ds = Seq2FunDataset(split="test", data_dir=DATA[a.species],
                        genome_path=GENOME[a.species], sequence_length=seq_len)
    print(f"{a.species} {a.model}: {cls} seq_len={seq_len} tracks={n_t} "
          f"N={len(ds)} kept={int(keep.sum())}")
    probs, labels = run_inference(model, ds, a.batch_size, a.num_workers, device)

    rows = []
    for ti in range(labels.shape[1]):
        y, s = labels[:, ti], probs[:, ti]
        r = dict(track_idx=ti, positives_all=int((y == 1).sum()),
                 positives_kept=int((y[keep] == 1).sum()))
        for tag, m in (("all", slice(None)), ("kept", keep)):
            yy, ss = y[m], s[m]
            if len(np.unique(yy)) < 2:
                r[f"auroc_{tag}"] = np.nan
                r[f"auprc_{tag}"] = np.nan
            else:
                r[f"auroc_{tag}"] = float(roc_auc_score(yy, ss))
                r[f"auprc_{tag}"] = float(average_precision_score(yy, ss))
        rows.append(r)
    df = pd.DataFrame(rows)
    df.insert(0, "model", a.model)
    df.insert(0, "species", a.species)
    df.to_csv(out, index=False)
    d = df.dropna(subset=["auroc_all", "auroc_kept"])
    print(f"mean AUROC all={d.auroc_all.mean():.5f} kept={d.auroc_kept.mean():.5f} "
          f"delta={d.auroc_kept.mean()-d.auroc_all.mean():+.5f}")
    print(f"mean AUPRC all={d.auprc_all.mean():.5f} kept={d.auprc_kept.mean():.5f} "
          f"delta={d.auprc_kept.mean()-d.auprc_all.mean():+.5f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
