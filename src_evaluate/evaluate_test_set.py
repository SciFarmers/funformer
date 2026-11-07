"""Test-set evaluation for seq2fun models.

Loads the best_model saved inside a checkpoint directory, runs inference on
the test split of a Seq2FunDataset, and writes per-track AUROC/AUPRC tables.

The model class and sequence_length are read automatically from
``{checkpoint_dir}/best_model/model_config.json`` — no need to specify them.

Output (written to --output-dir, default ``eval_results/test_set``):
    {exp_tag}_per_track.csv  — columns: track_idx, auroc, auprc
    {exp_tag}_summary.csv    — single-row summary: exp_name, auroc_mean,
                               auroc_median, auroc_std, auprc_mean, auprc_std

where ``exp_tag`` is the basename of --checkpoint-dir.

Usage:
    python -m src_evaluate.evaluate_test_set \\
        --checkpoint-dir checkpoints/funformer_entex_423_len20000_lr5e-4_b250_gac2 \\
        --data-dir /cluster/.../hf_ENTEX_423_beds \\
        --genome-path /cluster/.../Homo_sapiens.GRCh38.dna.primary_assembly.fa \\
        --num-tracks 423 \\
        --batch-size 1000 \\
        --num-workers 8 \\
        --output-dir eval_results/test_set
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.seq2fun import Seq2FunDataset
from src.seq2fun.baseline_models import DanQ, DeepATT, DeepFormer, DeepSEA
from src.seq2fun.funformer import FunFormer
from src.seq2fun.trainer import DualStrandWrapper

# ---------------------------------------------------------------------------
# Registry — must match trainer.py MODEL_CLASSES
# ---------------------------------------------------------------------------

MODEL_CLASSES = {
    "DeepSEA": DeepSEA,
    "DanQ": DanQ,
    "DeepFormer": DeepFormer,
    "DeepATT": DeepATT,
    "FunFormer": FunFormer,
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_backbone(best_model_dir: str) -> tuple:
    """Load backbone from *best_model_dir*.

    Reads model_config.json to discover the model class and sequence_length,
    then delegates to ModelClass.from_pretrained() (Seq2FunMixin) for the
    actual weight loading — exactly as vep_eqtl_finemap.py does.

    Returns (backbone, model_class_name, sequence_length).
    The backbone is on the appropriate device and in eval mode.
    """
    config_path = os.path.join(best_model_dir, "model_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No model_config.json found in '{best_model_dir}'.\n"
            "The checkpoint may pre-date auto-config saving; re-train or "
            "create model_config.json manually."
        )

    with open(config_path) as f:
        config = json.load(f)

    model_class_name: str = config["model_class"]
    backbone_kwargs: dict = dict(config["backbone_kwargs"])

    if model_class_name not in MODEL_CLASSES:
        raise ValueError(
            f"Unknown model_class '{model_class_name}'. "
            f"Expected one of {list(MODEL_CLASSES.keys())}."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use the Seq2FunMixin from_pretrained() — single source of truth for
    # weight loading logic (safetensors / pytorch_model.bin fallback).
    cls = MODEL_CLASSES[model_class_name]
    backbone = cls.from_pretrained(best_model_dir, device=device)
    # from_pretrained() already calls model.eval() and model.to(device)

    # n_targets and sequence_length are always persisted in model_config.json
    # via get_backbone_config() — no need for the caller to supply them.
    sequence_length = int(backbone_kwargs.get("sequence_length", 1000))
    n_targets = int(
        backbone_kwargs.get("n_targets")
        or backbone_kwargs.get("num_tracks")  # older checkpoints name it this way
        or backbone_kwargs.get("n_targets", 0)
    )

    return backbone, model_class_name, sequence_length, n_targets


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: DualStrandWrapper,
    dataset: Seq2FunDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run full-dataset inference.

    Returns
    -------
    probs  : ndarray (N, T)  — sigmoid probabilities
    labels : ndarray (N, T)  — ground-truth binary labels
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_logits = []
    all_labels = []

    model.eval()
    for batch in tqdm(loader, desc="Inference", unit="batch"):
        fwd_ids = batch["fwd_ids"].to(device)
        rev_ids = batch["rev_ids"].to(device)
        labels  = batch["labels"]

        out = model(fwd_ids=fwd_ids, rev_ids=rev_ids)
        all_logits.append(out["logits"].cpu().float())
        all_labels.append(labels.float())

    logits = torch.cat(all_logits, dim=0).numpy()   # (N, T)
    labels = torch.cat(all_labels, dim=0).numpy()   # (N, T)
    probs  = torch.sigmoid(torch.from_numpy(logits)).numpy()
    return probs, labels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_per_track_metrics(
    probs: np.ndarray, labels: np.ndarray
) -> pd.DataFrame:
    """Compute per-track AUROC and AUPRC.

    Tracks with only one class in the test set are skipped (NaN).
    """
    n_tracks = labels.shape[1]
    records = []
    for i in range(n_tracks):
        y_true = labels[:, i]
        y_score = probs[:, i]
        if len(np.unique(y_true)) < 2:
            records.append({"track_idx": i, "auroc": float("nan"), "auprc": float("nan")})
            continue
        try:
            auroc = float(roc_auc_score(y_true, y_score))
            auprc = float(average_precision_score(y_true, y_score))
        except Exception:
            auroc, auprc = float("nan"), float("nan")
        records.append({"track_idx": i, "auroc": auroc, "auprc": auprc})

    return pd.DataFrame(records)


def compute_summary(per_track: pd.DataFrame, exp_name: str) -> pd.DataFrame:
    auroc_vals = per_track["auroc"].dropna().values
    auprc_vals = per_track["auprc"].dropna().values
    return pd.DataFrame([{
        "exp_name":      exp_name,
        "n_tracks_eval": len(auroc_vals),
        "auroc_mean":    float(np.mean(auroc_vals))   if len(auroc_vals) else float("nan"),
        "auroc_median":  float(np.median(auroc_vals)) if len(auroc_vals) else float("nan"),
        "auroc_std":     float(np.std(auroc_vals))    if len(auroc_vals) else float("nan"),
        "auprc_mean":    float(np.mean(auprc_vals))   if len(auprc_vals) else float("nan"),
        "auprc_std":     float(np.std(auprc_vals))    if len(auprc_vals) else float("nan"),
    }])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained seq2fun model on the test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint-dir", required=True,
                   help="Top-level experiment dir, e.g. checkpoints/funformer_entex_423_len20000_lr5e-4_b250_gac2. "
                        "The script looks for best_model/ inside it. "
                        "model_class, sequence_length and n_targets are read from best_model/model_config.json.")
    p.add_argument("--data-dir", required=True,
                   help="Path to HuggingFace dataset directory (contains train/validation/test splits).")
    p.add_argument("--genome-path", required=True,
                   help="Path to FASTA file (samtools faidx index must exist alongside it).")
    p.add_argument("--batch-size", type=int, default=1000,
                   help="Inference batch size.")
    p.add_argument("--num-workers", type=int, default=8,
                   help="DataLoader worker processes.")
    p.add_argument("--output-dir", default="eval_results/test_set",
                   help="Directory where per_track and summary CSVs are written.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Optional cap on test samples (for debugging).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    exp_tag       = Path(args.checkpoint_dir).name
    best_model_dir = os.path.join(args.checkpoint_dir, "best_model")

    print(f"=== evaluate_test_set ===")
    print(f"  checkpoint_dir : {args.checkpoint_dir}")
    print(f"  best_model_dir : {best_model_dir}")
    print(f"  data_dir       : {args.data_dir}")
    print(f"  output_dir     : {args.output_dir}")

    # --- load model (model_class / sequence_length / n_targets from model_config.json) ---
    backbone, model_class_name, sequence_length, n_targets = load_backbone(best_model_dir)
    print(f"  model_class    : {model_class_name}")
    print(f"  sequence_length: {sequence_length}")
    print(f"  n_targets      : {n_targets}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device         : {device}")

    model = DualStrandWrapper(backbone).to(device)
    model.eval()

    # --- dataset ---
    test_dataset = Seq2FunDataset(
        split="test",
        data_dir=args.data_dir,
        genome_path=args.genome_path,
        sequence_length=sequence_length,
        max_samples=args.max_samples,
    )
    print(f"  test samples   : {len(test_dataset)}")

    # --- inference ---
    probs, labels = run_inference(
        model, test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    print(f"  probs shape    : {probs.shape}")

    # --- metrics ---
    per_track = compute_per_track_metrics(probs, labels)
    summary   = compute_summary(per_track, exp_tag)

    print(f"\nSummary:")
    print(summary.to_string(index=False))

    # --- save ---
    os.makedirs(args.output_dir, exist_ok=True)
    per_track_path = os.path.join(args.output_dir, f"{exp_tag}_per_track.csv")
    summary_path   = os.path.join(args.output_dir, f"{exp_tag}_summary.csv")

    per_track.to_csv(per_track_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"\nSaved:")
    print(f"  {per_track_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
