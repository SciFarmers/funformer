"""
Cross-species regulatory sequence classification evaluation.

Takes a model trained on Species A, runs it on Species B's test set genomic
intervals (using Species B's genome), and computes AUROC/AUPRC for matched
(assay, tissue) track pairs as defined in meta_info/matched_tracks.json.

For each matched group the model may have multiple source replicate tracks and
the target dataset may have multiple target replicate tracks.  AUROC is
computed for every source×target index pair independently, then averaged to
produce one number per (assay, tissue) group.

Usage:
    python -m src_evaluate.evaluate_cross_species \\
        --source-model-path checkpoints/funformer_entex_423_len20000_lr5e-4_b250_gac2/best_model \\
        --source-species    human \\
        --target-species    pig \\
        --target-data-dir   /cluster/.../hf_pig_beds \\
        --target-genome     /cluster/.../Sus_scrofa.Sscrofa11.1.dna.toplevel.fa \\
        --matched-tracks    meta_info/matched_tracks.json \\
        --model-tag         FunFormer_20k \\
        --batch-size        256 \\
        --num-workers       4 \\
        --out-dir           eval_results/cross_species/human_to_pig
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from src_evaluate.evaluate_test_set import load_backbone
from src.data.seq2fun import Seq2FunDataset
from src.seq2fun.trainer import DualStrandWrapper

# Canonical pair-key ordering (must match find_matched_tracks.py)
# MUST stay in the same order as SPECIES_ORDER in meta_info/find_matched_tracks.py:
# matched_tracks.json pair keys are derived from that order, so a mismatch here
# silently produces zero matched channels (this is what broke the first chicken
# batch -- 32 jobs "COMPLETED" with empty output). Append new species, never insert.
_SPECIES_ORDER = ["human", "pig", "cattle", "chicken", "salmon", "zebrafish"]


def _pair_key(sp_a: str, sp_b: str) -> str:
    if _SPECIES_ORDER.index(sp_a) <= _SPECIES_ORDER.index(sp_b):
        return f"{sp_a}_{sp_b}"
    return f"{sp_b}_{sp_a}"


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
    """Return (probs, labels) arrays of shape (N, T_source) and (N, T_target)."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    all_logits, all_labels = [], []
    model.eval()
    for batch in tqdm(loader, desc="Inference", unit="batch"):
        fwd_ids = batch["fwd_ids"].to(device)
        rev_ids = batch["rev_ids"].to(device)
        out = model(fwd_ids=fwd_ids, rev_ids=rev_ids)
        all_logits.append(out["logits"].cpu().float())
        all_labels.append(batch["labels"].float())
    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    probs  = torch.sigmoid(torch.from_numpy(logits)).numpy()
    return probs, labels


# ---------------------------------------------------------------------------
# Per-match AUROC / AUPRC
# ---------------------------------------------------------------------------

def evaluate_matches(
    probs:  np.ndarray,   # (N, n_source_tracks)
    labels: np.ndarray,   # (N, n_target_tracks)
    matches: list[dict],
    source_species: str,
    target_species: str,
) -> list[dict]:
    """
    For each (assay, tissue) group compute mean AUROC/AUPRC over all
    source_idx × target_idx pairs.
    """
    results = []
    for m in matches:
        src_idxs = m[source_species]
        tgt_idxs = m[target_species]
        aurocs, auprcs = [], []
        for si in src_idxs:
            for ti in tgt_idxs:
                y_score = probs[:, si]
                y_true  = labels[:, ti]
                if len(np.unique(y_true)) < 2:
                    continue
                try:
                    aurocs.append(float(roc_auc_score(y_true, y_score)))
                    auprcs.append(float(average_precision_score(y_true, y_score)))
                except Exception:
                    pass
        if not aurocs:
            continue
        results.append({
            "assay":   m["assay"],
            "tissue":  m["tissue"],
            "n_pairs": len(aurocs),
            "auroc":   float(np.mean(aurocs)),
            "auprc":   float(np.mean(auprcs)),
        })
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-species regulatory sequence classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-model-path", required=True,
                        help="Path to best_model dir of source species checkpoint.")
    parser.add_argument("--source-species", required=True,
                        choices=_SPECIES_ORDER)
    parser.add_argument("--target-species", required=True,
                        choices=_SPECIES_ORDER)
    parser.add_argument("--target-data-dir", required=True,
                        help="HF dataset dir for target species.")
    parser.add_argument("--target-genome", required=True,
                        help="FASTA path for target species genome.")
    parser.add_argument("--matched-tracks", default="meta_info/matched_tracks.json",
                        help="JSON file from find_matched_tracks.py.")
    parser.add_argument("--model-tag", default=None,
                        help="Short label for output filenames. Defaults to model dir name.")
    parser.add_argument("--batch-size",   type=int, default=256)
    parser.add_argument("--num-workers",  type=int, default=4)
    parser.add_argument("--out-dir",      required=True,
                        help="Output directory for per-match and summary CSVs.")
    args = parser.parse_args()

    if args.source_species == args.target_species:
        parser.error("source-species and target-species must differ.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_tag = args.model_tag or Path(args.source_model_path).parent.name

    print("=" * 70)
    print("Cross-species Evaluation")
    print(f"  Source : {args.source_species}  ({args.source_model_path})")
    print(f"  Target : {args.target_species}  ({args.target_data_dir})")
    print(f"  Model  : {model_tag}")
    print(f"  Device : {device}")
    print("=" * 70)

    # Load matched track index lookup
    with open(args.matched_tracks) as f:
        matched_tracks = json.load(f)
    key = _pair_key(args.source_species, args.target_species)
    if key not in matched_tracks:
        print(f"ERROR: no matched tracks for pair '{key}' in {args.matched_tracks}")
        return
    matches = matched_tracks[key]
    print(f"Matched groups for {key}: {len(matches)}")

    # Load source backbone (reads model_config.json automatically)
    backbone, model_class, seq_len, n_source = load_backbone(args.source_model_path)
    model = DualStrandWrapper(backbone).to(device)
    model.eval()
    print(f"Source model: {model_class}  seq_len={seq_len}  n_tracks={n_source}")

    # Build target test dataset using source model's sequence length
    print(f"Loading target test set: {args.target_data_dir}")
    dataset = Seq2FunDataset(
        split="test",
        data_dir=args.target_data_dir,
        genome_path=args.target_genome,
        sequence_length=seq_len,
    )
    print(f"  {len(dataset)} test examples  |  target genome: {args.target_genome}")

    # Inference
    probs, labels = run_inference(model, dataset, args.batch_size, args.num_workers, device)
    print(f"Predictions: {probs.shape}  Labels: {labels.shape}")

    # Evaluate matched pairs
    results = evaluate_matches(probs, labels, matches, args.source_species, args.target_species)
    if not results:
        print("WARNING: no valid matched pairs produced results (all single-class?).")
        return

    # Per-match CSV
    import pandas as pd
    per_match = pd.DataFrame(results)
    per_match.insert(0, "target_species", args.target_species)
    per_match.insert(0, "source_species", args.source_species)
    per_match.insert(0, "model", model_tag)
    pm_path = out_dir / f"{model_tag}_per_match.csv"
    per_match.to_csv(pm_path, index=False)
    print(f"\nPer-match results ({len(per_match)} groups):")
    for _, r in per_match.iterrows():
        print(f"  {r['assay']:12s} {r['tissue']:8s}  AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}  (n_pairs={r['n_pairs']})")

    # Summary CSV
    auroc_vals = per_match["auroc"].values
    auprc_vals = per_match["auprc"].values
    summary = pd.DataFrame([{
        "model":          model_tag,
        "source_species": args.source_species,
        "target_species": args.target_species,
        "n_matched":      len(results),
        "auroc_mean":     float(np.mean(auroc_vals)),
        "auroc_median":   float(np.median(auroc_vals)),
        "auroc_std":      float(np.std(auroc_vals)),
        "auprc_mean":     float(np.mean(auprc_vals)),
        "auprc_std":      float(np.std(auprc_vals)),
    }])
    sm_path = out_dir / f"{model_tag}_summary.csv"
    summary.to_csv(sm_path, index=False)

    print(f"\nSummary: AUROC mean={auroc_vals.mean():.4f}  median={np.median(auroc_vals):.4f}")
    print(f"Saved: {pm_path}")
    print(f"Saved: {sm_path}")
    print("DONE.")


if __name__ == "__main__":
    main()
