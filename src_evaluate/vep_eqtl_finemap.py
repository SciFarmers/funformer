#!/usr/bin/env python
"""
Variant Effect Prediction (VEP) for eQTL fine-mapped variants.

Computes per-track impact scores (sigmoid(alt) - sigmoid(ref)) for all 423
ENCODE tracks, averaged over forward and reverse complement strands.

Genome access uses src.bio.genome.Genome + Tokenizer + get_seq_vep().
Token IDs are converted to one-hot via tokens_to_onehot(), which matches
trainer.py exactly: vocab N=0,A=1,T=2,G=3,C=4 → channels A/N→0,T→1,G→2,C→3.

Variants on chromosomes absent from the genome (e.g. X, Y — excluded by
normalize_chrom) are filtered out before inference with a logged count.

Usage (run from workspace root):
    python -m src_evaluate.vep_eqtl_finemap \\
        --genome-path  /cluster/.../GRCh38.fa \\
        --variant-path bench_data/enformer_gtex_fine_vcf/Whole_Blood_pos.vcf \\
        --output-path  bench_results/vep_eqtl/DanQ_1e-4_1000/Whole_Blood_pos.csv.gz \\
        --model-path   checkpoints/DanQ_1e-4_1000/best_model \\
        --model-class  DanQ \\
        --window-size  1000
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.bio.genome import Genome, Tokenizer, normalize_chrom
from .utils import read_vcf_fine_mapped_enformer

from src.seq2fun.baseline_models import DeepSEA, DanQ, DeepFormer, DeepATT
from src.seq2fun.funformer import FunFormer


_MODEL_REGISTRY = {
    "DeepSEA": DeepSEA,
    "DanQ": DanQ,
    "DeepFormer": DeepFormer,
    "DeepATT": DeepATT,
    "FunFormer": FunFormer,
}


# ---------------------------------------------------------------------------
# Token → one-hot (matches trainer.py DualStrandWrapper exactly)
# ---------------------------------------------------------------------------

def tokens_to_onehot(ids: torch.Tensor) -> torch.Tensor:
    """NATGC token ids (B, L) -> one-hot float (B, 4, L).

    Vocab: N=0, A=1, T=2, G=3, C=4. Channels: A->0, T->1, G->2, C->3.

    N carries no channel. One-hot over the full five-symbol vocabulary and
    drop the pad column: index 0 is N, so removing it leaves an all-zero
    column there and shifts A/T/G/C down to 0/1/2/3 unchanged.

    An assembly gap therefore reaches the model as absence of evidence rather
    than as a base, and the encoding is reverse-complement equivariant.
    """
    return F.one_hot(ids.long(), 5).float()[..., 1:].transpose(-2, -1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VariantDataset(Dataset):
    """
    VEP dataset backed by Genome.get_seq_vep().

    Returns int64 token tensors for ref/alt × fwd/rev strands.
    Variants on chromosomes absent from the genome are filtered at init.
    """

    def __init__(self, data, genome: Genome, seq_len: int = 1000):
        if isinstance(data, str):
            df = read_vcf_fine_mapped_enformer(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("data must be a DataFrame or a file path string")

        self.genome = genome
        self.seq_len = seq_len
        self.df = self._filter_valid_chroms(df).reset_index(drop=True)

    def _filter_valid_chroms(self, df: pd.DataFrame) -> pd.DataFrame:
        valid = set(self.genome.chrom_sizes.keys())

        def ok(c):
            n = normalize_chrom(str(c))
            return n is not None and n in valid

        mask = df["chrom"].apply(ok)
        n_skip = (~mask).sum()
        if n_skip:
            skipped = df.loc[~mask, "chrom"].unique().tolist()
            print(f"  Filtering {n_skip} variants on chromosomes not in genome: {skipped}")
        return df[mask].copy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        (ref_fwd, ref_rev), (alt_fwd, alt_rev) = self.genome.get_seq_vep(
            chr=str(row["chrom"]),
            variant_pos=int(row["pos"]),
            ref=str(row["ref"]),
            alt=str(row["alt"]),
            seq_len=self.seq_len,
        )
        return {
            "ref_fwd": torch.from_numpy(ref_fwd.astype(np.int64)),
            "alt_fwd": torch.from_numpy(alt_fwd.astype(np.int64)),
            "ref_rev": torch.from_numpy(ref_rev.astype(np.int64)),
            "alt_rev": torch.from_numpy(alt_rev.astype(np.int64)),
        }

    @staticmethod
    def collate_fn(batch: list) -> dict:
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ---------------------------------------------------------------------------
# VEP model wrapper
# ---------------------------------------------------------------------------

class VEPModel(torch.nn.Module):
    """
    Wraps any backbone for VEP inference.

    Accepts token ID tensors (B, L) int64, converts to one-hot (B, 4, L)
    via tokens_to_onehot (identical to trainer.py), then computes
    sigmoid(alt) - sigmoid(ref) per track averaged over fwd and rev strands.
    """

    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        self.backbone = backbone

    def _predict(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, L) int64 → logits (B, T)."""
        out = self.backbone(tokens_to_onehot(ids))
        return out["logits"] if isinstance(out, dict) else out

    def forward(
        self,
        ref_fwd: torch.Tensor,
        alt_fwd: torch.Tensor,
        ref_rev: torch.Tensor,
        alt_rev: torch.Tensor,
    ) -> torch.Tensor:
        score_fwd = torch.sigmoid(self._predict(alt_fwd)) - torch.sigmoid(self._predict(ref_fwd))
        score_rev = torch.sigmoid(self._predict(alt_rev)) - torch.sigmoid(self._predict(ref_rev))
        return (score_fwd + score_rev) / 2.0


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(model_path: str, model_class: str, device: str) -> torch.nn.Module:
    if model_class not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model class: {model_class!r}. "
            f"Supported: {list(_MODEL_REGISTRY)}"
        )
    ModelClass = _MODEL_REGISTRY[model_class]
    print(f"Loading {model_class} from {model_path} ...")
    model = ModelClass.from_pretrained(model_path, device=device)
    model.eval()
    model.to(device)
    print(f"  {model_class} loaded on {device}")
    return model


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_vep(
    dataset: VariantDataset,
    model: VEPModel,
    batch_size: int = 64,
    device: str = "cuda",
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=VariantDataset.collate_fn,
    )
    model.eval()
    model.to(device)
    all_scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="VEP", unit="batch"):
            batch = {k: v.to(device) for k, v in batch.items()}
            all_scores.append(model(**batch).cpu().numpy())
    return np.concatenate(all_scores, axis=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VEP for eQTL fine-mapped variants (enformer GTEx VCF format)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--genome-path", required=True,
                        help="Path to GRCh38 FASTA (samtools-indexed).")
    parser.add_argument("--variant-path", required=True,
                        help="Enformer-style fine-mapped VCF file.")
    parser.add_argument("--output-path", required=True,
                        help="Output CSV path (variant_id × tracks).")
    parser.add_argument("--model-path", required=True,
                        help="Checkpoint directory (contains model_config.json).")
    parser.add_argument("--model-class", required=True,
                        choices=list(_MODEL_REGISTRY))
    parser.add_argument("--window-size", type=int, default=1000,
                        help="Sequence window length centred on each variant.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None,
                        help="Device (cuda/cpu). Auto-detected if not set.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N variants (testing).")

    args = parser.parse_args()

    if args.device is None:
        args.device = (
            "cuda" if torch.cuda.is_available() else
            ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    print(f"Device: {args.device}")

    print(f"Loading genome: {args.genome_path}")
    genome = Genome(args.genome_path, Tokenizer())
    print(f"  {len(genome.chrom_sizes)} chromosomes loaded")

    print(f"Loading variants: {args.variant_path}")
    df = read_vcf_fine_mapped_enformer(args.variant_path)
    if args.limit is not None:
        df = df.head(args.limit)
        print(f"  Limited to {args.limit} variants")
    print(f"  {len(df)} variants before chromosome filter")

    backbone = load_model(args.model_path, args.model_class, device=args.device)
    model = VEPModel(backbone)

    dataset = VariantDataset(df, genome, seq_len=args.window_size)
    print(f"  {len(dataset)} variants after chromosome filter")

    print(f"Running VEP (window={args.window_size}, batch={args.batch_size}) ...")
    scores = run_vep(dataset, model, batch_size=args.batch_size, device=args.device)
    print(f"  Predictions shape: {scores.shape}")

    variant_ids = (
        dataset.df["chrom"].astype(str) + ":" +
        dataset.df["pos"].astype(str) + ":" +
        dataset.df["ref"] + ":" +
        dataset.df["alt"]
    )
    out_df = pd.DataFrame(
        scores,
        index=variant_ids,
        columns=[f"track_{i}" for i in range(scores.shape[1])],
    )
    out_df.index.name = "variant_id"

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    out_df.to_csv(args.output_path, float_format="%.3e")
    print(f"Saved: {args.output_path}  ({out_df.shape[0]} variants × {out_df.shape[1]} tracks)")


if __name__ == "__main__":
    main()
