#!/usr/bin/env python
# coding: utf-8
"""
Variant Effect Prediction (VEP) for Seq2Fun, BinTEN, and Baseline Models

This script performs variant impact prediction for fine-mapped eQTL variants.

Usage:
    python vep_eqtl_finemap.py \
        --genome-path data/hg38.fa \
        --variant-path variants.vcf \
        --output-path results.csv \
        --model-path checkpoints/model \
        --model-class BinTEN1 \
        --window-size 10000
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# Import local utilities
from utils import Genome, VariantDataset, read_vcf_fine_mapped_enformer

# Add parent directory to path to import seq2fun modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from seq2fun.model import BinTEN1, BinTEN2, BinTEN3
from seq2fun.baseline_models import DeepSEA, DanQ, DeepFormer, DeepATT
from seq2fun.config import Seq2FunConfig

tqdm.pandas()


# Model registry
MODEL_CLASSES = {
    "BinTEN1": BinTEN1,
    "BinTEN2": BinTEN2,
    "BinTEN3": BinTEN3,
    "DeepSEA": DeepSEA,
    "DanQ": DanQ,
    "DeepFormer": DeepFormer,
    "DeepATT": DeepATT,
}


class VEPModel(torch.nn.Module):
    """
    Wrapper for variant effect prediction.
    Keeps all features and computes simple absolute difference between ref and alt.
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_predictions(self, x):
        """Get model prediction logits"""
        outputs = self.model(x)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        return logits

    def compute_score(self, y_ref, y_alt):
        """
        Compute variant impact score from reference and alternate predictions.
        Returns per-track probability differences.
        """
        score = torch.sigmoid(y_alt) - torch.sigmoid(y_ref)
        return score

    def forward(
        self,
        x_ref_fwd=None,
        x_alt_fwd=None,
        x_ref_rev=None,
        x_alt_rev=None,
    ):
        """
        Forward pass for VEP.
        
        Computes predictions for reference and alternate alleles on both strands,
        then averages the impact scores.
        
        Returns:
            scores: shape (batch_size, n_tracks) - per-track impact scores
        """
        # Forward strand
        y_ref_fwd = self.get_predictions(x_ref_fwd)
        y_alt_fwd = self.get_predictions(x_alt_fwd)
        score_fwd = self.compute_score(y_ref_fwd, y_alt_fwd)
        #return score_fwd ## faster computation

        # Reverse strand
        y_ref_rev = self.get_predictions(x_ref_rev)
        y_alt_rev = self.get_predictions(x_alt_rev)
        score_rev = self.compute_score(y_ref_rev, y_alt_rev)
        
        # Average scores from both strands
        final_score = (score_fwd + score_rev) / 2
        
        return final_score


def load_model(model_path, model_class, num_tracks=423, sequence_length=1000, device="cuda"):
    """
    Load a trained model from HuggingFace Trainer checkpoint.
    
    Args:
        model_path: Path to model checkpoint directory (from HF Trainer)
        model_class: Name of model class (e.g., "BinTEN1", "DeepSEA")
        num_tracks: Number of output tracks
        sequence_length: Input sequence length
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    if model_class not in MODEL_CLASSES:
        raise ValueError(f"Unknown model class: {model_class}. Available: {list(MODEL_CLASSES.keys())}")
    
    ModelClass = MODEL_CLASSES[model_class]
    
    # For HF Trainer models (BinTEN models), use from_pretrained
    if hasattr(ModelClass, 'from_pretrained'):
        print(f"Loading {model_class} from HF checkpoint: {model_path}")
        model = ModelClass.from_pretrained(model_path)
    else:
        # For baseline models, instantiate and load state dict
        print(f"Loading {model_class} from checkpoint: {model_path}")
        model = ModelClass(n_targets=num_tracks, sequence_length=sequence_length)
        
        # Find checkpoint file
        if os.path.isdir(model_path):
            ckpt_files = ["model.safetensors", "pytorch_model.bin", "model.pt"]
            for fname in ckpt_files:
                ckpt_path = os.path.join(model_path, fname)
                if os.path.exists(ckpt_path):
                    model_path = ckpt_path
                    break
        
        # Load state dict
        if model_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(model_path)
        else:
            state_dict = torch.load(model_path, map_location=device)
        
        # Load with strict=False to ignore extra keys like pos_weight
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    model.to(device)
    print(f"✓ Loaded {model_class} (tracks={num_tracks}, seq_len={sequence_length}) to {device}")
    
    return model


def run_vep(
    variant_dataset,
    model,
    batch_size=8,
    device="cuda",
):
    """
    Run variant effect prediction on a dataset of variants.
    
    Args:
        variant_dataset: VariantDataset instance
        model: VEPModel wrapper
        batch_size: Batch size for inference
        device: Device to use
        
    Returns:
        predictions: Array of per-track impact scores (n_variants, n_tracks)
    """
    dataloader = DataLoader(
        variant_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues
        collate_fn=variant_dataset.collate_fn
    )
    
    model.eval()
    model.to(device)
    
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running VEP"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get predictions
            scores = model(**batch)  # shape: (batch_size, n_tracks)
            
            # Move to CPU and append
            all_predictions.append(scores.cpu().numpy())
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)
    
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Variant Effect Prediction for fine-mapped eQTL variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument("--genome-path", required=True, 
                       help="Path to reference genome FASTA (must be indexed)")
    parser.add_argument("--variant-path", required=True, 
                       help="Path to variant file (VCF from fine-mapped enformer)")
    parser.add_argument("--output-path", required=True, 
                       help="Output CSV path for results")
    parser.add_argument("--model-path", required=True, 
                       help="Path to model checkpoint (directory or file)")
    parser.add_argument("--model-class", required=True, 
                       choices=list(MODEL_CLASSES.keys()),
                       help="Model architecture class")
    
    # Model configuration
    parser.add_argument("--window-size", type=int, default=1000, 
                       help="Sequence window size (default: 1000)")
    parser.add_argument("--num-tracks", type=int, default=423, 
                       help="Number of output tracks (default: 423)")
    
    # Performance options
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for inference (default: 8)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
                       help="Device to use (default: cuda if available)")
    
    # Testing options
    parser.add_argument("--limit", type=int, default=None, 
                       help="Limit total number of variants (for testing)")
    
    args = parser.parse_args()

    # Load genome
    print(f"Loading genome from {args.genome_path}...")
    genome = Genome(args.genome_path)
    print(f"Loaded genome with {len(genome.chrom_sizes)} chromosomes")

    # Load model
    print(f"\nLoading {args.model_class} model from {args.model_path}...")
    base_model = load_model(
        args.model_path,
        args.model_class,
        num_tracks=args.num_tracks,
        sequence_length=args.window_size,
        device=args.device
    )
    
    # Wrap in VEP model
    model_wrapper = VEPModel(base_model)
    model_wrapper.to(args.device)
    model_wrapper.eval()

    # Load variants
    print(f"\nLoading variants from {args.variant_path}...")
    variant_data = read_vcf_fine_mapped_enformer(args.variant_path)
    print(f"Loaded {len(variant_data)} variants")

    # Apply limit if specified
    if args.limit is not None:
        variant_data = variant_data.head(args.limit)
        print(f"Limited to {len(variant_data)} variants for testing")

    # Create variant dataset
    print(f"\nCreating variant dataset with window size {args.window_size}...")
    variant_dataset = VariantDataset(variant_data, genome, seq_len=args.window_size)

    # Create output directory
    output_dir = os.path.dirname(args.output_path)
    if output_dir != "" and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Run VEP
    print(f"\nRunning VEP on {len(variant_dataset)} variants...")
    predictions = run_vep(
        variant_dataset,
        model_wrapper,
        batch_size=args.batch_size,
        device=args.device,
    )

    print(f"Predictions shape: {predictions.shape}")

    # Create output DataFrame with variant IDs and per-track scores
    variant_ids = [
        f"{row.chrom}:{row.pos}:{row.ref}:{row.alt}"
        for _, row in variant_data.iterrows()
    ]
    
    # Create DataFrame with per-track scores
    df = pd.DataFrame(
        predictions,
        columns=[f"track_{i}" for i in range(predictions.shape[1])],
        index=variant_ids
    )
    df.index.name = "variant_id"

    # Save results
    df.to_csv(args.output_path)
    print(f"\nResults saved to {args.output_path}")
    print(f"Output shape: {df.shape}")


if __name__ == "__main__":
    main()
