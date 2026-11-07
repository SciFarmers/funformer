import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import wandb
from typing import Optional, Dict
from datetime import datetime
import subprocess
import platform

from src.seq2fun.baseline_models import DeepSEA, DanQ, DeepFormer, DeepATT
from src.seq2fun.funformer import FunFormer, FunFormerConfig
from src.data.seq2fun import Seq2FunDataset
from src.seq2fun.utils import compute_metrics

# ---------------------------------------------------------------------------
# Token encoding + dual-strand wrapper (shared by all models)
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
    return F.one_hot(ids.long(), 5).float()[..., 1:].transpose(1, 2)


class DualStrandWrapper(nn.Module):
    """Wraps any backbone to consume paired fwd/rev token IDs from Seq2FunDataset.

    Batch keys expected: fwd_ids (B, L), rev_ids (B, L), labels (B, T).

    Training  – randomly pick fwd or rev (50/50), convert to one-hot,
                pass to backbone; loss computed over the chosen strand.
    Eval/Test – run both strands, average logits, compute loss on the average.

    The backbone receives input_ids as a one-hot float tensor (B, 4, L).
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def set_pos_weight(self, pos_weight: torch.Tensor):
        self.backbone.set_pos_weight(pos_weight)

    def forward(
        self,
        fwd_ids: torch.Tensor,
        rev_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        fwd_enc = tokens_to_onehot(fwd_ids)  # (B, 4, L)
        rev_enc = tokens_to_onehot(rev_ids)  # (B, 4, L)

        if self.training:
            # Randomly pick forward or reverse strand each step.
            input_ids = fwd_enc if torch.rand(1).item() < 0.5 else rev_enc
            return self.backbone(input_ids, labels=labels.float())
        else:
            out_fwd = self.backbone(fwd_enc)
            out_rev = self.backbone(rev_enc)
            avg_logits = (out_fwd["logits"] + out_rev["logits"]) / 2.0
            outputs: Dict[str, torch.Tensor] = {"logits": avg_logits}
            if labels is not None:
                pw = getattr(self.backbone, "pos_weight", None)
                outputs["loss"] = F.binary_cross_entropy_with_logits(
                    avg_logits, labels.float(), pos_weight=pw, reduction="mean"
                )
            return outputs

# Pool sizes: 3-layer CNN compressing each ablation length to exactly 500 tokens.
_SEQ_LEN_TO_POOLS: dict = {
     1_000: (2, 1, 1),   #  1000 / 2 / 1 / 1 = 500
     5_000: (5, 2, 1),   #  5000 / 5 / 2 / 1 = 500
    10_000: (5, 2, 2),   # 10000 / 5 / 2 / 2 = 500
    20_000: (5, 4, 2),   # 20000 / 5 / 4 / 2 = 500
    30_000: (5, 4, 3),   # 30000 / 5 / 4 / 3 = 500
    50_000: (5, 5, 4),   # 50000 / 5 / 5 / 4 = 500
}

# Configure CUDA memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.set_float32_matmul_precision('medium')


def get_git_info():
    """Get current git branch and commit hash."""
    try:
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        return branch, commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def get_optimal_precision_settings():
    """Determine optimal precision settings based on hardware."""
    system = platform.system()
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    
    # Check for Apple Silicon
    is_apple_silicon = system == "Darwin" and platform.processor() == "arm"
    
    return {
        "has_cuda": has_cuda,
        "has_mps": has_mps,
        "is_apple_silicon": is_apple_silicon,
        "system": system
    }


MODEL_CLASSES = {
    "DeepSEA": DeepSEA,
    "DanQ": DanQ,
    "DeepFormer": DeepFormer,
    "DeepATT": DeepATT,
    "FunFormer": FunFormer,
}


def train_seq2fun(
    # Core parameters
    model_class: str = "DanQ",  # DeepSEA, DanQ, DeepFormer, DeepATT, FunFormer
    sequence_length: int = 1000,
    max_epochs: int = 50,
    patience: int = 5,
    batch_size: int = 8,
    lr: float = 5e-4,

    # Data paths
    data_dir: str = "datasets/hf_ENTEX_423_beds",
    genome_path: str = "data_raw/hg38.fa",
    num_tracks: int = 423,
    pos_weight_path: Optional[str] = None,

    # Training settings
    output_dir: str = "checkpoints",
    logging_steps: int = 300,

    # Precision settings
    precision: str = "auto",  # "fp16", "bf16", "fp32", "auto"

    # Optional
    max_samples: Optional[int] = None,
    exp_name: Optional[str] = None,
    use_wandb: bool = True,
    wandb_project: str = "seq2fun_length_tradeoff",
    seed: int = 42,

    # System settings
    num_workers: int = 4,
    pin_memory: bool = True,
    gradient_accumulation_steps: int = 1,

    # Learning rate schedule
    lr_scheduler_type: str = "linear",  # "linear", "cosine", "constant", "constant_with_warmup", etc.

    # ── FunFormer architecture (ignored for baseline models) ──────────────
    hidden_size: int = 512,
    num_hidden_layers: int = 1,
    num_attention_heads: int = 8,
    intermediate_size: int = 1024,
    projection_dim: int = 256,
    rope_theta: float = 10000.0,
    attn_dropout: float = 0.0,
):
    # Determine rank — works whether launched via torchrun, accelerate, or plain python.
    is_main_process = int(os.environ.get("LOCAL_RANK", "0")) == 0

    # Hardware info
    hw_info = get_optimal_precision_settings()
    if is_main_process:
        print(f"Hardware info: {hw_info}")

    # Random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if hw_info["has_cuda"]:
        torch.cuda.manual_seed_all(seed)
    
    # Precision settings
    use_fp16 = False
    use_bf16 = False
    if precision == "auto":
        if hw_info["has_cuda"]:
            use_fp16 = True
            if is_main_process: print("Auto-selected: FP16 (CUDA)")
        elif hw_info["is_apple_silicon"]:
            if is_main_process: print("Auto-selected: FP32 (Apple Silicon)")
        else:
            if is_main_process: print("Auto-selected: FP32 (CPU)")
    elif precision == "fp16":
        use_fp16 = hw_info["has_cuda"]
    elif precision == "bf16":
        use_bf16 = hw_info["has_cuda"] or hw_info["is_apple_silicon"]
    elif precision == "fp32":
        pass
    else:
        raise ValueError(f"Invalid precision: {precision}. Use 'auto', 'fp16', 'bf16', or 'fp32'")

    if hw_info["is_apple_silicon"]:
        pin_memory = False
    
    # Map string to model class
    if isinstance(model_class, str):
        if model_class not in MODEL_CLASSES:
            raise ValueError(f"Invalid model_class: {model_class}. Choose from {list(MODEL_CLASSES.keys())}")
        model_class_name = model_class
    else:
        model_class_name = model_class.__name__

    # ── Build model ────────────────────────────────────────────────────────
    is_funformer = (model_class_name == "FunFormer")

    if is_funformer:
        # Build FunFormerConfig + FunFormer backbone
        if sequence_length not in _SEQ_LEN_TO_POOLS:
            raise ValueError(
                f"FunFormer: sequence_length={sequence_length} not in known ablation set "
                f"{sorted(_SEQ_LEN_TO_POOLS.keys())}."
            )
        pool_sizes = _SEQ_LEN_TO_POOLS[sequence_length]
        if is_main_process:
            print(f"FunFormer auto pool sizes for seq_len={sequence_length}: {pool_sizes}")

        ff_config = FunFormerConfig(
            n_targets=num_tracks,
            sequence_length=sequence_length,
            cnn_pool_sizes=pool_sizes,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            projection_dim=projection_dim,
            rope_theta=rope_theta,
            attn_dropout=attn_dropout,
        )
        backbone = FunFormer(ff_config)
        n_total = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        if is_main_process:
            print(f"FunFormer  seq_len={sequence_length}  n_tokens={ff_config.n_tokens}"
                  f"  layers={num_hidden_layers}  params={n_total:,}")
    else:
        model_class_obj = MODEL_CLASSES[model_class_name]
        backbone = model_class_obj(num_tracks, sequence_length=sequence_length)

    # All models wrapped with DualStrandWrapper (handles fwd/rev token IDs → one-hot).
    model = DualStrandWrapper(backbone)

    # Experiment name
    if exp_name is None:
        branch, commit = get_git_info()
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        precision_suffix = f"_{precision}" if precision != "auto" else ""
        if branch and commit:
            exp_name = f"{branch}_{commit}_{model_class}_len{sequence_length}_b{batch_size}_gac{gradient_accumulation_steps}_{timestamp}{precision_suffix}"
        else:
            exp_name = f"{model_class}_no-git_{timestamp}{precision_suffix}"

    # Initialize wandb — rank 0 only; Trainer reuses the open run on rank 0.
    if is_main_process and use_wandb:
        wandb_config = {
            "model_class": model_class_name,
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "lr": lr,
            "max_epochs": max_epochs,
            "patience": patience,
            "precision": precision,
            "hardware": hw_info,
        }
        if is_funformer:
            wandb_config.update({
                "n_tokens": ff_config.n_tokens,
                "cnn_pool_sizes": list(ff_config.cnn_pool_sizes),
                "hidden_size": hidden_size,
                "num_hidden_layers": num_hidden_layers,
                "num_attention_heads": num_attention_heads,
                "intermediate_size": intermediate_size,
            })
        wandb.init(
            project=wandb_project,
            name=exp_name,
            config=wandb_config,
        )
    
    # Clear GPU
    if hw_info["has_cuda"]:
        torch.cuda.empty_cache()

    if pos_weight_path is not None and os.path.exists(pos_weight_path):
        pos_weight = torch.from_numpy(np.load(pos_weight_path)).float()
        if pos_weight.shape[0] != num_tracks:
            raise ValueError(f"pos_weight length {pos_weight.shape[0]} does not match num_tracks {num_tracks}")
        model.set_pos_weight(pos_weight)
        if is_main_process:
            print(f"Loaded pos_weight from {pos_weight_path}")

    # Datasets
    train_dataset = Seq2FunDataset(
        split="train",
        data_dir=data_dir,
        genome_path=genome_path,
        sequence_length=sequence_length,
        max_samples=max_samples,
    )
    val_dataset = Seq2FunDataset(
        split="validation",
        data_dir=data_dir,
        genome_path=genome_path,
        sequence_length=sequence_length,
        max_samples=max_samples,
    )
    test_dataset = Seq2FunDataset(
        split="test",
        data_dir=data_dir,
        genome_path=genome_path,
        sequence_length=sequence_length,
        max_samples=max_samples,
    )
    
    # TrainingArguments
    # All models optimise training loss for early stopping and best-model selection.
    best_metric, greater_is_better = "loss", False

    # Warmup is irrelevant for constant LR — disable it to keep semantics clean.
    warmup_ratio = 0.0 if lr_scheduler_type in ("constant", "constant_with_warmup") else 0.01

    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{exp_name}",
        run_name=exp_name,
        num_train_epochs=max_epochs,
        max_steps=-1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=greater_is_better,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=pin_memory,
        report_to="wandb" if use_wandb else "none",
        seed=seed,
        logging_steps=logging_steps,
        logging_first_step=True,
    )

    # Early stopping
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=patience,
        early_stopping_threshold=0.0
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )

    # Training info — rank 0 only
    if is_main_process:
        print(f"\nTraining Configuration:")
        print(f"  Experiment: {exp_name}")
        print(f"  Model class: {model_class}")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Max epochs: {max_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {lr}")
        print(f"  Precision: {'FP16' if use_fp16 else 'BF16' if use_bf16 else 'FP32'}")
        print(f"  Workers: {num_workers}")
        print(f"  Total train samples: {len(train_dataset)}")
        print(f"  Total val samples: {len(val_dataset)}")
        print(f"  Total test samples: {len(test_dataset)}")
    
    # Train
    try:
        trainer.train()
        if is_main_process:
            print(f"\nTraining completed successfully!")
        best_model_dir = f"{training_args.output_dir}/best_model"
        # load_best_model_at_end=True guarantees model holds the best weights here.
        if trainer.is_world_process_zero():
            model.backbone.save_pretrained(best_model_dir)
            print(f"Best model saved to: {best_model_dir}")

        ## Evaluate on test set
        if is_main_process:
            print(f"\nEvaluating on test set...")
        test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")
        if is_main_process:
            print(f"Test results: {test_results}")
            if use_wandb:
                wandb.log({f"test_{k}": v for k, v in test_results.items()})


    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nFailed with error: {e}")
        raise
    finally:
        if is_main_process and use_wandb:
            wandb.finish()


if __name__ == "__main__":
    from fire import Fire
    Fire(train_seq2fun)
    exit(0)
