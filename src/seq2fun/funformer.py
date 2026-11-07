"""FunFormer — Pure self-attention sequence-to-function model.

Uses:
  • Self-built Rotary Position Embedding (RoPE), adapted from OLMo Hybrid
    (olmo_hybrid_template/modeling_olmo_hybrid.py :: OlmoHybridRotaryEmbedding)
  • PyTorch built-in F.scaled_dot_product_attention — dispatches to
    FlashAttention-2 on CUDA automatically (no extra install needed).

Architecture:
  1. 3-layer CNN encoder with fixed channel progression:
         Conv(4→320,  kernel=15, pool=p1)
       → Conv(320→480, kernel=9,  pool=p2)
       → Conv(480→512, kernel=9,  pool=p3)
     Pool sizes compress each ablation length to exactly 500 tokens:
        1k seq:  pools=(2,1,1)  →   1000/2/1/1 = 500
        5k seq:  pools=(5,2,1)  →  5000/5/2/1 = 500
       10k seq:  pools=(5,2,2)  → 10000/5/2/2 = 500
       20k seq:  pools=(5,4,2)  → 20000/5/4/2 = 500
       30k seq:  pools=(5,4,3)  → 30000/5/4/3 = 500
       50k seq:  pools=(5,5,4)  → 50000/5/5/4 = 500

  2. N layers of bidirectional self-attention (full attention) with:
       - Pre-RMSNorm
       - Q/K per-head RMSNorm  (OLMo Hybrid style)
       - RoPE applied to Q/K
       - PyTorch SDPA (is_causal=False → bidirectional)
       - SwiGLU FFN with pre-RMSNorm


Input: input_ids (B, 4, L) one-hot float, labels (B, T).
Backbone receives one-hot input from DualStrandWrapper (in trainer.py).
"""

from __future__ import annotations

import math
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.seq2fun.baseline_models import Seq2FunMixin


# ---------------------------------------------------------------------------
# CNN building block
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv1d (same-padding) → BatchNorm → GELU → MaxPool → Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        pool_size: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.act  = nn.GELU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size) if pool_size > 1 else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.pool is not None:
            x = self.pool(x)
        return self.drop(x)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * xf).to(dtype)


# ---------------------------------------------------------------------------
# Rotary Position Embedding (self-built, no HF dependency)
#
# Adapted from olmo_hybrid_template/modeling_olmo_hybrid.py
#   OlmoHybridRotaryEmbedding.compute_default_rope_parameters
#   and the forward pass pattern.
# The key formula is identical to the original Su et al. RoPE paper:
#   inv_freq_i = 1 / theta^(2i/d),  i = 0,1,...,d/2-1
#   freqs[t]   = t * inv_freq
#   cos/sin from concatenated [freqs, freqs]
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Self-built RoPE. Registers inv_freq as a non-persistent buffer."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        # inv_freq shape: (head_dim / 2,)
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) of shape (seq_len, head_dim), in float32."""
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        # outer product → (seq_len, head_dim/2)
        freqs = torch.outer(positions, self.inv_freq)
        # concat → (seq_len, head_dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates the second half of the last dimension: [-x2, x1]."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q and k.

    Args:
        q, k : (B, H, L, head_dim)
        cos, sin : (L, head_dim)  — from RotaryEmbedding.forward()
    """
    q_dtype, k_dtype = q.dtype, k.dtype
    # broadcast: (1, 1, L, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out.to(q_dtype), k_out.to(k_dtype)


# ---------------------------------------------------------------------------
# Transformer block: bidirectional self-attention + SwiGLU FFN
# ---------------------------------------------------------------------------

class FunFormerBlock(nn.Module):
    """One transformer block: pre-norm attn (RoPE + SDPA) + pre-norm SwiGLU FFN.

    Attention is *bidirectional* (is_causal=False).
    F.scaled_dot_product_attention dispatches to FlashAttention-2 on CUDA
    when the input dtype is float16/bfloat16 — no extra install needed.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        rope: RotaryEmbedding,
        rms_eps: float = 1e-6,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.rope      = rope
        self.attn_dropout = attn_dropout

        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Per-head Q/K RMSNorm (OLMo Hybrid style — improves training stability)
        self.q_norm = RMSNorm(hidden_size, eps=rms_eps)
        self.k_norm = RMSNorm(hidden_size, eps=rms_eps)

        # Pre-norms
        self.attn_norm = RMSNorm(hidden_size, eps=rms_eps)
        self.ffn_norm  = RMSNorm(hidden_size, eps=rms_eps)

        # SwiGLU FFN: gate_proj + up_proj fused into down_proj
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape

        # ── Self-attention ───────────────────────────────────────────────
        residual = x
        h = self.attn_norm(x)

        # Project and normalise Q, K
        q = self.q_norm(self.q_proj(h))     # (B, L, H*d)
        k = self.k_norm(self.k_proj(h))
        v = self.v_proj(h)

        # Reshape to (B, H, L, head_dim) for multi-head ops
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        cos, sin = self.rope(L, x.device)
        q, k = apply_rope(q, k, cos, sin)

        # Bidirectional attention via PyTorch SDPA
        # On CUDA with fp16/bf16 this calls Flash Attention 2 automatically.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,             # bidirectional — no causal mask
        )                                # (B, H, L, head_dim)

        out = out.transpose(1, 2).reshape(B, L, -1)   # (B, L, hidden_size)
        out = self.o_proj(out)
        x = residual + out

        # ── SwiGLU FFN ───────────────────────────────────────────────────
        residual = x
        h = self.ffn_norm(x)
        ffn = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        x = residual + ffn

        return x


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class FunFormerConfig:
    """Configuration for FunFormer.

    3-layer CNN tower with fixed channel progression (4→320→480→hidden_size)
    followed by a Transformer encoder. Pool sizes compress all ablation lengths
    to exactly 500 transformer tokens:

        sequence_length= 1000: cnn_pool_sizes=(2, 1, 1) →  1000/2/1/1 = 500
        sequence_length= 5000: cnn_pool_sizes=(5, 2, 1) →  5000/5/2/1 = 500
        sequence_length=10000: cnn_pool_sizes=(5, 2, 2) → 10000/5/2/2 = 500
        sequence_length=20000: cnn_pool_sizes=(5, 4, 2) → 20000/5/4/2 = 500
        sequence_length=30000: cnn_pool_sizes=(5, 4, 3) → 30000/5/4/3 = 500
        sequence_length=50000: cnn_pool_sizes=(5, 5, 4) → 50000/5/5/4 = 500
    """
    # ── Task ─────────────────────────────────────────────────────────────
    n_targets: int         = 423
    sequence_length: int   = 10000

    # ── CNN encoder: 3 layers with fixed channel progression ───────────
    cnn_kernel_sizes: Tuple[int, int, int] = (15, 9, 9)   # per-layer kernel sizes
    cnn_channels: Tuple[int, int]          = (320, 480)   # out_channels for layers 1 & 2
                                                           # layer 3 always outputs hidden_size
    cnn_pool_sizes: Tuple[int, int, int]   = (5, 2, 2)    # (pool1, pool2, pool3)
    cnn_dropout: float                     = 0.2

    # ── Transformer encoder ───────────────────────────────────────────────
    hidden_size: int              = 512
    num_hidden_layers: int        = 1
    num_attention_heads: int      = 8
    intermediate_size: int        = 1024
    rope_theta: float             = 10000.0
    attn_dropout: float           = 0.0
    rms_norm_eps: float           = 1e-6

    # ── Classifier ───────────────────────────────────────────────────────
    projection_dim: int    = 256
    proj_dropout: float    = 0.2

    def __post_init__(self):
        if len(self.cnn_pool_sizes) != 3:
            raise ValueError(
                f"cnn_pool_sizes must have exactly 3 elements, got {len(self.cnn_pool_sizes)}"
            )
        if len(self.cnn_kernel_sizes) != 3:
            raise ValueError(
                f"cnn_kernel_sizes must have exactly 3 elements, got {len(self.cnn_kernel_sizes)}"
            )
        if len(self.cnn_channels) != 2:
            raise ValueError(
                f"cnn_channels must have exactly 2 elements (layers 1 & 2), got {len(self.cnn_channels)}"
            )
        if self.n_tokens != 500:
            raise ValueError(
                f"Pool sizes {self.cnn_pool_sizes} yield {self.n_tokens} tokens for "
                f"sequence_length={self.sequence_length}; expected 500."
            )

    @property
    def n_tokens(self) -> int:
        """Number of sequence positions after CNN pooling (target: 500)."""
        t = self.sequence_length
        for p in self.cnn_pool_sizes:
            t = t // p
        return t

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# FunFormer (main model)
# ---------------------------------------------------------------------------

class FunFormer(Seq2FunMixin, nn.Module):
    """Single-strand pure-attention sequence-to-function model.

    Encodes the forward strand through a shared CNN + transformer backbone.
    Dataset keys: input_ids (B, L), rev_ids (B, L, ignored), labels (B, T).
    """

    def __init__(self, config: Optional[FunFormerConfig] = None, **override_kwargs):
        super().__init__()
        if config is None:
            config = FunFormerConfig(**override_kwargs)
        self.config = config

        # ── CNN encoder ───────────────────────────────────────────────────
        k1, k2, k3 = config.cnn_kernel_sizes
        ch1, ch2   = config.cnn_channels
        p1, p2, p3 = config.cnn_pool_sizes
        self.conv1 = ConvBlock(4,    ch1,               kernel_size=k1, pool_size=p1, dropout=config.cnn_dropout)
        self.conv2 = ConvBlock(ch1,  ch2,               kernel_size=k2, pool_size=p2, dropout=config.cnn_dropout)
        self.conv3 = ConvBlock(ch2,  config.hidden_size, kernel_size=k3, pool_size=p3, dropout=config.cnn_dropout)

        # ── Shared RoPE ───────────────────────────────────────────────────
        head_dim  = config.hidden_size // config.num_attention_heads
        self.rope = RotaryEmbedding(head_dim=head_dim, theta=config.rope_theta)

        # ── Transformer blocks ────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            FunFormerBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                rope=self.rope,
                rms_eps=config.rms_norm_eps,
                attn_dropout=config.attn_dropout,
            )
            for _ in range(config.num_hidden_layers)
        ])
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # ── Classifier ────────────────────────────────────────────────────
        n_tokens = config.n_tokens
        self.projection = nn.Linear(config.hidden_size, config.projection_dim)
        self.proj_drop  = nn.Dropout(config.proj_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.projection_dim * n_tokens, config.projection_dim),
            nn.GELU(),
            nn.Linear(config.projection_dim, config.n_targets),
        )

        # pos_weight for class-balanced BCE loss (optional)
        self.register_buffer("pos_weight", None)

    def get_backbone_config(self) -> dict:
        return self.config.to_dict()

    @classmethod
    def _build_from_config(cls, backbone_kwargs: dict):
        # dataclasses.asdict stores tuples as lists; FunFormerConfig needs tuples.
        for key in ("cnn_pool_sizes", "cnn_kernel_sizes", "cnn_channels"):
            if key in backbone_kwargs and isinstance(backbone_kwargs[key], list):
                backbone_kwargs[key] = tuple(backbone_kwargs[key])
        return cls(**backbone_kwargs)  # FunFormer(**kwargs) builds FunFormerConfig internally

    def set_pos_weight(self, pos_weight: torch.Tensor):
        self.register_buffer("pos_weight", pos_weight)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode one-hot sequence (B, 4, L) through CNN + transformer → (B, n_tokens, hidden_size)."""
        x = self.conv1(x)            # (B, 320,         L//p1)
        x = self.conv2(x)            # (B, 480,         L//p1//p2)
        x = self.conv3(x)            # (B, hidden_size, L//p1//p2//p3 = n_tokens=500)
        x = x.transpose(1, 2)        # (B, n_tokens, hidden_size)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)    # (B, n_tokens, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,           # one-hot float (B, 4, L) from DualStrandWrapper
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        x = self._encode(input_ids)        # (B, n_tokens, hidden_size)

        x = self.projection(x)                        # (B, n_tokens, proj_dim)
        x = self.proj_drop(x)
        x = x.reshape(x.size(0), -1)                  # (B, n_tokens * proj_dim)
        logits = self.classifier(x)                   # (B, n_targets)

        outputs: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            outputs["loss"] = F.binary_cross_entropy_with_logits(
                logits, labels.float(), pos_weight=self.pos_weight, reduction="mean"
            )
        return outputs
