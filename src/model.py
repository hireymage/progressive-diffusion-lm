"""
Progressive-precision diffusion language model.

Architecture
------------
Bidirectional (non-causal) Transformer encoder.  The model takes a partially
masked token sequence and predicts the original token at every position.

Conditioning on the noise level (mask rate) is done by adding a sinusoidal
"step embedding" to every token position — analogous to time conditioning in
image diffusion models.

All weight matrices in attention and feed-forward sub-layers are
QuantizedLinear layers; their `bits` attribute is updated at runtime to
implement the progressive precision schedule.

The model operates identically whether it is used as the "baseline" (bits=16,
no quantisation) or the "progressive" model (bits per step from the schedule).
"""

import math
import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.utils

from .quantization import QuantizedLinear, set_model_bits
from .config import ModelConfig


# ---------------------------------------------------------------------------
# Sinusoidal step / noise-level embedding
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """
    Maps a scalar in [0, 1] (normalised diffusion step / mask rate) to a
    vector of dimension `dim` using sinusoidal features, then passes it
    through a small MLP.
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for sinusoidal embedding"
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def __call__(self, t: mx.array) -> mx.array:
        # t: (batch,) float in [0, 1]
        half = self.dim // 2
        freqs = mx.exp(
            -math.log(10000.0) * mx.arange(half, dtype=mx.float32) / half
        )  # (half,)
        t = t[:, None]  # (batch, 1)
        emb = t * freqs[None, :]  # (batch, half)
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)  # (batch, dim)
        return self.proj(emb)  # (batch, dim)


# ---------------------------------------------------------------------------
# Multi-head self-attention (bidirectional — no causal mask)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, bits: int = 16, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout

        self.q_proj = QuantizedLinear(d_model, d_model, bias=False, bits=bits)
        self.k_proj = QuantizedLinear(d_model, d_model, bias=False, bits=bits)
        self.v_proj = QuantizedLinear(d_model, d_model, bias=False, bits=bits)
        self.out_proj = QuantizedLinear(d_model, d_model, bias=True, bits=bits)

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        B, L, _ = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).reshape(B, L, H, D).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, H, D).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, H, D).transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # (B, H, L, L)

        if mask is not None:
            # mask: (B, 1, 1, L) True = keep, False = ignore
            scores = mx.where(mask, scores, mx.full(scores.shape, float("-inf")))

        attn = nn.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)

        if self.dropout > 0.0:
            # Simple dropout approximation via scaling (MLX lacks nn.Dropout in older ver)
            pass

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, bits: int = 16, dropout: float = 0.0):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, bits=bits, dropout=dropout)
        self.ff1 = QuantizedLinear(d_model, d_ff, bits=bits)
        self.ff2 = QuantizedLinear(d_ff, d_model, bits=bits)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff2(nn.gelu(self.ff1(self.ln2(x))))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DiffusionLM(nn.Module):
    """
    Masked-diffusion language model with configurable per-step precision.

    The forward pass:
        1. Embed input tokens (including [MASK] tokens at noised positions).
        2. Add sinusoidal noise-level embedding (broadcast over sequence).
        3. Add learned positional embedding.
        4. Pass through N Transformer blocks.
        5. Project to logits over the full vocabulary.

    The model predicts the original token at EVERY position (including
    non-masked positions), but the training loss is computed only at masked
    positions.

    Precision is controlled externally:
        model.set_bits(bits)   # e.g., 1, 2, 4, or 16 for full precision
    All QuantizedLinear layers in attention + FF sub-layers are updated.
    Embeddings and projections remain in float32 throughout.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        vocab_with_mask = cfg.vocab_size + 1  # regular vocab + 1 MASK token

        # Embeddings — kept in full precision
        self.token_embed = nn.Embedding(vocab_with_mask, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.step_embed = SinusoidalEmbedding(cfg.d_model)

        # Transformer blocks — contain QuantizedLinear layers
        initial_bits = 16 if cfg.model_type == "baseline" else cfg.precision_schedule[0]
        self.blocks = [
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, bits=initial_bits, dropout=cfg.dropout)
            for _ in range(cfg.n_layers)
        ]

        # Output head — full precision (predicting vocab is a high-precision op)
        self.ln_out = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)  # logits over clean vocab only

    def set_bits(self, bits: int) -> None:
        """Update precision of all QuantizedLinear layers in the model."""
        for block in self.blocks:
            for _, module in block.named_modules():
                if isinstance(module, QuantizedLinear):
                    module.set_bits(bits)

    def get_current_bits(self) -> int:
        """Return current bits of the first QuantizedLinear found."""
        for block in self.blocks:
            for _, module in block.named_modules():
                if isinstance(module, QuantizedLinear):
                    return module.bits
        return 16

    def __call__(
        self,
        token_ids: mx.array,      # (B, L) integer IDs; MASK_TOKEN at noised positions
        step_frac: mx.array,      # (B,) float in [0, 1]; 1.0 = fully noisy, 0.0 = clean
        pad_mask: mx.array = None, # (B, L) bool; True = real token, False = padding
    ) -> mx.array:
        """
        Returns logits: (B, L, vocab_size).
        Logits at masked positions are used for training loss.
        """
        B, L = token_ids.shape

        positions = mx.arange(L)[None, :]  # (1, L)
        x = self.token_embed(token_ids) + self.pos_embed(positions)

        # Add noise-level embedding (same vector broadcast across sequence positions)
        step_emb = self.step_embed(step_frac)  # (B, d_model)
        x = x + step_emb[:, None, :]           # (B, L, d_model)

        # Attention mask for padding (None = attend to all positions)
        attn_mask = None
        if pad_mask is not None:
            # (B, 1, 1, L) — True = positions to attend to
            attn_mask = pad_mask[:, None, None, :]

        for block in self.blocks:
            x = block(x, attn_mask)

        x = self.ln_out(x)
        logits = self.lm_head(x)  # (B, L, vocab_size)
        return logits

    def count_params(self) -> dict:
        """Count trainable parameters by component."""
        def _count(m):
            return sum(v.size for _, v in mlx.utils.tree_flatten(m.parameters()))

        return {
            "token_embed": _count(self.token_embed),
            "pos_embed": _count(self.pos_embed),
            "step_embed": _count(self.step_embed),
            "blocks_total": sum(_count(b) for b in self.blocks),
            "lm_head": _count(self.lm_head),
        }

    def total_params(self) -> int:
        return sum(v.size for _, v in mlx.utils.tree_flatten(self.parameters()))
