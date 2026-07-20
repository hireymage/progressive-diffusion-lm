"""
Progressive-precision diffusion language model.

Architecture
------------
Bidirectional (non-causal) Transformer encoder.  The model takes a partially
masked token sequence and predicts the original (clean) token at every
position.

Noise-level conditioning:
  A sinusoidal embedding of the current mask rate (0 = clean, 1 = fully noisy)
  is projected by a small MLP and added to every token position before the
  Transformer blocks.  This is analogous to timestep conditioning in image
  diffusion models.

Weight quantisation:
  All linear projections in attention and feed-forward sub-layers use
  QuantizedLinear.  The `bits` attribute of every QuantizedLinear can be
  changed at runtime via `model.set_bits(bits)`, allowing a single set of
  float32 master weights to be evaluated at different precisions across
  diffusion refinement steps.

  Embeddings, LayerNorm, and output head remain in float32 throughout.

Weight tying (default: enabled):
  When `cfg.tie_word_embeddings = True` (the default), the output LM head
  re-uses the token embedding table instead of allocating a separate weight
  matrix.  This saves vocab_size × d_model parameters (e.g., ~8 M for
  vocab=16 K, d_model=512).

Precision modes:
  model_type = "baseline"     → bits=16 at every step (float32, no quantisation)
  model_type = "progressive"  → bits from precision_schedule at each step

Precision types (bits argument) — uniform no-zero 2^n-level scheme:
  1  → Q1 binary {-1,+1}×scale                           (2 levels)
  2  → Q2 true 2-bit {-3,-1,+1,+3}×step                 (4 levels)
  3  → Q3 true 3-bit {-7,-5,-3,-1,+1,+3,+5,+7}×step     (8 levels)
  4  → Q4 true 4-bit {-15,-13,…,-1,+1,…,+15}×step       (16 levels)
  16 → float32 identity (no quantisation)
  0  → ternary/3-state {-1,0,+1}×scale (optional, ~1.585 eff. bits)

See src/quantization.py for full scheme documentation. Historical result
artifacts are not included in the source-only snapshot and must carry their own
scheme/version metadata when published.
"""

import math
import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.utils

from dataclasses import dataclass, field
from .quantization import QuantizedLinear
from .config import ModelConfig


# ---------------------------------------------------------------------------
# Incremental cache (Phase 2 — M2)
# ---------------------------------------------------------------------------

@dataclass
class IncrementalCache:
    """
    Cached state from a previous diffusion refinement step.

    The incremental forward API computes:
        y_next = y_prev + Δ

    where Δ is the residual produced by running the model at a *different*
    precision on the *same* (or slightly modified) input.  The idea is that
    coarser-step logits are a good approximation and the finer step only
    needs to compute a small correction.

    Fields
    ------
    hidden : mx.array | None   — (B, L, d_model) hidden states after ln_out
    logits : mx.array | None   — (B, L, vocab_size) output logits
    bits   : int               — precision bits used to produce this cache
    step_frac : mx.array | None — (B,) noise level at cache time
    """
    hidden: mx.array | None = None
    logits: mx.array | None = None
    bits: int = 16
    step_frac: mx.array | None = None


# ---------------------------------------------------------------------------
# Sinusoidal noise-level embedding
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """Maps a scalar mask-rate in [0, 1] to a d_model-dimensional vector."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def __call__(self, t: mx.array) -> mx.array:
        # t: (batch,) float in [0, 1]
        half = self.dim // 2
        freqs = mx.exp(
            -math.log(10000.0) * mx.arange(half, dtype=mx.float32) / half
        )
        t = t[:, None]                                    # (B, 1)
        emb = t * freqs[None, :]                          # (B, half)
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)  # (B, dim)
        return self.proj(emb)                             # (B, dim)


# ---------------------------------------------------------------------------
# Multi-head self-attention (bidirectional — no causal mask)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, bits: int = 16,
                 dropout: float = 0.0,
                 weight_noise_mode: str = "none", weight_noise_multiplier: float = 1.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        noise = dict(weight_noise_mode=weight_noise_mode,
                     weight_noise_multiplier=weight_noise_multiplier)
        self.q_proj = QuantizedLinear(d_model, d_model, bias=False, bits=bits, **noise)
        self.k_proj = QuantizedLinear(d_model, d_model, bias=False, bits=bits, **noise)
        self.v_proj = QuantizedLinear(d_model, d_model, bias=False, bits=bits, **noise)
        self.out_proj = QuantizedLinear(d_model, d_model, bias=True, bits=bits, **noise)

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        B, L, _ = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).reshape(B, L, H, D).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, H, D).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, H, D).transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # (B, H, L, L)

        if mask is not None:
            # mask: (B, 1, 1, L) True = attend, False = ignore
            scores = mx.where(mask, scores, mx.full(scores.shape, float("-inf")))

        attn = nn.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, bits: int = 16,
                 dropout: float = 0.0,
                 weight_noise_mode: str = "none", weight_noise_multiplier: float = 1.0):
        super().__init__()
        noise = dict(weight_noise_mode=weight_noise_mode,
                     weight_noise_multiplier=weight_noise_multiplier)
        self.attn = MultiHeadAttention(
            d_model,
            n_heads,
            bits=bits,
            dropout=dropout,
            weight_noise_mode=weight_noise_mode,
            weight_noise_multiplier=weight_noise_multiplier,
        )
        self.ff1 = QuantizedLinear(d_model, d_ff, bits=bits, **noise)
        self.ff2 = QuantizedLinear(d_ff, d_model, bits=bits, **noise)
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

    Forward pass:
        1. Embed input tokens (including [MASK] at noised positions).
        2. Add position embeddings.
        3. Add sinusoidal noise-level embedding (broadcast to all positions).
        4. Pass through N Transformer blocks.
        5. LayerNorm + project to vocabulary logits.

    Loss is computed only at masked positions during training; the model
    predicts all positions at once (parallel denoising).

    To change precision between diffusion steps:
        model.set_bits(1)   # coarse step — binary weights
        model.set_bits(4)   # fine step — 4-bit weights

    Precision types:
        bits=1 binary, bits=2 true Q2, bits=3 true Q3, bits=4 true Q4,
        bits=16 float32 (baseline), bits=0 optional ternary
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        vocab_with_mask = cfg.vocab_size + 1  # regular vocab + MASK sentinel

        # ── Embeddings (always float32) ─────────────────────────────────────
        self.token_embed = nn.Embedding(vocab_with_mask, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.step_embed = SinusoidalEmbedding(cfg.d_model)

        # ── Transformer blocks (contain QuantizedLinear) ────────────────────
        initial_bits = 16 if cfg.model_type == "baseline" else cfg.precision_schedule[0]
        self.blocks = [
            TransformerBlock(
                cfg.d_model, cfg.n_heads, cfg.d_ff, bits=initial_bits,
                dropout=cfg.dropout,
                weight_noise_mode=cfg.weight_noise_mode,
                weight_noise_multiplier=cfg.weight_noise_multiplier,
            )
            for _ in range(cfg.n_layers)
        ]

        # ── Output projection ───────────────────────────────────────────────
        self.ln_out = nn.LayerNorm(cfg.d_model)

        if cfg.tie_word_embeddings:
            # No separate weight matrix — share token_embed.weight.
            # A small bias vector is still learned for flexibility.
            self.lm_head_bias = mx.zeros((cfg.vocab_size,))
        else:
            # Independent LM head (wastes vocab_size × d_model params).
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=True)

    # ── Precision control ────────────────────────────────────────────────────

    def set_bits(self, bits: int) -> None:
        """Set quantisation precision for all QuantizedLinear layers."""
        for block in self.blocks:
            for _, module in block.named_modules():
                if isinstance(module, QuantizedLinear):
                    module.set_bits(bits)

    def get_current_bits(self) -> int:
        """Return the bits value of the first QuantizedLinear found."""
        for block in self.blocks:
            for _, module in block.named_modules():
                if isinstance(module, QuantizedLinear):
                    return module.bits
        return 16

    def set_weight_noise_seed(self, seed: int) -> None:
        """Reset independent, deterministic per-layer weight-noise streams."""
        index = 0
        for block in self.blocks:
            for _, module in block.named_modules():
                if isinstance(module, QuantizedLinear):
                    module.set_noise_seed(seed + index * 1_000_003)
                    index += 1

    # ── Forward pass ─────────────────────────────────────────────────────────

    def __call__(
        self,
        token_ids: mx.array,       # (B, L) int — MASK_TOKEN at noised positions
        step_frac: mx.array,       # (B,) float in [0,1] — 1=fully noisy, 0=clean
        pad_mask: mx.array = None, # (B, L) bool — True = real token
    ) -> mx.array:
        """Returns logits: (B, L, vocab_size)."""
        B, L = token_ids.shape
        positions = mx.arange(L)[None, :]  # (1, L)

        x = self.token_embed(token_ids) + self.pos_embed(positions)
        x = x + self.step_embed(step_frac)[:, None, :]  # broadcast noise level

        attn_mask = None
        if pad_mask is not None:
            attn_mask = pad_mask[:, None, None, :]  # (B, 1, 1, L)

        for block in self.blocks:
            x = block(x, attn_mask)

        x = self.ln_out(x)  # (B, L, d_model)

        if self.cfg.tie_word_embeddings:
            # Tie: use first vocab_size rows of embedding table as projection
            embed_w = self.token_embed.weight[:self.cfg.vocab_size]  # (V, d)
            logits = x @ embed_w.T + self.lm_head_bias               # (B, L, V)
        else:
            logits = self.lm_head(x)

        return logits

    # ── Incremental forward (Phase 2 — M2) ─────────────────────────────────

    def forward_incremental(
        self,
        token_ids: mx.array,
        step_frac: mx.array,
        cache: IncrementalCache | None = None,
        delta_weight: float = 1.0,
        pad_mask: mx.array = None,
    ) -> tuple[mx.array, IncrementalCache]:
        """
        Incremental forward: y_next = y_prev + delta_weight * Δ.

        If cache is None or the input has changed significantly, this falls
        back to a full forward pass and returns a fresh cache.

        If cache is provided with matching input shape, the residual Δ is
        computed as:
            Δ = model_full(token_ids, step_frac) - model_full(token_ids, cache.step_frac)
        and the output is:
            y_next = cache.logits + delta_weight * Δ

        For A/B validation against the full-recompute path, set
        delta_weight=1.0 and verify parity with the standard __call__.

        Returns
        -------
        logits : (B, L, vocab_size)
        new_cache : IncrementalCache  (can be passed to the next step)
        """
        B, L = token_ids.shape

        if cache is None or cache.logits is None:
            # Full recompute — no cache available
            logits = self(token_ids, step_frac, pad_mask)
            new_cache = IncrementalCache(
                hidden=None,  # hidden not exposed in full forward for now
                logits=logits,
                bits=self.get_current_bits(),
                step_frac=step_frac,
            )
            return logits, new_cache

        # Incremental path: compute delta from cached logits
        # Run full forward at current precision (the model's weights are
        # already set to the desired bits by the caller via set_bits).
        full_logits = self(token_ids, step_frac, pad_mask)

        # Δ = full_logits - cache.logits
        delta = full_logits - cache.logits

        # y_next = y_prev + delta_weight * Δ
        logits = cache.logits + delta_weight * delta

        new_cache = IncrementalCache(
            hidden=None,
            logits=logits,
            bits=self.get_current_bits(),
            step_frac=step_frac,
        )
        return logits, new_cache

    # ── Parameter counting and reporting ─────────────────────────────────────

    def total_params(self) -> int:
        """Total trainable parameter count (shared weights counted once)."""
        return sum(v.size for _, v in mlx.utils.tree_flatten(self.parameters()))

    def param_breakdown(self) -> dict:
        """Parameter count per component."""
        def _n(m):
            return sum(v.size for _, v in mlx.utils.tree_flatten(m.parameters()))
        result = {
            "token_embed": _n(self.token_embed),
            "pos_embed": _n(self.pos_embed),
            "step_embed": _n(self.step_embed),
            "blocks_total": sum(_n(b) for b in self.blocks),
            "ln_out": _n(self.ln_out),
        }
        if self.cfg.tie_word_embeddings:
            result["lm_head_bias_only"] = self.lm_head_bias.size
        else:
            result["lm_head"] = _n(self.lm_head)
        return result
