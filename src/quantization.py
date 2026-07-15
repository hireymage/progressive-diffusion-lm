"""
Quantization-aware training (QAT) linear layers.

Weight representations
----------------------
1-bit (binary):
    w_q = sign(w) → {-1, +1}  (0 maps to +1 by convention)
    w_approx = α * w_q  where α = mean(|w|) per output-row (optimal l2 scale)

    This is NOT ternary {-1, 0, +1}.  True binary is the primary experiment.
    The optional ternary mode is exposed via bits="ternary" (1.58 effective bits).

2-bit (symmetric uniform, 3 levels):
    levels {-1, 0, +1} scaled by per-row max.  Note: 3 distinct levels ≈ 1.58
    effective bits, the natural result of symmetric signed integer rounding.

4-bit (symmetric uniform, 15 levels):
    Integer codes {-7, …, 7} scaled by per-row max / 7.  15 distinct values.

Straight-Through Estimator (STE):
    Forward pass uses quantised weights.
    Backward pass passes gradients through to the full-precision (latent) weights
    unchanged, implemented as:
        w_ste = w + stop_gradient(w_q - w)
    so ∂L/∂w = ∂L/∂w_ste (identity gradient w.r.t. latent weights).
"""

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Per-row quantise helpers (return the float approximation, not integer codes)
# ---------------------------------------------------------------------------

def _quantize_1bit(w: mx.array) -> mx.array:
    """Binary {-1, +1} scaled by per-row mean-absolute-value."""
    scale = mx.mean(mx.abs(w), axis=-1, keepdims=True)
    scale = mx.maximum(scale, 1e-8)
    w_bin = mx.where(w >= 0, mx.ones_like(w), -mx.ones_like(w))
    return w_bin * scale


def _quantize_2bit(w: mx.array) -> mx.array:
    """Symmetric 3-level: {-1, 0, +1} * scale.  scale = per-row max(|w|)."""
    scale = mx.max(mx.abs(w), axis=-1, keepdims=True)
    scale = mx.maximum(scale, 1e-8)
    w_norm = w / scale          # in [-1, 1]
    w_q = mx.clip(mx.round(w_norm), -1.0, 1.0)
    return w_q * scale


def _quantize_4bit(w: mx.array) -> mx.array:
    """Symmetric 15-level: codes {-7…7} scaled by per-row max(|w|)/7."""
    n = 7.0
    scale = mx.max(mx.abs(w), axis=-1, keepdims=True)
    scale = mx.maximum(scale, 1e-8)
    w_norm = w / scale * n      # in approx [-7, 7]
    w_q = mx.clip(mx.round(w_norm), -n, n) / n
    return w_q * scale


def quantize_weights(w: mx.array, bits: int) -> mx.array:
    """
    Return a float approximation of w quantised to `bits` bits.
    bits=1  → binary ±scale
    bits=2  → ternary ×scale
    bits=4  → 15-level ×scale
    bits=16 → identity (no quantisation)
    """
    if bits == 1:
        return _quantize_1bit(w)
    elif bits == 2:
        return _quantize_2bit(w)
    elif bits == 4:
        return _quantize_4bit(w)
    else:
        return w   # full-precision pass-through


def ste_quantize(w: mx.array, bits: int) -> mx.array:
    """
    Straight-Through Estimator wrapper.
    Forward:  returns quantised weights.
    Backward: gradient flows to w unchanged (identity STE).
    """
    if bits >= 16:
        return w
    w_q = quantize_weights(w, bits)
    # STE: forward = w_q, backward = identity gradient w.r.t. w
    return w + mx.stop_gradient(w_q - w)


# ---------------------------------------------------------------------------
# Quantised linear layer
# ---------------------------------------------------------------------------

class QuantizedLinear(nn.Module):
    """
    Linear layer with quantization-aware training.

    Maintains full-precision (float32) master/latent weights for gradient
    updates.  During the forward pass, weights are quantised to `bits`
    precision using the STE.

    `bits` can be changed at any time via set_bits(); this allows the same
    layer to operate at different precisions during different diffusion steps
    without allocating separate weight tensors.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits

        # Xavier uniform init
        bound = (6.0 / (in_features + out_features)) ** 0.5
        self.weight = mx.random.uniform(-bound, bound, (out_features, in_features))
        self.bias = mx.zeros((out_features,)) if bias else None

    def set_bits(self, bits: int) -> None:
        self.bits = bits

    def __call__(self, x: mx.array) -> mx.array:
        w = ste_quantize(self.weight, self.bits)
        out = x @ w.T
        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# Utility: walk a model and set precision on all QuantizedLinear layers
# ---------------------------------------------------------------------------

def set_model_bits(model: nn.Module, bits: int) -> None:
    """Recursively set quantisation bits for all QuantizedLinear layers."""
    for _, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            module.set_bits(bits)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def bits_per_param(model: nn.Module) -> float:
    """
    Estimated average bits per weight parameter across all QuantizedLinear
    layers (excluding biases, which are always float32).
    """
    total_params = 0
    total_bits = 0
    for _, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            n = module.weight.size
            total_params += n
            total_bits += n * module.bits
    if total_params == 0:
        return 32.0  # fallback: assume float32
    return total_bits / total_params


def theoretical_storage_bytes(model: nn.Module) -> dict:
    """
    Report theoretical storage for quantised vs full-precision weights.
    NOTE: MLX stores master weights as float32 regardless; these numbers
    represent theoretical storage if a dedicated low-bit kernel were used.
    """
    total_q_bits = 0
    total_params = 0
    for _, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            n = module.weight.size
            total_params += n
            total_q_bits += n * module.bits
    fp32_bytes = total_params * 4
    q_bytes = total_q_bits // 8
    return {
        "fp32_bytes": fp32_bytes,
        "theoretical_quantised_bytes": q_bytes,
        "theoretical_compression_ratio": fp32_bytes / max(q_bytes, 1),
    }
