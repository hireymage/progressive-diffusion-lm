"""
Quantization-aware training (QAT) linear layers.

Weight representations (bits argument to QuantizedLinear / quantize_weights)
-----------------------------------------------------------------------------
Uniform no-zero symmetric scheme: 2^n levels, all non-zero.
General form: levels = {±1, ±3, …, ±(2^n - 1)} × step
              step   = max(|w|) / (2^n - 1) per output-row
              boundaries at ±2, ±4, …, ±(2^n - 2) × step

bits=1  — Q1 / Binary: 2 levels {-1, +1}.
            scale = mean(|w|) per output-row (optimal l2 approximation).
            w_approx = sign(w) * scale.  (0 maps to +1 by convention.)
            Effective bits: 1.0

bits=2  — Q2 / True 2-bit: 4 levels {-3, -1, +1, +3} × step.
            step = max(|w|) / 3.  Boundaries: ±2 × step.
            No zero level.  Effective bits: 2.0

bits=3  — Q3 / True 3-bit: 8 levels {-7, -5, -3, -1, +1, +3, +5, +7} × step.
            step = max(|w|) / 7.  Boundaries: ±2, ±4, ±6 × step.
            No zero level.  Effective bits: 3.0

bits=4  — Q4 / True 4-bit: 16 levels {-15, -13, …, -1, +1, …, +15} × step.
            step = max(|w|) / 15.  Boundaries: ±2, ±4, …, ±14 × step.
            No zero level.  Effective bits: 4.0
            NOTE: prior to this scheme the 4-bit mode used 15 levels
            {-7,…,+7}×scale/7 (with zero).  The ablation const_4bit variant
            was trained under the old scheme — see ptq_study.py for caveat.

bits=16 — Identity pass-through (no quantisation). Used for the baseline.
            Storage bits: 32.0 because master/checkpoint weights are float32.

bits=0  — Ternary (optional / experimental): 3 levels {-1, 0, +1} × scale.
            scale = max(|w|) per output-row.  Boundary: ±0.5 × scale.
            Effective bits ≈ log2(3) ≈ 1.585.  NOT part of the main Q1–Q4
            matrix; use bits=0 only for the optional ternary comparison.

Straight-Through Estimator (STE)
---------------------------------
Forward:  w_ste  =  quantize(w)       ← quantised weights used in matmul
Backward: ∂L/∂w  =  ∂L/∂w_ste        ← identity gradient to full-prec weights

Implemented as:
    w_ste = w + stop_gradient(quantize(w) - w)

The full-precision master weights (float32) are updated by the optimiser.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.utils

_LAST_INJECTED_NOISE: dict[int, mx.array] = {}

# Effective bits per scheme value.
# 0 = ternary sentinel (optional, ~1.585 bits); 3/4 are true 3-/4-bit.
EFFECTIVE_BITS = {0: math.log2(3), 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 16: 32.0}


# ---------------------------------------------------------------------------
# Pure quantisation helpers (return float approximations, no integer codes)
# ---------------------------------------------------------------------------

def _quantize_1bit(w: mx.array) -> mx.array:
    """
    Binary {-1, +1} with per-row mean-absolute-value scale.
    w_q = sign(w);  0 maps to +1 by convention.
    scale = mean(|w|, axis=-1, keepdims=True)
    w_approx = w_q * scale
    """
    scale = mx.mean(mx.abs(w), axis=-1, keepdims=True)
    scale = mx.maximum(scale, 1e-8)
    w_bin = mx.where(w >= 0, mx.ones_like(w), -mx.ones_like(w))
    return w_bin * scale


def _quantize_2bit(w: mx.array) -> mx.array:
    """
    Q2 / True 2-bit: 4 symmetric levels {-3, -1, +1, +3} × step.

    step = max(|w|) / 3 per output-row.
    Boundaries at ±2×step.  No zero level.  Effective bits: 2.0
    """
    w_max = mx.max(mx.abs(w), axis=-1, keepdims=True)
    step = mx.maximum(w_max / 3.0, 1e-8)
    w_norm = w / step
    w_sign = mx.where(w_norm >= 0, mx.ones_like(w_norm), -mx.ones_like(w_norm))
    w_mag = mx.where(mx.abs(w_norm) >= 2.0,
                     mx.full_like(w_norm, 3.0),
                     mx.ones_like(w_norm))
    return w_sign * w_mag * step


def _quantize_3bit(w: mx.array) -> mx.array:
    """
    Q3 / True 3-bit: 8 symmetric levels {-7, -5, -3, -1, +1, +3, +5, +7} × step.

    step = max(|w|) / 7 per output-row.
    Boundaries at ±2, ±4, ±6 × step.
    General formula: mag = 2·floor(|w_norm|/2)+1, capped at 7.
    No zero level.  Effective bits: 3.0
    """
    w_max = mx.max(mx.abs(w), axis=-1, keepdims=True)
    step = mx.maximum(w_max / 7.0, 1e-8)
    w_norm = w / step
    w_sign = mx.where(w_norm >= 0, mx.ones_like(w_norm), -mx.ones_like(w_norm))
    w_abs = mx.abs(w_norm)
    w_mag = mx.minimum(2.0 * mx.floor(w_abs / 2.0) + 1.0, 7.0)
    return w_sign * w_mag * step


def _quantize_ternary(w: mx.array) -> mx.array:
    """
    Ternary (optional): 3 levels {-1, 0, +1} × scale.

    scale = max(|w|) per output-row.
    Boundary at ±0.5×scale.  Effective bits ≈ log2(3) ≈ 1.585.
    Accessed via bits=0 as a sentinel value.  NOT part of the main Q1–Q4 matrix.
    """
    scale = mx.max(mx.abs(w), axis=-1, keepdims=True)
    scale = mx.maximum(scale, 1e-8)
    w_norm = w / scale
    w_q = mx.clip(mx.round(w_norm), -1.0, 1.0)
    return w_q * scale


def _quantize_4bit(w: mx.array) -> mx.array:
    """
    Q4 / True 4-bit: 16 symmetric levels {-15, -13, …, -1, +1, …, +15} × step.

    step = max(|w|) / 15 per output-row.
    Boundaries at ±2, ±4, …, ±14 × step.
    General formula: mag = 2·floor(|w_norm|/2)+1, capped at 15.
    No zero level.  Effective bits: 4.0
    """
    w_max = mx.max(mx.abs(w), axis=-1, keepdims=True)
    step = mx.maximum(w_max / 15.0, 1e-8)
    w_norm = w / step
    w_sign = mx.where(w_norm >= 0, mx.ones_like(w_norm), -mx.ones_like(w_norm))
    w_abs = mx.abs(w_norm)
    w_mag = mx.minimum(2.0 * mx.floor(w_abs / 2.0) + 1.0, 15.0)
    return w_sign * w_mag * step


def quantize_weights(w: mx.array, bits: int) -> mx.array:
    """
    Return float approximation of w at the given precision.

    All main levels (bits=1–4) use the no-zero symmetric scheme: 2^n levels.

    bits=1  → Q1  binary ×scale        (2 levels,  eff. 1.0 bits)
    bits=2  → Q2  true 2-bit ×step     (4 levels,  eff. 2.0 bits)
    bits=3  → Q3  true 3-bit ×step     (8 levels,  eff. 3.0 bits)
    bits=4  → Q4  true 4-bit ×step     (16 levels, eff. 4.0 bits)
    bits=16 → identity                 (no quantisation)
    bits=0  → ternary ×scale (optional, 3 levels, eff. ~1.585 bits)
    """
    if bits == 1:
        return _quantize_1bit(w)
    elif bits == 2:
        return _quantize_2bit(w)
    elif bits == 3:
        return _quantize_3bit(w)
    elif bits == 4:
        return _quantize_4bit(w)
    elif bits == 0:
        return _quantize_ternary(w)
    else:
        return w   # full-precision pass-through (bits=16 and any other value)


def ste_quantize(w: mx.array, bits: int) -> mx.array:
    """
    Straight-Through Estimator quantisation wrapper.

    Forward:  returns quantised weights  (float approximation of low-bit w).
    Backward: gradient passes through to full-precision w unchanged.

    Implementation:
        w_ste = w + stop_gradient(quantize(w) - w)
    """
    if bits >= 16:
        return w
    w_q = quantize_weights(w, bits)
    return w + mx.stop_gradient(w_q - w)


# ---------------------------------------------------------------------------
# Quantised linear layer
# ---------------------------------------------------------------------------

class QuantizedLinear(nn.Module):
    """
    Linear layer with quantization-aware training (QAT).

    Stores full-precision (float32) master weights that the optimiser updates.
    During the forward pass, weights are quantised to `bits` precision via STE
    so that gradients flow to the master weights.

    `bits` is a runtime attribute that can be changed at any time via
    `set_bits()`, enabling one model to switch precision between diffusion steps.

    Master weight dtype: float32 (always).
    Forward weight dtype: float32 (all operations simulated in float32).
    Checkpoint dtype: float32 (MLX .npz saves as float32).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 16,
        weight_noise_mode: str = "none",
        weight_noise_multiplier: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.weight_noise_mode = weight_noise_mode
        self.weight_noise_multiplier = weight_noise_multiplier
        self._noise_seed = 0
        self._noise_calls = 0

        bound = (6.0 / (in_features + out_features)) ** 0.5
        self.weight = mx.random.uniform(-bound, bound, (out_features, in_features))
        self.bias = mx.zeros((out_features,)) if bias else None

    def set_bits(self, bits: int) -> None:
        self.bits = bits

    def effective_bits(self) -> float:
        return EFFECTIVE_BITS.get(self.bits, float(self.bits))

    def set_noise_seed(self, seed: int) -> None:
        """Set this layer's independent deterministic noise stream."""
        self._noise_seed = int(seed)
        self._noise_calls = 0

    def _matched_noise(self) -> mx.array:
        residual = quantize_weights(self.weight, 1) - self.weight
        target_rms = mx.stop_gradient(
            mx.sqrt(mx.mean(residual * residual, axis=-1, keepdims=True))
        ) * self.weight_noise_multiplier
        key = mx.random.key(self._noise_seed + self._noise_calls)
        self._noise_calls += 1
        if self.weight_noise_mode == "gaussian_matched":
            return mx.random.normal(self.weight.shape, key=key) * target_rms
        amplitude = math.sqrt(3.0) * target_rms
        return mx.random.uniform(-1.0, 1.0, self.weight.shape, key=key) * amplitude

    def __call__(self, x: mx.array) -> mx.array:
        if self.training and self.bits >= 16 and self.weight_noise_mode != "none":
            noise = self._matched_noise()
            _LAST_INJECTED_NOISE[id(self)] = noise
            w = self.weight + noise
        else:
            _LAST_INJECTED_NOISE.pop(id(self), None)
            w = ste_quantize(self.weight, self.bits)
        out = x @ w.T
        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# Model-wide bit setting
# ---------------------------------------------------------------------------

def set_model_bits(model: nn.Module, bits: int) -> None:
    """Recursively set quantisation bits for all QuantizedLinear layers."""
    for _, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            module.set_bits(bits)


def quantization_noise_metrics(model: nn.Module) -> dict[str, float]:
    """Return element-weighted Q1 residual and latest injected-noise RMS."""
    residual_sumsq = mx.array(0.0)
    noise_sumsq = mx.array(0.0)
    count = 0
    for _, module in model.named_modules():
        if not isinstance(module, QuantizedLinear):
            continue
        residual = quantize_weights(module.weight, 1) - module.weight
        residual_sumsq = residual_sumsq + mx.sum(residual * residual)
        noise = _LAST_INJECTED_NOISE.get(id(module))
        if noise is not None:
            noise_sumsq = noise_sumsq + mx.sum(noise * noise)
        count += module.weight.size
    if count == 0:
        return {"q1_residual_rms": 0.0, "injected_noise_rms": 0.0}
    residual_rms = mx.sqrt(residual_sumsq / count)
    noise_rms = mx.sqrt(noise_sumsq / count)
    mx.eval(residual_rms, noise_rms)
    return {
        "q1_residual_rms": float(residual_rms),
        "injected_noise_rms": float(noise_rms),
    }


# ---------------------------------------------------------------------------
# Storage and precision reporting
# ---------------------------------------------------------------------------

def model_storage_report(model: nn.Module, precision_schedule: list[int]) -> dict:
    """
    Compute detailed storage and precision statistics for the model.

    Separates QuantizedLinear parameters (quantised during inference)
    from other parameters (always float32: embeddings, LayerNorm, LM head).

    Parameters are counted from model.parameters() (the ground truth).

    Returns actual FP32 master/checkpoint storage, temporal average step
    precision, and (only for constant schedules) a hypothetical packed lower
    bound including FP32 per-row scales. Dynamic schedules do not imply a
    packed model size because this implementation requantizes FP32 masters at
    runtime.
    """
    # Collect QuantizedLinear weight sizes and per-output-row scale counts.
    q_weight_params = 0
    q_scale_params = 0
    seen_ids = set()
    for _, module in model.named_modules():
        if isinstance(module, QuantizedLinear) and id(module) not in seen_ids:
            seen_ids.add(id(module))
            q_weight_params += module.weight.size
            q_scale_params += module.weight.shape[0]

    total_params = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
    non_q_params = total_params - q_weight_params

    # This is a temporal compute statistic, not a stored-model bit width.
    average_step_bits = float(
        sum(EFFECTIVE_BITS.get(b, float(b)) for b in precision_schedule)
        / len(precision_schedule)
    )

    fp32_total = total_params * 4
    bf16_total = total_params * 2

    # The current implementation/checkpoints always retain FP32 master weights.
    actual_model_bytes = fp32_total

    # A packed-storage lower bound is meaningful only for a constant schedule.
    # Progressive schedules requantize the FP32 master at runtime and therefore
    # need an explicit deployment representation contract before claiming size.
    unique_bits = set(precision_schedule)
    hypothetical_packed_bytes = None
    hypothetical_packed_compression = None
    if len(unique_bits) == 1:
        constant_bits = precision_schedule[0]
        if constant_bits == 16:
            hypothetical_packed_bytes = fp32_total
        else:
            storage_bits = EFFECTIVE_BITS.get(constant_bits, float(constant_bits))
            packed_weights = math.ceil(q_weight_params * storage_bits / 8)
            scale_bytes = q_scale_params * 4
            hypothetical_packed_bytes = packed_weights + scale_bytes + non_q_params * 4
        hypothetical_packed_compression = fp32_total / max(hypothetical_packed_bytes, 1)

    # Training memory: master (fp32) + gradients (fp32) + Adam m+v (fp32 each)
    # ≈ 4× the number of trainable parameters in bytes
    training_mem_mb = total_params * 4 * 4 / 1e6

    return {
        "total_params": total_params,
        "q_linear_weight_params": q_weight_params,
        "q_linear_scale_params": q_scale_params,
        "non_quantized_params": non_q_params,
        "fp32_total_bytes": fp32_total,
        "fp32_total_mb": fp32_total / 1e6,
        "bf16_total_bytes": bf16_total,
        "bf16_total_mb": bf16_total / 1e6,
        "actual_model_bytes": actual_model_bytes,
        "actual_model_mb": actual_model_bytes / 1e6,
        "actual_compression_vs_fp32": 1.0,
        "average_step_weight_bits": average_step_bits,
        "hypothetical_packed_bytes": hypothetical_packed_bytes,
        "hypothetical_packed_mb": (
            hypothetical_packed_bytes / 1e6
            if hypothetical_packed_bytes is not None
            else None
        ),
        "hypothetical_packed_compression_vs_fp32": hypothetical_packed_compression,
        # Backward-compatible aliases now report actual runtime storage, not a
        # schedule-average packed model that does not exist.
        "theoretical_q_bytes": actual_model_bytes,
        "theoretical_q_mb": actual_model_bytes / 1e6,
        "effective_avg_bits": average_step_bits,
        "compression_vs_fp32": 1.0,
        "compression_vs_bf16": bf16_total / actual_model_bytes,
        "training_memory_estimate_mb": training_mem_mb,
    }


def bits_per_param_from_schedule(precision_schedule: list[int]) -> float:
    """Average temporal weight bit-width per refinement step/MAC."""
    return sum(EFFECTIVE_BITS.get(b, float(b)) for b in precision_schedule) / len(precision_schedule)


def level_counts() -> dict:
    """Return the number of distinct quantisation levels per bits setting."""
    return {0: 3, 1: 2, 2: 4, 3: 8, 4: 16, 16: "float32 (continuous)"}
