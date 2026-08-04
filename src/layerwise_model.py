"""Layer-wise grouped-precision language-model prototype.

Unlike the legacy diffusion model, this module changes precision *between
Transformer layers*, not between complete model passes.  Its default schedule
is five layers each of Q1, Q2, Q4, Q8, and actual FP16 arithmetic.  A shared
LM head is applied after every layer from ``min_exit_layer`` onward, so a
sequence-wide controller may stop at any such layer (including within a
precision group, e.g. layer 8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import mlx.core as mx
import mlx.nn as nn

from .config import LayerwiseModelConfig
from .quantization import ste_quantize


PRECISION_PROXY_COSTS: Mapping[str, int] = {
    "q1": 1, "q2": 2, "q4": 4, "q8": 8, "fp16": 16, "fp32": 32,
}


def proxy_cost_for_schedule(precisions: list[str], exit_layer: int | None = None) -> int:
    """Cost proxy for executed 1-indexed layers (not packed-model storage)."""
    if exit_layer is None:
        exit_layer = len(precisions)
    if not 1 <= exit_layer <= len(precisions):
        raise ValueError("exit_layer must be in [1, len(precisions)]")
    try:
        return sum(PRECISION_PROXY_COSTS[name] for name in precisions[:exit_layer])
    except KeyError as error:
        raise ValueError(f"unknown precision {error.args[0]!r}") from error


def fp32_reference_cost(n_layers: int) -> int:
    return n_layers * PRECISION_PROXY_COSTS["fp32"]


class LayerwiseLinear(nn.Module):
    """QAT linear layer with explicit FP16 and FP32 execution modes.

    Master weights and checkpoints remain FP32.  ``fp16`` casts both the
    activation and weight for the matmul, then returns FP32 for stable residual
    accumulation.  Quantized modes retain STE training against FP32 masters.
    """

    _Q_BITS = {"q1": 1, "q2": 2, "q4": 4, "q8": 8}

    def __init__(self, in_features: int, out_features: int, precision: str,
                 bias: bool = True):
        super().__init__()
        if precision not in PRECISION_PROXY_COSTS:
            raise ValueError(f"unsupported layer precision {precision!r}")
        bound = (6.0 / (in_features + out_features)) ** 0.5
        self.weight = mx.random.uniform(-bound, bound, (out_features, in_features))
        self.bias = mx.zeros((out_features,)) if bias else None
        self.precision = precision

    def __call__(self, x: mx.array) -> mx.array:
        if self.precision == "fp16":
            # This is deliberately not legacy bits=16: the matmul itself uses
            # fp16 operands.  Residual streams remain fp32 afterwards.
            out = self.fp16_matmul(x).astype(mx.float32)
        elif self.precision == "fp32":
            # Keep this a distinct, actual FP32 matmul path; it is the fair
            # all-FP32 control for the layer-wise pilot.
            out = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        else:
            out = x @ ste_quantize(self.weight, self._Q_BITS[self.precision]).T
        if self.bias is not None:
            out = out + self.bias
        return out

    def fp16_matmul(self, x: mx.array) -> mx.array:
        """Expose the actual FP16 product for focused numerical tests."""
        if self.precision != "fp16":
            raise ValueError("fp16_matmul is only valid for precision='fp16'")
        return x.astype(mx.float16) @ self.weight.astype(mx.float16).T


class LayerwiseAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, precision: str, dropout: float):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = LayerwiseLinear(d_model, d_model, precision, bias=False)
        self.k_proj = LayerwiseLinear(d_model, d_model, precision, bias=False)
        self.v_proj = LayerwiseLinear(d_model, d_model, precision, bias=False)
        self.out_proj = LayerwiseLinear(d_model, d_model, precision)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        batch, length, d_model = x.shape
        heads, head_dim = self.n_heads, self.head_dim
        q = self.q_proj(x).reshape(batch, length, heads, head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch, length, heads, head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch, length, heads, head_dim).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = mx.where(mask, scores, mx.full(scores.shape, float("-inf")))
        attention = nn.softmax(scores.astype(mx.float32), axis=-1)
        attention = self.dropout(attention)
        out = (attention @ v).transpose(0, 2, 1, 3).reshape(batch, length, d_model)
        return self.out_proj(out)


class LayerwiseTransformerBlock(nn.Module):
    def __init__(self, cfg: LayerwiseModelConfig, precision: str):
        super().__init__()
        self.precision = precision
        self.attn = LayerwiseAttention(cfg.d_model, cfg.n_heads, precision, cfg.dropout)
        self.ff1 = LayerwiseLinear(cfg.d_model, cfg.d_ff, precision)
        self.ff2 = LayerwiseLinear(cfg.d_ff, cfg.d_model, precision)
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        x = x + self.attn(self.ln1(x), mask)
        return x + self.ff2(nn.gelu(self.ff1(self.ln2(x))))


@dataclass
class EarlyExitResult:
    logits: mx.array
    exit_layer: int
    proxy_cost: int
    mean_margin: float


class LayerwiseProgressiveLM(nn.Module):
    """25-layer prototype with shared intermediate prediction head."""

    def __init__(self, cfg: LayerwiseModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size + 1, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = [
            LayerwiseTransformerBlock(cfg, precision)
            for precision in cfg.layer_precisions
        ]
        self.ln_out = nn.LayerNorm(cfg.d_model)
        if cfg.tie_word_embeddings:
            self.lm_head_bias = mx.zeros((cfg.vocab_size,))
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def _head(self, hidden: mx.array) -> mx.array:
        hidden = self.ln_out(hidden)
        if self.cfg.tie_word_embeddings:
            return hidden @ self.token_embed.weight[:self.cfg.vocab_size].T + self.lm_head_bias
        return self.lm_head(hidden)

    def forward_intermediates(
        self, token_ids: mx.array, pad_mask: mx.array | None = None,
        exit_layer: int | None = None,
        requested_layers: tuple[int, ...] | None = None,
    ) -> dict[int, mx.array]:
        """Return shared-head logits after all eligible executed layers."""
        batch, length = token_ids.shape
        if length > self.cfg.max_seq_len:
            raise ValueError("token sequence exceeds max_seq_len")
        if exit_layer is None:
            exit_layer = self.cfg.n_layers
        if not self.cfg.min_exit_layer <= exit_layer <= self.cfg.n_layers:
            raise ValueError("exit_layer must be from min_exit_layer through n_layers")
        x = self.token_embed(token_ids) + self.pos_embed(mx.arange(length)[None, :])
        attention_mask = pad_mask[:, None, None, :] if pad_mask is not None else None
        requested = set(requested_layers) if requested_layers is not None else set(range(self.cfg.min_exit_layer, exit_layer + 1))
        if not requested.issubset(set(range(self.cfg.min_exit_layer, exit_layer + 1))):
            raise ValueError("requested_layers must be eligible executed exit layers")
        logits_by_layer: dict[int, mx.array] = {}
        for index, block in enumerate(self.blocks, start=1):
            x = block(x, attention_mask)
            if index in requested:
                logits_by_layer[index] = self._head(x)
            if index == exit_layer:
                break
        return logits_by_layer

    def __call__(self, token_ids: mx.array, pad_mask: mx.array | None = None,
                 exit_layer: int | None = None) -> mx.array:
        """Return logits at one requested sequence-wide exit layer."""
        outputs = self.forward_intermediates(token_ids, pad_mask, exit_layer)
        return outputs[max(outputs)]

    def early_exit(self, token_ids: mx.array, margin_threshold: float,
                   pad_mask: mx.array | None = None) -> EarlyExitResult:
        """Run once and stop all sequences together at the first confident layer."""
        batch, length = token_ids.shape
        if length > self.cfg.max_seq_len:
            raise ValueError("token sequence exceeds max_seq_len")
        x = self.token_embed(token_ids) + self.pos_embed(mx.arange(length)[None, :])
        attention_mask = pad_mask[:, None, None, :] if pad_mask is not None else None
        latest_logits, latest_margin = None, 0.0
        for index, block in enumerate(self.blocks, start=1):
            x = block(x, attention_mask)
            if index < self.cfg.min_exit_layer:
                continue
            logits = self._head(x)
            # ``mx.topk`` does not promise descending output order.  Sorting
            # makes the confidence definition explicit: largest minus runner-up.
            top_two = mx.sort(logits, axis=-1)[..., -2:]
            margin = mx.mean(top_two[..., 1] - top_two[..., 0])
            mx.eval(margin)
            latest_logits, latest_margin = logits, float(margin)
            if latest_margin >= margin_threshold:
                return EarlyExitResult(logits, index, proxy_cost_for_schedule(self.cfg.layer_precisions, index), latest_margin)
        assert latest_logits is not None
        return EarlyExitResult(latest_logits, self.cfg.n_layers,
                               proxy_cost_for_schedule(self.cfg.layer_precisions), latest_margin)

    def proxy_cost(self, exit_layer: int | None = None) -> int:
        return proxy_cost_for_schedule(self.cfg.layer_precisions, exit_layer)

    def fp32_reference_cost(self) -> int:
        return fp32_reference_cost(self.cfg.n_layers)


def masked_deep_supervision_loss(
    model: LayerwiseProgressiveLM, token_ids: mx.array, targets: mx.array,
    masked_positions: mx.array, pad_mask: mx.array | None = None,
    supervised_layers: tuple[int, ...] | None = None,
    layer_weights: tuple[float, ...] | None = None,
) -> mx.array:
    """Streaming masked CE over selected exits, without retaining 21 logits.

    The sum is constructed during the single forward pass.  This keeps only
    the current vocab-logit tensor live from Python's point of view, unlike
    ``forward_intermediates`` which is intentionally retained for analysis.
    MLX still keeps the required autograd graph, so the peak is not identical
    to inference; it does avoid an avoidable Python container of logits.
    """
    selected = set(supervised_layers or tuple(range(model.cfg.min_exit_layer, model.cfg.n_layers + 1)))
    if not selected or not selected.issubset(set(range(model.cfg.min_exit_layer, model.cfg.n_layers + 1))):
        raise ValueError("supervised_layers must be eligible, non-empty exit layers")
    ordered_selected = tuple(sorted(selected))
    if layer_weights is not None and (len(layer_weights) != len(ordered_selected) or any(weight <= 0 for weight in layer_weights)):
        raise ValueError("layer_weights must be positive and match supervised_layers")
    weights = dict(zip(ordered_selected, layer_weights or (1.0,) * len(ordered_selected)))
    batch, length = token_ids.shape
    if length > model.cfg.max_seq_len:
        raise ValueError("token sequence exceeds max_seq_len")
    flat_mask = masked_positions.reshape(-1).astype(mx.float32)
    n_masked = mx.maximum(mx.sum(flat_mask), mx.array(1.0))
    flat_targets = targets.reshape(-1)
    x = model.token_embed(token_ids) + model.pos_embed(mx.arange(length)[None, :])
    attention_mask = pad_mask[:, None, None, :] if pad_mask is not None else None
    total = mx.array(0.0)
    total_weight = 0.0
    for index, block in enumerate(model.blocks, start=1):
        x = block(x, attention_mask)
        if index in selected:
            logits = model._head(x)
            flat_logits = logits.reshape(-1, logits.shape[-1])
            log_probs = nn.log_softmax(flat_logits.astype(mx.float32), axis=-1)
            token_loss = -log_probs[mx.arange(flat_logits.shape[0]), flat_targets]
            weight = weights[index]
            total = total + weight * mx.sum(token_loss * flat_mask) / n_masked
            total_weight += weight
    return total / total_weight
