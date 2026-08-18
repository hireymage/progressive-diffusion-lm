"""PyTorch/CUDA implementation of the MLX layer-wise flexible LM."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


PRECISION_COSTS = {"q1": 1, "q2": 2, "q4": 4, "q8": 8, "fp16": 16, "fp32": 32}


def build_layer_precision_schedule(n_layers: int, precisions: list[str]) -> list[str]:
    ordered = sorted(precisions, key=PRECISION_COSTS.__getitem__)
    base, remainder = divmod(n_layers, len(ordered))
    return [name for index, name in enumerate(ordered)
            for _ in range(base + (index < remainder))]


def route_pool(n_layers: int = 25) -> dict[str, list[str]]:
    if n_layers != 25:
        raise ValueError("the production flexible route pool requires 25 layers")
    return {
        "q8_only": ["q8"] * n_layers,
        "q8_fp16": build_layer_precision_schedule(n_layers, ["q8", "fp16"]),
        "q2_q8_fp16": build_layer_precision_schedule(n_layers, ["q2", "q8", "fp16"]),
    }


def quantize_weight(weight: torch.Tensor, bits: int) -> torch.Tensor:
    maximum = weight.abs().amax(dim=-1, keepdim=True)
    if bits == 1:
        scale = weight.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.where(weight >= 0, torch.ones_like(weight), -torch.ones_like(weight)) * scale
    level = (1 << bits) - 1
    step = (maximum / level).clamp_min(1e-8)
    normalized = weight / step
    sign = torch.where(normalized >= 0, torch.ones_like(normalized), -torch.ones_like(normalized))
    magnitude = (2.0 * torch.floor(normalized.abs() / 2.0) + 1.0).clamp_max(float(level))
    return sign * magnitude * step


def ste_quantize(weight: torch.Tensor, bits: int) -> torch.Tensor:
    quantized = quantize_weight(weight, bits)
    return weight + (quantized - weight).detach()


@dataclass
class TorchLayerwiseConfig:
    vocab_size: int
    d_model: int = 64
    d_ff: int = 256
    n_heads: int = 4
    n_layers: int = 25
    max_seq_len: int = 256
    min_exit_layer: int = 5
    dropout: float = 0.0

    def mask_token_id(self) -> int:
        return self.vocab_size


class LayerwiseLinear(nn.Module):
    Q_BITS = {"q1": 1, "q2": 2, "q4": 4, "q8": 8}

    def __init__(self, in_features: int, out_features: int, precision: str, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)
        self.precision = precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.precision == "fp16":
            output = F.linear(x.to(torch.float16), self.weight.to(torch.float16), None).float()
        elif self.precision == "fp32":
            output = F.linear(x.float(), self.weight.float(), None)
        else:
            output = F.linear(x, ste_quantize(self.weight, self.Q_BITS[self.precision]), None)
        return output + self.bias if self.bias is not None else output


class LayerwiseAttention(nn.Module):
    def __init__(self, cfg: TorchLayerwiseConfig, precision: str):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = LayerwiseLinear(cfg.d_model, cfg.d_model, precision, False)
        self.k_proj = LayerwiseLinear(cfg.d_model, cfg.d_model, precision, False)
        self.v_proj = LayerwiseLinear(cfg.d_model, cfg.d_model, precision, False)
        self.out_proj = LayerwiseLinear(cfg.d_model, cfg.d_model, precision)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, width = x.shape
        shape = (batch, length, self.n_heads, self.head_dim)
        q = self.q_proj(x).reshape(shape).transpose(1, 2)
        k = self.k_proj(x).reshape(shape).transpose(1, 2)
        v = self.v_proj(x).reshape(shape).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attention = self.dropout(torch.softmax(scores.float(), dim=-1))
        output = torch.matmul(attention, v).transpose(1, 2).reshape(batch, length, width)
        return self.out_proj(output)


class LayerwiseBlock(nn.Module):
    def __init__(self, cfg: TorchLayerwiseConfig, precision: str):
        super().__init__()
        self.attn = LayerwiseAttention(cfg, precision)
        self.ff1 = LayerwiseLinear(cfg.d_model, cfg.d_ff, precision)
        self.ff2 = LayerwiseLinear(cfg.d_ff, cfg.d_model, precision)
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.precision = precision

    def set_precision(self, precision: str) -> None:
        self.precision = precision
        for layer in (self.attn.q_proj, self.attn.k_proj, self.attn.v_proj,
                      self.attn.out_proj, self.ff1, self.ff2):
            layer.precision = precision

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))


class TorchLayerwiseProgressiveLM(nn.Module):
    def __init__(self, cfg: TorchLayerwiseConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size + 1, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([LayerwiseBlock(cfg, "q8") for _ in range(cfg.n_layers)])
        self.ln_out = nn.LayerNorm(cfg.d_model)
        self.lm_head_bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        self.layer_precisions = ["q8"] * cfg.n_layers

    def set_layer_precisions(self, schedule: list[str]) -> None:
        if len(schedule) != self.cfg.n_layers or set(schedule) - set(PRECISION_COSTS):
            raise ValueError("invalid layer precision schedule")
        for block, precision in zip(self.blocks, schedule):
            block.set_precision(precision)
        self.layer_precisions = list(schedule)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.ln_out(hidden)
        return F.linear(hidden, self.token_embed.weight[:self.cfg.vocab_size], self.lm_head_bias)

    def forward_intermediates(self, token_ids: torch.Tensor, requested_layers: tuple[int, ...],
                              pad_mask: torch.Tensor | None = None) -> dict[int, torch.Tensor]:
        _, length = token_ids.shape
        positions = torch.arange(length, device=token_ids.device).unsqueeze(0)
        x = self.token_embed(token_ids) + self.pos_embed(positions)
        attention_mask = pad_mask[:, None, None, :] if pad_mask is not None else None
        requested = set(requested_layers)
        outputs = {}
        for index, block in enumerate(self.blocks, 1):
            x = block(x, attention_mask)
            if index in requested:
                outputs[index] = self.head(x)
            if index == max(requested):
                break
        return outputs

    def forward(self, token_ids: torch.Tensor, exit_layer: int | None = None,
                pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        layer = exit_layer or self.cfg.n_layers
        return self.forward_intermediates(token_ids, (layer,), pad_mask)[layer]


def masked_deep_supervision_loss(model: TorchLayerwiseProgressiveLM, token_ids: torch.Tensor,
                                 targets: torch.Tensor, mask: torch.Tensor,
                                 milestones=((5, .1), (10, .2), (15, .3), (20, .4), (25, 1.0))):
    layers = tuple(layer for layer, _ in milestones)
    logits = model.forward_intermediates(token_ids, layers)
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1).float()
    denominator = flat_mask.sum().clamp_min(1.0)
    total = torch.zeros((), device=token_ids.device)
    for layer, weight in milestones:
        token_loss = F.cross_entropy(logits[layer].reshape(-1, model.cfg.vocab_size),
                                     flat_targets, reduction="none")
        total = total + weight * (token_loss * flat_mask).sum() / denominator
    return total / sum(weight for _, weight in milestones)
