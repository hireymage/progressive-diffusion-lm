"""Focused tests for the separate layer-wise grouped-precision prototype."""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils

from src.config import LayerwiseModelConfig
from src.layerwise_model import (
    LayerwiseLinear, LayerwiseProgressiveLM, fp32_reference_cost, masked_deep_supervision_loss,
    proxy_cost_for_schedule,
)


def tiny_cfg(**overrides):
    values = dict(vocab_size=32, d_model=16, n_layers=25, n_heads=4, d_ff=32,
                  max_seq_len=12, min_exit_layer=5)
    values.update(overrides)
    return LayerwiseModelConfig(**values)


def test_default_schedule_and_proxy_accounting():
    cfg = tiny_cfg()
    assert cfg.layer_precisions == ["q1"] * 5 + ["q2"] * 5 + ["q4"] * 5 + ["q8"] * 5 + ["fp16"] * 5
    assert proxy_cost_for_schedule(cfg.layer_precisions, 8) == 11
    assert proxy_cost_for_schedule(cfg.layer_precisions) == 155
    assert fp32_reference_cost(25) == 800


def test_precisions_are_assigned_to_the_actual_blocks():
    cfg = tiny_cfg()
    model = LayerwiseProgressiveLM(cfg)
    assert [block.precision for block in model.blocks] == cfg.layer_precisions
    assert model.blocks[0].attn.q_proj.precision == "q1"
    assert model.blocks[5].ff1.precision == "q2"
    assert model.blocks[10].ff2.precision == "q4"
    assert model.blocks[15].attn.out_proj.precision == "q8"
    assert model.blocks[20].ff1.precision == "fp16"


def test_fp16_mode_uses_fp16_matmul_and_returns_finite_residual_dtype():
    linear = LayerwiseLinear(8, 4, "fp16")
    x = mx.random.normal((2, 8))
    raw_product = linear.fp16_matmul(x)
    output = linear(x)
    mx.eval(raw_product, output)
    assert raw_product.dtype == mx.float16
    assert output.dtype == mx.float32
    assert np.isfinite(np.array(output)).all()


def test_fp32_mode_is_an_actual_fp32_linear_path():
    linear = LayerwiseLinear(8, 4, "fp32")
    x = mx.random.normal((2, 8)).astype(mx.float16)
    output = linear(x)
    mx.eval(output)
    assert output.dtype == mx.float32
    np.testing.assert_allclose(np.array(output), np.array(x.astype(mx.float32) @ linear.weight.T + linear.bias), rtol=1e-5)


def test_intermediate_shared_head_and_inside_group_exit():
    cfg = tiny_cfg()
    model = LayerwiseProgressiveLM(cfg)
    tokens = mx.array([[1, 2, cfg.mask_token_id(), 4, 5, 6]])
    outputs = model.forward_intermediates(tokens, exit_layer=8)
    mx.eval(outputs)
    assert list(outputs) == [5, 6, 7, 8]
    assert all(logits.shape == (1, 6, cfg.vocab_size) for logits in outputs.values())
    assert model.proxy_cost(8) == 11


def test_requested_intermediates_do_not_materialize_other_exit_logits():
    cfg = tiny_cfg()
    model = LayerwiseProgressiveLM(cfg)
    outputs = model.forward_intermediates(mx.array([[1, 2, 3, 4, 5, 6]]), exit_layer=10,
                                         requested_layers=(5, 10))
    mx.eval(outputs)
    assert list(outputs) == [5, 10]


def test_early_exit_is_sequence_wide_and_can_stop_at_layer_eight():
    cfg = tiny_cfg()
    model = LayerwiseProgressiveLM(cfg)
    tokens = mx.array([[1, 2, 3, 4, 5, 6]])
    # Choose the observed layer-8 margin, forcing the controller to accept it
    # no later than layer 8 while still exercising an in-group candidate.
    layer_eight = model.forward_intermediates(tokens, exit_layer=8)[8]
    top_two = mx.sort(layer_eight, axis=-1)[..., -2:]
    threshold = float(mx.mean(top_two[..., 1] - top_two[..., 0]))
    result = model.early_exit(tokens, margin_threshold=threshold)
    mx.eval(result.logits)
    assert cfg.min_exit_layer <= result.exit_layer <= 8
    assert result.proxy_cost == model.proxy_cost(result.exit_layer)
    assert result.logits.shape == (1, 6, cfg.vocab_size)


def test_early_exit_does_not_execute_later_layers(monkeypatch):
    cfg = tiny_cfg()
    model = LayerwiseProgressiveLM(cfg)
    calls = []
    original_call = type(model.blocks[0]).__call__

    def counted_call(block, *args, **kwargs):
        calls.append(block.precision)
        return original_call(block, *args, **kwargs)

    monkeypatch.setattr(type(model.blocks[0]), "__call__", counted_call)
    result = model.early_exit(mx.array([[1, 2, 3, 4, 5, 6]]), margin_threshold=-1.0)
    mx.eval(result.logits)
    assert result.exit_layer == 5
    assert len(calls) == 5
    assert calls == ["q1"] * 5


def test_deep_supervision_is_masked_and_differentiable():
    cfg = tiny_cfg()
    model = LayerwiseProgressiveLM(cfg)
    targets = mx.array([[1, 2, 3, 4, 5, 6]])
    masked = mx.array([[False, True, False, True, False, True]])
    inputs = mx.where(masked, mx.full_like(targets, cfg.mask_token_id()), targets)

    loss_and_grad = nn.value_and_grad(model, masked_deep_supervision_loss)
    loss, grads = loss_and_grad(model, inputs, targets, masked)
    optimizer = optim.Adam(learning_rate=1e-3)
    optimizer.update(model, grads)
    mx.eval(loss, model.parameters())
    assert np.isfinite(float(loss)) and float(loss) > 0.0


def test_deep_supervision_reaches_early_and_late_blocks():
    cfg = tiny_cfg()
    model = LayerwiseProgressiveLM(cfg)
    targets = mx.array([[1, 2, 3, 4, 5, 6]])
    masked = mx.array([[False, True, False, True, False, True]])
    inputs = mx.where(masked, mx.full_like(targets, cfg.mask_token_id()), targets)
    loss_and_grad = nn.value_and_grad(model, masked_deep_supervision_loss)
    _, grads = loss_and_grad(model, inputs, targets, masked)
    flat = dict(mlx.utils.tree_flatten(grads))
    early = [np.array(value) for name, value in flat.items() if name.startswith("blocks.0.")]
    late = [np.array(value) for name, value in flat.items() if name.startswith("blocks.24.")]
    assert early and late
    assert any(np.any(np.abs(value) > 0) for value in early)
    assert any(np.any(np.abs(value) > 0) for value in late)
