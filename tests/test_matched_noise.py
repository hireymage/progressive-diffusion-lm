"""P1 matched-noise controls, developed one RED→GREEN behavior at a time."""
import json
from pathlib import Path

import numpy as np
import pytest
import mlx
import mlx.core as mx
import mlx.nn as nn

from src.config import ExperimentConfig, ModelConfig
from src.quantization import QuantizedLinear, quantize_weights, quantization_noise_metrics


def _np(value):
    mx.eval(value)
    return np.array(value.tolist())


def test_noise_config_defaults_are_behavior_preserving():
    cfg = ModelConfig()
    assert cfg.weight_noise_mode == "none"
    assert cfg.weight_noise_multiplier == 1.0


def test_noise_config_rejects_invalid_mode():
    with pytest.raises(ValueError, match="weight_noise_mode"):
        ModelConfig(weight_noise_mode="laplace")


def test_noise_config_rejects_negative_multiplier():
    with pytest.raises(ValueError, match="weight_noise_multiplier"):
        ModelConfig(weight_noise_multiplier=-0.01)


def test_none_mode_is_exact_forward_identity():
    layer = QuantizedLinear(4, 3, bias=False, bits=16, weight_noise_mode="none")
    x = mx.array([[1.0, -2.0, 0.5, 3.0]])
    expected = x @ layer.weight.T
    actual = layer(x)
    assert np.array_equal(_np(actual), _np(expected))


def test_matched_noise_is_training_only_and_eval_is_clean_fp32():
    layer = QuantizedLinear(
        256, 4, bias=False, bits=16, weight_noise_mode="gaussian_matched"
    )
    x = mx.ones((1, 256))
    clean = x @ layer.weight.T
    layer.train()
    noisy = layer(x)
    layer.eval()
    evaluated = layer(x)
    assert not np.array_equal(_np(noisy), _np(clean))
    assert np.array_equal(_np(evaluated), _np(clean))


def test_q1_qat_forward_is_unchanged_when_noise_mode_is_configured():
    layer = QuantizedLinear(
        32, 8, bias=False, bits=1, weight_noise_mode="gaussian_matched"
    )
    layer.train()
    x = mx.random.normal((2, 32))
    expected = x @ quantize_weights(layer.weight, 1).T
    assert np.array_equal(_np(layer(x)), _np(expected))


def test_gaussian_noise_matches_q1_residual_rms_per_row():
    layer = QuantizedLinear(
        65536, 4, bias=False, bits=16, weight_noise_mode="gaussian_matched"
    )
    layer.set_noise_seed(123)
    residual = quantize_weights(layer.weight, 1) - layer.weight
    target = mx.sqrt(mx.mean(residual * residual, axis=-1))
    actual = mx.sqrt(mx.mean(layer._matched_noise() ** 2, axis=-1))
    np.testing.assert_allclose(_np(actual), _np(target), rtol=0.02, atol=1e-7)


def test_model_propagates_noise_configuration_to_quantized_linears():
    from src.model import DiffusionLM

    cfg = ModelConfig(
        vocab_size=32, d_model=16, n_layers=1, n_heads=2, d_ff=32,
        max_seq_len=8, n_diffusion_steps=2, precision_schedule=[16, 16],
        model_type="baseline", weight_noise_mode="uniform_matched",
        weight_noise_multiplier=0.5,
    )
    model = DiffusionLM(cfg)
    layers = [m for _, m in model.named_modules() if isinstance(m, QuantizedLinear)]
    assert layers
    assert all(m.weight_noise_mode == "uniform_matched" for m in layers)
    assert all(m.weight_noise_multiplier == 0.5 for m in layers)


def test_uniform_noise_matches_q1_residual_rms_per_row():
    layer = QuantizedLinear(
        65536, 4, bias=False, bits=16, weight_noise_mode="uniform_matched"
    )
    layer.set_noise_seed(456)
    residual = quantize_weights(layer.weight, 1) - layer.weight
    target = mx.sqrt(mx.mean(residual * residual, axis=-1))
    actual = mx.sqrt(mx.mean(layer._matched_noise() ** 2, axis=-1))
    np.testing.assert_allclose(_np(actual), _np(target), rtol=0.01, atol=1e-7)


def test_matched_noise_has_finite_nonzero_weight_gradients():
    layer = QuantizedLinear(
        64, 8, bias=False, bits=16, weight_noise_mode="gaussian_matched"
    )
    layer.train()

    def loss_fn(module, x):
        return mx.sum(module(x) ** 2)

    loss, grads = nn.value_and_grad(layer, loss_fn)(layer, mx.ones((2, 64)))
    mx.eval(loss, grads)
    weight_grad = dict(mlx.utils.tree_flatten(grads))["weight"]
    grad = _np(weight_grad)
    assert np.isfinite(grad).all()
    assert np.any(grad != 0)


def test_quantization_noise_metrics_report_actual_injected_rms():
    layer = QuantizedLinear(
        4096, 4, bias=False, bits=16, weight_noise_mode="uniform_matched"
    )
    layer.train()
    layer(mx.ones((1, 4096)))
    metrics = quantization_noise_metrics(layer)
    assert metrics["q1_residual_rms"] > 0
    assert metrics["injected_noise_rms"] > 0
    assert metrics["injected_noise_rms"] == pytest.approx(
        metrics["q1_residual_rms"], rel=0.03
    )


def test_model_noise_seed_is_reproducible_and_independent_of_global_rng():
    from src.model import DiffusionLM

    cfg = ModelConfig(
        vocab_size=32, d_model=16, n_layers=1, n_heads=2, d_ff=32,
        max_seq_len=8, dropout=0.0, n_diffusion_steps=2,
        precision_schedule=[16, 16], model_type="baseline",
        weight_noise_mode="gaussian_matched",
    )
    model = DiffusionLM(cfg)
    model.train()
    x = mx.array([[1, 2, 3, 4]])
    t = mx.array([0.5])
    model.set_weight_noise_seed(77)
    first = model(x, t)
    mx.random.normal((1000,))
    model.set_weight_noise_seed(77)
    second = model(x, t)
    assert np.array_equal(_np(first), _np(second))
