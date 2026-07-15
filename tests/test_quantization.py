"""Tests for quantization-aware training layers."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.utils

from src.quantization import (
    quantize_weights,
    ste_quantize,
    QuantizedLinear,
    set_model_bits,
    bits_per_param,
    theoretical_storage_bytes,
)


class TestQuantize1Bit:
    """1-bit binary {-1, +1} quantization."""

    def test_values_are_binary(self):
        """All quantised values should be ±scale."""
        w = mx.random.normal((32, 64))
        w_q = quantize_weights(w, bits=1)
        mx.eval(w_q)
        # Unique absolute values should be one value (the scale)
        scale = mx.mean(mx.abs(w), axis=-1, keepdims=True)
        # Check that w_q / scale is always ±1
        ratio = w_q / scale
        mx.eval(ratio)
        ratio_np = np.array(ratio.tolist())
        assert np.allclose(np.abs(ratio_np), 1.0, atol=1e-5), \
            f"1-bit values should be ±1 × scale, got unique ratios: {np.unique(np.abs(ratio_np))}"

    def test_no_zeros(self):
        """1-bit binary must not produce zero weights."""
        w = mx.random.normal((16, 16))
        w_q = quantize_weights(w, bits=1)
        mx.eval(w_q)
        w_np = np.array(w_q.tolist())
        assert not np.any(w_np == 0), "1-bit quantisation produced zero weights (should be binary ±)"

    def test_not_ternary(self):
        """Verify we are NOT silently using ternary {-1, 0, +1}."""
        # Make a weight tensor with a near-zero value
        w = mx.array([[0.0, 1.0, -1.0, 0.5, -0.5]])
        w_q = quantize_weights(w, bits=1)
        mx.eval(w_q)
        w_np = np.array(w_q.tolist())
        # Zero weight should map to +scale (not 0)
        assert w_np[0, 0] > 0, f"Zero weight mapped to {w_np[0,0]}, expected +scale (binary convention)"

    def test_scale_per_row(self):
        """Each output row should have its own scale."""
        w = mx.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        w_q = quantize_weights(w, bits=1)
        mx.eval(w_q)
        w_np = np.array(w_q.tolist())
        # Row 0 scale ≈ mean(|[1,2,3]|) = 2.0; row 1 scale ≈ mean(|[10,20,30]|) = 20.0
        row0_scale = np.mean(np.abs(w_np[0]))
        row1_scale = np.mean(np.abs(w_np[1]))
        assert abs(row0_scale - 2.0) < 0.1, f"Row 0 scale should be ~2.0, got {row0_scale}"
        assert abs(row1_scale - 20.0) < 1.0, f"Row 1 scale should be ~20.0, got {row1_scale}"


class TestQuantize2Bit:
    """2-bit symmetric {-1, 0, +1} × scale quantization."""

    def test_three_levels(self):
        """2-bit symmetric produces at most 3 distinct magnitudes per row."""
        w = mx.random.normal((8, 128))
        w_q = quantize_weights(w, bits=2)
        mx.eval(w_q)
        w_np = np.array(w_q.tolist())
        # Each row should have values in {-scale, 0, +scale}
        for row in w_np:
            scale = np.max(np.abs(row))
            if scale < 1e-8:
                continue
            norm = row / scale
            unique_vals = np.unique(np.round(norm, 5))
            assert len(unique_vals) <= 3, f"2-bit should have ≤3 levels, got: {unique_vals}"

    def test_bounded(self):
        """Quantised values should stay within [-max(|w|), +max(|w|)]."""
        w = mx.random.normal((4, 64))
        w_q = quantize_weights(w, bits=2)
        mx.eval(w_q, w)
        w_np = np.array(w.tolist())
        wq_np = np.array(w_q.tolist())
        orig_max = np.max(np.abs(w_np), axis=-1, keepdims=True)
        assert np.all(np.abs(wq_np) <= orig_max + 1e-6), "2-bit values out of range"


class TestQuantize4Bit:
    """4-bit symmetric 15-level quantization."""

    def test_at_most_15_levels(self):
        """4-bit quantization should produce at most 15 distinct values per row."""
        w = mx.random.normal((4, 256))
        w_q = quantize_weights(w, bits=4)
        mx.eval(w_q, w)
        w_np = np.array(w.tolist())
        wq_np = np.array(w_q.tolist())
        for i, row in enumerate(wq_np):
            scale = np.max(np.abs(row))
            if scale < 1e-8:
                continue
            norm = np.round(row / scale * 7)
            unique_vals = np.unique(norm)
            assert len(unique_vals) <= 15, \
                f"Row {i}: 4-bit should have ≤15 levels, got {len(unique_vals)}: {unique_vals}"

    def test_finer_than_2bit(self):
        """4-bit should have more levels than 2-bit for the same data."""
        mx.random.seed(0)
        w = mx.random.normal((1, 256))
        w2 = quantize_weights(w, bits=2)
        w4 = quantize_weights(w, bits=4)
        mx.eval(w2, w4)
        w2_np = np.array(w2.tolist())
        w4_np = np.array(w4.tolist())
        n_levels_2 = len(np.unique(np.round(w2_np[0], 6)))
        n_levels_4 = len(np.unique(np.round(w4_np[0], 6)))
        assert n_levels_4 >= n_levels_2, "4-bit should have >= levels as 2-bit"


class TestSTE:
    """Straight-Through Estimator gradient flow."""

    def test_ste_forward_is_quantised(self):
        """Forward pass of STE should match direct quantization."""
        w = mx.random.normal((8, 16))
        w_ste = ste_quantize(w, bits=1)
        w_q = quantize_weights(w, bits=1)
        mx.eval(w_ste, w_q)
        w_ste_np = np.array(w_ste.tolist())
        w_q_np = np.array(w_q.tolist())
        assert np.allclose(w_ste_np, w_q_np, atol=1e-6), \
            "STE forward should equal direct quantization"

    def test_ste_gradient_flows(self):
        """Gradient should flow through STE to latent weights."""
        w = mx.array([[1.0, -2.0, 0.5]])

        def loss_fn(w):
            w_ste = ste_quantize(w, bits=1)
            return (w_ste ** 2).sum()

        grad_fn = mx.grad(loss_fn)
        grads = grad_fn(w)
        mx.eval(grads)
        grads_np = np.array(grads.tolist())
        # Gradients should be non-zero (flowing through STE)
        assert not np.all(grads_np == 0), "STE should not block all gradients"

    def test_fullprec_passthrough(self):
        """bits=16 should be identity."""
        w = mx.random.normal((4, 8))
        w_out = ste_quantize(w, bits=16)
        mx.eval(w, w_out)
        assert np.allclose(
            np.array(w.tolist()),
            np.array(w_out.tolist()),
            atol=1e-6,
        ), "bits=16 should be identity"


class TestQuantizedLinear:
    """QuantizedLinear layer tests."""

    def test_forward_shape(self):
        layer = QuantizedLinear(64, 32, bias=True, bits=4)
        x = mx.random.normal((2, 8, 64))
        out = layer(x)
        mx.eval(out)
        assert out.shape == (2, 8, 32), f"Expected (2,8,32), got {out.shape}"

    def test_set_bits_changes_output(self):
        """Changing bits should change the quantised output."""
        mx.random.seed(1)
        layer = QuantizedLinear(32, 16, bias=False, bits=1)
        x = mx.random.normal((1, 4, 32))

        out_1bit = layer(x)
        layer.set_bits(4)
        out_4bit = layer(x)
        mx.eval(out_1bit, out_4bit)

        o1 = np.array(out_1bit.tolist())
        o4 = np.array(out_4bit.tolist())
        assert not np.allclose(o1, o4), "Different bits should produce different outputs"

    def test_gradient_to_latent_weights(self):
        """Gradients should reach the latent weight parameter."""
        layer = QuantizedLinear(8, 4, bias=False, bits=1)

        def loss_fn(layer, x):
            return layer(x).sum()

        x = mx.ones((1, 2, 8))
        loss_and_grad = nn.value_and_grad(layer, loss_fn)
        loss, grads = loss_and_grad(layer, x)
        mx.eval(loss, grads)

        # Check gradient exists for weight
        flat_grads = dict(mlx.utils.tree_flatten(grads))
        has_weight_grad = any("weight" in k for k in flat_grads)
        assert has_weight_grad, f"No weight gradient found in: {list(flat_grads.keys())}"

        # Gradient should not be all zeros
        weight_grads = [v for k, v in flat_grads.items() if "weight" in k]
        for wg in weight_grads:
            wg_np = np.array(wg.tolist())
            assert not np.all(wg_np == 0), "Weight gradient is all zeros"


class TestPrecisionSchedule:
    """Precision schedule selection and model-wide bits setting."""

    def test_set_model_bits(self):
        """set_model_bits should update all QuantizedLinear layers."""
        from src.model import DiffusionLM
        from src.config import ModelConfig

        cfg = ModelConfig(d_model=64, n_layers=2, n_heads=2, d_ff=128,
                          n_diffusion_steps=4, precision_schedule=[1, 1, 2, 4],
                          model_type="progressive")
        model = DiffusionLM(cfg)

        model.set_bits(1)
        assert model.get_current_bits() == 1

        model.set_bits(4)
        assert model.get_current_bits() == 4

    def test_schedule_length_matches_steps(self):
        """precision_schedule length must equal n_diffusion_steps."""
        from src.config import ModelConfig
        cfg = ModelConfig(n_diffusion_steps=8, precision_schedule=[1, 1, 1, 1, 2, 2, 4, 4])
        assert len(cfg.precision_schedule) == cfg.n_diffusion_steps


def run_all():
    """Run all tests and report results."""
    test_classes = [
        TestQuantize1Bit,
        TestQuantize2Bit,
        TestQuantize4Bit,
        TestSTE,
        TestQuantizedLinear,
        TestPrecisionSchedule,
    ]

    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(cls) if m.startswith("test_")]
        for method in methods:
            name = f"{cls.__name__}.{method}"
            try:
                getattr(instance, method)()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
                failed += 1
                errors.append((name, e))

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  {name}: {err}")
    return failed == 0


if __name__ == "__main__":
    print("Running quantization tests...")
    success = run_all()
    sys.exit(0 if success else 1)
