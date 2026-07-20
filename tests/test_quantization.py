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
    model_storage_report,
    bits_per_param_from_schedule,
    EFFECTIVE_BITS,
)


class TestQuantize1Bit:
    """1-bit binary {-1, +1} quantization."""

    def test_values_are_binary(self):
        """All quantised values should be ±scale."""
        w = mx.random.normal((32, 64))
        w_q = quantize_weights(w, bits=1)
        mx.eval(w_q)
        scale = mx.mean(mx.abs(w), axis=-1, keepdims=True)
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
        """Verify zero input maps to +scale (not 0) — binary convention."""
        w = mx.array([[0.0, 1.0, -1.0, 0.5, -0.5]])
        w_q = quantize_weights(w, bits=1)
        mx.eval(w_q)
        w_np = np.array(w_q.tolist())
        assert w_np[0, 0] > 0, f"Zero weight mapped to {w_np[0,0]}, expected +scale (binary convention)"

    def test_scale_per_row(self):
        """Each output row should have its own scale = mean(|w|)."""
        w = mx.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        w_q = quantize_weights(w, bits=1)
        mx.eval(w_q)
        w_np = np.array(w_q.tolist())
        row0_scale = np.mean(np.abs(w_np[0]))
        row1_scale = np.mean(np.abs(w_np[1]))
        assert abs(row0_scale - 2.0) < 0.1, f"Row 0 scale should be ~2.0, got {row0_scale}"
        assert abs(row1_scale - 20.0) < 1.0, f"Row 1 scale should be ~20.0, got {row1_scale}"

    def test_exactly_two_distinct_values_per_row(self):
        """Each row must contain exactly 2 distinct values: +scale and -scale."""
        mx.random.seed(42)
        w = mx.random.normal((8, 64))
        w_q = quantize_weights(w, bits=1)
        mx.eval(w_q)
        w_np = np.array(w_q.tolist())
        for i, row in enumerate(w_np):
            vals = np.unique(np.round(row, 8))
            assert len(vals) == 2, f"Row {i}: expected 2 distinct values, got {len(vals)}: {vals}"
            assert vals[0] < 0 and vals[1] > 0, f"Row {i}: expected ±scale, got {vals}"


class TestQuantize2BitTrue:
    """True 2-bit quantization: exactly 4 levels {-3,-1,+1,+3}×step."""

    def test_four_levels(self):
        """bits=2 can produce up to 4 distinct levels per row; verify all 4 are reachable."""
        # Construct input guaranteed to span all 4 regions:
        # small negative, large negative, small positive, large positive
        w = mx.array([[-0.5, -3.5, 0.5, 3.5]])  # step = 3.5/3 ≈ 1.17; norms ≈ -0.43,-3,0.43,3
        w_q = quantize_weights(w, bits=2)
        mx.eval(w_q, w)
        w_np = np.array(w.tolist())
        wq_np = np.array(w_q.tolist())
        step = np.max(np.abs(w_np)) / 3.0
        norm = np.round(wq_np / step, 3)
        unique_vals = np.unique(norm)
        assert len(unique_vals) == 4, \
            f"Expected exactly 4 levels, got {len(unique_vals)}: {unique_vals}"
        assert set(unique_vals.tolist()) == {-3.0, -1.0, 1.0, 3.0}, \
            f"Levels should be exactly {{-3,-1,+1,+3}}, got {unique_vals}"

    def test_no_zero_level(self):
        """True 2-bit has no zero level — all 4 values are nonzero."""
        mx.random.seed(13)
        w = mx.random.normal((8, 128))
        w_q = quantize_weights(w, bits=2)
        mx.eval(w_q)
        w_np = np.array(w_q.tolist())
        assert not np.any(np.abs(w_np) < 1e-7), \
            "bits=2 (true 2-bit) must not produce zero-valued weights"

    def test_levels_are_pm1_pm3_times_step(self):
        """Quantised values should be in {-3,-1,+1,+3}×step."""
        w = mx.array([[5.0, -3.0, 1.0, -1.0, 4.5, -4.5, 0.1, -0.1]])
        w_q = quantize_weights(w, bits=2)
        mx.eval(w_q, w)
        w_np = np.array(w.tolist())
        wq_np = np.array(w_q.tolist())
        step = np.max(np.abs(w_np)) / 3.0
        norm = np.round(wq_np / step, 4)
        allowed = {-3.0, -1.0, 1.0, 3.0}
        for v in norm.flatten():
            assert v in allowed, f"Value {v} not in allowed set {allowed}"

    def test_not_ternary(self):
        """bits=2 must produce 4 levels, not 3 (distinguish from ternary)."""
        mx.random.seed(99)
        w = mx.random.normal((4, 512))
        w_q = quantize_weights(w, bits=2)
        mx.eval(w_q)
        w_np = np.array(w_q.tolist())
        all_levels = set()
        for row in w_np:
            step = np.max(np.abs(row)) / 3.0
            if step < 1e-7:
                continue
            norm = set(np.unique(np.round(row / step, 3)).tolist())
            all_levels |= norm
        # Should see all 4 distinct normalized levels across the batch
        assert len(all_levels) == 4, \
            f"bits=2 should produce 4 levels globally, got {len(all_levels)}: {sorted(all_levels)}"
        assert 0.0 not in {round(v, 2) for v in all_levels}, \
            f"Zero found in bits=2 levels — should not exist"

    def test_effective_bits(self):
        assert EFFECTIVE_BITS[2] == 2.0, "bits=2 effective bits should be 2.0"


class TestQuantize3BitTrue:
    """True 3-bit quantization: 8 non-zero odd levels from -7 to +7."""

    def test_eight_levels(self):
        """bits=3 exposes all eight Q3 levels when each region is sampled."""
        w = mx.array([[-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0]])
        w_q = quantize_weights(w, bits=3)
        mx.eval(w_q)
        w_np = np.array(w_q.tolist())
        assert set(np.unique(np.round(w_np[0], 6)).tolist()) == {
            -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0,
        }

    def test_has_no_zero_level(self):
        """True Q3 uses eight non-zero levels."""
        mx.random.seed(3)
        w = mx.random.normal((8, 256))
        w_q = quantize_weights(w, bits=3)
        mx.eval(w_q)
        w_np = np.array(w_q.tolist())
        assert not np.any(np.abs(w_np) < 1e-7), "bits=3 (true Q3) must not produce zero weights"

    def test_effective_bits(self):
        assert EFFECTIVE_BITS[3] == 3.0

    def test_distinct_from_2bit(self):
        """bits=3 (8 levels) must differ from bits=2 (4 levels)."""
        mx.random.seed(21)
        w = mx.random.normal((4, 256))
        w2 = quantize_weights(w, bits=2)
        w3 = quantize_weights(w, bits=3)
        mx.eval(w2, w3)
        w2_np = np.array(w2.tolist())
        w3_np = np.array(w3.tolist())
        assert not np.allclose(w2_np, w3_np, atol=1e-5), \
            "bits=2 (true Q2) and bits=3 (true Q3) should produce different results"

    def test_bounded(self):
        """Quantised values should stay within [-max(|w|), +max(|w|)]."""
        w = mx.random.normal((4, 64))
        w_q = quantize_weights(w, bits=3)
        mx.eval(w_q, w)
        w_np = np.array(w.tolist())
        wq_np = np.array(w_q.tolist())
        orig_max = np.max(np.abs(w_np), axis=-1, keepdims=True)
        assert np.all(np.abs(wq_np) <= orig_max + 1e-6), "Q3 values out of range"


class TestQuantizeTernary:
    """Optional ternary quantization: bits=0, 3 levels {-1,0,+1}×scale."""

    def test_three_levels(self):
        w = mx.array([[-1.0, -0.1, 0.0, 0.1, 1.0]])
        w_q = quantize_weights(w, bits=0)
        mx.eval(w_q)
        assert set(np.unique(np.array(w_q.tolist())).tolist()) == {-1.0, 0.0, 1.0}

    def test_has_zero_level(self):
        w = mx.array([[-1.0, 0.0, 1.0]])
        w_q = quantize_weights(w, bits=0)
        mx.eval(w_q)
        assert np.any(np.abs(np.array(w_q.tolist())) < 1e-7)

    def test_effective_bits_is_log2_3(self):
        import math
        assert abs(EFFECTIVE_BITS[0] - math.log2(3)) < 1e-6

    def test_bounded(self):
        w = mx.random.normal((4, 64))
        w_q = quantize_weights(w, bits=0)
        mx.eval(w_q, w)
        w_np = np.array(w.tolist())
        wq_np = np.array(w_q.tolist())
        orig_max = np.max(np.abs(w_np), axis=-1, keepdims=True)
        assert np.all(np.abs(wq_np) <= orig_max + 1e-6)


class TestQuantize4Bit:
    """True 4-bit quantization: 16 non-zero odd levels from -15 to +15."""

    def test_sixteen_levels(self):
        """bits=4 exposes all 16 Q4 levels when each region is sampled."""
        w = mx.array([[
            -15.0, -13.0, -11.0, -9.0, -7.0, -5.0, -3.0, -1.0,
            1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0,
        ]])
        w_q = quantize_weights(w, bits=4)
        mx.eval(w_q)
        wq_np = np.array(w_q.tolist())
        assert len(np.unique(np.round(wq_np[0], 6))) == 16

    def test_more_levels_than_2bit(self):
        """4-bit should produce more distinct values than true 2-bit."""
        mx.random.seed(0)
        w = mx.random.normal((1, 512))
        w2 = quantize_weights(w, bits=2)
        w4 = quantize_weights(w, bits=4)
        mx.eval(w2, w4)
        w2_np = np.array(w2.tolist())
        w4_np = np.array(w4.tolist())
        n_levels_2 = len(np.unique(np.round(w2_np[0], 6)))
        n_levels_4 = len(np.unique(np.round(w4_np[0], 6)))
        assert n_levels_4 > n_levels_2, \
            f"4-bit ({n_levels_4} levels) should exceed 2-bit ({n_levels_2} levels)"

    def test_effective_bits(self):
        assert EFFECTIVE_BITS[4] == 4.0, "bits=4 effective bits should be 4.0"

    def test_has_no_zero_intentionally(self):
        """True Q4 has no zero code, including for exact-zero input."""
        w = mx.array([[0.0, 0.001, -0.001, 10.0, -10.0]])
        w_q = quantize_weights(w, bits=4)
        mx.eval(w_q)
        wq_np = np.array(w_q.tolist())
        assert not np.any(np.abs(wq_np) < 1e-7)


class TestQuantize8Bit:
    """True 8-bit quantization: 256 non-zero odd levels from -255 to +255."""

    def test_256_levels(self):
        """bits=8 exposes all 256 Q8 levels when each region is sampled."""
        # Construct weights that span all 256 levels: -255, -253, ..., 253, 255
        levels = [float(v) for v in range(-255, 256, 2)]  # 256 values
        w = mx.array([levels])
        w_q = quantize_weights(w, bits=8)
        mx.eval(w_q)
        wq_np = np.array(w_q.tolist())
        assert len(np.unique(np.round(wq_np[0], 4))) == 256, \
            f"Expected 256 distinct levels, got {len(np.unique(np.round(wq_np[0], 4)))}"

    def test_more_levels_than_4bit(self):
        """8-bit should produce more distinct values than 4-bit."""
        mx.random.seed(42)
        w = mx.random.normal((4, 512))
        w4 = quantize_weights(w, bits=4)
        w8 = quantize_weights(w, bits=8)
        mx.eval(w4, w8)
        w4_np = np.array(w4.tolist())
        w8_np = np.array(w8.tolist())
        n_levels_4 = len(np.unique(np.round(w4_np.flatten(), 6)))
        n_levels_8 = len(np.unique(np.round(w8_np.flatten(), 6)))
        assert n_levels_8 > n_levels_4, \
            f"8-bit ({n_levels_8} levels) should exceed 4-bit ({n_levels_4} levels)"

    def test_no_zero_level(self):
        """True Q8 has no zero level — all 256 values are nonzero."""
        mx.random.seed(7)
        w = mx.random.normal((8, 256))
        w_q = quantize_weights(w, bits=8)
        mx.eval(w_q)
        w_np = np.array(w_q.tolist())
        assert not np.any(np.abs(w_np) < 1e-7), \
            "bits=8 (true Q8) must not produce zero weights"

    def test_levels_are_odd_multiples_of_step(self):
        """Quantised values should be in {-255,-253,…,+253,+255}×step."""
        w = mx.array([[255.0, -255.0, 127.0, -127.0, 1.0, -1.0, 0.5, -0.5]])
        w_q = quantize_weights(w, bits=8)
        mx.eval(w_q, w)
        w_np = np.array(w.tolist())
        wq_np = np.array(w_q.tolist())
        step = np.max(np.abs(w_np)) / 255.0
        norm = np.round(wq_np / step, 4)
        allowed = {float(v) for v in range(-255, 256, 2)}
        for v in norm.flatten():
            assert v in allowed, f"Value {v} not in allowed 256 odd levels"

    def test_effective_bits(self):
        assert EFFECTIVE_BITS[8] == 8.0, "bits=8 effective bits should be 8.0"

    def test_bounded(self):
        """Quantised values should stay within [-max(|w|), +max(|w|)]."""
        mx.random.seed(11)
        w = mx.random.normal((4, 64))
        w_q = quantize_weights(w, bits=8)
        mx.eval(w_q, w)
        w_np = np.array(w.tolist())
        wq_np = np.array(w_q.tolist())
        orig_max = np.max(np.abs(w_np), axis=-1, keepdims=True)
        assert np.all(np.abs(wq_np) <= orig_max + 1e-5), "Q8 values out of range"

    def test_ste_forward_matches(self):
        """STE forward for bits=8 matches direct quantization."""
        mx.random.seed(23)
        w = mx.random.normal((4, 64))
        w_ste = ste_quantize(w, bits=8)
        w_q = quantize_weights(w, bits=8)
        mx.eval(w_ste, w_q)
        assert np.allclose(
            np.array(w_ste.tolist()), np.array(w_q.tolist()), atol=1e-5
        ), "STE forward should equal direct quantization for bits=8"

    def test_close_to_original(self):
        """8-bit quantization should be closer to original than 4-bit."""
        mx.random.seed(77)
        w = mx.random.normal((8, 128))
        w4 = quantize_weights(w, bits=4)
        w8 = quantize_weights(w, bits=8)
        mx.eval(w, w4, w8)
        w_np = np.array(w.tolist())
        err4 = np.mean((w_np - np.array(w4.tolist()))**2)
        err8 = np.mean((w_np - np.array(w8.tolist()))**2)
        assert err8 < err4, \
            f"8-bit MSE ({err8:.6f}) should be less than 4-bit MSE ({err4:.6f})"

    def test_bits_per_param_schedule(self):
        """bits_per_param_from_schedule with 8-bit values."""
        schedule = [1, 2, 4, 8]
        avg = bits_per_param_from_schedule(schedule)
        expected = (1.0 + 2.0 + 4.0 + 8.0) / 4
        assert abs(avg - expected) < 1e-6, f"Expected {expected}, got {avg}"


class TestQuantize16Bit:
    """bits=16 identity pass-through."""

    def test_identity(self):
        w = mx.random.normal((8, 32))
        w_out = quantize_weights(w, bits=16)
        mx.eval(w, w_out)
        assert np.allclose(
            np.array(w.tolist()), np.array(w_out.tolist()), atol=1e-6
        ), "bits=16 should be identity"

    def test_effective_bits(self):
        assert EFFECTIVE_BITS[16] == 32.0


class TestSTE:
    """Straight-Through Estimator gradient flow."""

    def test_ste_forward_is_quantised(self):
        """Forward pass of STE should match direct quantization."""
        w = mx.random.normal((8, 16))
        w_ste = ste_quantize(w, bits=1)
        w_q = quantize_weights(w, bits=1)
        mx.eval(w_ste, w_q)
        assert np.allclose(
            np.array(w_ste.tolist()), np.array(w_q.tolist()), atol=1e-6
        ), "STE forward should equal direct quantization"

    def test_ste_gradient_flows(self):
        """Gradient should flow through STE to latent weights."""
        w = mx.array([[1.0, -2.0, 0.5]])

        def loss_fn(w):
            return (ste_quantize(w, bits=1) ** 2).sum()

        grads = mx.grad(loss_fn)(w)
        mx.eval(grads)
        grads_np = np.array(grads.tolist())
        assert not np.all(grads_np == 0), "STE should not block all gradients"

    def test_fullprec_passthrough(self):
        """bits=16 should be identity."""
        w = mx.random.normal((4, 8))
        w_out = ste_quantize(w, bits=16)
        mx.eval(w, w_out)
        assert np.allclose(
            np.array(w.tolist()), np.array(w_out.tolist()), atol=1e-6
        ), "bits=16 STE should be identity"

    def test_ste_2bit(self):
        """STE forward for bits=2 matches quantize_weights bits=2."""
        w = mx.random.normal((4, 32))
        w_ste = ste_quantize(w, bits=2)
        w_q = quantize_weights(w, bits=2)
        mx.eval(w_ste, w_q)
        assert np.allclose(
            np.array(w_ste.tolist()), np.array(w_q.tolist()), atol=1e-6
        )

    def test_ste_3bit(self):
        """STE forward for true Q3 matches direct bits=3 quantization."""
        w = mx.random.normal((4, 32))
        w_ste = ste_quantize(w, bits=3)
        w_q = quantize_weights(w, bits=3)
        mx.eval(w_ste, w_q)
        assert np.allclose(
            np.array(w_ste.tolist()), np.array(w_q.tolist()), atol=1e-6
        )

    def test_ste_ternary(self):
        """STE forward for optional ternary matches direct bits=0 quantization."""
        w = mx.random.normal((4, 32))
        w_ste = ste_quantize(w, bits=0)
        w_q = quantize_weights(w, bits=0)
        mx.eval(w_ste, w_q)
        assert np.allclose(
            np.array(w_ste.tolist()), np.array(w_q.tolist()), atol=1e-6
        )


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

        assert not np.allclose(
            np.array(out_1bit.tolist()), np.array(out_4bit.tolist())
        ), "Different bits should produce different outputs"

    def test_gradient_to_latent_weights(self):
        """Gradients should reach the latent weight parameter."""
        layer = QuantizedLinear(8, 4, bias=False, bits=1)

        def loss_fn(layer, x):
            return layer(x).sum()

        x = mx.ones((1, 2, 8))
        loss_and_grad = nn.value_and_grad(layer, loss_fn)
        loss, grads = loss_and_grad(layer, x)
        mx.eval(loss, grads)

        flat_grads = dict(mlx.utils.tree_flatten(grads))
        has_weight_grad = any("weight" in k for k in flat_grads)
        assert has_weight_grad, f"No weight gradient found in: {list(flat_grads.keys())}"

        for k, v in flat_grads.items():
            if "weight" in k:
                assert not np.all(np.array(v.tolist()) == 0), "Weight gradient is all zeros"

    def test_no_bias_option(self):
        layer = QuantizedLinear(16, 8, bias=False, bits=2)
        assert layer.bias is None

    def test_effective_bits_property(self):
        layer = QuantizedLinear(8, 4, bits=3)
        assert layer.effective_bits() == 3.0

        import math
        layer.set_bits(0)
        assert abs(layer.effective_bits() - math.log2(3)) < 1e-6


class TestPrecisionSchedule:
    """Precision schedule selection and model-wide bits setting."""

    def test_set_model_bits(self):
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
        from src.config import ModelConfig
        cfg = ModelConfig(n_diffusion_steps=8, precision_schedule=[1, 1, 1, 1, 2, 2, 4, 4])
        assert len(cfg.precision_schedule) == cfg.n_diffusion_steps

    def test_bits_per_param_from_schedule(self):
        import math
        schedule = [1, 1, 2, 4]
        avg = bits_per_param_from_schedule(schedule)
        # 1-bit: 1.0, 1-bit: 1.0, 2-bit: 2.0, 4-bit: 4.0 → avg = 2.0
        expected = (1.0 + 1.0 + 2.0 + 4.0) / 4
        assert abs(avg - expected) < 1e-6, f"Expected {expected}, got {avg}"

    def test_bits_per_param_ternary(self):
        import math
        schedule = [1, 1, 0, 4]
        avg = bits_per_param_from_schedule(schedule)
        expected = (1.0 + 1.0 + math.log2(3) + 4.0) / 4
        assert abs(avg - expected) < 1e-6

    def test_bits_per_param_true_q3(self):
        schedule = [1, 1, 3, 4]
        avg = bits_per_param_from_schedule(schedule)
        assert abs(avg - 2.25) < 1e-6


class TestStorageReport:
    """model_storage_report correctness."""

    def _make_model(self, tie=True):
        from src.model import DiffusionLM
        from src.config import ModelConfig
        cfg = ModelConfig(
            d_model=64, n_layers=2, n_heads=2, d_ff=128,
            vocab_size=200, max_seq_len=32,
            n_diffusion_steps=4, precision_schedule=[1, 2, 2, 4],
            model_type="progressive", tie_word_embeddings=tie,
        )
        return DiffusionLM(cfg), cfg

    def test_report_keys(self):
        model, cfg = self._make_model()
        report = model_storage_report(model, cfg.precision_schedule)
        required = [
            "total_params", "q_linear_weight_params", "non_quantized_params",
            "fp32_total_mb", "bf16_total_mb", "theoretical_q_mb",
            "effective_avg_bits", "compression_vs_fp32", "compression_vs_bf16",
            "training_memory_estimate_mb",
        ]
        for key in required:
            assert key in report, f"Missing key in storage report: {key}"

    def test_params_add_up(self):
        model, cfg = self._make_model()
        report = model_storage_report(model, cfg.precision_schedule)
        assert report["q_linear_weight_params"] + report["non_quantized_params"] == report["total_params"]

    def test_compression_with_tied_vs_untied(self):
        """Tied model should have fewer total params (no LM head weight matrix)."""
        model_tied, cfg = self._make_model(tie=True)
        model_untied, _ = self._make_model(tie=False)
        r_tied = model_storage_report(model_tied, cfg.precision_schedule)
        r_untied = model_storage_report(model_untied, cfg.precision_schedule)
        assert r_tied["total_params"] < r_untied["total_params"], \
            "Tied model should have fewer total params"

    def test_fp32_bytes_formula(self):
        model, cfg = self._make_model()
        report = model_storage_report(model, cfg.precision_schedule)
        expected_mb = report["total_params"] * 4 / 1e6
        assert abs(report["fp32_total_mb"] - expected_mb) < 0.001

    def test_fp32_identity_uses_32_storage_bits(self):
        assert EFFECTIVE_BITS[16] == 32.0
        assert bits_per_param_from_schedule([16, 16, 16, 16]) == 32.0

    def test_fp32_identity_report_has_no_theoretical_compression(self):
        model, _ = self._make_model()
        report = model_storage_report(model, [16, 16, 16, 16])
        assert report["actual_model_bytes"] == report["fp32_total_bytes"]
        assert report["actual_compression_vs_fp32"] == 1.0

    def test_progressive_schedule_is_temporal_not_model_storage(self):
        model, _ = self._make_model()
        report = model_storage_report(model, [1, 1, 2, 4])
        assert report["actual_model_bytes"] == report["fp32_total_bytes"]
        assert report["actual_compression_vs_fp32"] == 1.0
        assert report["average_step_weight_bits"] == 2.0
        assert report["hypothetical_packed_bytes"] is None

    def test_constant_low_bit_schedule_can_report_hypothetical_packed_lower_bound(self):
        model, _ = self._make_model()
        report = model_storage_report(model, [1, 1, 1, 1])
        assert report["hypothetical_packed_bytes"] < report["fp32_total_bytes"]
        assert report["hypothetical_packed_compression_vs_fp32"] > 1.0


def run_all():
    """Run all tests and report results."""
    test_classes = [
        TestQuantize1Bit,
        TestQuantize2BitTrue,
        TestQuantize3BitTrue,
        TestQuantizeTernary,
        TestQuantize4Bit,
        TestQuantize8Bit,
        TestQuantize16Bit,
        TestSTE,
        TestQuantizedLinear,
        TestPrecisionSchedule,
        TestStorageReport,
    ]

    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = sorted(m for m in dir(cls) if m.startswith("test_"))
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
