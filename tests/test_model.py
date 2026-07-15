"""Tests for the DiffusionLM model architecture."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from src.model import DiffusionLM, SinusoidalEmbedding, MultiHeadAttention, TransformerBlock
from src.config import ModelConfig


def make_small_cfg(**overrides):
    defaults = dict(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_seq_len=32,
        dropout=0.0,
        n_diffusion_steps=4,
        precision_schedule=[1, 1, 2, 4],
        model_type="progressive",
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


class TestSinusoidalEmbedding:
    def test_output_shape(self):
        emb = SinusoidalEmbedding(64)
        t = mx.array([0.0, 0.25, 0.5, 0.75])
        out = emb(t)
        mx.eval(out)
        assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"

    def test_different_times_differ(self):
        emb = SinusoidalEmbedding(64)
        t1 = mx.array([0.1])
        t2 = mx.array([0.9])
        o1 = emb(t1)
        o2 = emb(t2)
        mx.eval(o1, o2)
        assert not np.allclose(
            np.array(o1.tolist()),
            np.array(o2.tolist()),
            atol=1e-4,
        ), "Different times should produce different embeddings"


class TestMultiHeadAttention:
    def test_output_shape(self):
        attn = MultiHeadAttention(d_model=64, n_heads=4, bits=16)
        x = mx.random.normal((2, 10, 64))
        out = attn(x)
        mx.eval(out)
        assert out.shape == (2, 10, 64), f"Expected (2,10,64), got {out.shape}"

    def test_bidirectional(self):
        """Attention mask should be absent (bidirectional)."""
        attn = MultiHeadAttention(d_model=32, n_heads=2, bits=16)
        # With a non-uniform input, no causal masking means output at pos 0
        # should depend on all positions.  We just verify the output is finite
        # and matches the sequence length — a causal model would zero out
        # future attention scores.
        x = mx.random.normal((1, 6, 32))
        out_no_mask = attn(x, mask=None)
        # Causal-like mask where only position 0 can see itself
        causal_mask = mx.array([[[[True, False, False, False, False, False]]]])
        out_with_mask = attn(x, mask=causal_mask)
        mx.eval(out_no_mask, out_with_mask)
        out_no_mask_np = np.array(out_no_mask.tolist())
        out_with_mask_np = np.array(out_with_mask.tolist())
        # Outputs should differ when a mask is applied
        assert not np.allclose(out_no_mask_np, out_with_mask_np, atol=1e-4), \
            "Applying an attention mask should change the output"


class TestTransformerBlock:
    def test_residual_shape(self):
        block = TransformerBlock(d_model=64, n_heads=4, d_ff=128, bits=16)
        x = mx.random.normal((2, 8, 64))
        out = block(x)
        mx.eval(out)
        assert out.shape == x.shape, f"Block should preserve shape, got {out.shape}"


class TestDiffusionLM:
    def test_forward_shape(self):
        cfg = make_small_cfg()
        model = DiffusionLM(cfg)
        mask_id = cfg.mask_token_id()

        B, L = 2, 16
        x = mx.array([[1, 2, mask_id, 4, 5, mask_id, 7, 8, 9, 10, 11, 12, 13, mask_id, 15, 16]] * B)
        t = mx.array([0.5, 0.3])
        logits = model(x, t)
        mx.eval(logits)
        assert logits.shape == (B, L, cfg.vocab_size), \
            f"Expected ({B}, {L}, {cfg.vocab_size}), got {logits.shape}"

    def test_baseline_uses_full_precision(self):
        """Baseline model should have bits=16 in all linear layers."""
        cfg = make_small_cfg(model_type="baseline")
        model = DiffusionLM(cfg)
        assert model.get_current_bits() == 16, \
            f"Baseline should start at 16 bits, got {model.get_current_bits()}"

    def test_progressive_starts_at_first_schedule_bit(self):
        """Progressive model should start at the first precision_schedule value."""
        cfg = make_small_cfg(model_type="progressive", precision_schedule=[1, 1, 2, 4])
        model = DiffusionLM(cfg)
        # Initial bits = first in schedule
        assert model.get_current_bits() == 1, \
            f"Progressive should start at 1 bit, got {model.get_current_bits()}"

    def test_set_bits_changes_precision(self):
        cfg = make_small_cfg()
        model = DiffusionLM(cfg)

        model.set_bits(1)
        b1 = model.get_current_bits()
        model.set_bits(4)
        b4 = model.get_current_bits()
        assert b1 == 1 and b4 == 4

    def test_param_count_positive(self):
        cfg = make_small_cfg()
        model = DiffusionLM(cfg)
        n = model.total_params()
        assert n > 0, f"Model should have parameters, got {n}"
        print(f"  Small test model params: {n:,}")

    def test_mask_token_id_is_vocab_size(self):
        cfg = make_small_cfg(vocab_size=100)
        assert cfg.mask_token_id() == 100

    def test_gradient_flows_to_model(self):
        """A gradient pass should not error."""
        cfg = make_small_cfg()
        model = DiffusionLM(cfg)
        mask_id = cfg.mask_token_id()

        def loss_fn(model, x, t):
            logits = model(x, t)
            return logits.sum()

        x = mx.array([[1, mask_id, 3, 4, 5, 6, 7, 8]])
        t = mx.array([0.5])
        loss_and_grad = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad(model, x, t)
        mx.eval(loss, grads)
        assert float(loss) != 0.0, "Loss should be non-zero"

    def test_1bit_vs_4bit_different_output(self):
        """1-bit and 4-bit should produce measurably different logits."""
        cfg = make_small_cfg()
        model = DiffusionLM(cfg)
        x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        t = mx.array([0.5])

        model.set_bits(1)
        logits_1 = model(x, t)
        model.set_bits(4)
        logits_4 = model(x, t)
        mx.eval(logits_1, logits_4)

        l1 = np.array(logits_1.tolist())
        l4 = np.array(logits_4.tolist())
        assert not np.allclose(l1, l4, atol=1e-4), \
            "1-bit and 4-bit should produce different logits"


def run_all():
    test_classes = [
        TestSinusoidalEmbedding,
        TestMultiHeadAttention,
        TestTransformerBlock,
        TestDiffusionLM,
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
                import traceback
                print(f"  FAIL  {name}: {e}")
                traceback.print_exc()
                failed += 1
                errors.append((name, e))

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    print("Running model tests...")
    success = run_all()
    sys.exit(0 if success else 1)
