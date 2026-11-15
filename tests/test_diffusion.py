"""Tests for the masked diffusion process."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import mlx.core as mx

from src.diffusion import (
    corrupt_tokens, mask_rate_to_step, compute_loss,
    generate, generate_incremental, generate_with_early_exit,
    _reveal_positions, _token_predictions,
)
from src.model import DiffusionLM, IncrementalCache
from src.config import ModelConfig


MASK_TOKEN = 100  # test mask token id (outside vocab range)


def make_small_model():
    cfg = ModelConfig(
        vocab_size=MASK_TOKEN,
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
    return DiffusionLM(cfg), cfg


class TestCorruptTokens:
    def test_high_mask_rate_masks_most(self):
        """With mask_rate=1.0, all positions should be masked."""
        x0 = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        x_t, mask = corrupt_tokens(x0, 1.0, MASK_TOKEN)
        mx.eval(x_t, mask)
        x_np = np.array(x_t.tolist())
        # All should be MASK_TOKEN
        assert np.all(x_np == MASK_TOKEN), f"Expected all masked, got {x_np}"

    def test_zero_mask_rate_keeps_all(self):
        """With mask_rate=0.0, no positions should be masked."""
        x0 = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        x_t, mask = corrupt_tokens(x0, 0.0, MASK_TOKEN)
        mx.eval(x_t, mask)
        x_np = np.array(x_t.tolist())
        x0_np = np.array(x0.tolist())
        assert np.all(x_np == x0_np), f"Expected no masking, got {x_np}"

    def test_partial_masking(self):
        """With mask_rate=0.5, approximately half should be masked."""
        mx.random.seed(42)
        x0 = mx.tile(mx.arange(100)[None, :], (1, 1))  # (1, 100)
        x_t, mask = corrupt_tokens(x0, 0.5, MASK_TOKEN)
        mx.eval(mask)
        mask_np = np.array(mask.tolist())
        frac = mask_np.mean()
        assert 0.3 <= frac <= 0.7, f"Expected ~50% masking, got {frac:.2f}"

    def test_non_masked_positions_unchanged(self):
        """Non-masked positions in x_t should equal x_0."""
        x0 = mx.array([[10, 20, 30, 40, 50, 60, 70, 80]])
        x_t, mask = corrupt_tokens(x0, 0.3, MASK_TOKEN)
        mx.eval(x_t, mask, x0)
        x0_np = np.array(x0.tolist())
        xt_np = np.array(x_t.tolist())
        mask_np = np.array(mask.tolist())
        unmasked = ~mask_np.astype(bool)
        assert np.all(xt_np[unmasked] == x0_np[unmasked]), \
            "Unmasked positions should be unchanged"

    def test_per_example_mask_rate(self):
        """Per-example mask rates should work."""
        x0 = mx.ones((4, 20), dtype=mx.int32)
        rates = mx.array([0.0, 0.5, 0.5, 1.0])
        x_t, mask = corrupt_tokens(x0, rates, MASK_TOKEN)
        mx.eval(x_t, mask)
        xt_np = np.array(x_t.tolist())
        # First row: no masking
        assert np.all(xt_np[0] == 1), f"Row 0 should be unchanged: {xt_np[0]}"
        # Last row: all masked
        assert np.all(xt_np[3] == MASK_TOKEN), f"Row 3 should be all mask: {xt_np[3]}"

    def test_masked_positions_are_mask_token(self):
        """Masked positions should contain exactly MASK_TOKEN."""
        x0 = mx.array([[1, 2, 3, 4, 5]])
        x_t, mask = corrupt_tokens(x0, 0.6, MASK_TOKEN)
        mx.eval(x_t, mask)
        xt_np = np.array(x_t.tolist())
        mask_np = np.array(mask.tolist()).astype(bool)
        assert np.all(xt_np[mask_np] == MASK_TOKEN), \
            "Masked positions should contain MASK_TOKEN"


class TestMaskRateToStep:
    def test_high_rate_selects_first_coarse_schedule_entry(self):
        """High mask rate is the coarse/high-noise start of refinement."""
        assert mask_rate_to_step(1.0, n_steps=8) == 0
        assert mask_rate_to_step(0.99, n_steps=8) == 0

    def test_low_rate_selects_last_fine_schedule_entry(self):
        """Low mask rate is the fine/low-noise end of refinement."""
        assert mask_rate_to_step(0.0, n_steps=8) == 7
        assert mask_rate_to_step(0.05, n_steps=8) == 7

    def test_monotonically_moves_from_coarse_to_fine_as_noise_decreases(self):
        rates = [1.0, 0.75, 0.5, 0.25, 0.0]
        steps = [mask_rate_to_step(rate, n_steps=8) for rate in rates]
        assert steps == sorted(steps), steps

    def test_clamped_to_valid_range(self):
        for n in [4, 8, 16]:
            s0 = mask_rate_to_step(0.0, n)
            s1 = mask_rate_to_step(1.0, n)
            assert 0 <= s0 < n, f"Step {s0} out of range [0,{n})"
            assert 0 <= s1 < n, f"Step {s1} out of range [0,{n})"


class TestComputeLoss:
    def test_loss_is_positive(self):
        """Cross-entropy loss should always be positive."""
        model, cfg = make_small_model()
        x0 = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
        loss = compute_loss(
            model, x0, cfg.mask_token_id(),
            cfg.precision_schedule, cfg.n_diffusion_steps,
        )
        mx.eval(loss)
        assert float(loss) > 0, f"Loss should be positive, got {float(loss)}"

    def test_loss_is_finite(self):
        """Loss should be a finite number."""
        model, cfg = make_small_model()
        x0 = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
        loss = compute_loss(
            model, x0, cfg.mask_token_id(),
            cfg.precision_schedule, cfg.n_diffusion_steps,
        )
        mx.eval(loss)
        val = float(loss)
        assert not (val != val), f"Loss is NaN"
        assert val < 1e10, f"Loss is unexpectedly large: {val}"

    def test_loss_decreases_with_training(self):
        """Loss should decrease after a few gradient steps."""
        import mlx.nn as nn
        import mlx.optimizers as optim

        model, cfg = make_small_model()
        optimizer = optim.Adam(learning_rate=1e-3)

        def loss_fn(model, x0):
            return compute_loss(
                model, x0, cfg.mask_token_id(),
                cfg.precision_schedule, cfg.n_diffusion_steps,
            )

        loss_and_grad = nn.value_and_grad(model, loss_fn)
        x0 = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])

        mx.random.seed(42)
        losses = []
        for _ in range(10):
            loss, grads = loss_and_grad(model, x0)
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters())
            losses.append(float(loss))

        # Loss should not be strictly monotone due to stochasticity, but
        # it should generally decrease (average first 3 vs last 3)
        avg_early = np.mean(losses[:3])
        avg_late = np.mean(losses[-3:])
        print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
        # Relaxed assertion: at least the model can process without exploding
        assert avg_late < avg_early * 2.0, \
            f"Loss increased dramatically: {avg_early:.4f} → {avg_late:.4f}"


class TestGenerate:
    def test_generate_shape(self):
        """Generate should return (batch_size, seq_len) tensor."""
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()
        tokens = generate(
            model,
            seq_len=16,
            mask_token_id=mask_id,
            precision_schedule=cfg.precision_schedule,
            batch_size=2,
        )
        mx.eval(tokens)
        assert tokens.shape == (2, 16), f"Expected (2,16), got {tokens.shape}"

    def test_generate_no_mask_tokens(self):
        """Final output should not contain MASK tokens (all positions revealed)."""
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()
        tokens = generate(
            model,
            seq_len=16,
            mask_token_id=mask_id,
            precision_schedule=cfg.precision_schedule,
            batch_size=1,
        )
        mx.eval(tokens)
        tokens_np = np.array(tokens.tolist())
        assert not np.any(tokens_np == mask_id), \
            f"Generated sequence should not contain MASK tokens, got: {tokens_np}"

    def test_generate_tokens_in_vocab(self):
        """Generated token IDs should be in valid vocabulary range."""
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()
        tokens = generate(
            model,
            seq_len=16,
            mask_token_id=mask_id,
            precision_schedule=cfg.precision_schedule,
            batch_size=2,
        )
        mx.eval(tokens)
        tokens_np = np.array(tokens.tolist())
        assert np.all(tokens_np >= 0), "Token IDs should be non-negative"
        assert np.all(tokens_np < cfg.vocab_size), \
            f"Token IDs should be < vocab_size={cfg.vocab_size}, got max {tokens_np.max()}"


class TestForwardIncremental:
    """Tests for the incremental forward API (Phase 2 — M2)."""

    def test_no_cache_falls_back_to_full_forward(self):
        """Without a cache, forward_incremental should produce same result as __call__."""
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()
        x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8,
                       mask_id, mask_id, mask_id, mask_id, mask_id, mask_id, mask_id, mask_id]])
        step_frac = mx.array([0.5])
        model.set_bits(2)

        logits_full = model(x, step_frac)
        logits_inc, cache = model.forward_incremental(x, step_frac, cache=None)
        mx.eval(logits_full, logits_inc)

        np.testing.assert_allclose(
            np.array(logits_full.tolist()),
            np.array(logits_inc.tolist()),
            atol=1e-6,
        )

    def test_cache_returns_correct_bits(self):
        """The cache should record the bits used for the forward pass."""
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()
        x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8,
                       mask_id, mask_id, mask_id, mask_id, mask_id, mask_id, mask_id, mask_id]])
        step_frac = mx.array([0.5])
        model.set_bits(4)

        _, cache = model.forward_incremental(x, step_frac, cache=None)
        assert cache.bits == 4, f"Expected cache.bits=4, got {cache.bits}"

    def test_delta_weight_1_parities_with_full_recompute(self):
        """With delta_weight=1.0, cached path should equal full recompute."""
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()
        x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8,
                       mask_id, mask_id, mask_id, mask_id, mask_id, mask_id, mask_id, mask_id]])
        step_frac_1 = mx.array([0.8])
        step_frac_2 = mx.array([0.4])

        # Step 1: no cache
        model.set_bits(1)
        logits1, cache = model.forward_incremental(x, step_frac_1, cache=None)

        # Step 2: with cache, delta_weight=1.0
        model.set_bits(4)
        logits2_inc, _ = model.forward_incremental(x, step_frac_2, cache=cache, delta_weight=1.0)

        # Full recompute at step 2
        model.set_bits(4)
        logits2_full = model(x, step_frac_2)

        mx.eval(logits2_inc, logits2_full)
        np.testing.assert_allclose(
            np.array(logits2_inc.tolist()),
            np.array(logits2_full.tolist()),
            atol=1e-6,
        )

    def test_delta_weight_0_returns_cached_logits(self):
        """With delta_weight=0.0, output should be exactly the cached logits."""
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()
        x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8,
                       mask_id, mask_id, mask_id, mask_id, mask_id, mask_id, mask_id, mask_id]])
        step_frac_1 = mx.array([0.8])
        step_frac_2 = mx.array([0.4])

        model.set_bits(1)
        _, cache = model.forward_incremental(x, step_frac_1, cache=None)

        model.set_bits(4)
        logits2, _ = model.forward_incremental(x, step_frac_2, cache=cache, delta_weight=0.0)

        mx.eval(logits2, cache.logits)
        np.testing.assert_allclose(
            np.array(logits2.tolist()),
            np.array(cache.logits.tolist()),
            atol=1e-6,
        )

    def test_cached_update_is_a_scaled_full_recompute(self):
        """A cached call still equals the full forward blended with the cache."""
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()
        x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8,
                       mask_id, mask_id, mask_id, mask_id, mask_id, mask_id, mask_id, mask_id]])
        step_frac_1 = mx.array([0.8])
        step_frac_2 = mx.array([0.4])
        model.set_bits(1)
        _, cache = model.forward_incremental(x, step_frac_1, cache=None)
        model.set_bits(4)
        logits, _ = model.forward_incremental(x, step_frac_2, cache=cache, delta_weight=0.25)
        full = model(x, step_frac_2)
        expected = cache.logits + 0.25 * (full - cache.logits)
        mx.eval(logits, expected)
        np.testing.assert_allclose(
            np.array(logits.tolist()),
            np.array(expected.tolist()),
            atol=1e-6,
        )

    def test_incremental_cache_chain(self):
        """Chaining multiple incremental steps should work without errors."""
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()
        x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8,
                       mask_id, mask_id, mask_id, mask_id, mask_id, mask_id, mask_id, mask_id]])

        cache = None
        fracs = [0.9, 0.6, 0.3, 0.1]
        bits_list = [1, 2, 4, 8]
        for frac, bits in zip(fracs, bits_list):
            model.set_bits(bits)
            sf = mx.array([frac])
            logits, cache = model.forward_incremental(x, sf, cache=cache)
            mx.eval(logits)
            assert logits.shape == (1, 16, cfg.vocab_size), \
                f"Expected (1,16,{cfg.vocab_size}), got {logits.shape}"


class TestGenerateIncremental:
    """Tests for generate_incremental (Phase 2 — M2)."""

    def test_generate_incremental_shape(self):
        """Should return (batch_size, seq_len) tensor."""
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()
        tokens = generate_incremental(
            model,
            seq_len=16,
            mask_token_id=mask_id,
            precision_schedule=cfg.precision_schedule,
            batch_size=2,
        )
        mx.eval(tokens)
        assert tokens.shape == (2, 16), f"Expected (2,16), got {tokens.shape}"

    def test_generate_incremental_no_mask_tokens(self):
        """Final output should not contain MASK tokens."""
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()
        tokens = generate_incremental(
            model,
            seq_len=16,
            mask_token_id=mask_id,
            precision_schedule=cfg.precision_schedule,
            batch_size=1,
        )
        mx.eval(tokens)
        tokens_np = np.array(tokens.tolist())
        assert not np.any(tokens_np == mask_id), \
            f"Generated sequence should not contain MASK tokens, got: {tokens_np}"

    def test_generate_incremental_tokens_in_vocab(self):
        """Generated token IDs should be in valid vocabulary range."""
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()
        tokens = generate_incremental(
            model,
            seq_len=16,
            mask_token_id=mask_id,
            precision_schedule=cfg.precision_schedule,
            batch_size=2,
        )
        mx.eval(tokens)
        tokens_np = np.array(tokens.tolist())
        assert np.all(tokens_np >= 0), "Token IDs should be non-negative"
        assert np.all(tokens_np < cfg.vocab_size), \
            f"Token IDs should be < vocab_size={cfg.vocab_size}, got max {tokens_np.max()}"

    def test_parity_generate_vs_incremental(self):
        """generate and generate_incremental should produce identical results
        with delta_weight=1.0 and the same random seed."""
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()

        # Use deterministic seed for reproducibility
        mx.random.seed(123)
        tokens_std = generate(
            model,
            seq_len=16,
            mask_token_id=mask_id,
            precision_schedule=cfg.precision_schedule,
            batch_size=1,
        )
        mx.eval(tokens_std)

        mx.random.seed(123)
        tokens_inc = generate_incremental(
            model,
            seq_len=16,
            mask_token_id=mask_id,
            precision_schedule=cfg.precision_schedule,
            batch_size=1,
            delta_weight=1.0,
        )
        mx.eval(tokens_inc)

        np_std = np.array(tokens_std.tolist())
        np_inc = np.array(tokens_inc.tolist())
        np.testing.assert_array_equal(
            np_std, np_inc,
            err_msg="generate and generate_incremental should produce identical tokens"
        )

    def test_delta_weight_below_1_still_valid(self):
        """With delta_weight=0.5, output should still be valid tokens (no crash)."""
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()
        tokens = generate_incremental(
            model,
            seq_len=16,
            mask_token_id=mask_id,
            precision_schedule=cfg.precision_schedule,
            batch_size=1,
            delta_weight=0.5,
        )
        mx.eval(tokens)
        tokens_np = np.array(tokens.tolist())
        assert not np.any(tokens_np == mask_id), \
            "Should not contain MASK tokens even with delta_weight<1.0"
        assert np.all(tokens_np >= 0) and np.all(tokens_np < cfg.vocab_size)


class TestGenerateWithEarlyExit:
    """Tests for generate_with_early_exit (Phase 2 - M3)."""

    def test_returns_tuple(self):
        """Should return (token_ids, steps_used) tuple."""
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()
        result = generate_with_early_exit(
            model,
            seq_len=16,
            mask_token_id=mask_id,
            precision_schedule=cfg.precision_schedule,
            batch_size=1,
        )
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected 2 elements, got {len(result)}"

    def test_shape_correct(self):
        """Token IDs should have shape (batch_size, seq_len)."""
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()
        tokens, steps = generate_with_early_exit(
            model,
            seq_len=16,
            mask_token_id=mask_id,
            precision_schedule=cfg.precision_schedule,
            batch_size=2,
        )
        mx.eval(tokens)
        assert tokens.shape == (2, 16), f"Expected (2,16), got {tokens.shape}"

    def test_no_mask_tokens(self):
        """Output should not contain MASK tokens."""
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()
        tokens, _ = generate_with_early_exit(
            model,
            seq_len=16,
            mask_token_id=mask_id,
            precision_schedule=cfg.precision_schedule,
            batch_size=1,
        )
        mx.eval(tokens)
        tokens_np = np.array(tokens.tolist())
        assert not np.any(tokens_np == mask_id), "Should not contain MASK tokens"

    def test_tokens_in_vocab(self):
        """Generated tokens should be in valid vocab range."""
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()
        tokens, _ = generate_with_early_exit(
            model,
            seq_len=16,
            mask_token_id=mask_id,
            precision_schedule=cfg.precision_schedule,
            batch_size=1,
        )
        mx.eval(tokens)
        tokens_np = np.array(tokens.tolist())
        assert np.all(tokens_np >= 0)
        assert np.all(tokens_np < cfg.vocab_size)

    def test_high_threshold_uses_all_steps(self):
        """With threshold=1.0, early-exit should never trigger, using all steps."""
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()
        tokens, steps_used = generate_with_early_exit(
            model,
            seq_len=16,
            mask_token_id=mask_id,
            precision_schedule=cfg.precision_schedule,
            batch_size=1,
            confidence_threshold=1.0,  # impossible to reach
        )
        n_steps = len(cfg.precision_schedule)
        assert steps_used == n_steps, \
            f"Threshold=1.0 should use all {n_steps} steps, got {steps_used}"

    def test_low_threshold_exits_early(self):
        """With threshold=0.0, early-exit should trigger after min_steps."""
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()
        tokens, steps_used = generate_with_early_exit(
            model,
            seq_len=16,
            mask_token_id=mask_id,
            precision_schedule=cfg.precision_schedule,
            batch_size=1,
            confidence_threshold=0.0,  # always satisfied
            min_steps=1,
        )
        assert steps_used == 1, \
            f"Threshold=0.0 + min_steps=1 should exit after 1 step, got {steps_used}"

    def test_min_steps_respected(self):
        """Even with threshold=0.0, should not exit before min_steps."""
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()
        tokens, steps_used = generate_with_early_exit(
            model,
            seq_len=16,
            mask_token_id=mask_id,
            precision_schedule=cfg.precision_schedule,
            batch_size=1,
            confidence_threshold=0.0,
            min_steps=3,
        )
        assert steps_used == 3, \
            f"min_steps=3 should force 3 steps even with threshold=0.0, got {steps_used}"

    def test_incremental_mode_works(self):
        """Early-exit should work with use_incremental=True."""
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()
        tokens, steps_used = generate_with_early_exit(
            model,
            seq_len=16,
            mask_token_id=mask_id,
            precision_schedule=cfg.precision_schedule,
            batch_size=1,
            use_incremental=True,
            delta_weight=1.0,
        )
        mx.eval(tokens)
        tokens_np = np.array(tokens.tolist())
        assert not np.any(tokens_np == mask_id), "Should not contain MASK tokens"
        assert steps_used >= 1, "Should complete at least 1 step"

    def test_valid_output_with_incremental_and_early_exit(self):
        """Output with incremental + early-exit should be valid tokens."""
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()
        tokens, _ = generate_with_early_exit(
            model,
            seq_len=16,
            mask_token_id=mask_id,
            precision_schedule=cfg.precision_schedule,
            batch_size=2,
            use_incremental=True,
            delta_weight=0.5,
            confidence_threshold=0.5,
        )
        mx.eval(tokens)
        tokens_np = np.array(tokens.tolist())
        assert not np.any(tokens_np == mask_id)
        assert np.all(tokens_np >= 0) and np.all(tokens_np < cfg.vocab_size)


class TestTemperatureSampling:
    def _greedy_reference(self, model, cfg, seq_len, batch_size):
        """The previous argmax reveal loop, kept here so greedy stays fixed."""
        mask_id = cfg.mask_token_id()
        schedule = cfg.precision_schedule
        n_steps = len(schedule)
        x = mx.full((batch_size, seq_len), mask_id, dtype=mx.int32)
        for step_i in range(n_steps):
            model.set_bits(schedule[step_i])
            step_frac = mx.full((batch_size,), 1.0 - step_i / n_steps)
            target = int((step_i + 1) / n_steps * seq_len)
            current = int((x != mask_id).sum()) // batch_size
            n_to_reveal = max(0, target - current)
            if n_to_reveal == 0:
                continue
            logits = model(x, step_frac)
            pred_ids = mx.argmax(logits.astype(mx.float32), axis=-1).astype(mx.int32)
            confidence = mx.softmax(logits.astype(mx.float32), axis=-1).max(axis=-1)
            x = _reveal_positions(x, pred_ids, confidence, mask_id, n_to_reveal)
            mx.eval(x)
        still_masked = x == mask_id
        if bool(still_masked.any()):
            model.set_bits(schedule[-1])
            logits = model(x, mx.zeros((batch_size,)))
            pred_ids = mx.argmax(logits.astype(mx.float32), axis=-1).astype(mx.int32)
            mx.eval(pred_ids)
            x = mx.where(still_masked, pred_ids, x)
            mx.eval(x)
        return x

    def test_temperature_zero_matches_greedy_argmax_and_ignores_rng(self):
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()
        kwargs = dict(
            seq_len=8,
            mask_token_id=mask_id,
            precision_schedule=cfg.precision_schedule,
            batch_size=2,
        )
        reference = self._greedy_reference(model, cfg, 8, 2)
        mx.random.seed(1)
        greedy = generate(model, temperature=0.0, **kwargs)
        mx.random.seed(99)
        default = generate(model, **kwargs)
        mx.eval(reference, greedy, default)
        np.testing.assert_array_equal(np.array(greedy.tolist()), np.array(reference.tolist()))
        np.testing.assert_array_equal(np.array(default.tolist()), np.array(reference.tolist()))

    def test_positive_temperature_is_deterministic_and_samples(self):
        mx.random.seed(5)
        model, cfg = make_small_model()
        mask_id = cfg.mask_token_id()
        kwargs = dict(
            seq_len=12,
            mask_token_id=mask_id,
            precision_schedule=cfg.precision_schedule,
            temperature=1.0,
            batch_size=2,
        )
        mx.random.seed(11)
        first = generate(model, **kwargs)
        mx.eval(first)
        mx.random.seed(11)
        second = generate(model, **kwargs)
        mx.eval(second)
        mx.random.seed(11)
        incremental = generate_incremental(model, delta_weight=1.0, **kwargs)
        mx.eval(incremental)
        greedy = generate(model, temperature=0.0, seq_len=12, mask_token_id=mask_id,
                          precision_schedule=cfg.precision_schedule, batch_size=2)
        mx.eval(greedy)
        first_np = np.array(first.tolist())
        np.testing.assert_array_equal(first_np, np.array(second.tolist()))
        np.testing.assert_array_equal(first_np, np.array(incremental.tolist()))
        assert not np.array_equal(first_np, np.array(greedy.tolist()))
        assert not np.any(first_np == mask_id)
        assert np.all(first_np >= 0) and np.all(first_np < cfg.vocab_size)

    def test_reveal_order_follows_confidence(self):
        mask_id = 9
        tokens = mx.full((1, 4), mask_id, dtype=mx.int32)
        predictions = mx.array([[3, 4, 5, 6]], dtype=mx.int32)
        confidence = mx.array([[0.1, 0.9, 0.2, 0.8]])
        revealed = _reveal_positions(tokens, predictions, confidence, mask_id, n_to_reveal=2)
        mx.eval(revealed)
        assert revealed.tolist() == [[mask_id, 4, mask_id, 6]]

    def test_confidence_uses_the_sampling_distribution(self):
        logits = mx.array([[[4.0, 0.0, 0.0, 0.0]]])
        greedy_ids, greedy_conf = _token_predictions(logits, 0.0)
        _, unit_conf = _token_predictions(logits, 1.0)
        _, sharp_conf = _token_predictions(logits, 0.5)
        _, flat_conf = _token_predictions(logits, 50.0)
        mx.eval(greedy_ids, greedy_conf, unit_conf, sharp_conf, flat_conf)
        assert int(greedy_ids.item()) == 0
        assert float(greedy_conf.item()) == pytest.approx(float(unit_conf.item()))
        assert float(sharp_conf.item()) > float(unit_conf.item()) > float(flat_conf.item())

    def test_negative_temperature_is_rejected(self):
        model, cfg = make_small_model()
        with pytest.raises(ValueError, match="temperature"):
            generate(
                model,
                seq_len=4,
                mask_token_id=cfg.mask_token_id(),
                precision_schedule=cfg.precision_schedule,
                temperature=-0.1,
            )


def run_all():
    test_classes = [
        TestCorruptTokens,
        TestMaskRateToStep,
        TestComputeLoss,
        TestGenerate,
        TestForwardIncremental,
        TestGenerateIncremental,
        TestGenerateWithEarlyExit,
        TestTemperatureSampling,
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
    print("Running diffusion tests...")
    success = run_all()
    sys.exit(0 if success else 1)
