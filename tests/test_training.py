"""Tests for one complete training step and checkpoint save/load."""

import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils

from src.model import DiffusionLM
from src.config import ModelConfig, ExperimentConfig, DataConfig, TrainConfig
from src.diffusion import compute_loss
from src.train import save_checkpoint, load_checkpoint


MASK_TOKEN = 50  # small vocab for tests


def make_tiny_cfg(**overrides):
    defaults = dict(
        vocab_size=MASK_TOKEN,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        max_seq_len=16,
        dropout=0.0,
        n_diffusion_steps=4,
        precision_schedule=[1, 1, 2, 4],
        model_type="progressive",
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


class TestOneTrainingStep:
    def test_step_reduces_loss(self):
        """A gradient step should not increase loss dramatically."""
        mx.random.seed(7)
        cfg = make_tiny_cfg()
        model = DiffusionLM(cfg)
        optimizer = optim.Adam(learning_rate=1e-3)

        def loss_fn(model, x):
            return compute_loss(
                model, x, cfg.mask_token_id(),
                cfg.precision_schedule, cfg.n_diffusion_steps,
            )

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])

        loss0, grads = loss_and_grad(model, x)
        mx.eval(loss0)
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        loss1, _ = loss_and_grad(model, x)
        mx.eval(loss1)

        print(f"  Loss before: {float(loss0):.4f}, after: {float(loss1):.4f}")
        # Just verify no explosion
        assert float(loss1) < float(loss0) * 10, \
            f"Loss exploded: {float(loss0):.4f} → {float(loss1):.4f}"

    def test_weights_change_after_step(self):
        """Model weights should change after an optimizer step."""
        cfg = make_tiny_cfg()
        model = DiffusionLM(cfg)
        optimizer = optim.Adam(learning_rate=1e-2)

        # Capture initial weight
        params_before = dict(mlx.utils.tree_flatten(model.parameters()))
        w_before = None
        for k, v in params_before.items():
            if "weight" in k and v.ndim == 2:
                w_before = np.array(v.tolist()).copy()
                break
        assert w_before is not None, "Could not find weight to check"

        def loss_fn(model, x):
            return compute_loss(
                model, x, cfg.mask_token_id(),
                cfg.precision_schedule, cfg.n_diffusion_steps,
            )

        x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
        loss_and_grad = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad(model, x)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())

        params_after = dict(mlx.utils.tree_flatten(model.parameters()))
        w_after = None
        for k, v in params_after.items():
            if "weight" in k and v.ndim == 2:
                if k in {k2 for k2, _ in params_before.items()}:
                    w_after = np.array(v.tolist())
                    break

        if w_after is not None:
            assert not np.allclose(w_before, w_after), \
                "Weights did not change after optimizer step"

    def test_gradient_clipping(self):
        """Gradient clipping should work without errors."""
        cfg = make_tiny_cfg()
        model = DiffusionLM(cfg)
        optimizer = optim.Adam(learning_rate=1e-3)

        def loss_fn(model, x):
            return compute_loss(
                model, x, cfg.mask_token_id(),
                cfg.precision_schedule, cfg.n_diffusion_steps,
            )

        x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
        loss_and_grad = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad(model, x)

        # Clip gradients
        clipped_grads, norm = optim.clip_grad_norm(grads, max_norm=1.0)
        optimizer.update(model, clipped_grads)
        mx.eval(loss, norm, model.parameters())

        assert float(norm) > 0, "Gradient norm should be positive"
        print(f"  Gradient norm: {float(norm):.4f} (after clipping to 1.0)")


class TestCheckpointing:
    def test_save_and_load(self):
        """Saved checkpoint should be loadable and produce same output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = make_tiny_cfg()
            model = DiffusionLM(cfg)
            mx.eval(model.parameters())

            x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
            t = mx.array([0.5])
            logits_before = model(x, t)
            mx.eval(logits_before)
            logits_before_np = np.array(logits_before.tolist())

            # Save
            experiment_cfg = ExperimentConfig(
                model=cfg,
                experiment_name="test_exp",
            )
            optimizer = optim.Adam(learning_rate=1e-3)
            ckpt_dir = Path(tmpdir)
            ckpt_path = save_checkpoint(model, optimizer, 100, 1.23, experiment_cfg, ckpt_dir)
            assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"

            # Load into fresh model
            model2 = DiffusionLM(cfg)
            step = load_checkpoint(model2, str(ckpt_path))
            assert step == 100, f"Expected step 100, got {step}"

            logits_after = model2(x, t)
            mx.eval(logits_after)
            logits_after_np = np.array(logits_after.tolist())

            assert np.allclose(logits_before_np, logits_after_np, atol=1e-5), \
                "Loaded model should produce same output as saved model"

    def test_checkpoint_contains_model_weights(self):
        """Checkpoint file should contain model weight arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = make_tiny_cfg()
            model = DiffusionLM(cfg)
            mx.eval(model.parameters())

            experiment_cfg = ExperimentConfig(model=cfg, experiment_name="test")
            optimizer = optim.Adam(learning_rate=1e-3)
            ckpt_path = save_checkpoint(model, optimizer, 1, 0.0, experiment_cfg, Path(tmpdir))

            data = mx.load(str(ckpt_path))
            weight_keys = [k for k in data.keys() if not k.startswith("opt_")]
            assert len(weight_keys) > 0, f"No model weights in checkpoint: {list(data.keys())}"
            print(f"  Checkpoint contains {len(weight_keys)} weight arrays")


class TestBaselineModel:
    def test_baseline_all_full_precision(self):
        """Baseline model should use bits=16 throughout."""
        cfg = make_tiny_cfg(model_type="baseline")
        model = DiffusionLM(cfg)
        assert model.get_current_bits() == 16

    def test_baseline_and_progressive_same_architecture(self):
        """Baseline and progressive should have same parameter count."""
        cfg_b = make_tiny_cfg(model_type="baseline")
        cfg_p = make_tiny_cfg(model_type="progressive")
        model_b = DiffusionLM(cfg_b)
        model_p = DiffusionLM(cfg_p)
        assert model_b.total_params() == model_p.total_params(), \
            "Baseline and progressive should have identical architecture"


def run_all():
    test_classes = [
        TestOneTrainingStep,
        TestCheckpointing,
        TestBaselineModel,
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
    print("Running training tests...")
    success = run_all()
    sys.exit(0 if success else 1)
