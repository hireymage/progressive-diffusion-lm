import numpy as np
import pytest

from src.config import ExperimentConfig, ModelConfig, TrainConfig
from src.data import BatchIterator
from src.evaluate import estimate_flop_savings, eval_model
from src.quantization import EFFECTIVE_BITS
from src.model import DiffusionLM


def _small_baseline():
    cfg = ExperimentConfig(
        model=ModelConfig(
            vocab_size=32,
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_ff=32,
            max_seq_len=8,
            dropout=0.0,
            n_diffusion_steps=4,
            precision_schedule=[16, 16, 16, 16],
            model_type="baseline",
        ),
        train=TrainConfig(batch_size=2),
    )
    return DiffusionLM(cfg.model), cfg


def test_eval_model_replays_identical_batches_and_masks_with_same_seed():
    model, cfg = _small_baseline()
    model.train()
    data = np.arange(8 * 8, dtype=np.int32).reshape(8, 8) % cfg.model.vocab_size

    first = eval_model(
        model,
        BatchIterator(data, batch_size=2, seed=17),
        cfg,
        n_steps=3,
        rng_seed=919,
    )
    second = eval_model(
        model,
        BatchIterator(data, batch_size=2, seed=17),
        cfg,
        n_steps=3,
        rng_seed=919,
    )

    assert first["val_loss"] == second["val_loss"]
    assert first["val_accuracy"] == second["val_accuracy"]
    assert model.training is False


def test_flop_ratio_uses_fp32_effective_bits():
    schedule = [1, 1, 1, 1, 2, 2, 4, 4]
    report = estimate_flop_savings(schedule)
    average = sum(EFFECTIVE_BITS[bits] for bits in schedule) / len(schedule)
    assert report["avg_effective_bits"] == pytest.approx(average)
    assert report["fp32_effective_bits"] == pytest.approx(EFFECTIVE_BITS[16])
    assert report["theoretical_flop_reduction_vs_fp32"] == pytest.approx(
        EFFECTIVE_BITS[16] / average
    )
    assert "bf16" not in report
    assert "bfloat" not in report["note"].lower()


def test_constant_fp32_schedule_has_no_theoretical_savings():
    report = estimate_flop_savings([16, 16, 16, 16])
    assert report["theoretical_flop_reduction_vs_fp32"] == pytest.approx(1.0)
