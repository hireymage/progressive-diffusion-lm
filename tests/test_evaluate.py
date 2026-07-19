import numpy as np

from src.config import ExperimentConfig, ModelConfig, TrainConfig
from src.data import BatchIterator
from src.evaluate import eval_model
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
