"""Strict P1 metric-schema and train/eval-mode regression tests."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np

from src.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from src.data import BatchIterator
from src.model import DiffusionLM
from src.train import MetricsLogger, evaluate
import src.train as train_module


REQUIRED_TRAIN_COLUMNS = {
    "step",
    "train_loss",
    "gradient_norm",
    "q1_residual_rms",
    "injected_noise_rms",
    "bits_used",
    "lr",
    "elapsed_s",
}
REQUIRED_EVAL_FIELDS = {
    "step",
    "val_loss",
    "val_perplexity",
    "val_accuracy",
    "generalization_gap",
    "train_loss",
    "bits_used",
}


def tiny_cfg(noise_mode: str = "gaussian_matched") -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="p1-metrics-test",
        model=ModelConfig(
            vocab_size=32,
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_ff=32,
            max_seq_len=8,
            dropout=0.0,
            n_diffusion_steps=2,
            precision_schedule=[16, 16],
            model_type="baseline",
            weight_noise_mode=noise_mode,
        ),
    )


def test_metrics_logger_writes_required_finite_machine_readable_schema(tmp_path: Path):
    logger = MetricsLogger(tmp_path, "schema")
    logger.log_train(
        step=10,
        loss=2.5,
        gradient_norm=1.25,
        q1_residual_rms=0.125,
        injected_noise_rms=0.124,
        bits=16,
        lr=3e-4,
        elapsed=1.5,
    )
    logger.log_eval(step=10, val_loss=2.75, val_acc=0.2, bits=16, train_loss=2.5)
    logger.close()

    with (tmp_path / "schema/train_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert set(rows[0]) == REQUIRED_TRAIN_COLUMNS
    for field in REQUIRED_TRAIN_COLUMNS - {"step", "bits_used"}:
        assert math.isfinite(float(rows[0][field])), field

    history = json.loads((tmp_path / "schema/eval_history.json").read_text())
    assert set(history[0]) == REQUIRED_EVAL_FIELDS
    assert history[0]["generalization_gap"] == 0.25
    assert history[0]["val_perplexity"] == pytest.approx(math.exp(2.75), rel=1e-6)
    for field in REQUIRED_EVAL_FIELDS - {"step", "bits_used"}:
        assert math.isfinite(float(history[0][field])), field


def test_metrics_logger_rejects_nan_and_inf(tmp_path: Path):
    logger = MetricsLogger(tmp_path, "nonfinite")
    try:
        with pytest.raises(ValueError, match="finite"):
            logger.log_train(
                step=1,
                loss=float("nan"),
                gradient_norm=1.0,
                q1_residual_rms=0.1,
                injected_noise_rms=0.1,
                bits=16,
                lr=1e-3,
                elapsed=1.0,
            )
        with pytest.raises(ValueError, match="finite"):
            logger.log_eval(
                step=1,
                val_loss=float("inf"),
                val_acc=0.1,
                bits=16,
                train_loss=1.0,
            )
    finally:
        logger.close()


def test_evaluate_disables_train_only_noise_and_restores_training_mode():
    class TrackingModel(DiffusionLM):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.forward_training_modes = []

        def __call__(self, *args, **kwargs):
            self.forward_training_modes.append(self.training)
            return super().__call__(*args, **kwargs)

    cfg = tiny_cfg()
    model = TrackingModel(cfg.model)
    model.set_weight_noise_seed(123)
    model.train()
    data = np.tile(np.arange(1, 9, dtype=np.int32), (4, 1))
    metrics = evaluate(model, BatchIterator(data, batch_size=2, seed=7), cfg, n_steps=1)
    assert model.forward_training_modes == [False]
    assert model.training is True
    assert all(math.isfinite(float(value)) for value in metrics.values())


def test_evaluate_preserves_eval_mode():
    cfg = tiny_cfg()
    model = DiffusionLM(cfg.model)
    model.eval()
    data = np.tile(np.arange(1, 9, dtype=np.int32), (4, 1))
    evaluate(model, BatchIterator(data, batch_size=2, seed=7), cfg, n_steps=1)
    assert model.training is False


def test_train_enters_train_mode_and_seeds_layer_noise(monkeypatch, tmp_path: Path):
    observations = {}

    class TrackingModel(DiffusionLM):
        def set_weight_noise_seed(self, seed):
            observations["seed"] = seed
            return super().set_weight_noise_seed(seed)

        def train(self, mode=True):
            observations.setdefault("train_modes", []).append(mode)
            return super().train(mode)

    data = np.tile(np.arange(1, 9, dtype=np.int32), (4, 1))
    monkeypatch.setattr(train_module, "DiffusionLM", TrackingModel)
    monkeypatch.setattr(train_module, "build_and_cache_dataset", lambda **_: (data, data))

    cfg = tiny_cfg()
    cfg.data = DataConfig(seq_len=8)
    cfg.train = TrainConfig(
        batch_size=2,
        max_steps=1,
        warmup_steps=0,
        eval_every=1,
        eval_steps=1,
        log_every=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        results_dir=str(tmp_path / "results"),
        save_checkpoints=False,
        seed=919,
    )
    train_module.train(cfg)

    assert observations["seed"] == 919
    assert observations["train_modes"][0] is True
    assert observations["train_modes"][-1] is True
    rows = list(csv.DictReader((tmp_path / "results/p1-metrics-test/train_metrics.csv").open()))
    assert rows and all(math.isfinite(float(rows[0][field])) for field in REQUIRED_TRAIN_COLUMNS - {"step", "bits_used"})


def test_train_performs_final_evaluation_when_interval_is_not_reached(monkeypatch, tmp_path: Path):
    data = np.tile(np.arange(1, 9, dtype=np.int32), (4, 1))
    monkeypatch.setattr(train_module, "build_and_cache_dataset", lambda **_: (data, data))

    cfg = tiny_cfg()
    cfg.data = DataConfig(seq_len=8)
    cfg.train = TrainConfig(
        batch_size=2,
        max_steps=1,
        warmup_steps=0,
        eval_every=2,
        eval_steps=1,
        log_every=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        results_dir=str(tmp_path / "results"),
        save_checkpoints=False,
        seed=919,
    )
    train_module.train(cfg)
    summary = json.loads(
        (tmp_path / "results/p1-metrics-test/final_summary.json").read_text()
    )
    assert math.isfinite(summary["best_val_loss"])


# Imported late to keep the MLX imports grouped above and lint-free.
import pytest
