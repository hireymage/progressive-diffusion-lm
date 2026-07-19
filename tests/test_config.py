import pytest

from src.config import DataConfig, ModelConfig, TrainConfig


def test_model_config_rejects_invalid_schedule_and_mode():
    with pytest.raises(ValueError, match="n_diffusion_steps"):
        ModelConfig(n_diffusion_steps=0, precision_schedule=[])
    with pytest.raises(ValueError, match="precision_schedule length"):
        ModelConfig(n_diffusion_steps=2, precision_schedule=[1])
    with pytest.raises(ValueError, match="unsupported precision"):
        ModelConfig(n_diffusion_steps=1, precision_schedule=[8])
    with pytest.raises(ValueError, match="model_type"):
        ModelConfig(model_type="unknown")


def test_regularization_ranges_are_validated():
    with pytest.raises(ValueError, match="dropout"):
        ModelConfig(dropout=1.0)
    with pytest.raises(ValueError, match="weight_decay"):
        TrainConfig(weight_decay=-0.1)


def test_data_and_training_ranges_are_validated():
    with pytest.raises(ValueError, match="train_split"):
        DataConfig(train_split=1.0)
    with pytest.raises(ValueError, match="batch_size"):
        TrainConfig(batch_size=0)
    with pytest.raises(ValueError, match="max_steps"):
        TrainConfig(max_steps=0)
