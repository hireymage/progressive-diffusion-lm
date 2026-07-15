"""Configuration for the progressive diffusion LM experiment."""
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class ModelConfig:
    # Vocabulary
    vocab_size: int = 16000
    # MASK token is at vocab_size index (not in tokenizer vocab — appended)
    pad_token_id: int = 0

    # Architecture
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 256
    dropout: float = 0.1

    # Diffusion
    n_diffusion_steps: int = 8
    # bits per refinement step; length == n_diffusion_steps
    # step 0 is the "coarsest" (most noise), step T-1 is "finest" (least noise)
    precision_schedule: List[int] = field(
        default_factory=lambda: [1, 1, 1, 1, 2, 2, 4, 4]
    )

    # "baseline" uses full-precision (no quantisation) at every step
    # "progressive" uses precision_schedule
    model_type: str = "progressive"

    def mask_token_id(self) -> int:
        return self.vocab_size  # one past the end of the regular vocab


@dataclass
class DataConfig:
    dataset_name: str = "wikimedia/wikipedia"
    dataset_config: str = "20231101.en"
    max_articles: int = 1000
    max_text_bytes: int = 10_000_000   # 10 MB
    seq_len: int = 256
    tokenizer_path: str = "tokenizer/wiki_bpe"
    data_cache_dir: str = "data/cache"
    train_split: float = 0.95


@dataclass
class TrainConfig:
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 10_000
    warmup_steps: int = 500
    grad_clip: float = 1.0

    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 500
    resume_from: Optional[str] = None

    eval_every: int = 250
    eval_steps: int = 50

    log_every: int = 10
    seed: int = 42


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    experiment_name: str = "experiment"

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            d = json.load(f)
        return cls(
            model=ModelConfig(**d.get("model", {})),
            data=DataConfig(**d.get("data", {})),
            train=TrainConfig(**d.get("train", {})),
            experiment_name=d.get("experiment_name", "experiment"),
        )

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "model": asdict(self.model),
                    "data": asdict(self.data),
                    "train": asdict(self.train),
                    "experiment_name": self.experiment_name,
                },
                f,
                indent=2,
            )
