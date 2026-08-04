"""Configuration for the progressive diffusion LM experiment.

Precision schedule notes
-------------------------
bits=1  — binary {-1, +1}                       2 levels, 1.0 eff. bits
bits=2  — true 2-bit {-3,-1,+1,+3}×step        4 levels, 2.0 eff. bits
bits=3  — true 3-bit odd levels × step          8 non-zero levels, 3.0 eff. bits
bits=4  — true 4-bit odd levels × step         16 non-zero levels, 4.0 eff. bits
bits=8  — symmetric 8-bit (256 levels)           8.0 eff. bits
bits=16 — no quantisation (full-precision baseline)
bits=0  — optional ternary {-1,0,+1}×scale      3 levels, ~1.585 eff. bits
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class ModelConfig:
    # Vocabulary
    vocab_size: int = 16000
    pad_token_id: int = 0

    # Architecture
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 256
    dropout: float = 0.0

    # Weight tying: share token_embedding weights with the LM head projection.
    # Reduces parameters by vocab_size × d_model (e.g., ~8M for 16K vocab / 512 dim).
    tie_word_embeddings: bool = True

    # Diffusion
    n_diffusion_steps: int = 8
    # bits per refinement step; length == n_diffusion_steps.
    # step 0 = coarsest (highest mask rate), step T-1 = finest.
    # bits=3 is true Q3; use bits=0 only for optional ternary experiments.
    precision_schedule: List[int] = field(
        default_factory=lambda: [1, 1, 1, 1, 2, 2, 4, 4]
    )

    # "baseline"    — bits=16 at every step (no quantisation)
    # "progressive" — bits from precision_schedule
    model_type: str = "progressive"

    # Training-only FP32 weight-noise control.  The default is an exact no-op.
    weight_noise_mode: str = "none"
    weight_noise_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.n_diffusion_steps <= 0:
            raise ValueError("n_diffusion_steps must be positive")
        if len(self.precision_schedule) != self.n_diffusion_steps:
            raise ValueError(
                "precision_schedule length must equal n_diffusion_steps"
            )
        unsupported = sorted(set(self.precision_schedule) - {0, 1, 2, 3, 4, 8, 16})
        if unsupported:
            raise ValueError(f"unsupported precision values: {unsupported}")
        if self.model_type not in {"baseline", "progressive"}:
            raise ValueError("model_type must be 'baseline' or 'progressive'")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.d_model <= 0 or self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be positive and divisible by n_heads")
        allowed = {"none", "gaussian_matched", "uniform_matched"}
        if self.weight_noise_mode not in allowed:
            raise ValueError(
                f"weight_noise_mode must be one of {sorted(allowed)}, "
                f"got {self.weight_noise_mode!r}"
            )
        if self.weight_noise_multiplier < 0:
            raise ValueError("weight_noise_multiplier must be nonnegative")

    def mask_token_id(self) -> int:
        """MASK token ID is one past the regular vocabulary."""
        return self.vocab_size


LAYERWISE_PRECISIONS = ("q1", "q2", "q4", "q8", "fp16")


@dataclass
class LayerwiseModelConfig:
    """Configuration for the separate layer-wise progressive LM prototype.

    This is intentionally independent from :class:`ModelConfig`: legacy
    ``bits=16`` means the FP32 identity path and must not be reinterpreted as
    FP16.  The new prototype uses explicit string precision names instead.
    """

    vocab_size: int = 256
    pad_token_id: int = 0
    d_model: int = 256
    n_layers: int = 25
    n_heads: int = 8
    d_ff: int = 1024
    max_seq_len: int = 64
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    min_exit_layer: int = 5
    layer_precisions: List[str] = field(
        default_factory=lambda: ["q1"] * 5 + ["q2"] * 5 + ["q4"] * 5
        + ["q8"] * 5 + ["fp16"] * 5
    )

    def __post_init__(self) -> None:
        if self.n_layers <= 0 or len(self.layer_precisions) != self.n_layers:
            raise ValueError("layer_precisions length must equal positive n_layers")
        unsupported = sorted(set(self.layer_precisions) - set(LAYERWISE_PRECISIONS))
        if unsupported:
            raise ValueError(f"unsupported layer precisions: {unsupported}")
        if not 1 <= self.min_exit_layer <= self.n_layers:
            raise ValueError("min_exit_layer must be in [1, n_layers]")
        if self.d_model <= 0 or self.n_heads <= 0 or self.d_model % self.n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if self.d_ff <= 0 or self.max_seq_len <= 0 or self.vocab_size <= 1:
            raise ValueError("vocab_size, d_ff, and max_seq_len must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def mask_token_id(self) -> int:
        return self.vocab_size


@dataclass
class DataConfig:
    dataset_name: str = "wikimedia/wikipedia"
    dataset_config: str = "20231101.en"
    dataset_revision: Optional[str] = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
    max_articles: int = 1000
    max_text_bytes: int = 10_000_000   # 10 MB
    seq_len: int = 256
    tokenizer_path: str = "tokenizer/wiki_bpe"
    data_cache_dir: str = "data/cache"
    train_split: float = 0.95

    def __post_init__(self) -> None:
        if not 0.0 < self.train_split < 1.0:
            raise ValueError("train_split must be between 0 and 1")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.max_articles <= 0 or self.max_text_bytes <= 0:
            raise ValueError("dataset collection limits must be positive")


@dataclass
class TrainConfig:
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    max_steps: int = 10_000
    warmup_steps: int = 500
    grad_clip: float = 1.0

    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 500
    resume_from: Optional[str] = None

    eval_every: int = 250
    eval_steps: int = 50

    log_every: int = 10

    # Results directory — training metrics CSV and final JSON are saved here.
    results_dir: str = "results"

    # Set False to skip all checkpoint saves (metric-only runs, e.g. ablation study).
    # Metrics CSV/JSON are always saved regardless of this flag.
    save_checkpoints: bool = True

    seed: int = 42

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be nonnegative")
        for name, value in {
            "checkpoint_every": self.checkpoint_every,
            "eval_every": self.eval_every,
            "eval_steps": self.eval_steps,
            "log_every": self.log_every,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")


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
