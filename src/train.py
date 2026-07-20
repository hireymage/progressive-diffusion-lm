"""
Training loop for the progressive-precision diffusion language model.

Usage
-----
python -m src.train --config configs/progressive_1_2_4.json
python -m src.train --config configs/baseline.json
python -m src.train --config configs/smoke_test.json

The training loop:
  1. Build / load the dataset (cached numpy arrays).
  2. Initialise the model (DiffusionLM) and AdamW optimiser.
  3. For each step:
       a. Sample a batch of token chunks.
       b. Compute masked-diffusion loss (which sets model bits internally).
       c. Compute gradients via MLX value_and_grad.
       d. Apply gradient clipping + Adam update.
       e. Log loss, eval on validation set periodically.
       f. Save checkpoint periodically and on finish.

Metrics output
--------------
Results are saved to {results_dir}/{experiment_name}/:
  train_metrics.csv   — step, train_loss, bits_used, lr, elapsed_s
  eval_history.json   — list of {step, val_loss, val_accuracy, bits} dicts
  final_summary.json  — config, storage report, best val loss, total time

Checkpoints
-----------
Saved as NPZ files in {checkpoint_dir}/{experiment_name}/step_{N}.npz.
Each checkpoint has step_{N}.json metadata; latest_meta.json is only the latest pointer.
The NPZ contains model and optimizer state. Restart is warm, not bit-exact,
because iterator/RNG/logger state is not restored.
"""

import csv
import os
import sys
import json
import time
import argparse
import dataclasses
import math
import numpy as np
from pathlib import Path

import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils

from .config import ExperimentConfig, ModelConfig, TrainConfig
from .model import DiffusionLM
from .diffusion import compute_loss
from .data import build_and_cache_dataset, BatchIterator
from .quantization import model_storage_report, quantization_noise_metrics


# ---------------------------------------------------------------------------
# Metrics logger
# ---------------------------------------------------------------------------

class MetricsLogger:
    """Write finite, machine-readable training/evaluation artifacts."""

    TRAIN_FIELDS = [
        "step", "train_loss", "gradient_norm", "q1_residual_rms",
        "injected_noise_rms", "bits_used", "lr", "elapsed_s",
    ]

    def __init__(self, results_dir: Path, experiment_name: str):
        self.dir = results_dir / experiment_name
        self.dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.dir / "train_metrics.csv"
        self.eval_path = self.dir / "eval_history.json"
        self.final_path = self.dir / "final_summary.json"
        self._eval_history: list = []

        self._csv_file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self.TRAIN_FIELDS)
        self._writer.writeheader()

    @staticmethod
    def _require_finite(**values: float) -> None:
        nonfinite = [name for name, value in values.items() if not math.isfinite(float(value))]
        if nonfinite:
            raise ValueError(f"metrics must be finite; invalid fields: {', '.join(nonfinite)}")

    def log_train(
        self,
        step: int,
        loss: float,
        gradient_norm: float,
        q1_residual_rms: float,
        injected_noise_rms: float,
        bits: int,
        lr: float,
        elapsed: float,
    ) -> None:
        self._require_finite(
            train_loss=loss,
            gradient_norm=gradient_norm,
            q1_residual_rms=q1_residual_rms,
            injected_noise_rms=injected_noise_rms,
            lr=lr,
            elapsed_s=elapsed,
        )
        self._writer.writerow({
            "step": step,
            "train_loss": round(loss, 6),
            "gradient_norm": round(gradient_norm, 6),
            "q1_residual_rms": round(q1_residual_rms, 8),
            "injected_noise_rms": round(injected_noise_rms, 8),
            "bits_used": bits,
            "lr": round(lr, 8),
            "elapsed_s": round(elapsed, 2),
        })
        self._csv_file.flush()

    def log_eval(
        self,
        step: int,
        val_loss: float,
        val_acc: float,
        bits: int,
        train_loss: float,
    ) -> None:
        val_perplexity = math.exp(val_loss)
        generalization_gap = val_loss - train_loss
        self._require_finite(
            val_loss=val_loss,
            val_perplexity=val_perplexity,
            val_accuracy=val_acc,
            generalization_gap=generalization_gap,
            train_loss=train_loss,
        )
        record = {
            "step": step,
            "val_loss": round(val_loss, 6),
            "val_perplexity": round(val_perplexity, 6),
            "val_accuracy": round(val_acc, 6),
            "generalization_gap": round(generalization_gap, 6),
            "train_loss": round(train_loss, 6),
            "bits_used": bits,
        }
        self._eval_history.append(record)
        with open(self.eval_path, "w") as f:
            json.dump(self._eval_history, f, indent=2)

    def save_final(
        self,
        cfg: ExperimentConfig,
        storage_report: dict,
        best_val_loss: float,
        total_seconds: float,
    ) -> None:
        self._require_finite(best_val_loss=best_val_loss, total_training_seconds=total_seconds)
        summary = {
            "experiment_name": cfg.experiment_name,
            "model_type": cfg.model.model_type,
            "weight_noise_mode": cfg.model.weight_noise_mode,
            "seed": cfg.train.seed,
            "precision_schedule": cfg.model.precision_schedule,
            "tie_word_embeddings": cfg.model.tie_word_embeddings,
            "max_steps": cfg.train.max_steps,
            "best_val_loss": round(best_val_loss, 6),
            "total_training_seconds": round(total_seconds, 1),
            "storage": storage_report,
            "config": {
                "model": dataclasses.asdict(cfg.model),
                "train": dataclasses.asdict(cfg.train),
                "data": dataclasses.asdict(cfg.data),
            },
        }
        with open(self.final_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  [METRICS] Final summary saved to {self.final_path}")

    def close(self) -> None:
        self._csv_file.close()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: DiffusionLM,
    optimizer: optim.Optimizer,
    step: int,
    val_loss: float,
    cfg: ExperimentConfig,
    ckpt_dir: Path,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"step_{step:07d}.npz"

    weights = dict(mlx.utils.tree_flatten(model.parameters()))
    opt_state = {}
    for k, v in mlx.utils.tree_flatten(optimizer.state):
        if not isinstance(v, mx.array):
            raise TypeError(f"Optimizer state leaf {k!r} is not an MLX array")
        opt_state[f"opt_{k}"] = v
    if not opt_state:
        raise RuntimeError("Optimizer state is empty")

    mx.savez(str(ckpt_path), **{**weights, **opt_state})

    meta = {
        "step": step,
        "val_loss": val_loss,
        "experiment_name": cfg.experiment_name,
    }
    checkpoint_meta_path = ckpt_path.with_suffix(".json")
    with open(checkpoint_meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    with open(ckpt_dir / "latest_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return ckpt_path


def load_checkpoint(
    model: DiffusionLM,
    ckpt_path: str,
    optimizer: optim.Optimizer | None = None,
) -> int:
    """Load model and optional optimizer state; return this checkpoint's step."""
    data = mx.load(ckpt_path)
    model_weights = {k: v for k, v in data.items() if not k.startswith("opt_")}
    model.load_weights(list(model_weights.items()))
    mx.eval(model.parameters())

    if optimizer is not None:
        flat_opt_state = [
            (k.removeprefix("opt_"), v)
            for k, v in data.items()
            if k.startswith("opt_")
        ]
        if not flat_opt_state:
            raise ValueError(f"Checkpoint {ckpt_path} has no optimizer state")
        optimizer.state = mlx.utils.tree_unflatten(flat_opt_state)
        mx.eval(optimizer.state)

    checkpoint_meta_path = Path(ckpt_path).with_suffix(".json")
    if checkpoint_meta_path.exists():
        with open(checkpoint_meta_path) as f:
            meta = json.load(f)
        return meta.get("step", 0)
    stem = Path(ckpt_path).stem
    if stem.startswith("step_"):
        try:
            return int(stem.split("_")[1])
        except ValueError:
            pass
    return 0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: DiffusionLM,
    val_iter: "BatchIterator",
    cfg: ExperimentConfig,
    n_steps: int = 50,
) -> dict:
    """Evaluate with training-only noise disabled, restoring caller mode."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0

    mask_token_id = cfg.model.mask_token_id()
    precision_schedule = (
        cfg.model.precision_schedule
        if cfg.model.model_type == "progressive"
        else [16] * cfg.model.n_diffusion_steps
    )

    try:
        for _ in range(n_steps):
            batch_np = next(val_iter)
            x0 = mx.array(batch_np, dtype=mx.int32)

            from .diffusion import corrupt_tokens, mask_rate_to_step
            mask_rates = mx.random.uniform(low=0.1, high=1.0, shape=(x0.shape[0],))
            x_t, mask = corrupt_tokens(x0, mask_rates, mask_token_id)
            mean_rate = float(mask_rates.mean())
            step_idx = mask_rate_to_step(mean_rate, cfg.model.n_diffusion_steps)
            bits = precision_schedule[step_idx]
            model.set_bits(bits)

            logits = model(x_t, mask_rates)
            mx.eval(logits)

            vocab_size = logits.shape[-1]
            flat_logits = logits.reshape(-1, vocab_size)
            flat_targets = x0.reshape(-1)
            flat_mask = mask.reshape(-1).astype(mx.float32)

            log_probs = nn.log_softmax(flat_logits, axis=-1)
            token_loss = -log_probs[mx.arange(flat_logits.shape[0]), flat_targets]
            n_masked = flat_mask.sum()
            loss = (token_loss * flat_mask).sum() / mx.maximum(n_masked, mx.array(1.0))

            preds = flat_logits.argmax(axis=-1)
            correct = ((preds == flat_targets) * flat_mask).sum()

            mx.eval(loss, correct, n_masked)
            total_loss += float(loss)
            total_correct += float(correct)
            total_masked += float(n_masked)
    finally:
        model.train(was_training)

    return {
        "val_loss": total_loss / n_steps,
        "val_accuracy": total_correct / max(total_masked, 1),
    }


# ---------------------------------------------------------------------------
# Learning rate schedule (linear warmup + cosine decay)
# ---------------------------------------------------------------------------

def lr_schedule(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))


def build_optimizer(cfg: TrainConfig) -> optim.AdamW:
    """Build the configured decoupled-weight-decay optimizer."""
    return optim.AdamW(
        learning_rate=cfg.learning_rate,
        betas=[0.9, 0.95],
        weight_decay=cfg.weight_decay,
    )


# ---------------------------------------------------------------------------
# Storage report printer
# ---------------------------------------------------------------------------

def print_storage_report(report: dict, model_type: str, schedule: list) -> None:
    print("\n── Storage & Parameter Report ────────────────────────────────")
    print(f"  Total parameters:          {report['total_params']:>12,}")
    print(f"  QuantizedLinear weights:   {report['q_linear_weight_params']:>12,}")
    print(f"  Non-quantized params:      {report['non_quantized_params']:>12,}")
    print(f"  Precision schedule:        {schedule}  (model_type={model_type})")
    print(f"  Average step weight bits:  {report['average_step_weight_bits']:.3f} bits")
    print()
    print(f"  Actual model storage:      {report['actual_model_mb']:>8.1f} MB  "
          f"(FP32 master/checkpoint weights)")
    print(f"  BF16 storage (hyp.):       {report['bf16_total_mb']:>8.1f} MB")
    if report["hypothetical_packed_mb"] is not None:
        print(f"  Packed lower bound:        {report['hypothetical_packed_mb']:>8.1f} MB  "
              f"(constant schedule only; includes FP32 row scales)")
        print(f"  Hyp. compression vs FP32:  "
              f"{report['hypothetical_packed_compression_vs_fp32']:.1f}×")
    else:
        print("  Packed lower bound:             n/a  "
              "(progressive schedule needs a deployment representation contract)")
    print(f"  Actual compression vs FP32: {report['actual_compression_vs_fp32']:.1f}×")
    print()
    print(f"  Training memory est.:      {report['training_memory_estimate_mb']:.1f} MB  "
          f"(master + grads + Adam m/v, all FP32)")
    print("── Note: quantized ops are SIMULATED in FP32 via STE ─────────")
    print()


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: ExperimentConfig) -> None:
    print("=" * 60)
    print(f"Experiment: {cfg.experiment_name}")
    print(f"Model type: {cfg.model.model_type}")
    print(f"Precision schedule: {cfg.model.precision_schedule}")
    print(f"Weight tying: {cfg.model.tie_word_embeddings}")
    print("=" * 60)

    mx.random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    # ---- Dataset -----------------------------------------------------------
    train_data, val_data = build_and_cache_dataset(
        tokenizer_path=cfg.data.tokenizer_path,
        cache_dir=cfg.data.data_cache_dir,
        seq_len=cfg.data.seq_len,
        max_articles=cfg.data.max_articles,
        max_text_bytes=cfg.data.max_text_bytes,
        dataset_name=cfg.data.dataset_name,
        dataset_config=cfg.data.dataset_config,
        dataset_revision=cfg.data.dataset_revision,
        train_split=cfg.data.train_split,
        seed=cfg.train.seed,
    )

    train_iter = BatchIterator(train_data, cfg.train.batch_size, seed=cfg.train.seed)
    val_iter = BatchIterator(val_data, cfg.train.batch_size, seed=cfg.train.seed + 1)

    # ---- Model -------------------------------------------------------------
    model = DiffusionLM(cfg.model)
    mx.eval(model.parameters())
    # Training-only noise uses deterministic independent per-layer streams.
    model.set_weight_noise_seed(cfg.train.seed)
    model.train()

    precision_schedule = (
        cfg.model.precision_schedule
        if cfg.model.model_type == "progressive"
        else [16] * cfg.model.n_diffusion_steps
    )

    storage = model_storage_report(model, precision_schedule)
    print_storage_report(storage, cfg.model.model_type, precision_schedule)

    # ---- Metrics logger ----------------------------------------------------
    results_dir = Path(cfg.train.results_dir)
    logger = MetricsLogger(results_dir, cfg.experiment_name)

    # ---- Optimiser ---------------------------------------------------------
    optimizer = build_optimizer(cfg.train)

    mask_token_id = cfg.model.mask_token_id()

    # ---- Resume from checkpoint -------------------------------------------
    start_step = 0
    ckpt_dir = Path(cfg.train.checkpoint_dir) / cfg.experiment_name
    if cfg.train.resume_from:
        print(f"Resuming from {cfg.train.resume_from}")
        start_step = load_checkpoint(model, cfg.train.resume_from, optimizer=optimizer)
        print(f"  Resumed at step {start_step}")
    else:
        meta_path = ckpt_dir / "latest_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            latest_step = meta.get("step", 0)
            latest_ckpt = ckpt_dir / f"step_{latest_step:07d}.npz"
            if latest_ckpt.exists():
                print(f"Auto-resuming from {latest_ckpt}")
                start_step = load_checkpoint(model, str(latest_ckpt), optimizer=optimizer)
                print(f"  Resumed at step {start_step}")

    # ---- Loss function for MLX value_and_grad ------------------------------
    def loss_fn(model, batch_x0):
        return compute_loss(
            model,
            batch_x0,
            mask_token_id,
            precision_schedule,
            cfg.model.n_diffusion_steps,
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # ---- Training loop -----------------------------------------------------
    best_val_loss = float("inf")
    running_loss = 0.0
    running_gradient_norm = 0.0
    running_q1_residual_rms = 0.0
    running_injected_noise_rms = 0.0
    latest_logged_train_loss: float | None = None
    last_step_loss: float | None = None
    t0 = time.time()
    train_start = time.time()

    print(f"Starting training from step {start_step} to {cfg.train.max_steps}")

    for step in range(start_step, cfg.train.max_steps):
        lr = lr_schedule(step, cfg.train.warmup_steps, cfg.train.max_steps, cfg.train.learning_rate)
        optimizer.learning_rate = lr

        batch_np = next(train_iter)
        x0 = mx.array(batch_np, dtype=mx.int32)

        loss, grads = loss_and_grad(model, x0)
        # ``clip_grad_norm`` returns the global L2 norm before clipping.
        grads, pre_clip_gradient_norm = optim.clip_grad_norm(grads, cfg.train.grad_clip)
        optimizer.update(model, grads)
        mx.eval(loss, pre_clip_gradient_norm, model.parameters())

        running_loss += float(loss)
        last_step_loss = float(loss)
        noise_metrics = quantization_noise_metrics(model)
        running_gradient_norm += float(pre_clip_gradient_norm)
        running_q1_residual_rms += noise_metrics["q1_residual_rms"]
        running_injected_noise_rms += noise_metrics["injected_noise_rms"]

        # ---- Logging -------------------------------------------------------
        if (step + 1) % cfg.train.log_every == 0:
            elapsed = time.time() - t0
            avg_loss = running_loss / cfg.train.log_every
            avg_gradient_norm = running_gradient_norm / cfg.train.log_every
            avg_q1_residual_rms = running_q1_residual_rms / cfg.train.log_every
            avg_injected_noise_rms = running_injected_noise_rms / cfg.train.log_every
            latest_logged_train_loss = avg_loss
            bits_now = model.get_current_bits()
            print(
                f"Step {step+1:6d}/{cfg.train.max_steps} | "
                f"loss={avg_loss:.4f} | bits={bits_now} | "
                f"lr={lr:.2e} | {elapsed:.1f}s"
            )
            logger.log_train(
                step + 1, avg_loss, avg_gradient_norm, avg_q1_residual_rms,
                avg_injected_noise_rms, bits_now, lr, elapsed,
            )
            running_loss = 0.0
            running_gradient_norm = 0.0
            running_q1_residual_rms = 0.0
            running_injected_noise_rms = 0.0
            t0 = time.time()

        # ---- Evaluation ----------------------------------------------------
        if (step + 1) % cfg.train.eval_every == 0:
            if latest_logged_train_loss is None:
                raise ValueError(
                    "evaluation requires a comparable logged train loss; "
                    "set log_every <= eval_every"
                )
            metrics = evaluate(model, val_iter, cfg, n_steps=cfg.train.eval_steps)
            val_loss = metrics["val_loss"]
            val_acc = metrics["val_accuracy"]
            bits_now = model.get_current_bits()
            print(
                f"  [EVAL] val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )
            logger.log_eval(
                step + 1, val_loss, val_acc, bits_now, latest_logged_train_loss
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if cfg.train.save_checkpoints:
                    best_path = save_checkpoint(model, optimizer, step + 1, val_loss, cfg, ckpt_dir)
                    print(f"  [BEST] Saved best model to {best_path}")
                else:
                    print(f"  [BEST] val_loss={val_loss:.4f} (checkpoints disabled)")

        # ---- Checkpointing -------------------------------------------------
        if cfg.train.save_checkpoints and (step + 1) % cfg.train.checkpoint_every == 0:
            ckpt_path = save_checkpoint(model, optimizer, step + 1, float("inf"), cfg, ckpt_dir)
            print(f"  [CKPT] Saved checkpoint to {ckpt_path}")

    # Ensure every completed run has at least one finite validation metric, even
    # when max_steps is not divisible by eval_every.
    if not math.isfinite(best_val_loss):
        metrics = evaluate(model, val_iter, cfg, n_steps=cfg.train.eval_steps)
        best_val_loss = metrics["val_loss"]
        train_loss_for_gap = latest_logged_train_loss
        if train_loss_for_gap is None:
            train_loss_for_gap = last_step_loss
        if train_loss_for_gap is not None:
            logger.log_eval(
                cfg.train.max_steps,
                best_val_loss,
                metrics["val_accuracy"],
                model.get_current_bits(),
                train_loss_for_gap,
            )
        print(
            f"  [FINAL EVAL] val_loss={best_val_loss:.4f}  "
            f"val_acc={metrics['val_accuracy']:.4f}"
        )

    # Final checkpoint and summary
    if cfg.train.save_checkpoints:
        ckpt_path = save_checkpoint(model, optimizer, cfg.train.max_steps, best_val_loss, cfg, ckpt_dir)
    else:
        ckpt_path = Path("(checkpoints disabled)")
    total_time = time.time() - train_start
    logger.save_final(cfg, storage, best_val_loss, total_time)
    logger.close()

    print(f"\nTraining complete.  Final checkpoint: {ckpt_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total training time: {total_time:.1f}s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train progressive diffusion LM")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_json(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
