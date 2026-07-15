"""
Training loop for the progressive-precision diffusion language model.

Usage
-----
python -m src.train --config configs/progressive_1_2_4.json
python -m src.train --config configs/baseline.json
python -m src.train --config configs/smoke_test.json

The training loop:
  1. Build / load the dataset (cached numpy arrays).
  2. Initialise the model (DiffusionLM) and Adam optimiser.
  3. For each step:
       a. Sample a batch of token chunks.
       b. Compute masked-diffusion loss (which sets model bits internally).
       c. Compute gradients via MLX value_and_grad.
       d. Apply gradient clipping + Adam update.
       e. Log loss, eval on validation set periodically.
       f. Save checkpoint periodically and on finish.

Checkpoints
-----------
Saved as NPZ files in {checkpoint_dir}/{experiment_name}/step_{N}.npz
Metadata (config, step, best_val_loss) saved in checkpoint_dir/meta.json.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils

from .config import ExperimentConfig, ModelConfig
from .model import DiffusionLM
from .diffusion import compute_loss
from .data import build_and_cache_dataset, BatchIterator


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: DiffusionLM,
    optimizer: optim.Adam,
    step: int,
    val_loss: float,
    cfg: ExperimentConfig,
    ckpt_dir: Path,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"step_{step:07d}.npz"

    # Save model weights
    weights = dict(mlx.utils.tree_flatten(model.parameters()))
    # Save optimizer state
    opt_state = {}
    try:
        opt_flat = dict(mlx.utils.tree_flatten(optimizer.state))
        for k, v in opt_flat.items():
            if isinstance(v, mx.array):
                opt_state[f"opt_{k}"] = v
    except Exception:
        pass  # optimizer state may not be serialisable in all MLX versions

    all_arrays = {**weights, **opt_state}
    mx.savez(str(ckpt_path), **all_arrays)

    # Save metadata alongside
    meta = {
        "step": step,
        "val_loss": val_loss,
        "experiment_name": cfg.experiment_name,
    }
    meta_path = ckpt_dir / "latest_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return ckpt_path


def load_checkpoint(
    model: DiffusionLM,
    ckpt_path: str,
) -> int:
    """Load model weights from checkpoint. Returns the step number."""
    data = mx.load(ckpt_path)
    # Separate model weights from optimizer state
    model_weights = {k: v for k, v in data.items() if not k.startswith("opt_")}
    model.load_weights(list(model_weights.items()))
    mx.eval(model.parameters())

    # Try to read step from meta
    meta_path = Path(ckpt_path).parent / "latest_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("step", 0)
    # Infer from filename
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
    val_iter: BatchIterator,
    cfg: ExperimentConfig,
    n_steps: int = 50,
) -> dict:
    """Run evaluation on validation set.  Returns dict of metrics."""
    total_loss = 0.0
    total_correct = 0
    total_masked = 0

    mask_token_id = cfg.model.mask_token_id()
    precision_schedule = cfg.model.precision_schedule if cfg.model.model_type == "progressive" else [16] * cfg.model.n_diffusion_steps

    for i in range(n_steps):
        batch_np = next(val_iter)
        x0 = mx.array(batch_np, dtype=mx.int32)

        # Compute loss
        from .diffusion import corrupt_tokens, mask_rate_to_step
        mask_rates = mx.random.uniform(low=0.1, high=1.0, shape=(x0.shape[0],))
        x_t, mask = corrupt_tokens(x0, mask_rates, mask_token_id)
        mean_rate = float(mask_rates.mean())
        step_idx = mask_rate_to_step(mean_rate, cfg.model.n_diffusion_steps)
        bits = precision_schedule[step_idx]
        model.set_bits(bits)

        logits = model(x_t, mask_rates)
        mx.eval(logits)

        V = logits.shape[-1]
        flat_logits = logits.reshape(-1, V)
        flat_targets = x0.reshape(-1)
        flat_mask = mask.reshape(-1).astype(mx.float32)

        log_probs = nn.log_softmax(flat_logits, axis=-1)
        token_loss = -log_probs[mx.arange(flat_logits.shape[0]), flat_targets]
        n_masked = flat_mask.sum()
        loss = (token_loss * flat_mask).sum() / mx.maximum(n_masked, mx.array(1.0))

        # Accuracy at masked positions
        preds = flat_logits.argmax(axis=-1)
        correct = ((preds == flat_targets) * flat_mask).sum()

        mx.eval(loss, correct, n_masked)
        total_loss += float(loss)
        total_correct += float(correct)
        total_masked += float(n_masked)

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


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: ExperimentConfig) -> None:
    print("=" * 60)
    print(f"Experiment: {cfg.experiment_name}")
    print(f"Model type: {cfg.model.model_type}")
    print(f"Precision schedule: {cfg.model.precision_schedule}")
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
        dataset_config=cfg.data.dataset_config,
        train_split=cfg.data.train_split,
        seed=cfg.train.seed,
    )

    train_iter = BatchIterator(train_data, cfg.train.batch_size, seed=cfg.train.seed)
    val_iter = BatchIterator(val_data, cfg.train.batch_size, seed=cfg.train.seed + 1)

    # ---- Model -------------------------------------------------------------
    model = DiffusionLM(cfg.model)
    n_params = model.total_params()
    print(f"\nModel parameters: {n_params:,}")
    approx_mb = n_params * 4 / 1024 / 1024
    print(f"Approx model size (float32): {approx_mb:.1f} MB")
    print(f"Approx memory with Adam states: {approx_mb * 3:.1f} MB\n")

    mx.eval(model.parameters())

    # ---- Optimiser ---------------------------------------------------------
    optimizer = optim.Adam(
        learning_rate=cfg.train.learning_rate,
        betas=[0.9, 0.95],
    )

    mask_token_id = cfg.model.mask_token_id()
    precision_schedule = (
        cfg.model.precision_schedule
        if cfg.model.model_type == "progressive"
        else [16] * cfg.model.n_diffusion_steps
    )

    # ---- Resume from checkpoint -------------------------------------------
    start_step = 0
    ckpt_dir = Path(cfg.train.checkpoint_dir) / cfg.experiment_name
    if cfg.train.resume_from:
        print(f"Resuming from {cfg.train.resume_from}")
        start_step = load_checkpoint(model, cfg.train.resume_from)
        print(f"  Resumed at step {start_step}")
    else:
        # Check for latest checkpoint
        meta_path = ckpt_dir / "latest_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            latest_step = meta.get("step", 0)
            latest_ckpt = ckpt_dir / f"step_{latest_step:07d}.npz"
            if latest_ckpt.exists():
                print(f"Auto-resuming from {latest_ckpt}")
                start_step = load_checkpoint(model, str(latest_ckpt))
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
    t0 = time.time()

    print(f"Starting training from step {start_step} to {cfg.train.max_steps}")

    for step in range(start_step, cfg.train.max_steps):
        # Update learning rate
        lr = lr_schedule(step, cfg.train.warmup_steps, cfg.train.max_steps, cfg.train.learning_rate)
        optimizer.learning_rate = lr

        # Fetch batch
        batch_np = next(train_iter)
        x0 = mx.array(batch_np, dtype=mx.int32)

        # Forward + backward
        loss, grads = loss_and_grad(model, x0)

        # Gradient clipping
        grads, total_norm = optim.clip_grad_norm(grads, cfg.train.grad_clip)

        # Optimiser update
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())

        running_loss += float(loss)

        # ---- Logging -------------------------------------------------------
        if (step + 1) % cfg.train.log_every == 0:
            elapsed = time.time() - t0
            avg_loss = running_loss / cfg.train.log_every
            bits_now = model.get_current_bits()
            print(
                f"Step {step+1:6d}/{cfg.train.max_steps} | "
                f"loss={avg_loss:.4f} | bits={bits_now} | "
                f"lr={lr:.2e} | {elapsed:.1f}s"
            )
            running_loss = 0.0
            t0 = time.time()

        # ---- Evaluation ----------------------------------------------------
        if (step + 1) % cfg.train.eval_every == 0:
            metrics = evaluate(model, val_iter, cfg, n_steps=cfg.train.eval_steps)
            val_loss = metrics["val_loss"]
            val_acc = metrics["val_accuracy"]
            print(
                f"  [EVAL] val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = save_checkpoint(model, optimizer, step + 1, val_loss, cfg, ckpt_dir)
                print(f"  [BEST] Saved best model to {best_path}")

        # ---- Checkpointing -------------------------------------------------
        if (step + 1) % cfg.train.checkpoint_every == 0:
            ckpt_path = save_checkpoint(model, optimizer, step + 1, float("inf"), cfg, ckpt_dir)
            print(f"  [CKPT] Saved checkpoint to {ckpt_path}")

    # Final checkpoint
    ckpt_path = save_checkpoint(model, optimizer, cfg.train.max_steps, best_val_loss, cfg, ckpt_dir)
    print(f"\nTraining complete.  Final checkpoint: {ckpt_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")


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
