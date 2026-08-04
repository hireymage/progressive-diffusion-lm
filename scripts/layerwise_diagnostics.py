#!/usr/bin/env python3
"""Offline diagnostics for the layer-wise seq256 experiment.

This script never builds data or contacts the network.  It operates only on a
verified local cache selected by :func:`scripts.layerwise_pilot.select_cache`.
Every invocation writes one self-contained JSON report.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.layerwise_pilot import select_cache
from src.config import LayerwiseModelConfig
from src.data import load_tokenizer
from src.layerwise_model import LayerwiseProgressiveLM, masked_deep_supervision_loss

MASK_RATES = (0.15, 0.30, 0.50, 0.75, 1.00)

# Each strategy sees exactly the same fixed 100 sequences and is evaluated on
# the same 50% mask.  They differ only in the corruption curriculum (and C's
# intentionally conservative learning rate).
QUALITY_GATE_STRATEGIES = {
    "A": {"schedule": "0.50:40000", "lr": 1e-3},
    "B": {"schedule": "0.15:12000,0.30:12000,0.50:16000", "lr": 1e-3},
    "C": {"schedule": "0.15:8000,0.30:8000,0.50:12000,0.75:12000", "lr": 7e-4},
}


def parse_mask_schedule(value: str) -> tuple[tuple[float, int], ...]:
    """Parse ``RATE:STEPS,...`` with no implicit or random schedule state."""
    try:
        parts = tuple((float(rate), int(steps)) for rate, steps in
                      (part.strip().split(":") for part in value.split(",")))
    except (TypeError, ValueError) as exc:
        raise ValueError("mask schedule must be RATE:STEPS[,RATE:STEPS...]") from exc
    if not parts or any(not 0 < rate <= 1 or steps <= 0 for rate, steps in parts):
        raise ValueError("schedule rates must be in (0, 1] and steps positive")
    return parts


def schedule_rate(schedule: tuple[tuple[float, int], ...], step: int) -> float:
    if step < 1:
        raise ValueError("step is one-indexed and must be positive")
    for rate, duration in schedule:
        if step <= duration:
            return rate
        step -= duration
    return schedule[-1][0]


def gate_streak(previous: int, accuracy: float, threshold: float) -> int:
    """Consecutive reports satisfying the gate (small, testable early stop)."""
    return previous + 1 if accuracy >= threshold else 0


def atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def deterministic_mask(shape: tuple[int, int], rate: float, seed: int) -> np.ndarray:
    """Return a reproducible non-empty Bernoulli mask for each sequence."""
    if not 0.0 < rate <= 1.0:
        raise ValueError("mask rate must be in (0, 1]")
    rng = np.random.RandomState(seed)
    mask = rng.random_sample(shape) < rate
    mask[:, 0] = True
    return mask


def corrupt(batch: np.ndarray, mask_id: int, rate: float, seed: int):
    mask = deterministic_mask(batch.shape, rate, seed)
    return (mx.array(np.where(mask, mask_id, batch), dtype=mx.int32),
            mx.array(batch, dtype=mx.int32), mx.array(mask))


def masked_metrics(logits: mx.array, targets: mx.array, mask: mx.array) -> dict:
    flat_logits = logits.reshape(-1, logits.shape[-1]).astype(mx.float32)
    truth = targets.reshape(-1)
    flat_mask = mask.reshape(-1).astype(mx.float32)
    n = mx.maximum(mx.sum(flat_mask), mx.array(1.0))
    ce = -nn.log_softmax(flat_logits, axis=-1)[mx.arange(flat_logits.shape[0]), truth]
    loss = mx.sum(ce * flat_mask) / n
    correct = mx.sum((mx.argmax(flat_logits, axis=-1) == truth) * flat_mask)
    mx.eval(loss, correct, n)
    return {"loss": float(loss), "accuracy": float(correct) / float(n), "masked_tokens": int(n)}


def aggregate_masked_metrics(rows: list[dict]) -> dict:
    """Combine per-batch masked means without giving short batches extra weight."""
    n = sum(row["masked_tokens"] for row in rows)
    if not n:
        return {"loss": 0.0, "accuracy": 0.0, "masked_tokens": 0}
    return {"loss": sum(row["loss"] * row["masked_tokens"] for row in rows) / n,
            "accuracy": sum(row["accuracy"] * row["masked_tokens"] for row in rows) / n,
            "masked_tokens": n}


def evaluate_in_chunks(model: LayerwiseProgressiveLM, targets_np: np.ndarray, mask_np: np.ndarray,
                       batch_size: int, capture_reconstructions: int = 0) -> tuple[dict, list[np.ndarray]]:
    """Evaluate without materialising logits for more than one small batch.

    ``capture_reconstructions`` is deliberately capped by callers at two; the
    rest of the fixed set contributes metrics only and is promptly released.
    """
    if batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    rows, reconstructions = [], []
    for start in range(0, len(targets_np), batch_size):
        target = targets_np[start:start + batch_size]
        mask = mask_np[start:start + batch_size]
        x = mx.array(np.where(mask, model.cfg.mask_token_id(), target), dtype=mx.int32)
        targets = mx.array(target, dtype=mx.int32)
        mask_mx = mx.array(mask)
        logits = model(x, exit_layer=model.cfg.n_layers)
        rows.append(masked_metrics(logits, targets, mask_mx))
        needed = capture_reconstructions - len(reconstructions)
        if needed > 0:
            predicted = np.array(mx.argmax(logits, axis=-1))[:needed]
            reconstructions.extend(np.where(mask[:needed], predicted, target[:needed]))
        # MLX is lazy: force the per-batch work before advancing so the next
        # iteration cannot retain one giant graph/output set.
        mx.eval(logits)
    return aggregate_masked_metrics(rows), reconstructions


def count_tokens(data: np.ndarray, vocab_size: int, chunk_rows: int = 4096) -> np.ndarray:
    counts = np.zeros(vocab_size, dtype=np.int64)
    for start in range(0, len(data), chunk_rows):
        ids = np.asarray(data[start:start + chunk_rows]).reshape(-1)
        if np.any((ids < 0) | (ids >= vocab_size)):
            raise ValueError("cache contains token ids outside tokenizer vocabulary")
        counts += np.bincount(ids, minlength=vocab_size)
    return counts


def frequency_summary(counts: np.ndarray, tokenizer, top_k: int) -> dict:
    total = int(counts.sum())
    probs = counts[counts > 0] / max(total, 1)
    order = np.argsort(-counts, kind="stable")[:top_k]
    top = [{"id": int(i), "token": tokenizer.decode([int(i)]), "count": int(counts[i]),
            "frequency": float(counts[i] / max(total, 1))} for i in order]
    return {"total_tokens": total, "unique_tokens": int(np.count_nonzero(counts)),
            "entropy_nats": float(-(probs * np.log(probs)).sum()),
            "perplexity": float(math.exp(-(probs * np.log(probs)).sum())),
            "top_tokens": top, "top1_id": int(order[0]), "top1_frequency": float(counts[order[0]] / max(total, 1))}


def build_model(a, vocab_size: int) -> LayerwiseProgressiveLM:
    cfg = LayerwiseModelConfig(vocab_size=vocab_size, d_model=a.d_model, d_ff=a.d_ff,
        n_heads=a.n_heads, n_layers=a.n_layers, min_exit_layer=a.n_layers,
        max_seq_len=256, layer_precisions=["fp32"] * a.n_layers)
    return LayerwiseProgressiveLM(cfg)


def load_weights_only(model: LayerwiseProgressiveLM, checkpoint: Path) -> None:
    if not checkpoint.exists():
        raise FileNotFoundError(f"mask-sweep requires an existing layer-wise FP32 checkpoint: {checkpoint}")
    payload = mx.load(str(checkpoint))
    weights = [(key, value) for key, value in payload.items() if not key.startswith("opt_")]
    try:
        model.load_weights(weights, strict=True)
    except TypeError:  # MLX versions before strict= support
        model.load_weights(weights)
    mx.eval(model.parameters())


def run_frequency(a, train, val, tokenizer) -> dict:
    train_info = frequency_summary(count_tokens(train, a.vocab_size), tokenizer, a.top_k)
    val_info = frequency_summary(count_tokens(val, a.vocab_size), tokenizer, a.top_k)
    # Predicting the train top-1 everywhere is intentionally evaluated against
    # val: this is the comparable collapsed-model baseline.
    top1 = train_info["top1_id"]
    val_top1_accuracy = float(np.mean(np.asarray(val) == top1))
    return {"mode": "frequency", "train": train_info, "val": val_info,
            "constant_train_top1": {"id": top1, "token": tokenizer.decode([top1]),
                                    "val_accuracy_all_positions": val_top1_accuracy,
                                    "note": "For position-independent masks, this equals masked accuracy in expectation."}}


def _overfit_checkpoint(model, optimizer, directory: Path, kind: str, metadata: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{kind}.npz"
    payload = dict(mlx.utils.tree_flatten(model.parameters()))
    payload.update({"opt_" + k: v for k, v in mlx.utils.tree_flatten(optimizer.state)})
    mx.savez(str(path), **payload)
    atomic_json_write(directory / f"{kind}.json", metadata)


def _load_overfit_checkpoint(model, optimizer, path: Path) -> dict:
    data = mx.load(str(path))
    model.load_weights([(k, v) for k, v in data.items() if not k.startswith("opt_")])
    opt = [(k.removeprefix("opt_"), v) for k, v in data.items() if k.startswith("opt_")]
    if not opt:
        raise ValueError("overfit checkpoint has no optimizer state")
    optimizer.state = mlx.utils.tree_unflatten(opt)
    mx.eval(model.parameters(), optimizer.state)
    return json.loads(path.with_suffix(".json").read_text())


def _overfit_loss(model, x, targets, mask, a):
    if a.auxiliary_loss == "final-only":
        exits, weights = (a.n_layers,), None
    else:
        exits, weights = tuple(layer for layer, _ in a.milestone_weights), tuple(weight for _, weight in a.milestone_weights)
    return masked_deep_supervision_loss(model, x, targets, mask, supervised_layers=exits, layer_weights=weights)


def run_overfit(a, train, tokenizer) -> dict:
    if len(train) < a.overfit_sequences:
        raise ValueError("cache has fewer rows than --overfit-sequences")
    fixed = np.asarray(train[:a.overfit_sequences], dtype=np.int32)
    mx.random.seed(a.seed)
    model = build_model(a, a.vocab_size)
    optimizer = optim.AdamW(learning_rate=a.lr)
    grad_fn = nn.value_and_grad(model, lambda m, x, t, mask: _overfit_loss(m, x, t, mask, a))
    checkpoint_dir = a.checkpoint_dir
    # Reserving two checkpoint files catches accidental long runs on a nearly
    # full volume before any model state is written.
    estimate = sum(np.asarray(v).nbytes for _, v in mlx.utils.tree_flatten(model.parameters())) * 4
    free = __import__("shutil").disk_usage(a.output.parent).free
    needed = int(a.min_free_gb * 1024**3) + 2 * estimate
    if free < needed:
        raise RuntimeError(f"disk budget guard: {free / 1024**3:.1f} GiB free, need {needed / 1024**3:.1f} GiB")
    history, start_step, streak, best_accuracy = [], 0, 0, -1.0
    if a.resume:
        resume_path = checkpoint_dir / "latest.npz"
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume requested but no checkpoint exists: {resume_path}")
        saved = _load_overfit_checkpoint(model, optimizer, resume_path)
        start_step, streak, best_accuracy = int(saved["step"]), int(saved["gate_streak"]), float(saved["best_accuracy"])
        history = saved.get("history", [])
        if saved["schedule"] != [[r, n] for r, n in a.mask_schedule]:
            raise ValueError("resume schedule differs from checkpoint metadata")
    start = time.time()
    for step in range(start_step + 1, a.steps + 1):
        # No random sampling: each contiguous batch cycles through the fixed 100.
        offset = ((step - 1) * a.batch_size) % len(fixed)
        indices = (np.arange(a.batch_size) + offset) % len(fixed)
        batch = fixed[indices]
        train_rate = schedule_rate(a.mask_schedule, step)
        x, targets, mask = corrupt(batch, model.cfg.mask_token_id(), train_rate, a.seed + step)
        loss, grads = grad_fn(model, x, targets, mask)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        if step == 1 or step % a.report_every == 0 or step == a.steps:
            # Fixed evaluation masks make the reported curve comparable.
            eval_mask = deterministic_mask(fixed.shape, a.gate_mask_rate, a.seed + 900_000)
            metrics, _ = evaluate_in_chunks(model, fixed, eval_mask, a.eval_batch_size)
            streak = gate_streak(streak, metrics["accuracy"], a.gate_accuracy)
            best_accuracy = max(best_accuracy, metrics["accuracy"])
            metrics.update({"step": step, "train_objective": float(loss), "train_mask_rate": train_rate,
                            "gate_streak": streak, "processed_tokens": step * a.batch_size * fixed.shape[1],
                            "exposures_per_sequence": step * a.batch_size / len(fixed)})
            history.append(metrics)
            metadata = {"step": step, "gate_streak": streak, "best_accuracy": best_accuracy,
                        "history": history, "schedule": [[r, n] for r, n in a.mask_schedule],
                        "seed": a.seed, "batch_size": a.batch_size, "overfit_sequences": a.overfit_sequences}
            _overfit_checkpoint(model, optimizer, checkpoint_dir, "latest", metadata)
            if metrics["accuracy"] >= best_accuracy:
                _overfit_checkpoint(model, optimizer, checkpoint_dir, "best", metadata)
            partial = {"mode": "overfit", "status": "running", "history": history, "latest": metrics,
                       "checkpoint_dir": str(checkpoint_dir)}
            atomic_json_write(a.output, partial)
            if streak >= a.gate_reports:
                break
    eval_mask = deterministic_mask(fixed.shape, a.gate_mask_rate, a.seed + 900_000)
    final, reconstructions = evaluate_in_chunks(model, fixed, eval_mask, a.eval_batch_size, capture_reconstructions=2)
    decode = lambda ids: tokenizer.decode([int(i) for i in ids])
    # The all-position value is derived from the masked aggregate: unmasked
    # positions are copied exactly by construction.
    final["reconstruction_accuracy_all_positions"] = float(
        (final["accuracy"] * final["masked_tokens"] + (fixed.size - final["masked_tokens"])) / fixed.size)
    final["quality_gate"] = {"threshold": a.gate_accuracy, "consecutive_reports_required": a.gate_reports,
        "consecutive_reports": streak, "passed": streak >= a.gate_reports,
        "meaning": "masked accuracy on a fixed mask over all fixed training sequences"}
    return {"mode": "overfit", "architecture": {"d_model": a.d_model, "d_ff": a.d_ff, "n_heads": a.n_heads, "n_layers": a.n_layers},
            "fixed_sequence_indices": list(range(a.overfit_sequences)), "masking": {"schedule": a.mask_schedule, "gate_rate": a.gate_mask_rate, "seed": a.seed + 900_000},
            "steps": history[-1]["step"] if history else start_step, "max_steps": a.steps, "batch_size": a.batch_size,
            "processed_tokens": (history[-1]["processed_tokens"] if history else start_step * a.batch_size * fixed.shape[1]),
            "exposures_per_sequence": (history[-1]["exposures_per_sequence"] if history else start_step * a.batch_size / len(fixed)),
            "checkpoint_policy": "latest+best including optimizer and metadata", "elapsed_seconds": time.time() - start,
            "history": history, "final": final,
            "examples": [{"target": decode(fixed[i]), "reconstruction": decode(reconstructions[i])} for i in range(len(reconstructions))]}


def run_mask_sweep(a, train, val, tokenizer) -> dict:
    model = build_model(a, a.vocab_size)
    load_weights_only(model, a.checkpoint)
    top1 = int(np.argmax(count_tokens(train, a.vocab_size)))
    rows = []
    # A fixed leading validation slice avoids random-batch variance across rates.
    sample = np.asarray(val[:min(a.eval_sequences, len(val))], dtype=np.int32)
    for index, rate in enumerate(MASK_RATES):
        mask_np = deterministic_mask(sample.shape, rate, a.seed + index)
        metrics, _ = evaluate_in_chunks(model, sample, mask_np, a.eval_batch_size)
        metrics.update({"mask_rate": rate, "constant_top1_id": top1,
                        "constant_top1_token": tokenizer.decode([top1]),
                        "constant_top1_accuracy": float(np.mean((sample == top1)[mask_np]))})
        rows.append(metrics)
    return {"mode": "mask-sweep", "checkpoint": str(a.checkpoint), "checkpoint_loaded": True,
            "eval_sequence_indices": list(range(len(sample))), "seed": a.seed, "rows": rows,
            "limits": "This measures one supplied FP32 layer-wise checkpoint; it does not evaluate random initialization or legacy diffusion checkpoints."}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("frequency", "overfit", "mask-sweep"))
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache")
    p.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer/wiki_bpe")
    p.add_argument("--output", type=Path, default=ROOT / "results/layerwise/diagnostics.json")
    p.add_argument("--seed", type=int, default=20260804); p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--d-model", type=int, default=64); p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4); p.add_argument("--n-layers", type=int, default=25)
    p.add_argument("--vocab-size", type=int, default=None, help="normally inferred from tokenizer")
    p.add_argument("--checkpoint", type=Path, help="FP32 layer-wise .npz required for mask-sweep")
    p.add_argument("--eval-sequences", type=int, default=32); p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--overfit-sequences", type=int, default=100); p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=4); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--overfit-mask-rate", type=float, default=.50); p.add_argument("--report-every", type=int, default=100)
    p.add_argument("--gate-accuracy", type=float, default=.95)
    p.add_argument("--mask-schedule", help="training curriculum, e.g. 0.15:12000,0.30:12000,0.50:16000")
    p.add_argument("--strategy", choices=tuple(QUALITY_GATE_STRATEGIES), help="bounded 100-sequence quality-gate preset A/B/C")
    p.add_argument("--gate-mask-rate", type=float, default=.50, help="fixed evaluation corruption rate")
    p.add_argument("--gate-reports", type=int, default=3, help="consecutive passing reports required to stop")
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "results/layerwise/overfit-checkpoints")
    p.add_argument("--resume", action="store_true"); p.add_argument("--min-free-gb", type=float, default=10.0)
    p.add_argument("--auxiliary-loss", choices=("final-only", "weighted-milestones"), default="final-only")
    p.add_argument("--milestone-weights", default="5:0.1,10:0.2,15:0.3,20:0.4,25:1.0")
    return p


def main() -> None:
    a = parser().parse_args()
    if a.steps < 1 or a.steps > 40000:
        raise ValueError("--steps must be in [1, 40000]")
    if a.strategy:
        preset = QUALITY_GATE_STRATEGIES[a.strategy]
        if a.mask_schedule is None:
            a.mask_schedule = preset["schedule"]
        # A user-provided LR remains authoritative for reproducibility.
        if a.lr == parser().get_default("lr"):
            a.lr = preset["lr"]
    a.mask_schedule = parse_mask_schedule(a.mask_schedule or f"{a.overfit_mask_rate}:{a.steps}")
    try:
        a.milestone_weights = tuple((int(layer), float(weight)) for layer, weight in
                                    (part.split(":") for part in a.milestone_weights.split(",")))
    except ValueError as exc:
        raise ValueError("--milestone-weights must be LAYER:WEIGHT[,LAYER:WEIGHT...]") from exc
    if a.auxiliary_loss == "weighted-milestones" and (not a.milestone_weights or
            any(layer < 1 or layer > a.n_layers or weight <= 0 for layer, weight in a.milestone_weights)):
        raise ValueError("milestone layers must be valid and weights positive")
    train, val, meta, meta_path = select_cache(a.cache_dir, seq_len=256)
    tokenizer = load_tokenizer(str(a.tokenizer))
    inferred_vocab = tokenizer.get_vocab_size()
    a.vocab_size = inferred_vocab if a.vocab_size is None else a.vocab_size
    if a.vocab_size != inferred_vocab:
        raise ValueError("--vocab-size must match the local tokenizer; checkpoint compatibility cannot be guessed")
    if a.mode == "mask-sweep" and a.checkpoint is None:
        raise ValueError("--checkpoint is required for mask-sweep (weights are never random)")
    result = {"offline_only": True, "cache_meta_path": str(meta_path), "cache_metadata": meta,
              "tokenizer": str(a.tokenizer), "seed": a.seed}
    if a.mode == "frequency": result["result"] = run_frequency(a, train, val, tokenizer)
    elif a.mode == "overfit": result["result"] = run_overfit(a, train, tokenizer)
    else: result["result"] = run_mask_sweep(a, train, val, tokenizer)
    atomic_json_write(a.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
