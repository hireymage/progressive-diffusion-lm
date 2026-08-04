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
import sys
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


def run_overfit(a, train, tokenizer) -> dict:
    if len(train) < a.overfit_sequences:
        raise ValueError("cache has fewer rows than --overfit-sequences")
    fixed = np.asarray(train[:a.overfit_sequences], dtype=np.int32)
    mx.random.seed(a.seed)
    model = build_model(a, a.vocab_size)
    optimizer = optim.AdamW(learning_rate=a.lr)
    grad_fn = nn.value_and_grad(model, lambda m, x, t, mask:
        masked_deep_supervision_loss(m, x, t, mask, supervised_layers=(a.n_layers,)))
    history = []
    start = time.time()
    for step in range(1, a.steps + 1):
        # No random sampling: each contiguous batch cycles through the fixed 100.
        offset = ((step - 1) * a.batch_size) % len(fixed)
        indices = (np.arange(a.batch_size) + offset) % len(fixed)
        batch = fixed[indices]
        x, targets, mask = corrupt(batch, model.cfg.mask_token_id(), a.overfit_mask_rate, a.seed + step)
        loss, grads = grad_fn(model, x, targets, mask)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        if step == 1 or step % a.report_every == 0 or step == a.steps:
            # Fixed evaluation masks make the reported curve comparable.
            eval_mask = deterministic_mask(fixed.shape, a.overfit_mask_rate, a.seed + 900_000)
            metrics, _ = evaluate_in_chunks(model, fixed, eval_mask, a.eval_batch_size)
            metrics.update({"step": step, "train_objective": float(loss)})
            history.append(metrics)
    eval_mask = deterministic_mask(fixed.shape, a.overfit_mask_rate, a.seed + 900_000)
    final, reconstructions = evaluate_in_chunks(model, fixed, eval_mask, a.eval_batch_size, capture_reconstructions=2)
    decode = lambda ids: tokenizer.decode([int(i) for i in ids])
    # The all-position value is derived from the masked aggregate: unmasked
    # positions are copied exactly by construction.
    final["reconstruction_accuracy_all_positions"] = float(
        (final["accuracy"] * final["masked_tokens"] + (fixed.size - final["masked_tokens"])) / fixed.size)
    final["quality_gate"] = {"threshold": a.gate_accuracy, "passed": final["accuracy"] >= a.gate_accuracy,
        "meaning": "masked accuracy on a fixed mask over all fixed training sequences"}
    return {"mode": "overfit", "architecture": {"d_model": a.d_model, "d_ff": a.d_ff, "n_heads": a.n_heads, "n_layers": a.n_layers},
            "fixed_sequence_indices": list(range(a.overfit_sequences)), "masking": {"rate": a.overfit_mask_rate, "seed": a.seed + 900_000},
            "steps": a.steps, "batch_size": a.batch_size, "elapsed_seconds": time.time() - start,
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
    return p


def main() -> None:
    a = parser().parse_args()
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
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
