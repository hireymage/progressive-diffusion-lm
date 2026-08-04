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
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.layerwise_pilot import select_cache
from src.config import LayerwiseModelConfig
from src.data import load_tokenizer
from src.layerwise_model import (LayerwiseProgressiveLM, masked_deep_supervision_loss,
                                 build_layer_precision_schedule, proxy_cost_for_schedule)

MASK_RATES = (0.15, 0.30, 0.50, 0.75, 1.00)
POLICY_CONFIDENCE_THRESHOLDS = (0.50, 0.70, 0.80, 0.90, 0.95, 0.99)
PROGRESSIVE_PRECISIONS = ["q1"] * 5 + ["q2"] * 5 + ["q4"] * 5 + ["q8"] * 5 + ["fp16"] * 5

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
                       batch_size: int, capture_reconstructions: int = 0,
                       exit_layer: int | None = None) -> tuple[dict, list[np.ndarray]]:
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
        logits = model(x, exit_layer=model.cfg.n_layers if exit_layer is None else exit_layer)
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
    if a.model_variant == "progressive":
        if a.n_layers != 25:
            raise ValueError("--model-variant progressive requires --n-layers 25")
        precisions = PROGRESSIVE_PRECISIONS
    elif a.model_variant == "flexible":
        if a.n_layers != 25:
            raise ValueError("--model-variant flexible requires --n-layers 25")
        # The schedule is switched at runtime; this canonical route is also
        # restored after multi-route evaluation.
        precisions = flexible_route_pool(a.n_layers)["q8_only"]
    else:
        precisions = ["fp32"] * a.n_layers
    min_exit_layer = (min(layer for layer, _ in a.milestone_weights)
                      if a.auxiliary_loss == "weighted-milestones" else a.n_layers)
    cfg = LayerwiseModelConfig(vocab_size=vocab_size, d_model=a.d_model, d_ff=a.d_ff,
        n_heads=a.n_heads, n_layers=a.n_layers, min_exit_layer=min_exit_layer,
        max_seq_len=256, layer_precisions=precisions)
    return LayerwiseProgressiveLM(cfg)


def flexible_route_pool(n_layers: int) -> dict[str, list[str]]:
    """The fixed deployment routes for the shared-master flexible experiment."""
    if n_layers != 25:
        raise ValueError("flexible route pool requires 25 layers")
    return {
        "q8_only": ["q8"] * n_layers,
        "q8_fp16": build_layer_precision_schedule(n_layers, ["q8", "fp16"]),
        "q2_q8_fp16": build_layer_precision_schedule(n_layers, ["q2", "q8", "fp16"]),
    }


def route_for_training_step(route_pool: dict[str, list[str]], step: int) -> tuple[str, list[str]]:
    """Choose one route per one-indexed update, with no random route state."""
    if step < 1:
        raise ValueError("step is one-indexed and must be positive")
    name = tuple(route_pool)[(step - 1) % len(route_pool)]
    return name, route_pool[name]


def evaluate_routes(model, targets_np, mask_np, batch_size, route_pool: dict[str, list[str]]) -> dict:
    """Evaluate every flexible route against exactly one caller-owned mask."""
    rows = {}
    for name, schedule in route_pool.items():
        model.set_layer_precisions(schedule)
        metrics, _ = evaluate_in_chunks(model, targets_np, mask_np, batch_size)
        metrics["proxy_full_cost"] = proxy_cost_for_schedule(schedule)
        rows[name] = metrics
    # Gate values are intentionally pessimistic: a passing report means each
    # independently executable route passed on the same masked tokens.
    return {"per_route": rows,
            "loss": max(row["loss"] for row in rows.values()),
            "accuracy": min(row["accuracy"] for row in rows.values()),
            "masked_tokens": min(row["masked_tokens"] for row in rows.values())}


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


def build_exit_sweep_model(a, vocab_size: int) -> LayerwiseProgressiveLM:
    """Build a progressive model whose requested exits are always eligible.

    Checkpoint tensor shapes do not depend on ``min_exit_layer``.  This keeps
    an evaluation sweep independent from whether the original run used an
    auxiliary intermediate-loss objective.
    """
    if a.model_variant != "progressive" or a.n_layers != 25:
        raise ValueError("exit-sweep requires --model-variant progressive and --n-layers 25")
    layers = tuple(layer for layer, _ in a.milestone_weights)
    if not layers or any(layer < 1 or layer > a.n_layers for layer in layers):
        raise ValueError("exit-sweep milestone layers must be within the model")
    cfg = LayerwiseModelConfig(vocab_size=vocab_size, d_model=a.d_model, d_ff=a.d_ff,
        n_heads=a.n_heads, n_layers=a.n_layers, min_exit_layer=min(layers),
        max_seq_len=256, layer_precisions=PROGRESSIVE_PRECISIONS)
    return LayerwiseProgressiveLM(cfg)


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
    payload = dict(tree_flatten(model.parameters()))
    payload.update({"opt_" + k: v for k, v in tree_flatten(optimizer.state)})
    mx.savez(str(path), **payload)
    atomic_json_write(directory / f"{kind}.json", metadata)


def _load_overfit_checkpoint(model, optimizer, path: Path) -> dict:
    data = mx.load(str(path))
    model.load_weights([(k, v) for k, v in data.items() if not k.startswith("opt_")])
    opt = [(k.removeprefix("opt_"), v) for k, v in data.items() if k.startswith("opt_")]
    if not opt:
        raise ValueError("overfit checkpoint has no optimizer state")
    optimizer.state = tree_unflatten(opt)
    mx.eval(model.parameters(), optimizer.state)
    return json.loads(path.with_suffix(".json").read_text())


def _overfit_loss(model, x, targets, mask, a):
    if a.auxiliary_loss == "final-only":
        exits, weights = (a.n_layers,), None
    else:
        ordered = tuple(sorted(a.milestone_weights))
        exits, weights = tuple(layer for layer, _ in ordered), tuple(weight for _, weight in ordered)
    return masked_deep_supervision_loss(model, x, targets, mask, supervised_layers=exits, layer_weights=weights)


def validate_resume_metadata(saved: dict, a) -> None:
    """Reject a state whose deterministic data/model contract changed."""
    if saved["schedule"] != [[r, n] for r, n in a.mask_schedule]:
        raise ValueError("resume schedule differs from checkpoint metadata")
    if saved.get("model_variant") != a.model_variant:
        raise ValueError("resume model variant differs from checkpoint metadata")
    if a.model_variant == "flexible":
        expected = flexible_route_pool(a.n_layers)
        if saved.get("route_pool") != expected:
            raise ValueError("resume route pool differs from checkpoint metadata")


def run_overfit(a, train, tokenizer) -> dict:
    if len(train) < a.overfit_sequences:
        raise ValueError("cache has fewer rows than --overfit-sequences")
    fixed = np.asarray(train[:a.overfit_sequences], dtype=np.int32)
    mx.random.seed(a.seed)
    model = build_model(a, a.vocab_size)
    route_pool = flexible_route_pool(a.n_layers) if a.model_variant == "flexible" else None
    optimizer = optim.AdamW(learning_rate=a.lr)
    grad_fn = nn.value_and_grad(model, lambda m, x, t, mask: _overfit_loss(m, x, t, mask, a))
    checkpoint_dir = a.checkpoint_dir
    # Reserving two checkpoint files catches accidental long runs on a nearly
    # full volume before any model state is written.
    estimate = sum(np.asarray(v).nbytes for _, v in tree_flatten(model.parameters())) * 4
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
        validate_resume_metadata(saved, a)
    start = time.time()
    for step in range(start_step + 1, a.steps + 1):
        # No random sampling: each contiguous batch cycles through the fixed 100.
        offset = ((step - 1) * a.batch_size) % len(fixed)
        indices = (np.arange(a.batch_size) + offset) % len(fixed)
        batch = fixed[indices]
        train_rate = schedule_rate(a.mask_schedule, step)
        x, targets, mask = corrupt(batch, model.cfg.mask_token_id(), train_rate, a.seed + step)
        train_route = None
        if route_pool is not None:
            train_route, schedule = route_for_training_step(route_pool, step)
            model.set_layer_precisions(schedule)
        loss, grads = grad_fn(model, x, targets, mask)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        if step == 1 or step % a.report_every == 0 or step == a.steps:
            # Fixed evaluation masks make the reported curve comparable.
            eval_mask = deterministic_mask(fixed.shape, a.gate_mask_rate, a.seed + 900_000)
            if route_pool is None:
                metrics, _ = evaluate_in_chunks(model, fixed, eval_mask, a.eval_batch_size)
            else:
                metrics = evaluate_routes(model, fixed, eval_mask, a.eval_batch_size, route_pool)
                # Evaluation finishes on the last route, so explicitly restore
                # the next deterministic training schedule before continuing.
                next_route, next_schedule = route_for_training_step(route_pool, step + 1)
                model.set_layer_precisions(next_schedule)
                metrics["next_training_route"] = next_route
            streak = gate_streak(streak, metrics["accuracy"], a.gate_accuracy)
            best_accuracy = max(best_accuracy, metrics["accuracy"])
            metrics.update({"step": step, "train_objective": float(loss), "train_mask_rate": train_rate,
                            "gate_streak": streak, "processed_tokens": step * a.batch_size * fixed.shape[1],
                            "exposures_per_sequence": step * a.batch_size / len(fixed)})
            if train_route is not None:
                metrics["training_route"] = train_route
            history.append(metrics)
            metadata = {"step": step, "gate_streak": streak, "best_accuracy": best_accuracy,
                        "history": history, "schedule": [[r, n] for r, n in a.mask_schedule],
                        "seed": a.seed, "batch_size": a.batch_size, "overfit_sequences": a.overfit_sequences,
                        "model_variant": a.model_variant, "route_pool": route_pool}
            _overfit_checkpoint(model, optimizer, checkpoint_dir, "latest", metadata)
            if metrics["accuracy"] >= best_accuracy:
                _overfit_checkpoint(model, optimizer, checkpoint_dir, "best", metadata)
            partial = {"mode": "overfit", "model_variant": a.model_variant, "status": "running", "history": history, "latest": metrics,
                       "checkpoint_dir": str(checkpoint_dir)}
            atomic_json_write(a.output, partial)
            if streak >= a.gate_reports:
                break
    eval_mask = deterministic_mask(fixed.shape, a.gate_mask_rate, a.seed + 900_000)
    if route_pool is None:
        final, reconstructions = evaluate_in_chunks(model, fixed, eval_mask, a.eval_batch_size, capture_reconstructions=2)
    else:
        final = evaluate_routes(model, fixed, eval_mask, a.eval_batch_size, route_pool)
        # Capture examples from the canonical all-Q8 route, preserving the
        # existing examples schema while reporting all route metrics above.
        model.set_layer_precisions(route_pool["q8_only"])
        _canonical, reconstructions = evaluate_in_chunks(model, fixed, eval_mask, a.eval_batch_size, capture_reconstructions=2)
    decode = lambda ids: tokenizer.decode([int(i) for i in ids])
    # The all-position value is derived from the masked aggregate: unmasked
    # positions are copied exactly by construction.
    final["reconstruction_accuracy_all_positions"] = float(
        (final["accuracy"] * final["masked_tokens"] + (fixed.size - final["masked_tokens"])) / fixed.size)
    final["quality_gate"] = {"threshold": a.gate_accuracy, "consecutive_reports_required": a.gate_reports,
        "consecutive_reports": streak, "passed": streak >= a.gate_reports,
        "meaning": "masked accuracy on a fixed mask over all fixed training sequences"}
    return {"mode": "overfit", "model_variant": a.model_variant,
            "architecture": {"d_model": a.d_model, "d_ff": a.d_ff, "n_heads": a.n_heads, "n_layers": a.n_layers,
                             "layer_precisions": list(model.cfg.layer_precisions),
                             **({"route_pool": route_pool, "active_schedule": list(model.cfg.layer_precisions)} if route_pool is not None else {})},
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


def run_exit_sweep(a, train) -> dict:
    """Measure fixed-mask reconstruction at selected progressive exits.

    Each exit is evaluated in a separate chunked pass.  In particular, this
    avoids keeping five [batch, sequence, vocab] logits tensors alive merely
    to produce a layer comparison.
    """
    if a.eval_sequences < 1:
        raise ValueError("--eval-sequences must be positive for exit-sweep")
    sample = np.asarray(train[:min(a.eval_sequences, len(train))], dtype=np.int32)
    if len(sample) < a.eval_sequences:
        raise ValueError("cache has fewer rows than --eval-sequences")
    model = build_exit_sweep_model(a, a.vocab_size)
    load_weights_only(model, a.checkpoint)
    mask_seed = a.seed + 900_000
    mask_np = deterministic_mask(sample.shape, a.gate_mask_rate, mask_seed)
    rows = []
    for layer, _weight in a.milestone_weights:
        metrics, _ = evaluate_in_chunks(model, sample, mask_np, a.eval_batch_size,
                                        exit_layer=layer)
        metrics.update({"layer": layer,
                        "precision": model.cfg.layer_precisions[layer - 1],
                        "proxy_cost": proxy_cost_for_schedule(model.cfg.layer_precisions, layer)})
        rows.append(metrics)
    return {"mode": "exit-sweep", "checkpoint": str(a.checkpoint), "checkpoint_loaded": True,
            "eval_sequence_indices": list(range(a.eval_sequences)),
            "masking": {"rate": a.gate_mask_rate, "seed": mask_seed},
            "rows": rows,
            "limits": "Each row uses the same fixed training sequences and exact mask; exits are evaluated in separate chunks."}


def collect_masked_predictions(model: LayerwiseProgressiveLM, targets_np: np.ndarray,
                               mask_np: np.ndarray, batch_size: int, exit_layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect only compact masked-token outputs for one exit layer.

    Vocab-sized logits are consumed within each batch, then released.  The
    returned arrays are one prediction, confidence, and target per masked token.
    """
    if batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    predictions, confidences, truths = [], [], []
    for start in range(0, len(targets_np), batch_size):
        target = targets_np[start:start + batch_size]
        mask = mask_np[start:start + batch_size]
        x = mx.array(np.where(mask, model.cfg.mask_token_id(), target), dtype=mx.int32)
        logits = model(x, exit_layer=exit_layer).astype(mx.float32)
        log_probs = nn.log_softmax(logits, axis=-1)
        predicted = np.asarray(mx.argmax(log_probs, axis=-1))
        confidence = np.asarray(mx.exp(mx.max(log_probs, axis=-1)))
        mx.eval(log_probs)
        predictions.append(predicted[mask])
        confidences.append(confidence[mask])
        truths.append(target[mask])
    return (np.concatenate(predictions), np.concatenate(confidences), np.concatenate(truths))


def summarize_routing(exit_indices: np.ndarray, predictions: list[np.ndarray], truths: np.ndarray,
                      layers: tuple[int, ...], costs: tuple[float, ...]) -> dict:
    """Summarize a token-wise simulated route; indices name milestone entries."""
    if len(exit_indices) != len(truths):
        raise ValueError("routing and target sizes differ")
    if len(exit_indices) == 0:
        raise ValueError("policy simulation needs at least one masked token")
    if np.any(exit_indices < 0) or np.any(exit_indices >= len(layers)):
        raise ValueError("routing chose an unknown milestone")
    selected = np.asarray([predictions[i][j] for j, i in enumerate(exit_indices)])
    counts = np.bincount(exit_indices, minlength=len(layers))
    mean_cost = float(np.mean(np.asarray(costs)[exit_indices]))
    full_cost = float(costs[-1])
    return {"accuracy": float(np.mean(selected == truths)), "mean_proxy_cost": mean_cost,
            "savings_vs_full": float(1.0 - mean_cost / full_cost),
            "exit_distribution": [{"layer": int(layer), "count": int(count)}
                                  for layer, count in zip(layers, counts)],
            "masked_tokens": int(len(truths))}


def simulate_stable_confidence_policy(predictions: list[np.ndarray], confidences: list[np.ndarray],
                                      truths: np.ndarray, layers: tuple[int, ...], costs: tuple[float, ...],
                                      threshold: float) -> dict:
    """Route to earliest stable confident exit, with final-layer fallback.

    The first milestone has no earlier prediction to compare, so it cannot be a
    stable exit under this policy.  This deliberately conservative convention
    makes the stated "stability versus previous milestone" condition literal.
    """
    route = np.full(len(truths), len(layers) - 1, dtype=np.int32)
    unresolved = np.ones(len(truths), dtype=bool)
    for index in range(1, len(layers)):
        eligible = unresolved & (confidences[index] >= threshold) & (predictions[index] == predictions[index - 1])
        route[eligible] = index
        unresolved[eligible] = False
    result = summarize_routing(route, predictions, truths, layers, costs)
    result["confidence_threshold"] = float(threshold)
    return result


def simulate_oracle_earliest_correct(predictions: list[np.ndarray], truths: np.ndarray,
                                     layers: tuple[int, ...], costs: tuple[float, ...]) -> dict:
    """Ground-truth upper bound; intentionally not a deployable controller."""
    route = np.full(len(truths), len(layers) - 1, dtype=np.int32)
    unresolved = np.ones(len(truths), dtype=bool)
    for index, prediction in enumerate(predictions):
        correct = unresolved & (prediction == truths)
        route[correct] = index
        unresolved[correct] = False
    result = summarize_routing(route, predictions, truths, layers, costs)
    result["label"] = "ground-truth oracle earliest-correct upper bound (non-deployable)"
    return result


def run_policy_sweep(a, train) -> dict:
    """Evaluate an algorithmic token-wise early-exit policy on a fixed mask."""
    if a.eval_sequences < 1:
        raise ValueError("--eval-sequences must be positive for policy-sweep")
    if len(train) < a.eval_sequences:
        raise ValueError("cache has fewer rows than --eval-sequences")
    layers = tuple(layer for layer, _ in a.milestone_weights)
    if layers != tuple(sorted(set(layers))) or not layers or layers[-1] != a.n_layers:
        raise ValueError("policy-sweep milestones must be strictly increasing and end at --n-layers for fallback")
    sample = np.asarray(train[:a.eval_sequences], dtype=np.int32)
    model = build_exit_sweep_model(a, a.vocab_size)
    load_weights_only(model, a.checkpoint)
    costs = tuple(float(proxy_cost_for_schedule(model.cfg.layer_precisions, layer)) for layer in layers)
    mask_seed = a.seed + 900_000
    mask_np = deterministic_mask(sample.shape, a.gate_mask_rate, mask_seed)
    predictions, confidences, truths = [], [], None
    for layer in layers:
        pred, confidence, current_truths = collect_masked_predictions(model, sample, mask_np, a.eval_batch_size, layer)
        predictions.append(pred)
        confidences.append(confidence)
        if truths is None:
            truths = current_truths
        elif not np.array_equal(truths, current_truths):
            raise RuntimeError("masked targets changed between milestone passes")
    fixed_exits = []
    for index, layer in enumerate(layers):
        route = np.full(len(truths), index, dtype=np.int32)
        row = summarize_routing(route, predictions, truths, layers, costs)
        row["layer"] = int(layer)
        fixed_exits.append(row)
    return {"mode": "policy-sweep", "checkpoint": str(a.checkpoint), "checkpoint_loaded": True,
            "eval_sequence_indices": list(range(a.eval_sequences)),
            "masking": {"rate": a.gate_mask_rate, "seed": mask_seed},
            "milestones": [{"layer": int(layer), "precision": model.cfg.layer_precisions[layer - 1], "proxy_cost": cost}
                           for layer, cost in zip(layers, costs)],
            "policy": {"kind": "token-wise simulated stable-confidence routing",
                       "confidence_metric": "top-1 softmax probability",
                       "thresholds": list(POLICY_CONFIDENCE_THRESHOLDS),
                       "rows": [simulate_stable_confidence_policy(predictions, confidences, truths, layers, costs, threshold)
                                for threshold in POLICY_CONFIDENCE_THRESHOLDS]},
            "fixed_exits": fixed_exits,
            "oracle": simulate_oracle_earliest_correct(predictions, truths, layers, costs),
            "limits": "Algorithmic token-wise/oracle simulation only: it is not an executable sparse speedup or current sequence-wide controller. Proxy-cost savings are simulated per-token accounting, not measured runtime savings."}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("frequency", "overfit", "mask-sweep", "exit-sweep", "policy-sweep"))
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache")
    p.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer/wiki_bpe")
    p.add_argument("--output", type=Path, default=ROOT / "results/layerwise/diagnostics.json")
    p.add_argument("--seed", type=int, default=20260804); p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--d-model", type=int, default=64); p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4); p.add_argument("--n-layers", type=int, default=25)
    p.add_argument("--model-variant", choices=("fp32", "progressive", "flexible"), default="fp32")
    p.add_argument("--vocab-size", type=int, default=None, help="normally inferred from tokenizer")
    p.add_argument("--checkpoint", type=Path, help="layer-wise .npz required for mask-sweep, exit-sweep, or policy-sweep")
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


def validate_mode_args(a) -> None:
    """Validate mode-specific inputs before accessing cache or tokenizer."""
    if a.mode in ("mask-sweep", "exit-sweep", "policy-sweep") and a.checkpoint is None:
        raise ValueError(f"--checkpoint is required for {a.mode} (weights are never random)")


def main() -> None:
    a = parser().parse_args()
    validate_mode_args(a)
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
    result = {"offline_only": True, "cache_meta_path": str(meta_path), "cache_metadata": meta,
              "tokenizer": str(a.tokenizer), "seed": a.seed}
    if a.mode == "frequency": result["result"] = run_frequency(a, train, val, tokenizer)
    elif a.mode == "overfit": result["result"] = run_overfit(a, train, tokenizer)
    elif a.mode == "mask-sweep": result["result"] = run_mask_sweep(a, train, val, tokenizer)
    elif a.mode == "exit-sweep": result["result"] = run_exit_sweep(a, train)
    else: result["result"] = run_policy_sweep(a, train)
    atomic_json_write(a.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
