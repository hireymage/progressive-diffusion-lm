#!/usr/bin/env python3
"""Benchmark alternating versus joint-route MLX updates from one checkpoint.

Both arms start from the exact same model and optimizer checkpoint.  The
comparison is matched by route forward/backward passes: the alternating arm
performs one route per optimizer update, while the joint arm averages gradients
from all routes on one shared batch and performs one optimizer update.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.layerwise_diagnostics import route_for_training_step
from scripts.train_cswiki_flexible import (
    build_model,
    corrupt_50,
    evaluate_routes,
    fixed_batch,
    load_checkpoint,
    milestone_weights,
    route_pool,
    select_verified_cswiki_cache,
)
from src.data import load_tokenizer
from src.layerwise_model import masked_deep_supervision_loss


def atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def average_gradient_trees(gradients: list[dict]) -> dict:
    """Average identically structured MLX gradient trees."""
    if not gradients:
        raise ValueError("at least one gradient tree is required")
    flattened = [tree_flatten(tree) for tree in gradients]
    keys = [key for key, _ in flattened[0]]
    if any([key for key, _ in row] != keys for row in flattened[1:]):
        raise ValueError("gradient trees do not have identical structure")
    scale = 1.0 / len(flattened)
    return tree_unflatten([
        (key, sum(row[index][1] for row in flattened) * scale)
        for index, key in enumerate(keys)
    ])


def checkpoint_contract(metadata: dict) -> dict:
    keys = ("cache_train_sha256", "cache_val_sha256", "route_pool", "strategy", "architecture")
    missing = [key for key in keys if key not in metadata]
    if missing:
        raise ValueError(f"checkpoint metadata is missing contract keys: {missing}")
    return {key: metadata[key] for key in keys}


def build_arm(checkpoint_path: Path, tokenizer, learning_rate: float):
    metadata = json.loads(checkpoint_path.with_suffix(".json").read_text())
    n_layers, d_model, d_ff, n_heads, seq_len = metadata["architecture"]
    model = build_model(tokenizer.get_vocab_size(), d_model=d_model, d_ff=d_ff,
                        n_heads=n_heads, n_layers=n_layers, seq_len=seq_len)
    optimizer = optim.AdamW(learning_rate=learning_rate)
    load_checkpoint(model, optimizer, checkpoint_path, checkpoint_contract(metadata))
    milestones = milestone_weights(n_layers)
    grad_fn = nn.value_and_grad(model, lambda m, x, targets, mask: masked_deep_supervision_loss(
        m, x, targets, mask,
        supervised_layers=tuple(layer for layer, _ in milestones),
        layer_weights=tuple(weight for _, weight in milestones)))
    return model, optimizer, grad_fn, metadata


def finite_loss_and_gradient(loss, gradients) -> tuple[float, float]:
    _, norm = optim.clip_grad_norm(gradients, float("inf"))
    mx.eval(loss, norm)
    loss_value, norm_value = float(loss), float(norm)
    if not np.isfinite(loss_value) or not np.isfinite(norm_value):
        raise FloatingPointError(f"non-finite benchmark value: loss={loss_value}, gradient_norm={norm_value}")
    return loss_value, norm_value


def run_arm(mode: str, *, checkpoint_path: Path, tokenizer, train, val, route_passes: int,
            batch_size: int, eval_steps: int, seed: int, learning_rate: float) -> dict:
    model, optimizer, grad_fn, metadata = build_arm(checkpoint_path, tokenizer, learning_rate)
    pool = route_pool(model.cfg.n_layers)
    routes = tuple(pool)
    updates = route_passes if mode == "alternating" else route_passes // len(routes)
    baseline = evaluate_routes(model, val, batch_size, eval_steps, seed + 900_000)
    losses, gradient_norms = [], []
    began = time.perf_counter()
    for update in range(1, updates + 1):
        batch_seed = seed + update
        x, targets, mask = corrupt_50(fixed_batch(train, batch_size, batch_seed),
                                      model.cfg.mask_token_id(), seed + 10_000 + update)
        if mode == "alternating":
            route, schedule = route_for_training_step(pool, update)
            model.set_layer_precisions(schedule)
            loss, gradients = grad_fn(model, x, targets, mask)
            loss_value, norm_value = finite_loss_and_gradient(loss, gradients)
            losses.append(loss_value)
        else:
            route_losses, route_gradients = [], []
            for route in routes:
                model.set_layer_precisions(pool[route])
                loss, gradients = grad_fn(model, x, targets, mask)
                loss_value, _ = finite_loss_and_gradient(loss, gradients)
                route_losses.append(loss_value)
                route_gradients.append(gradients)
            gradients = average_gradient_trees(route_gradients)
            _, norm_value = finite_loss_and_gradient(mx.array(sum(route_losses) / len(route_losses)), gradients)
            losses.extend(route_losses)
        gradient_norms.append(norm_value)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state)
    train_seconds = time.perf_counter() - began
    final = evaluate_routes(model, val, batch_size, eval_steps, seed + 900_000)
    return {
        "mode": mode,
        "source_step": metadata["step"],
        "optimizer_updates": updates,
        "route_passes": route_passes,
        "unique_batches": updates,
        "train_seconds": train_seconds,
        "route_passes_per_second": route_passes / train_seconds,
        "optimizer_updates_per_second": updates / train_seconds,
        "mean_train_loss": float(np.mean(losses)),
        "mean_gradient_norm": float(np.mean(gradient_norms)),
        "baseline": baseline,
        "final": final,
        "worst_loss_delta": final["loss"] - baseline["loss"],
        "worst_accuracy_delta": final["accuracy"] - baseline["accuracy"],
    }


def run(args) -> dict:
    if args.route_passes < 3 or args.route_passes % 3:
        raise ValueError("--route-passes must be a positive multiple of 3")
    train, val, cache_metadata, cache_meta_path = select_verified_cswiki_cache(args.cache_dir)
    tokenizer = load_tokenizer(cache_metadata["tokenizer"])
    arms = []
    for mode in ("alternating", "joint"):
        arms.append(run_arm(mode, checkpoint_path=args.checkpoint, tokenizer=tokenizer,
                            train=train, val=val, route_passes=args.route_passes,
                            batch_size=args.batch_size, eval_steps=args.eval_steps,
                            seed=args.seed, learning_rate=args.lr))
    alternating, joint = arms
    result = {
        "status": "complete",
        "device": "m1-256",
        "comparison": "matched route forward/backward passes",
        "source_checkpoint": str(args.checkpoint),
        "cache_meta_path": str(cache_meta_path),
        "route_passes": args.route_passes,
        "arms": {row["mode"]: row for row in arms},
        "joint_speedup_route_passes": joint["route_passes_per_second"] / alternating["route_passes_per_second"],
        "note": "Joint uses one shared batch for three routes and averages their gradients before one optimizer update.",
    }
    atomic_json_write(args.output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--route-passes", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
