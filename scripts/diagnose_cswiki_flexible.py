#!/usr/bin/env python3
"""Post-training diagnostics for a checksum-verified Czech flexible checkpoint.

This is deliberately validation-only: it reuses the strict cswiki cache loader
and refuses both iCloud outputs and replacement of a historical report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.layerwise_diagnostics import (atomic_json_write, deterministic_mask,
    evaluate_in_chunks, flexible_route_pool, proxy_cost_for_schedule)
from scripts.train_cswiki_flexible import (MILESTONES, N_LAYERS, build_model,
    ensure_outside_icloud, select_verified_cswiki_cache)
from src.data import load_tokenizer

EXIT_LAYERS = tuple(layer for layer, _ in MILESTONES)


def load_weights_only(model, checkpoint: Path) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"--checkpoint must name an existing best .npz checkpoint: {checkpoint}")
    payload = mx.load(str(checkpoint))
    weights = [(key, value) for key, value in payload.items() if not key.startswith("opt_")]
    try:
        model.load_weights(weights, strict=True)
    except TypeError:
        model.load_weights(weights)
    mx.eval(model.parameters())


def decode(tokenizer, sequence: np.ndarray) -> str:
    return tokenizer.decode([int(token) for token in sequence])


def refinement_steps(model, tokenizer, sample: np.ndarray, steps: int) -> list[str]:
    """Fill a deterministic quarter of remaining positions on each of four passes."""
    current = np.full(sample.shape, model.cfg.mask_token_id(), dtype=np.int32)
    outputs = []
    display_mask = tokenizer.token_to_id("[MASK]")
    if display_mask is None:
        raise ValueError("Czech tokenizer is missing [MASK]")
    for pass_index in range(steps):
        logits = model(mx.array(current, dtype=mx.int32), exit_layer=N_LAYERS)
        probabilities = mx.softmax(logits.astype(mx.float32), axis=-1)
        predictions = np.asarray(mx.argmax(probabilities, axis=-1), dtype=np.int32)
        confidence = np.asarray(mx.max(probabilities, axis=-1))
        mx.eval(probabilities)
        for row in range(len(current)):
            remaining = np.flatnonzero(current[row] == model.cfg.mask_token_id())
            # Dividing by the passes left guarantees that pass four fills all
            # positions while each pass still predicts many tokens together.
            take = int(np.ceil(len(remaining) / (steps - pass_index)))
            chosen = remaining[np.argsort(-confidence[row, remaining], kind="stable")[:take]]
            current[row, chosen] = predictions[row, chosen]
        visible = current[0].copy()
        visible[visible == model.cfg.mask_token_id()] = display_mask
        outputs.append(decode(tokenizer, visible))
    return outputs


def run(a) -> dict:
    if a.eval_sequences < 1 or a.examples < 1 or a.refinement_steps != 4:
        raise ValueError("--eval-sequences/--examples must be positive and --refinement-steps must be exactly 4")
    if a.output.exists():
        raise FileExistsError(f"Refusing to overwrite historical diagnostic report: {a.output}")
    train, val, meta, meta_path = select_verified_cswiki_cache(a.cache_dir)
    del train  # The diagnostic must never inspect training sequences.
    if len(val) < a.eval_sequences:
        raise ValueError("held-out cswiki validation split has fewer rows than --eval-sequences")
    tokenizer = load_tokenizer(meta["tokenizer"])
    model = build_model(tokenizer.get_vocab_size())
    load_weights_only(model, a.checkpoint)
    model.eval()
    sample = np.asarray(val[:a.eval_sequences], dtype=np.int32)
    mask_seed = a.seed + 900_000
    mask = deterministic_mask(sample.shape, .5, mask_seed)
    per_route, worst_by_exit = {}, []
    for route, schedule in flexible_route_pool(N_LAYERS).items():
        model.set_layer_precisions(schedule)
        rows = []
        for layer in EXIT_LAYERS:
            metrics, _ = evaluate_in_chunks(model, sample, mask, a.eval_batch_size, exit_layer=layer)
            metrics.update({"layer": layer, "precision": schedule[layer - 1],
                            "proxy_cost": proxy_cost_for_schedule(schedule, layer)})
            rows.append(metrics)
        recon_metrics, reconstructions = evaluate_in_chunks(model, sample, mask, a.eval_batch_size,
                                                              capture_reconstructions=min(a.examples, len(sample)))
        examples = [{"target": decode(tokenizer, sample[index]),
                     "masked_reconstruction": decode(tokenizer, reconstructions[index])}
                    for index in range(len(reconstructions))]
        refinements = refinement_steps(model, tokenizer, sample[:1], a.refinement_steps)
        per_route[route] = {"schedule": schedule, "route_exit_rows": rows,
                            "masked_reconstruction": {**recon_metrics, "examples": examples},
                            "all_mask_refinements": refinements}
    for index, layer in enumerate(EXIT_LAYERS):
        candidates = {route: result["route_exit_rows"][index] for route, result in per_route.items()}
        name = max(candidates, key=lambda route: (candidates[route]["loss"], -candidates[route]["accuracy"]))
        worst = dict(candidates[name])
        worst["worst_route"] = name
        worst_by_exit.append(worst)
    return {"status": "complete", "language": "cswiki-only", "checkpoint": str(a.checkpoint),
            "checkpoint_loaded": True, "cache_meta_path": str(meta_path), "cache_metadata": meta,
            "split": "held-out validation only", "eval_sequence_indices": list(range(a.eval_sequences)),
            "masking": {"rate": .5, "seed": mask_seed, "identical_across_routes_and_exits": True},
            "per_route": per_route, "worst_route_by_exit": worst_by_exit,
            "output_path": str(a.output),
            "limits": ["Only checksum-verified cswiki-cache-v1 validation rows are used; train rows are not inspected.",
                       "Route costs are cumulative precision proxies, not measured runtime.",
                       "All-mask refinements are deterministic diagnostic decoding, not a trained diffusion sampler."]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--eval-sequences", type=int, default=32); p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--examples", type=int, default=2); p.add_argument("--refinement-steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260804)
    a = p.parse_args(); ensure_outside_icloud(a.output)
    result = run(a); atomic_json_write(a.output, result); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
