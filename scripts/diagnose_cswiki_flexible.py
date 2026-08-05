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


def prompt_continuation(model, tokenizer, prompt: str = "Kočka leze dírou,",
                        max_new_tokens: int = 4, passes: int = 4) -> dict:
    """Keep Czech prompt ids immutable while confidence-ranking four infill passes."""
    if passes != 4 or max_new_tokens < 1:
        raise ValueError("prompt continuation requires exactly 4 passes and positive --max-new-tokens")
    encoded = np.asarray(tokenizer.encode(prompt).ids, dtype=np.int32)
    display_mask = tokenizer.token_to_id("[MASK]")
    bos_id, eos_id = tokenizer.token_to_id("[BOS]"), tokenizer.token_to_id("[EOS]")
    if display_mask is None or bos_id is None or eos_id is None:
        raise ValueError("Czech tokenizer is missing [MASK], [BOS], or [EOS]")
    if len(encoded) < 2 or encoded[0] != bos_id or encoded[-1] != eos_id:
        raise ValueError("Czech tokenizer must encode the prompt with [BOS] and terminal [EOS]")
    # Do not infill after EOS: reserve a fixed terminal EOS after the requested
    # number of new content tokens, retaining BOS and all prompt content.
    prompt_ids = encoded[:-1]
    if len(prompt_ids) + max_new_tokens + 1 > model.cfg.max_seq_len:
        raise ValueError("prompt token length plus --max-new-tokens must fit the model sequence length")
    current = np.full((1, len(prompt_ids) + max_new_tokens + 1), model.cfg.mask_token_id(), dtype=np.int32)
    current[0, :len(prompt_ids)] = prompt_ids
    current[0, -1] = eos_id
    generated = []
    for pass_index in range(passes):
        logits = model(mx.array(current, dtype=mx.int32), exit_layer=N_LAYERS)
        probabilities = mx.softmax(logits.astype(mx.float32), axis=-1)
        prediction = np.asarray(mx.argmax(probabilities, axis=-1), dtype=np.int32)
        confidence = np.asarray(mx.max(probabilities, axis=-1))
        mx.eval(probabilities)
        remaining = np.flatnonzero(current[0, len(prompt_ids):-1] == model.cfg.mask_token_id()) + len(prompt_ids)
        take = int(np.ceil(len(remaining) / (passes - pass_index)))
        chosen = remaining[np.argsort(-confidence[0, remaining], kind="stable")[:take]]
        current[0, chosen] = prediction[0, chosen]
        # This assignment is intentionally repeated after every pass: no model
        # prediction can ever replace the exact encoded Czech prompt prefix or EOS.
        current[0, :len(prompt_ids)] = prompt_ids
        current[0, -1] = eos_id
        visible = current[0].copy()
        visible[visible == model.cfg.mask_token_id()] = display_mask
        generated.append(decode(tokenizer, visible))
    return {"prompt": prompt, "prompt_token_ids": prompt_ids.tolist(), "terminal_eos_id": int(eos_id), "max_new_tokens": max_new_tokens,
            "passes": passes, "continuation_token_ids": current[0].tolist(), "refinements": generated}


def parse_character_spans(value: str | None, text: str) -> tuple[tuple[int, int], ...]:
    """Parse explicit START:END spans, defaulting to the two requested words."""
    if value is None:
        spans = []
        for word in ("Kocka", "dirou"):
            start = text.find(word)
            if start < 0:
                raise ValueError(f"default diacritics repair word is absent: {word}")
            spans.append((start, start + len(word)))
        return tuple(spans)
    try:
        spans = tuple((int(start), int(end)) for start, end in
                      (part.strip().split(":") for part in value.split(",")))
    except ValueError as exc:
        raise ValueError("--repair-spans must be START:END[,START:END...]") from exc
    if not spans or any(start < 0 or end <= start or end > len(text) for start, end in spans):
        raise ValueError("repair character spans must be non-empty and within --input-text")
    return spans


def diacritics_repair(model, tokenizer, text: str = "Kocka leze dirou.",
                      spans: tuple[tuple[int, int], ...] | None = None, passes: int = 4) -> dict:
    """Mask only token offsets overlapping requested character spans, then infill."""
    if passes != 4:
        raise ValueError("diacritics repair requires exactly 4 confidence-ranked passes")
    encoding = tokenizer.encode(text)
    token_ids = np.asarray(encoding.ids, dtype=np.int32)
    offsets = tuple(tuple(offset) for offset in encoding.offsets)
    if len(token_ids) != len(offsets):
        raise ValueError("Czech tokenizer encoding must provide one offset per token")
    chosen_spans = parse_character_spans(None, text) if spans is None else spans
    mask_id = model.cfg.mask_token_id()
    display_mask = tokenizer.token_to_id("[MASK]")
    if display_mask is None:
        raise ValueError("Czech tokenizer is missing [MASK]")
    masked = [index for index, (start, end) in enumerate(offsets)
              if end > start and any(start < span_end and end > span_start for span_start, span_end in chosen_spans)]
    if not masked:
        raise ValueError("repair spans overlap no tokenizer tokens")
    current = token_ids[None, :].copy()
    current[0, masked] = mask_id
    fixed_indices = [index for index in range(len(token_ids)) if index not in masked]
    fixed_ids = token_ids[fixed_indices].tolist()
    refinements = []
    for pass_index in range(passes):
        logits = model(mx.array(current, dtype=mx.int32), exit_layer=N_LAYERS)
        probabilities = mx.softmax(logits.astype(mx.float32), axis=-1)
        prediction = np.asarray(mx.argmax(probabilities, axis=-1), dtype=np.int32)
        confidence = np.asarray(mx.max(probabilities, axis=-1))
        mx.eval(probabilities)
        remaining = np.asarray([index for index in masked if current[0, index] == mask_id], dtype=np.int32)
        take = int(np.ceil(len(remaining) / (passes - pass_index)))
        selected = remaining[np.argsort(-confidence[0, remaining], kind="stable")[:take]]
        current[0, selected] = prediction[0, selected]
        current[0, fixed_indices] = fixed_ids
        visible = current[0].copy()
        visible[visible == mask_id] = display_mask
        refinements.append(decode(tokenizer, visible))
    return {"input": text, "character_spans": [list(span) for span in chosen_spans],
            "masked_token_indices": masked, "fixed_token_ids": {str(index): int(token_ids[index]) for index in fixed_indices},
            "passes": passes, "refinements": refinements, "final_text": decode(tokenizer, current[0])}


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
    if getattr(a, "mode", "validation-diagnostics") == "prompt-continuation":
        per_route = {}
        for route, schedule in flexible_route_pool(N_LAYERS).items():
            model.set_layer_precisions(schedule)
            per_route[route] = {"schedule": schedule,
                                "prompt_continuation": prompt_continuation(
                                    model, tokenizer, prompt=a.prompt, max_new_tokens=a.max_new_tokens)}
        return {"status": "complete", "language": "cswiki-only", "mode": "prompt-continuation",
                "checkpoint": str(a.checkpoint), "checkpoint_loaded": True,
                "cache_meta_path": str(meta_path), "tokenizer": meta["tokenizer"],
                "per_route": per_route, "output_path": str(a.output),
                "limits": ["Prompt prefix is held fixed by token id on every pass.",
                           "Four confidence-ranked infill passes are diagnostic decoding, not measured serving latency."]}
    if getattr(a, "mode", "validation-diagnostics") == "diacritics-repair":
        spans = parse_character_spans(a.repair_spans, a.input_text)
        per_route = {}
        for route, schedule in flexible_route_pool(N_LAYERS).items():
            model.set_layer_precisions(schedule)
            per_route[route] = {"schedule": schedule,
                                "diacritics_repair": diacritics_repair(model, tokenizer, a.input_text, spans)}
        return {"status": "complete", "language": "cswiki-only", "mode": "diacritics-repair",
                "checkpoint": str(a.checkpoint), "checkpoint_loaded": True,
                "cache_meta_path": str(meta_path), "tokenizer": meta["tokenizer"],
                "per_route": per_route, "output_path": str(a.output),
                "limits": ["Only token offsets overlapping requested character spans may be changed.",
                           "Four confidence-ranked infill passes are diagnostic decoding, not measured serving latency."]}
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
    p.add_argument("--mode", choices=("validation-diagnostics", "prompt-continuation", "diacritics-repair"), default="validation-diagnostics")
    p.add_argument("--eval-sequences", type=int, default=32); p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--examples", type=int, default=2); p.add_argument("--refinement-steps", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--prompt", default="Kočka leze dírou,", help="exact Czech prefix retained by prompt-continuation")
    p.add_argument("--input-text", default="Kocka leze dirou.", help="input for diacritics-repair")
    p.add_argument("--repair-spans", help="optional character spans START:END[,START:END...] for diacritics-repair")
    p.add_argument("--seed", type=int, default=20260804)
    a = p.parse_args(); ensure_outside_icloud(a.output)
    result = run(a); atomic_json_write(a.output, result); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
