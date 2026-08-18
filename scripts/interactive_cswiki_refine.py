#!/usr/bin/env python3
"""Interactive full-sentence refinement for a Czech flexible checkpoint.

Unlike ``interactive_cswiki_flexible.py`` this diagnostic is allowed to replace
the whole encoded input sentence content on every pass.  It keeps only BOS/EOS
fixed, so it is a risky but useful probe of whether the mask/diffusion model can
revise its own earlier guesses.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_cswiki_flexible import load_weights_only
from scripts.layerwise_diagnostics import flexible_route_pool
from scripts.train_cswiki_flexible import build_model, select_verified_cswiki_cache
from src.data import load_tokenizer


def full_sentence_refine(model, tokenizer, text: str, *, passes: int = 6,
                         temperature: float = 0.0) -> dict:
    if passes < 1:
        raise ValueError("--passes must be positive")
    if temperature < 0:
        raise ValueError("--temperature must be non-negative")
    encoded = np.asarray(tokenizer.encode(text).ids, dtype=np.int32)
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    if bos_id is None or eos_id is None:
        raise ValueError("Czech tokenizer is missing [BOS] or [EOS]")
    if len(encoded) < 3 or encoded[0] != bos_id or encoded[-1] != eos_id:
        raise ValueError("Czech tokenizer must encode text with [BOS] and terminal [EOS]")
    if len(encoded) > model.cfg.max_seq_len:
        raise ValueError("input is longer than model max_seq_len")

    current = encoded[None, :].copy()
    refinements = []
    for _pass_index in range(passes):
        logits = model(mx.array(current, dtype=mx.int32), exit_layer=model.cfg.n_layers)
        logits = logits.astype(mx.float32)
        if temperature > 0:
            probabilities = mx.softmax(logits / temperature, axis=-1)
            # MLX categorical sampling has changed APIs across versions; for a
            # diagnostic tool, deterministic argmax fallback is preferable to a
            # version-specific failure.
            prediction = np.asarray(mx.argmax(probabilities, axis=-1), dtype=np.int32)
            mx.eval(probabilities)
        else:
            prediction = np.asarray(mx.argmax(logits, axis=-1), dtype=np.int32)
        current[0, 1:-1] = prediction[0, 1:-1]
        current[0, 0] = bos_id
        current[0, -1] = eos_id
        refinements.append(tokenizer.decode([int(token) for token in current[0]]))
    return {"input": text, "passes": passes, "temperature": temperature,
            "token_count": int(len(encoded)), "refinements": refinements,
            "final_text": refinements[-1]}


def run_interactive(a, input_fn: Callable[[str], str] = input,
                    output_fn: Callable[[str], None] = print) -> None:
    _train, _val, cache_meta, _meta_path = select_verified_cswiki_cache(a.cache_dir)
    del _train, _val
    tokenizer = load_tokenizer(cache_meta["tokenizer"])
    model = build_model(tokenizer.get_vocab_size(), d_model=a.d_model, d_ff=a.d_ff,
                        n_heads=a.n_heads, n_layers=a.n_layers, seq_len=a.seq_len)
    load_weights_only(model, a.checkpoint)
    model.eval()
    model.set_layer_precisions(flexible_route_pool(a.n_layers)[a.route])
    output_fn(f"Český flexible refine je připraven ({a.route}); ukončení: /konec")
    while True:
        try:
            text = input_fn("> ")
        except EOFError:
            break
        if text.strip() == "/konec":
            break
        if not text.strip():
            continue
        result = full_sentence_refine(model, tokenizer, text, passes=a.passes,
                                      temperature=a.temperature)
        for index, refinement in enumerate(result["refinements"], start=1):
            output_fn(f"{index}: {refinement}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--route", choices=tuple(flexible_route_pool(25)), default="q2_q8_fp16")
    p.add_argument("--passes", type=int, default=6)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=25)
    p.add_argument("--seq-len", type=int, default=256)
    return p


def main() -> None:
    run_interactive(parser().parse_args())


if __name__ == "__main__":
    main()
