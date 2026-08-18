#!/usr/bin/env python3
"""Interactive Czech prompt continuation from one loaded flexible checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_cswiki_flexible import load_weights_only, prompt_continuation
from scripts.layerwise_diagnostics import flexible_route_pool
from scripts.train_cswiki_flexible import N_LAYERS, build_model, select_verified_cswiki_cache
from src.data import load_tokenizer


def run_interactive(a, input_fn: Callable[[str], str] = input,
                    output_fn: Callable[[str], None] = print) -> None:
    """Load local Czech artifacts once, then serve prompts until ``/konec``."""
    if a.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    train, val, cache_meta, _meta_path = select_verified_cswiki_cache(a.cache_dir)
    del train, val
    tokenizer = load_tokenizer(cache_meta["tokenizer"])
    model = build_model(tokenizer.get_vocab_size())
    load_weights_only(model, a.checkpoint)
    model.eval()
    schedule = flexible_route_pool(N_LAYERS)[a.route]
    model.set_layer_precisions(schedule)
    output_fn(f"Český flexible model je připraven ({a.route}); ukončení: /konec")
    while True:
        try:
            prompt = input_fn("> ")
        except EOFError:
            break
        if prompt.strip() == "/konec":
            break
        if not prompt:
            continue
        result = prompt_continuation(model, tokenizer, prompt=prompt,
                                     max_new_tokens=a.max_new_tokens, passes=4)
        output_fn(result["refinements"][-1])


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--route", choices=tuple(flexible_route_pool(N_LAYERS)), default="q8_only")
    p.add_argument("--max-new-tokens", type=int, default=4)
    return p


def main() -> None:
    run_interactive(parser().parse_args())


if __name__ == "__main__":
    main()
