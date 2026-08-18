#!/usr/bin/env python3
"""Batch prompt-continuation diagnostics across cswiki flexible checkpoints.

The runner is intentionally read-only with respect to checkpoints.  It discovers
immutable ``step_*.npz`` checkpoints when present, falls back to ``latest`` and
``best`` when historical step weights are unavailable, and writes one JSONL row
per model/checkpoint/route/prompt combination.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_cswiki_flexible import load_weights_only, prompt_continuation
from scripts.layerwise_diagnostics import flexible_route_pool
from scripts.train_cswiki_flexible import build_model, ensure_outside_icloud, select_verified_cswiki_cache
from src.data import load_tokenizer

STEP_RE = re.compile(r"step_(\d{7})\.npz$")
DEFAULT_PROMPTS = (
    "Praha je hlavní",
    "Česká republika se nachází",
    "Božena Němcová",
    "V roce 1918 vzniklo",
    "Kočka leze dírou",
    "Karlova univerzita byla založena",
    "Morava je historická",
    "Jan Hus byl",
    "Český jazyk patří mezi",
    "Nejvyšší hora České republiky je",
)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def architecture_from_metadata(meta: dict, fallback: dict) -> dict:
    arch = meta.get("architecture")
    if isinstance(arch, list) and len(arch) == 5:
        n_layers, d_model, d_ff, n_heads, seq_len = arch
        return {"n_layers": int(n_layers), "d_model": int(d_model), "d_ff": int(d_ff),
                "n_heads": int(n_heads), "seq_len": int(seq_len)}
    if isinstance(arch, dict):
        return {"n_layers": int(arch.get("n_layers", fallback["n_layers"])),
                "d_model": int(arch.get("d_model", fallback["d_model"])),
                "d_ff": int(arch.get("d_ff", fallback["d_ff"])),
                "n_heads": int(arch.get("n_heads", fallback["n_heads"])),
                "seq_len": int(arch.get("seq_len", fallback["seq_len"]))}
    return dict(fallback)


def discover_checkpoints(checkpoint_dir: Path, include_latest_best: bool) -> list[dict]:
    rows = []
    for path in sorted(checkpoint_dir.glob("step_*.npz")):
        match = STEP_RE.match(path.name)
        if not match:
            continue
        rows.append({"kind": path.stem, "step": int(match.group(1)), "path": path,
                     "metadata": load_json(path.with_suffix(".json"))})
    if include_latest_best:
        for kind in ("best", "latest"):
            path = checkpoint_dir / f"{kind}.npz"
            if path.exists():
                meta = load_json(path.with_suffix(".json"))
                rows.append({"kind": kind, "step": int(meta.get("step", -1)), "path": path,
                             "metadata": meta})
    rows.sort(key=lambda row: (row["step"] if row["step"] >= 0 else 10**18, row["kind"]))
    return rows


def load_prompts(path: Path | None, count: int) -> list[str]:
    if path:
        prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    else:
        prompts = [f"{prompt} {index}" if index else prompt
                   for index in range((count + len(DEFAULT_PROMPTS) - 1) // len(DEFAULT_PROMPTS))
                   for prompt in DEFAULT_PROMPTS]
    return prompts[:count]


def generation_state(result: dict) -> dict:
    """Classify whether a fixed-width continuation ended naturally or by cap."""
    token_ids = result["continuation_token_ids"]
    prompt_ids = result["prompt_token_ids"]
    eos_id = result["terminal_eos_id"]
    generated = token_ids[len(prompt_ids):-1]
    eos_offsets = [index for index, token in enumerate(generated) if token == eos_id]
    if eos_offsets:
        stop_reason = "eos"
        generated_before_stop = eos_offsets[0]
        prematurely_ended = False
    else:
        stop_reason = "max_new_tokens"
        generated_before_stop = len(generated)
        prematurely_ended = True
    return {"stop_reason": stop_reason,
            "prematurely_ended": prematurely_ended,
            "generated_token_count": generated_before_stop,
            "requested_max_new_tokens": result["max_new_tokens"],
            "generated_token_ids": generated,
            "eos_in_generated_tokens": bool(eos_offsets),
            "eos_generated_offsets": eos_offsets,
            "exact_prompt_preserved": token_ids[:len(prompt_ids)] == prompt_ids}


def default_exit_layers(n_layers: int) -> tuple[int, ...]:
    base = (5, 10, 15, 20, n_layers)
    return tuple(sorted({layer for layer in base if 1 <= layer <= n_layers}))


def exit_state(model, result: dict, exit_layers: tuple[int, ...]) -> dict:
    """Estimate earliest layer where generated tokens match final decode ids.

    This is a diagnostic proxy for token-level early-exit: it re-runs the model
    with the prompt fixed and generated positions masked, then checks at which
    exit layer each final generated token would already be predicted.
    """
    token_ids = result["continuation_token_ids"]
    prompt_len = len(result["prompt_token_ids"])
    generated_positions = list(range(prompt_len, len(token_ids) - 1))
    if not generated_positions:
        return {"exit_layers": list(exit_layers), "token_count": 0, "early_exit_token_ratio": 0.0,
                "mean_exit_layer": None, "mean_layers_saved": 0.0, "tokens": []}
    masked = np.asarray(token_ids, dtype=np.int32)[None, :]
    mask_id = model.cfg.mask_token_id()
    masked[0, generated_positions] = mask_id
    final_ids = np.asarray(token_ids, dtype=np.int32)
    predictions_by_layer = {}
    confidences_by_layer = {}
    for layer in exit_layers:
        logits = model(mx.array(masked, dtype=mx.int32), exit_layer=layer).astype(mx.float32)
        probabilities = mx.softmax(logits, axis=-1)
        predictions_by_layer[layer] = np.asarray(mx.argmax(probabilities, axis=-1), dtype=np.int32)[0]
        confidences_by_layer[layer] = np.asarray(mx.max(probabilities, axis=-1), dtype=np.float32)[0]
        mx.eval(probabilities)
    token_rows = []
    for position in generated_positions:
        chosen_layer = exit_layers[-1]
        matched = False
        confidence = float(confidences_by_layer[chosen_layer][position])
        for layer in exit_layers:
            if int(predictions_by_layer[layer][position]) == int(final_ids[position]):
                chosen_layer = layer
                confidence = float(confidences_by_layer[layer][position])
                matched = True
                break
        token_rows.append({"position": position, "generated_offset": position - prompt_len,
                           "token_id": int(final_ids[position]), "exit_layer": int(chosen_layer),
                           "max_layer": int(exit_layers[-1]), "early_exited": bool(chosen_layer < exit_layers[-1]),
                           "layers_saved": int(exit_layers[-1] - chosen_layer),
                           "exit_confidence": confidence,
                           "exit_reason": "matched_final_token" if matched else "forced_final_layer"})
    exit_layers_used = [row["exit_layer"] for row in token_rows]
    saved = [row["layers_saved"] for row in token_rows]
    early = [row["early_exited"] for row in token_rows]
    return {"exit_layers": list(exit_layers), "token_count": len(token_rows),
            "early_exit_token_ratio": float(sum(early) / len(early)),
            "mean_exit_layer": float(sum(exit_layers_used) / len(exit_layers_used)),
            "min_exit_layer": int(min(exit_layers_used)), "max_exit_layer": int(max(exit_layers_used)),
            "mean_layers_saved": float(sum(saved) / len(saved)),
            "tokens": token_rows}


def run(a) -> dict:
    if a.prompts < 1 or a.max_new_tokens < 1:
        raise ValueError("--prompts and --max-new-tokens must be positive")
    ensure_outside_icloud(a.output_dir)
    output_jsonl = a.output_dir / "generations.jsonl"
    output_summary = a.output_dir / "summary.json"
    output_csv = a.output_dir / "summary.csv"
    if output_jsonl.exists() or output_summary.exists() or output_csv.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic output in {a.output_dir}")

    _train, _val, cache_meta, cache_meta_path = select_verified_cswiki_cache(a.cache_dir)
    del _train, _val
    tokenizer = load_tokenizer(cache_meta["tokenizer"])
    prompts = load_prompts(a.prompt_file, a.prompts)
    routes = tuple(a.routes or flexible_route_pool(25).keys())
    model_rows = json.loads(a.models_json.read_text())
    summary_rows, total_rows = [], 0
    started = time.time()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w") as out:
        for model_spec in model_rows:
            name = model_spec["name"]
            checkpoint_dir = Path(model_spec["checkpoint_dir"])
            fallback_arch = {"n_layers": int(model_spec.get("n_layers", 25)),
                             "d_model": int(model_spec["d_model"]),
                             "d_ff": int(model_spec["d_ff"]),
                             "n_heads": int(model_spec.get("n_heads", 4)),
                             "seq_len": int(model_spec.get("seq_len", 256))}
            checkpoints = discover_checkpoints(checkpoint_dir, a.include_latest_best)
            if not checkpoints:
                summary_rows.append({"model": name, "status": "no-checkpoints", "checkpoint_dir": str(checkpoint_dir)})
                continue
            for ckpt in checkpoints:
                arch = architecture_from_metadata(ckpt["metadata"], fallback_arch)
                model = build_model(tokenizer.get_vocab_size(), **arch)
                load_weights_only(model, ckpt["path"])
                model.eval()
                pool = flexible_route_pool(arch["n_layers"])
                unknown = sorted(set(routes) - set(pool))
                if unknown:
                    raise ValueError(f"unsupported route(s) for {name}: {unknown}")
                route_count = 0
                stop_counts = {}
                early_exit_tokens = total_exit_tokens = 0
                mean_exit_layer_sum = mean_layers_saved_sum = 0.0
                exit_rows = 0
                for route in routes:
                    model.set_layer_precisions(pool[route])
                    route_count += 1
                    exit_layers = default_exit_layers(arch["n_layers"])
                    for index, prompt in enumerate(prompts):
                        result = prompt_continuation(model, tokenizer, prompt=prompt,
                                                     max_new_tokens=a.max_new_tokens, passes=4)
                        final_text = result["refinements"][-1]
                        state = generation_state(result)
                        exits = exit_state(model, result, exit_layers) if a.measure_exits else None
                        stop_counts[state["stop_reason"]] = stop_counts.get(state["stop_reason"], 0) + 1
                        if exits:
                            early_exit_tokens += round(exits["early_exit_token_ratio"] * exits["token_count"])
                            total_exit_tokens += exits["token_count"]
                            mean_exit_layer_sum += exits["mean_exit_layer"] or 0.0
                            mean_layers_saved_sum += exits["mean_layers_saved"]
                            exit_rows += 1
                        row = {"model": name, "checkpoint_kind": ckpt["kind"], "checkpoint_step": ckpt["step"],
                               "checkpoint": str(ckpt["path"]), "route": route, "prompt_index": index,
                               "prompt": prompt, "final_text": final_text,
                               "generation_state": state,
                               "exit_state": exits,
                               "refinements": result["refinements"],
                               "architecture": arch}
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        total_rows += 1
                summary_rows.append({"model": name, "checkpoint_kind": ckpt["kind"], "checkpoint_step": ckpt["step"],
                                     "routes": route_count, "prompts": len(prompts),
                                     "rows": route_count * len(prompts), "stop_counts": stop_counts,
                                     "exit_summary": {
                                         "measured": bool(a.measure_exits),
                                         "early_exit_token_ratio": (early_exit_tokens / total_exit_tokens) if total_exit_tokens else 0.0,
                                         "mean_exit_layer": (mean_exit_layer_sum / exit_rows) if exit_rows else None,
                                         "mean_layers_saved": (mean_layers_saved_sum / exit_rows) if exit_rows else 0.0,
                                         "token_count": total_exit_tokens},
                                     "architecture": arch})
    summary = {"status": "complete", "mode": "batch-prompt-continuation",
               "cache_meta_path": str(cache_meta_path), "tokenizer": cache_meta["tokenizer"],
               "models_json": str(a.models_json), "routes": list(routes), "prompt_count": len(prompts),
               "max_new_tokens": a.max_new_tokens, "jsonl": str(output_jsonl),
               "summary_csv": str(output_csv), "rows": total_rows,
               "elapsed_seconds": time.time() - started, "checkpoints": summary_rows,
               "limits": ["Only existing checkpoint files are tested; historical step checkpoints cannot be reconstructed from report metrics.",
                          "Prompt prefix is held fixed and generation is diagnostic continuation, not chat/infill."]}
    atomic_text(output_summary, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "checkpoint_kind", "checkpoint_step", "routes", "prompts",
                                                    "rows", "stopped_by_eos", "stopped_by_max_new_tokens",
                                                    "early_exit_token_ratio", "mean_exit_layer", "mean_layers_saved"))
        writer.writeheader()
        for row in summary_rows:
            if "checkpoint_kind" in row:
                writer.writerow({"model": row["model"], "checkpoint_kind": row["checkpoint_kind"],
                                 "checkpoint_step": row["checkpoint_step"], "routes": row["routes"],
                                 "prompts": row["prompts"], "rows": row["rows"],
                                 "stopped_by_eos": row.get("stop_counts", {}).get("eos", 0),
                                 "stopped_by_max_new_tokens": row.get("stop_counts", {}).get("max_new_tokens", 0),
                                 "early_exit_token_ratio": row.get("exit_summary", {}).get("early_exit_token_ratio"),
                                 "mean_exit_layer": row.get("exit_summary", {}).get("mean_exit_layer"),
                                 "mean_layers_saved": row.get("exit_summary", {}).get("mean_layers_saved")})
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--models-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prompt-file", type=Path)
    p.add_argument("--prompts", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--routes", nargs="+", choices=tuple(flexible_route_pool(25)))
    p.add_argument("--include-latest-best", action="store_true")
    p.add_argument("--measure-exits", action=argparse.BooleanOptionalAction, default=True,
                   help="estimate token-level early-exit layer for generated positions")
    print(json.dumps(run(p.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
