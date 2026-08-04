#!/usr/bin/env python3
"""Train the first shared-master flexible model on verified Czech Wikipedia only.

The input cache must have been produced by ``cswiki_pipeline build-cache``.
This program is deliberately offline: it only re-hashes local cache files and
never falls back to the English cache builder or to a network dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
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

from scripts.layerwise_diagnostics import flexible_route_pool, route_for_training_step
from src.config import LayerwiseModelConfig
from src.data import load_tokenizer
from src.layerwise_model import LayerwiseProgressiveLM, masked_deep_supervision_loss, proxy_cost_for_schedule

N_LAYERS, D_MODEL, D_FF, N_HEADS, SEQ_LEN = 25, 64, 256, 4, 256
MILESTONES = ((5, .1), (10, .2), (15, .3), (20, .4), (25, 1.0))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def select_verified_cswiki_cache(cache_dir: Path) -> tuple[np.ndarray, np.ndarray, dict, Path]:
    """Re-hash and select one genuine Czech seq256 cache, never an enwiki cache."""
    valid = []
    for meta_path in cache_dir.glob("meta_seq256_*.json"):
        meta = json.loads(meta_path.read_text())
        source = meta.get("source", {})
        suffix = meta_path.name.removeprefix("meta_seq256_").removesuffix(".json")
        train = cache_dir / f"train_seq256_{suffix}.npy"
        val = cache_dir / f"val_seq256_{suffix}.npy"
        if (meta.get("format") != "cswiki-cache-v1" or meta.get("seq_len") != SEQ_LEN
                or not re.fullmatch(r"cswiki-\d{8}-pages-articles\.xml\.bz2",
                                    str(source.get("dump_filename", "")))
                or not re.fullmatch(r"[0-9a-f]{40}", str(source.get("sha1", "")))):
            continue
        if not train.exists() or not val.exists():
            continue
        if sha256_file(train) != meta.get("train_sha256") or sha256_file(val) != meta.get("val_sha256"):
            raise ValueError(f"cswiki cache checksum mismatch: {meta_path}")
        tokenizer_dir = Path(meta.get("tokenizer", ""))
        tokenizer_file = tokenizer_dir / "tokenizer.json"
        if not tokenizer_file.exists() or sha256_file(tokenizer_file) != meta.get("tokenizer_sha256"):
            raise ValueError(f"cswiki tokenizer provenance mismatch: {meta_path}")
        if not meta.get("n_train_chunks") or not meta.get("n_val_chunks"):
            raise ValueError(f"cswiki cache has an empty split: {meta_path}")
        valid.append((meta.get("total_tokens", 0), train, val, meta, meta_path))
    if not valid:
        raise FileNotFoundError("No checksum-verified cswiki-cache-v1 seq256 cache was found; English caches are ignored.")
    _, train, val, meta, meta_path = max(valid, key=lambda row: row[0])
    return np.load(train, mmap_mode="r"), np.load(val, mmap_mode="r"), meta, meta_path


def fixed_batch(data: np.ndarray, batch_size: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.asarray(data[rng.randint(0, len(data), size=batch_size)], dtype=np.int32)


def corrupt_50(batch: np.ndarray, mask_id: int, seed: int) -> tuple[mx.array, mx.array, mx.array]:
    mask = np.random.RandomState(seed).random_sample(batch.shape) < .5
    mask[:, 0] = True
    return mx.array(np.where(mask, mask_id, batch), dtype=mx.int32), mx.array(batch, dtype=mx.int32), mx.array(mask)


def build_model(vocab_size: int) -> LayerwiseProgressiveLM:
    cfg = LayerwiseModelConfig(vocab_size=vocab_size, d_model=D_MODEL, d_ff=D_FF, n_heads=N_HEADS,
        n_layers=N_LAYERS, min_exit_layer=5, max_seq_len=SEQ_LEN,
        layer_precisions=flexible_route_pool(N_LAYERS)["q8_only"])
    return LayerwiseProgressiveLM(cfg)


def evaluate_routes(model, val: np.ndarray, batch_size: int, eval_steps: int, seed: int) -> dict:
    """Held-out, fixed-mask route metrics; report the pessimistic route summary."""
    pool, rows = flexible_route_pool(N_LAYERS), {}
    was_training = model.training
    model.eval()
    try:
        for name, schedule in pool.items():
            model.set_layer_precisions(schedule)
            total_loss = total_correct = total_tokens = 0.0
            for index in range(eval_steps):
                batch = fixed_batch(val, batch_size, seed + index)
                x, targets, mask = corrupt_50(batch, model.cfg.mask_token_id(), seed + 100_000 + index)
                logits = model(x, exit_layer=N_LAYERS).reshape(-1, model.cfg.vocab_size).astype(mx.float32)
                truth, flat_mask = targets.reshape(-1), mask.reshape(-1).astype(mx.float32)
                n = mx.maximum(mx.sum(flat_mask), mx.array(1.0))
                losses = -nn.log_softmax(logits, axis=-1)[mx.arange(logits.shape[0]), truth]
                loss_sum = mx.sum(losses * flat_mask)
                correct = mx.sum((mx.argmax(logits, axis=-1) == truth) * flat_mask)
                mx.eval(loss_sum, correct, n)
                total_loss += float(loss_sum); total_correct += float(correct); total_tokens += float(n)
            loss = total_loss / total_tokens
            rows[name] = {"loss": loss, "accuracy": total_correct / total_tokens,
                          "masked_tokens": int(total_tokens), "perplexity": float(np.exp(loss)),
                          "proxy_full_cost": proxy_cost_for_schedule(schedule)}
    finally:
        model.train(was_training)
    worst_name = max(rows, key=lambda name: rows[name]["loss"])
    finite = all(np.isfinite(row["loss"]) and np.isfinite(row["perplexity"])
                 for row in rows.values())
    return {"per_route": rows, "worst_route": worst_name,
            "loss": rows[worst_name]["loss"], "accuracy": min(row["accuracy"] for row in rows.values()),
            "masked_tokens": rows[worst_name]["masked_tokens"], "perplexity": rows[worst_name]["perplexity"],
            "quality_gate": {"passed": finite, "metric": "worst-route held-out loss",
                             "decision": "all routes must be finite; best checkpoint minimizes the worst route"}}


def checkpoint(model, optimizer, directory: Path, kind: str, metadata: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{kind}.npz"
    temporary = directory / f".{kind}.part.npz"
    payload = dict(tree_flatten(model.parameters()))
    payload.update({"opt_" + key: value for key, value in tree_flatten(optimizer.state)})
    mx.savez(str(temporary), **payload)
    os.replace(temporary, target)
    atomic_json_write(directory / f"{kind}.json", metadata)


def load_checkpoint(model, optimizer, path: Path, expected: dict) -> dict:
    metadata = json.loads(path.with_suffix(".json").read_text())
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"resume metadata mismatch for {key}")
    data = mx.load(str(path))
    model.load_weights([(key, value) for key, value in data.items() if not key.startswith("opt_")])
    state = [(key.removeprefix("opt_"), value) for key, value in data.items() if key.startswith("opt_")]
    if not state:
        raise ValueError("checkpoint has no optimizer state")
    optimizer.state = tree_unflatten(state)
    mx.eval(model.parameters(), optimizer.state)
    return metadata


def ensure_outside_icloud(path: Path | str) -> None:
    if "iCloud" in str(Path(path).expanduser().resolve()):
        raise ValueError("--output and --checkpoint-dir must be outside iCloud and explicitly supplied")


def run(a) -> dict:
    if not 1 <= a.steps <= 40000:
        raise ValueError("--steps must be in [1, 40000]")
    if a.batch_size < 1 or a.eval_steps < 1 or a.eval_every < 1:
        raise ValueError("batch size, eval steps, and eval interval must be positive")
    if not a.resume and a.output.exists():
        raise FileExistsError(f"Refusing to overwrite historical report: {a.output}")
    if not a.resume and any((a.checkpoint_dir / name).exists()
                            for name in ("latest.npz", "latest.json", "best.npz", "best.json")):
        raise FileExistsError(f"Refusing to overwrite historical checkpoints: {a.checkpoint_dir}")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    train, val, cache_meta, cache_meta_path = select_verified_cswiki_cache(a.cache_dir)
    tokenizer = load_tokenizer(cache_meta["tokenizer"])
    mx.random.seed(a.seed)
    model, optimizer = build_model(tokenizer.get_vocab_size()), optim.AdamW(learning_rate=a.lr)
    pool = flexible_route_pool(N_LAYERS)
    contract = {"cache_train_sha256": cache_meta["train_sha256"], "cache_val_sha256": cache_meta["val_sha256"],
                "route_pool": pool, "strategy": "A-constant-50pct", "architecture": [N_LAYERS, D_MODEL, D_FF, N_HEADS, SEQ_LEN]}
    estimate = sum(np.asarray(value).nbytes for _, value in tree_flatten(model.parameters())) * 4
    if shutil.disk_usage(a.output.parent).free < int(a.min_free_gb * 1024**3) + 2 * estimate:
        raise RuntimeError("disk budget guard failed for latest+best checkpoints")
    start_step, best_loss, history = 0, float("inf"), []
    latest = a.checkpoint_dir / "latest.npz"
    if a.resume:
        if not latest.exists(): raise FileNotFoundError("--resume requires latest.npz")
        restored = load_checkpoint(model, optimizer, latest, contract)
        start_step, best_loss, history = int(restored["step"]), float(restored["best_loss"]), restored.get("history", [])
    grad_fn = nn.value_and_grad(model, lambda m, x, t, mask: masked_deep_supervision_loss(
        m, x, t, mask, supervised_layers=tuple(layer for layer, _ in MILESTONES),
        layer_weights=tuple(weight for _, weight in MILESTONES)))
    began = time.time()
    for step in range(start_step + 1, a.steps + 1):
        route, schedule = route_for_training_step(pool, step)
        model.set_layer_precisions(schedule)
        x, targets, mask = corrupt_50(fixed_batch(train, a.batch_size, a.seed + step), model.cfg.mask_token_id(), a.seed + 10_000 + step)
        loss, grads = grad_fn(model, x, targets, mask); optimizer.update(model, grads); mx.eval(loss, model.parameters())
        if step % a.eval_every == 0 or step == a.steps:
            report = evaluate_routes(model, val, a.batch_size, a.eval_steps, a.seed + 900_000)
            # Avoid runtime precision leakage after the three route passes.
            next_route, next_schedule = route_for_training_step(pool, step + 1); model.set_layer_precisions(next_schedule)
            report.update({"step": step, "training_route": route, "next_training_route": next_route,
                           "train_loss": float(loss), "elapsed_seconds": time.time() - began})
            history.append(report)
            metadata = contract | {"step": step, "best_loss": min(best_loss, report["loss"]), "history": history}
            checkpoint(model, optimizer, a.checkpoint_dir, "latest", metadata)
            if report["loss"] < best_loss:
                best_loss = report["loss"]; checkpoint(model, optimizer, a.checkpoint_dir, "best", metadata)
            atomic_json_write(a.output, {"status": "running", "cache_meta_path": str(cache_meta_path), "history": history})
    final = history[-1] if history else None
    result = {"status": "complete", "language": "cswiki-only", "strategy": "A-constant-50pct",
              "architecture": {"n_layers": N_LAYERS, "d_model": D_MODEL, "d_ff": D_FF, "n_heads": N_HEADS, "seq_len": SEQ_LEN,
                               "route_pool": pool, "active_schedule": list(model.cfg.layer_precisions)},
              "cache_meta_path": str(cache_meta_path), "cache_metadata": cache_meta, "checkpoint_policy": "atomic latest+best; best is minimum worst-route held-out loss",
              "steps": a.steps, "history": history, "final": final}
    atomic_json_write(a.output, result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=40000); p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-steps", type=int, default=32); p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260804); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-free-gb", type=float, default=10.0); p.add_argument("--resume", action="store_true")
    a = p.parse_args(); ensure_outside_icloud(a.output); ensure_outside_icloud(a.checkpoint_dir)
    print(json.dumps(run(a), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
