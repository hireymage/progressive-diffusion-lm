#!/usr/bin/env python3
"""Resume the Czech flexible d64 model on PyTorch/CUDA without restarting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.torch_layerwise_model import masked_deep_supervision_loss, route_pool
from src.torch_mlx_checkpoint import convert_mlx_checkpoint, load_torch_checkpoint, save_torch_checkpoint


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def select_cache(cache_dir: Path, seq_len: int):
    valid = []
    for metadata_path in cache_dir.glob(f"meta_seq{seq_len}_*.json"):
        metadata = json.loads(metadata_path.read_text())
        source = metadata.get("source", {})
        suffix = metadata_path.stem.removeprefix(f"meta_seq{seq_len}_")
        train = cache_dir / f"train_seq{seq_len}_{suffix}.npy"
        val = cache_dir / f"val_seq{seq_len}_{suffix}.npy"
        if (metadata.get("format") != "cswiki-cache-v1"
                or not re.fullmatch(r"cswiki-\d{8}-pages-articles\.xml\.bz2", source.get("dump_filename", ""))
                or not train.exists() or not val.exists()):
            continue
        if sha256_file(train) != metadata.get("train_sha256") or sha256_file(val) != metadata.get("val_sha256"):
            raise ValueError(f"cache checksum mismatch: {metadata_path}")
        valid.append((metadata.get("total_tokens", 0), train, val, metadata))
    if not valid:
        raise FileNotFoundError("no verified Czech seq256 cache found")
    _, train, val, metadata = max(valid)
    return np.load(train, mmap_mode="r"), np.load(val, mmap_mode="r"), metadata


def fixed_batch(data, batch_size: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.asarray(data[rng.randint(0, len(data), size=batch_size)], dtype=np.int64)


def corrupt(batch: np.ndarray, mask_id: int, seed: int, device: torch.device):
    mask = np.random.RandomState(seed).random_sample(batch.shape) < .5
    mask[:, 0] = True
    tokens = np.where(mask, mask_id, batch)
    return (torch.from_numpy(tokens).to(device), torch.from_numpy(batch).to(device),
            torch.from_numpy(mask).to(device))


@torch.no_grad()
def evaluate_routes(model, val, batch_size: int, eval_steps: int, seed: int, device):
    was_training = model.training
    model.eval()
    rows = {}
    for name, schedule in route_pool(model.cfg.n_layers).items():
        model.set_layer_precisions(schedule)
        total_loss = total_correct = total_tokens = 0.0
        for index in range(eval_steps):
            batch = fixed_batch(val, batch_size, seed + index)
            tokens, targets, mask = corrupt(batch, model.cfg.mask_token_id(), seed + 100_000 + index, device)
            logits = model(tokens).reshape(-1, model.cfg.vocab_size).float()
            truth, flat_mask = targets.reshape(-1), mask.reshape(-1)
            losses = F.cross_entropy(logits, truth, reduction="none")
            total_loss += float(losses[flat_mask].sum())
            total_correct += float((logits.argmax(-1)[flat_mask] == truth[flat_mask]).sum())
            total_tokens += int(flat_mask.sum())
        loss = total_loss / total_tokens
        rows[name] = {"loss": loss, "accuracy": total_correct / total_tokens,
                      "masked_tokens": total_tokens, "perplexity": math.exp(loss)}
    model.train(was_training)
    worst = max(rows, key=lambda name: rows[name]["loss"])
    return {"per_route": rows, "worst_route": worst, "loss": rows[worst]["loss"],
            "accuracy": min(row["accuracy"] for row in rows.values()),
            "perplexity": rows[worst]["perplexity"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mlx-checkpoint", type=Path)
    source.add_argument("--resume", type=Path, help="PyTorch latest.pt checkpoint")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True, help="total target step, not additional steps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=32)
    parser.add_argument("--archive-every", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if "iCloud" in str(args.output_dir.resolve()):
        raise ValueError("CUDA outputs must be outside iCloud")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mlx_checkpoint:
        model, optimizer, metadata = convert_mlx_checkpoint(args.mlx_checkpoint, device, args.lr)
    else:
        model, optimizer, metadata = load_torch_checkpoint(args.resume, device, args.lr)
    start_step = int(metadata["step"])
    if args.steps <= start_step:
        raise ValueError(f"--steps must exceed checkpoint step {start_step}")
    train, val, cache_metadata = select_cache(args.cache_dir, model.cfg.max_seq_len)
    if metadata.get("cache_train_sha256") != cache_metadata.get("train_sha256"):
        raise ValueError("training cache does not match checkpoint")
    if metadata.get("cache_val_sha256") != cache_metadata.get("val_sha256"):
        raise ValueError("validation cache does not match checkpoint")

    history = list(metadata.get("history", []))
    best_loss = float(metadata.get("best_loss", float("inf")))
    pool = route_pool(model.cfg.n_layers)
    routes = tuple(pool)
    began = time.time()
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        route = routes[(step - 1) % len(routes)]
        model.set_layer_precisions(pool[route])
        batch = fixed_batch(train, args.batch_size, args.seed + step)
        tokens, targets, mask = corrupt(batch, model.cfg.mask_token_id(), args.seed + 10_000 + step, device)
        optimizer.zero_grad(set_to_none=True)
        loss = masked_deep_supervision_loss(model, tokens, targets, mask)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip or float("inf"))
        if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite value at step {step}: loss={loss}, grad={gradient_norm}")
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            report = evaluate_routes(model, val, args.batch_size, args.eval_steps, args.seed + 900_000, device)
            report.update({"step": step, "training_route": route, "train_loss": float(loss.detach()),
                           "gradient_norm": float(gradient_norm), "backend": "pytorch-cuda",
                           "elapsed_seconds": time.time() - began})
            history.append(report)
            current = metadata | {"step": step, "history": history,
                                  "best_loss": min(best_loss, report["loss"]), "backend": "pytorch-cuda"}
            save_torch_checkpoint(args.output_dir / "latest.pt", model, optimizer, current)
            if report["loss"] < best_loss:
                best_loss = report["loss"]
                save_torch_checkpoint(args.output_dir / "best.pt", model, optimizer, current)
            if args.archive_every and step % args.archive_every == 0:
                archive = args.output_dir / f"step_{step:07d}.pt"
                if not archive.exists():
                    save_torch_checkpoint(archive, model, optimizer, current)
            atomic_json(args.output_dir / "report.json", {"status": "running", "history": history})
    atomic_json(args.output_dir / "report.json", {"status": "complete", "step": args.steps,
                                                  "backend": "pytorch-cuda", "history": history})


if __name__ == "__main__":
    main()
