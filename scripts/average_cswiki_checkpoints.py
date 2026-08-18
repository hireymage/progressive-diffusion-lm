#!/usr/bin/env python3
"""Average route-specific MLX checkpoints into one resumable checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np


CONTRACT_KEYS = (
    "cache_train_sha256", "cache_val_sha256", "route_pool", "strategy", "architecture"
)


def merge_checkpoints(inputs: list[Path], output: Path) -> dict:
    if len(inputs) < 2:
        raise ValueError("at least two checkpoints are required")
    payloads = [mx.load(str(path)) for path in inputs]
    keyset = set(payloads[0])
    if any(set(payload) != keyset for payload in payloads[1:]):
        raise ValueError("checkpoint key sets differ")

    metadata = [json.loads(path.with_suffix(".json").read_text()) for path in inputs]
    steps = {int(item["step"]) for item in metadata}
    if len(steps) != 1:
        raise ValueError(f"checkpoint steps differ: {sorted(steps)}")
    for key in CONTRACT_KEYS:
        values = [item.get(key) for item in metadata]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"checkpoint contract differs for {key}")

    merged = {}
    for key in sorted(keyset):
        arrays = [np.asarray(payload[key]) for payload in payloads]
        if any(array.shape != arrays[0].shape for array in arrays[1:]):
            raise ValueError(f"shape mismatch for {key}")
        if np.issubdtype(arrays[0].dtype, np.floating):
            value = np.mean(np.stack(arrays, axis=0), axis=0, dtype=np.float64).astype(arrays[0].dtype)
            if not np.isfinite(value).all():
                raise FloatingPointError(f"non-finite merged value for {key}")
        else:
            if any(not np.array_equal(array, arrays[0]) for array in arrays[1:]):
                raise ValueError(f"non-floating state differs for {key}")
            value = arrays[0]
        merged[key] = mx.array(value)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.part.npz")
    mx.savez(str(temporary), **merged)
    os.replace(temporary, output)

    branch_finals = [item.get("history", [])[-1] for item in metadata if item.get("history")]
    conservative = max(branch_finals, key=lambda row: float(row.get("loss", float("-inf")))) if branch_finals else None
    base_history = metadata[0].get("history", [])[:-1] if metadata[0].get("history") else []
    history = base_history + ([conservative | {
        "distributed_merge": True,
        "merged_routes": [item.get("training_route") for item in branch_finals],
    }] if conservative else [])
    result = metadata[0] | {
        "step": steps.pop(),
        "best_loss": min(float(item.get("best_loss", float("inf"))) for item in metadata),
        "history": history,
        "distributed_merge": {
            "method": "equal-weight local-SGD checkpoint and Adam-state average",
            "inputs": [str(path) for path in inputs],
            "routes": [item.get("training_route") for item in branch_finals],
        },
    }
    output.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge_checkpoints(args.input, args.output)
    print(json.dumps({"output": str(args.output), "step": result["step"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
