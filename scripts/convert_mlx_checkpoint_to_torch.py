#!/usr/bin/env python3
"""Convert one MLX checkpoint, including AdamW state, to PyTorch format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.torch_mlx_checkpoint import convert_mlx_checkpoint, save_torch_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    model, optimizer, metadata = convert_mlx_checkpoint(args.input, torch.device(args.device), args.lr)
    metadata = metadata | {"backend": "pytorch", "converted_from": str(args.input)}
    save_torch_checkpoint(args.output, model, optimizer, metadata)
    print(json.dumps({"output": str(args.output), "step": metadata["step"],
                      "optimizer_parameters": len(optimizer.state)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
