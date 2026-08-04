#!/usr/bin/env python3
"""Small node-smokes for the layer-wise grouped-precision prototype.

Run one mode per node:
  python scripts/layerwise_smoke.py --mode schedule    # m1-256
  python scripts/layerwise_smoke.py --mode train       # m1-512
  python scripts/layerwise_smoke.py --mode early-exit  # m4-air
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import LayerwiseModelConfig
from src.layerwise_model import LayerwiseProgressiveLM, masked_deep_supervision_loss


def smoke_config() -> LayerwiseModelConfig:
    return LayerwiseModelConfig(
        vocab_size=64, d_model=64, n_heads=4, d_ff=128, max_seq_len=16,
    )


def run_schedule() -> dict:
    cfg, model = smoke_config(), None
    model = LayerwiseProgressiveLM(cfg)
    tokens = mx.array([[1, 2, cfg.mask_token_id(), 4, 5, 6, 7, 8]])
    outputs = model.forward_intermediates(tokens, exit_layer=8)
    mx.eval(outputs)
    return {
        "schedule": cfg.layer_precisions,
        "exit_layers": list(outputs),
        "layer_8_proxy_cost": model.proxy_cost(8),
        "full_proxy_cost": model.proxy_cost(),
        "fp32_reference_cost": model.fp32_reference_cost(),
    }


def run_train() -> dict:
    mx.random.seed(20260804)
    cfg, model = smoke_config(), None
    model, optimizer = LayerwiseProgressiveLM(cfg), optim.Adam(learning_rate=1e-3)
    targets = mx.array([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
    masked = mx.array([[False, True, False, True, True, False, False, True],
                       [True, False, True, False, False, True, True, False]])
    inputs = mx.where(masked, mx.full_like(targets, cfg.mask_token_id()), targets)
    value_and_grad = nn.value_and_grad(model, masked_deep_supervision_loss)
    losses = []
    for _ in range(3):
        loss, grads = value_and_grad(model, inputs, targets, masked)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        losses.append(float(loss))
    return {"steps": 3, "losses": losses, "final_loss": losses[-1],
            "supervised_exits": list(range(cfg.min_exit_layer, cfg.n_layers + 1))}


def run_early_exit() -> dict:
    cfg, model = smoke_config(), None
    model = LayerwiseProgressiveLM(cfg)
    tokens = mx.array([[cfg.mask_token_id()] * 8])
    # Deliberately accepts the first eligible layer.  Calibration is a later,
    # held-out-data experiment, not a property claimed by this smoke.
    result = model.early_exit(tokens, margin_threshold=-1.0)
    mx.eval(result.logits)
    return {"exit_layer": result.exit_layer, "proxy_cost": result.proxy_cost,
            "mean_margin": result.mean_margin, "logits_shape": list(result.logits.shape)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("schedule", "train", "early-exit"), required=True)
    args = parser.parse_args()
    runners = {"schedule": run_schedule, "train": run_train, "early-exit": run_early_exit}
    print(json.dumps(runners[args.mode](), indent=2))


if __name__ == "__main__":
    main()
