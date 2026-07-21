#!/usr/bin/env python3
"""Phase 2 inference evaluation: compare generate() vs generate_incremental()
vs generate_with_early_exit() on speed, quality, and step savings.

Pipeline:
  1. Train a small model for N steps (with checkpoint saving).
  2. Load the checkpoint.
  3. Run each generation mode K times and measure:
     - wall-clock latency
     - steps used (for early-exit modes)
     - token-level agreement with standard generate()
  4. Save results as JSON.

Usage:
  .venv/bin/python scripts/eval_inference.py --node m1-256 --steps 500 --gen-repeats 5
"""
from __future__ import annotations

import argparse
import json
import time
import os
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ExperimentConfig, ModelConfig, DataConfig, TrainConfig
from src.model import DiffusionLM
from src.diffusion import (
    generate,
    generate_incremental,
    generate_with_early_exit,
)
from src.train import train, load_checkpoint, build_optimizer
from src.data import build_and_cache_dataset, BatchIterator


def build_config(node: str, steps: int) -> ExperimentConfig:
    """Build a small config for quick inference eval."""
    model = ModelConfig(
        vocab_size=16000,
        d_model=256,
        n_layers=4,
        n_heads=8,
        d_ff=1024,
        max_seq_len=128,
        n_diffusion_steps=8,
        precision_schedule=[1, 2, 4, 8, 8, 4, 2, 1],  # progressive-up
        dropout=0.0,
    )
    data = DataConfig(
        seq_len=128,
        max_articles=200,
        max_text_bytes=2_000_000,
    )
    train = TrainConfig(
        batch_size=16,
        max_steps=steps,
        learning_rate=3e-4,
        weight_decay=0.0,
        warmup_steps=50,
        eval_every=999,
        eval_steps=1,
        log_every=100,
        save_checkpoints=True,
        checkpoint_every=999999,
        checkpoint_dir="checkpoints/inference_eval",
        results_dir="results/inference_eval",
        seed=201,
    )
    return ExperimentConfig(
        experiment_name=f"inference_eval_{node}",
        model=model,
        data=data,
        train=train,
    )


def measure_generation(
    model,
    mask_token_id: int,
    seq_len: int,
    precision_schedule: list[int],
    mode: str,
    gen_repeats: int,
    confidence_threshold: float = 0.95,
    min_steps: int = 1,
) -> dict:
    """Run generation in given mode and measure latency + steps."""
    latencies = []
    steps_used_list = []
    all_tokens = []

    for i in range(gen_repeats):
        t0 = time.perf_counter()

        if mode == "standard":
            tokens = generate(
                model, seq_len, mask_token_id, precision_schedule,
                batch_size=1,
            )
            steps_used = len(precision_schedule)
        elif mode == "incremental":
            tokens = generate_incremental(
                model, seq_len, mask_token_id, precision_schedule,
                batch_size=1,
                delta_weight=1.0,
            )
            steps_used = len(precision_schedule)
        elif mode == "early_exit":
            tokens, steps_used = generate_with_early_exit(
                model, seq_len, mask_token_id, precision_schedule,
                confidence_threshold=confidence_threshold,
                min_steps=min_steps,
                batch_size=1,
                use_incremental=False,
            )
        elif mode == "early_exit_incremental":
            tokens, steps_used = generate_with_early_exit(
                model, seq_len, mask_token_id, precision_schedule,
                confidence_threshold=confidence_threshold,
                min_steps=min_steps,
                batch_size=1,
                use_incremental=True,
                delta_weight=1.0,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        mx.eval(tokens)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)
        steps_used_list.append(steps_used)
        all_tokens.append(tokens.tolist())

    return {
        "mode": mode,
        "latency_mean_s": round(sum(latencies) / len(latencies), 4),
        "latency_min_s": round(min(latencies), 4),
        "latency_max_s": round(max(latencies), 4),
        "latencies_s": [round(l, 4) for l in latencies],
        "steps_mean": sum(steps_used_list) / len(steps_used_list),
        "steps_list": steps_used_list,
        "confidence_threshold": confidence_threshold if "early_exit" in mode else None,
        "seq_len": seq_len,
        "n_steps_max": len(precision_schedule),
        "tokens_sample": all_tokens[0] if all_tokens else None,
    }


def run_eval(node: str, train_steps: int, gen_repeats: int) -> dict:
    """Full inference evaluation pipeline."""
    cfg = build_config(node, train_steps)

    print(f"\n{'='*60}")
    print(f"Phase 2 Inference Evaluation — node={node}")
    print(f"{'='*60}\n")

    # Step 1: Train (saves checkpoint at the end)
    print("[1/3] Training small model...")
    train(cfg)

    # Step 2: Load checkpoint
    print("[2/3] Loading checkpoint...")
    ckpt_dir = Path(cfg.train.checkpoint_dir) / cfg.experiment_name
    meta_path = ckpt_dir / "latest_meta.json"
    if not meta_path.exists():
        # Try finding the final checkpoint
        ckpts = sorted(ckpt_dir.glob("step_*.npz"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
        ckpt_path = str(ckpts[-1])
    else:
        with open(meta_path) as f:
            meta = json.load(f)
        ckpt_path = str(ckpt_dir / f"step_{meta['step']:07d}.npz")

    model = DiffusionLM(cfg.model)
    load_checkpoint(model, ckpt_path, optimizer=None)
    mx.eval(model.parameters())
    model.eval()

    mask_token_id = cfg.model.mask_token_id()
    seq_len = cfg.model.max_seq_len
    precision_schedule = cfg.model.precision_schedule

    # Step 3: Run inference comparisons
    print(f"[3/3] Running inference comparisons ({gen_repeats} repeats each)...\n")

    results = []

    # Standard and incremental baselines
    for mode in ["standard", "incremental"]:
        r = measure_generation(
            model, mask_token_id, seq_len, precision_schedule,
            mode, gen_repeats,
        )
        results.append(r)
        print(f"  {mode:.<30s} latency={r['latency_mean_s']:.3f}s  steps={r['steps_mean']:.1f}")

    # Early exit at multiple thresholds.
    # For a small vocab model with low confidence, we also test very low
    # thresholds to actually trigger early exit.
    thresholds = [0.01, 0.02, 0.03, 0.05, 0.10, 0.50]
    for thresh in thresholds:
        for use_inc in [False, True]:
            mode = "early_exit_incremental" if use_inc else "early_exit"
            r = measure_generation(
                model, mask_token_id, seq_len, precision_schedule,
                mode, gen_repeats,
                confidence_threshold=thresh,
                min_steps=1,
            )
            results.append(r)
            inc_label = " +inc" if use_inc else ""
            print(f"  {mode}{inc_label} (t={thresh}):.{30-len(mode)-len(inc_label)-12:d}s"
                  f"  latency={r['latency_mean_s']:.3f}s  steps={r['steps_mean']:.1f}/{r['n_steps_max']}")

    # Compute agreement with standard mode
    standard_tokens = results[0]["tokens_sample"]
    for r in results[1:]:
        if r["tokens_sample"] is not None:
            agree = sum(
                1 for a, b in zip(standard_tokens[0], r["tokens_sample"][0])
                if a == b
            ) / len(standard_tokens[0])
            r["token_agreement_with_standard"] = round(agree, 4)
        else:
            r["token_agreement_with_standard"] = None

    # Summary
    summary = {
        "node": node,
        "train_steps": train_steps,
        "gen_repeats": gen_repeats,
        "seq_len": seq_len,
        "precision_schedule": precision_schedule,
        "n_steps_max": len(precision_schedule),
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Save
    out_dir = Path("results/inference_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{node}_inference_eval.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'='*75}")
    print(f"{'Mode':<35} {'Latency(s)':>10} {'Steps':>8} {'Speedup':>8} {'Agree%':>7}")
    print(f"{'-'*75}")
    baseline_latency = results[0]["latency_mean_s"]
    for r in results:
        mode_label = r["mode"]
        if r.get("confidence_threshold") is not None:
            mode_label += f" (t={r['confidence_threshold']})"
            if r.get("use_incremental"):
                mode_label += " +inc"
        speedup = baseline_latency / r["latency_mean_s"] if r["latency_mean_s"] > 0 else 0
        agree = r.get("token_agreement_with_standard")
        agree_str = f"{agree*100:.1f}%" if agree is not None else "—"
        print(f"{mode_label:<35} {r['latency_mean_s']:>10.3f} {r['steps_mean']:>8.1f} {speedup:>7.2f}x {agree_str:>7}")
    print(f"{'='*75}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 inference evaluation")
    parser.add_argument("--node", type=str, required=True,
                        choices=["m1-256", "m1-512", "m4-air"])
    parser.add_argument("--steps", type=int, default=2000,
                        help="Quick training steps before inference eval")
    parser.add_argument("--gen-repeats", type=int, default=5,
                        help="Number of generation repeats per mode for timing")
    args = parser.parse_args()

    run_eval(args.node, args.steps, args.gen_repeats)