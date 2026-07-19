"""
Evaluation and comparison script.

Usage
-----
python -m src.evaluate \\
    --baseline checkpoints/baseline/step_0010000.npz \\
    --progressive checkpoints/progressive/step_0010000.npz \\
    --config configs/progressive_1_2_4.json

Reports
-------
For each model:
  * Validation loss
  * Masked-token reconstruction accuracy
  * Parameter count
  * Theoretical weight storage (full-precision vs quantised)
  * Estimated bits per parameter
  * Inference speed (tokens/second)

Comparison:
  * Progressive vs Baseline on all metrics

IMPORTANT DISCLAIMER (printed in report):
  All low-bit operations in this implementation are SIMULATED using float32
  MLX operations.  Theoretical storage and FLOPs savings are estimated from
  the bit-width specification and do NOT reflect actual wall-clock speedups.
  Real speedups require dedicated low-bit hardware kernels (e.g., Apple Neural
  Engine 1-bit MACs, or custom MLX Metal shaders).
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .config import ExperimentConfig, ModelConfig
from .model import DiffusionLM
from .diffusion import corrupt_tokens, mask_rate_to_step
from .data import build_and_cache_dataset, BatchIterator
from .quantization import model_storage_report, bits_per_param_from_schedule
from .train import load_checkpoint


DISCLAIMER = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT: This is an experimental proof-of-concept.

All 1-bit, 2-bit, and 4-bit weight operations are SIMULATED
using float32 MLX operations via Straight-Through Estimation.

  * Theoretical storage savings are real (if weights were stored
    as integers).
  * Wall-clock speed numbers reflect float32 simulation overhead,
    NOT actual low-bit hardware performance.
  * Real speedups require:
      - Custom low-bit Metal kernels for Apple Silicon
      - Apple Neural Engine support for 1-bit MAC operations
      - Or dedicated hardware (e.g., 1-bit ASIC)

Do not interpret simulation speed as real-world 1-bit performance.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def load_model(ckpt_path: str, cfg: ExperimentConfig) -> DiffusionLM:
    model = DiffusionLM(cfg.model)
    load_checkpoint(model, ckpt_path)
    return model


def eval_model(
    model: DiffusionLM,
    val_iter: BatchIterator,
    cfg: ExperimentConfig,
    n_steps: int = 100,
    label: str = "model",
    rng_seed: int | None = None,
) -> dict:
    """Compute validation metrics on deterministic fixtures when seeded."""
    model.eval()
    if rng_seed is not None:
        mx.random.seed(rng_seed)
    mask_token_id = cfg.model.mask_token_id()
    precision_schedule = (
        cfg.model.precision_schedule
        if cfg.model.model_type == "progressive"
        else [16] * cfg.model.n_diffusion_steps
    )

    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    n_tokens_per_step = []

    t_start = time.time()
    for _ in range(n_steps):
        batch_np = next(val_iter)
        x0 = mx.array(batch_np, dtype=mx.int32)
        B, L = x0.shape

        mask_rates = mx.random.uniform(low=0.1, high=1.0, shape=(B,))
        x_t, mask = corrupt_tokens(x0, mask_rates, mask_token_id)

        mean_rate = float(mask_rates.mean())
        step_idx = mask_rate_to_step(mean_rate, cfg.model.n_diffusion_steps)
        bits = precision_schedule[step_idx]
        model.set_bits(bits)

        logits = model(x_t, mask_rates)
        mx.eval(logits)

        V = logits.shape[-1]
        flat_logits = logits.reshape(-1, V)
        flat_targets = x0.reshape(-1)
        flat_mask = mask.reshape(-1).astype(mx.float32)

        log_probs = nn.log_softmax(flat_logits, axis=-1)
        token_loss = -log_probs[mx.arange(flat_logits.shape[0]), flat_targets]
        n_masked = flat_mask.sum()
        loss = (token_loss * flat_mask).sum() / mx.maximum(n_masked, mx.array(1.0))

        preds = flat_logits.argmax(axis=-1)
        correct = ((preds == flat_targets) * flat_mask).sum()

        mx.eval(loss, correct, n_masked)
        total_loss += float(loss)
        total_correct += float(correct)
        total_masked += float(n_masked)
        n_tokens_per_step.append(int(n_masked))

    elapsed = time.time() - t_start
    total_tokens_processed = sum(n_tokens_per_step)
    tokens_per_sec = total_tokens_processed / elapsed

    storage = model_storage_report(model, precision_schedule)

    return {
        "label": label,
        "val_loss": total_loss / n_steps,
        "val_accuracy": total_correct / max(total_masked, 1),
        "masked_tokens_per_sec": tokens_per_sec,
        "n_params": storage["total_params"],
        "q_linear_weight_params": storage["q_linear_weight_params"],
        "non_quantized_params": storage["non_quantized_params"],
        "average_step_weight_bits": storage["average_step_weight_bits"],
        "actual_model_mb": storage["actual_model_mb"],
        "actual_compression_vs_fp32": storage["actual_compression_vs_fp32"],
        "hypothetical_packed_mb": storage["hypothetical_packed_mb"],
        "hypothetical_packed_compression_vs_fp32": storage[
            "hypothetical_packed_compression_vs_fp32"
        ],
    }


def measure_inference_speed(
    model: DiffusionLM,
    cfg: ExperimentConfig,
    seq_len: int = 64,
    batch_size: int = 4,
    n_warmup: int = 5,
    n_measure: int = 20,
) -> dict:
    """Measure generation (refinement) speed."""
    from .diffusion import generate

    precision_schedule = (
        cfg.model.precision_schedule
        if cfg.model.model_type == "progressive"
        else [16] * cfg.model.n_diffusion_steps
    )
    mask_token_id = cfg.model.mask_token_id()

    # Warmup
    for _ in range(n_warmup):
        tokens = generate(model, seq_len, mask_token_id, precision_schedule, batch_size=batch_size)
        mx.eval(tokens)

    # Measure
    t0 = time.time()
    for _ in range(n_measure):
        tokens = generate(model, seq_len, mask_token_id, precision_schedule, batch_size=batch_size)
        mx.eval(tokens)
    elapsed = time.time() - t0

    total_tokens = n_measure * batch_size * seq_len
    return {
        "gen_tokens_per_sec": total_tokens / elapsed,
        "gen_ms_per_sequence": elapsed / (n_measure * batch_size) * 1000,
        "n_diffusion_steps": len(precision_schedule),
    }


def precision_report(model: DiffusionLM, cfg: ExperimentConfig) -> str:
    """Human-readable precision schedule report."""
    if cfg.model.model_type == "baseline":
        return "  Baseline: all steps use full precision (no quantisation)"
    lines = ["  Progressive precision schedule:"]
    schedule = cfg.model.precision_schedule
    for i, bits in enumerate(schedule):
        lines.append(f"    Step {i+1:2d}: {bits}-bit weights")
    return "\n".join(lines)


def estimate_flop_savings(precision_schedule: list[int]) -> dict:
    """Estimate theoretical FLOPs reduction vs full-precision baseline."""
    avg_bits = np.mean(precision_schedule)
    full_bits = 16.0  # bfloat16 baseline
    # Linear layer FLOPs scale roughly with bit-width (for MACs)
    theoretical_reduction = full_bits / avg_bits
    return {
        "avg_bits": avg_bits,
        "theoretical_flop_reduction_vs_bf16": theoretical_reduction,
        "note": "Simulated; actual speedup requires dedicated low-bit kernels",
    }


def print_report(results: list[dict], flop_info: dict = None) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(DISCLAIMER)

    for r in results:
        print(f"\n{'─'*40}")
        print(f"  Model: {r['label']}")
        print(f"{'─'*40}")
        print(f"  Validation loss:        {r['val_loss']:.4f}")
        print(f"  Masked-token accuracy:  {r['val_accuracy']:.4f}  ({r['val_accuracy']*100:.1f}%)")
        print(f"  Parameters (total):     {r['n_params']:,}")
        print(f"    QuantizedLinear:       {r['q_linear_weight_params']:,}")
        print(f"    Non-quantized:         {r['non_quantized_params']:,}")
        print(f"  Avg step weight bits:     {r['average_step_weight_bits']:.3f}")
        print(f"  Actual model storage:     {r['actual_model_mb']:.2f} MB (FP32)")
        print(f"  Actual vs FP32:           {r['actual_compression_vs_fp32']:.1f}×")
        if r["hypothetical_packed_mb"] is not None:
            print(f"  Packed lower bound:       {r['hypothetical_packed_mb']:.2f} MB")
            print(f"  Hyp. packed vs FP32:      "
                  f"{r['hypothetical_packed_compression_vs_fp32']:.1f}×")
        else:
            print("  Packed lower bound:       n/a for dynamic schedule")
        print(f"  Throughput (masked tok/s): {r['masked_tokens_per_sec']:,.0f}")

        if "gen_tokens_per_sec" in r:
            print(f"  Generation speed:      {r['gen_tokens_per_sec']:,.0f} tokens/s")
            print(f"  Per-sequence latency:  {r['gen_ms_per_sequence']:.1f} ms")
            print(f"  Refinement steps:      {r['n_diffusion_steps']}")

    if flop_info:
        print(f"\n{'─'*40}")
        print("  Theoretical Compute Savings (Progressive vs Baseline)")
        print(f"{'─'*40}")
        print(f"  Average bits/step:      {flop_info['avg_bits']:.2f}")
        print(f"  vs BF16 baseline:       {flop_info['theoretical_flop_reduction_vs_bf16']:.1f}×")
        print(f"  ⚠ {flop_info['note']}")

    if len(results) == 2:
        r_b, r_p = results[0], results[1]
        print(f"\n{'─'*40}")
        print("  Comparison: Progressive vs Baseline")
        print(f"{'─'*40}")
        loss_delta = r_p['val_loss'] - r_b['val_loss']
        acc_delta = r_p['val_accuracy'] - r_b['val_accuracy']
        print(f"  Loss difference:  {loss_delta:+.4f}  (positive = worse)")
        print(f"  Acc difference:   {acc_delta:+.4f}  (positive = better)")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate progressive vs baseline diffusion LM")
    parser.add_argument("--baseline", type=str, help="Baseline checkpoint path")
    parser.add_argument("--progressive", type=str, help="Progressive checkpoint path")
    parser.add_argument("--config", type=str, required=True, help="Config JSON")
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--measure-speed", action="store_true")
    parser.add_argument("--save-results", type=str, default=None,
                        help="Path to save JSON results (e.g. results/comparison.json)")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_json(args.config)

    train_data, val_data = build_and_cache_dataset(
        tokenizer_path=cfg.data.tokenizer_path,
        cache_dir=cfg.data.data_cache_dir,
        seq_len=cfg.data.seq_len,
        max_articles=cfg.data.max_articles,
        max_text_bytes=cfg.data.max_text_bytes,
        dataset_name=cfg.data.dataset_name,
        dataset_config=cfg.data.dataset_config,
        dataset_revision=cfg.data.dataset_revision,
        train_split=cfg.data.train_split,
        seed=cfg.train.seed,
    )
    results = []
    flop_info = None
    eval_seed = cfg.train.seed + 2

    if args.baseline:
        b_cfg = ExperimentConfig.from_json(args.config)
        b_cfg.model.model_type = "baseline"
        model = load_model(args.baseline, b_cfg)
        r = eval_model(
            model,
            BatchIterator(val_data, cfg.train.batch_size, seed=eval_seed),
            b_cfg,
            n_steps=args.eval_steps,
            label="Baseline (full precision)",
            rng_seed=eval_seed,
        )
        if args.measure_speed:
            speed = measure_inference_speed(model, b_cfg)
            r.update(speed)
        results.append(r)

    if args.progressive:
        p_cfg = ExperimentConfig.from_json(args.config)
        p_cfg.model.model_type = "progressive"
        model = load_model(args.progressive, p_cfg)
        r = eval_model(
            model,
            BatchIterator(val_data, cfg.train.batch_size, seed=eval_seed),
            p_cfg,
            n_steps=args.eval_steps,
            label="Progressive (1→2→4-bit)",
            rng_seed=eval_seed,
        )
        if args.measure_speed:
            speed = measure_inference_speed(model, p_cfg)
            r.update(speed)
        results.append(r)
        flop_info = estimate_flop_savings(p_cfg.model.precision_schedule)

    print_report(results, flop_info)

    if args.save_results:
        import json as _json
        out_path = Path(args.save_results)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            _json.dump({"results": results, "flop_info": flop_info}, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
