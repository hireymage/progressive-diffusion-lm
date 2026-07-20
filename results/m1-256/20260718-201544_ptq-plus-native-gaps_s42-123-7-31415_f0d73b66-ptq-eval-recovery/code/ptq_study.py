#!/usr/bin/env python3
"""
Post-Training Quantization (PTQ) study — Direct/Naive PTQ vs Native Low-Bit Training.

Research question
-----------------
Given the exact same quantization rule, is it better to train a model with
low-bit weights from the beginning (native QAT), or to apply that quantization
only after high-precision training (direct/naive PTQ)?

Experiment design
-----------------
Phase 1 — Train 3 high-precision baselines (seeds 42, 123, 7) with
           save_checkpoints=True so final weights can be loaded for PTQ.
           Uses identical config to the ablation full-phase baselines.

Phase 2 — Apply direct/naive PTQ to each checkpoint at bits ∈ {1, 2, 3, 4, 16}.
           Optional: also evaluate ternary (bits=0, 3 levels) with --include-ternary.
           No additional training; weights are loaded FP32 then evaluated
           through the quantize_weights() function at inference time.

Phase 3 — Load native results from completed ablation study (no recomputation)
           and produce the full comparison table.

PTQ method: Direct / Naive PTQ
-------------------------------
This study applies the same quantize_weights() function used by native QAT
directly to the FP32-trained weights at evaluation time only.

  • Native training:  weights optimised under STE gradient pressure at the
                      target bit-width throughout 10k training steps.
  • Direct/Naive PTQ: weights optimised at full precision (FP32), then
                      quantize_weights() applied at eval time only.

Both paths call model.set_bits(bits) before each forward pass, triggering
ste_quantize() → quantize_weights() on QuantizedLinear weights.  At eval time
there are no gradients, so STE vs direct quantise is equivalent.

THIS IS INTENTIONAL.  The experiment asks whether STE gradient pressure during
training provides a quality benefit over post-hoc quantization of the same
weights.  This is NOT a claim about the quality of modern calibrated PTQ
methods (GPTQ, AWQ, etc.), which use additional calibration data, weight
reconstruction, or fine-tuning.

Precision levels (main matrix)
-------------------------------
All main levels use the no-zero symmetric 2^n-level scheme.

bits=1   Q1  binary    {-1,+1}×scale           2 levels   eff. 1.0 bits
bits=2   Q2  true 2b   {-3,-1,+1,+3}×step      4 levels   eff. 2.0 bits
bits=3   Q3  true 3b   {-7,…,-1,+1,…,+7}×step  8 levels   eff. 3.0 bits
bits=4   Q4  true 4b   {-15,…,-1,+1,…,+15}×step 16 levels eff. 4.0 bits
bits=16  FP32 pass-through (no quantisation)              eff. 16.0 bits

Optional level (separate, labelled explicitly)
----------------------------------------------
bits=0   ternary / 3-state  {-1,0,+1}×scale    3 levels   eff. ~1.585 bits
         Evaluated with --include-ternary; not part of the main Q1–Q4 matrix.

Scheme-change caveat (Q4)
-------------------------
The Q4 quantization was updated from the prior 15-level with-zero scheme
({-7,…,+7}×scale/7) to the current 16-level no-zero scheme to be consistent
with Q1–Q3.  The native ablation const_4bit variant was trained under the OLD
15-level scheme.  The PTQ Q4 vs native const_4bit comparison therefore has a
quantization-scheme mismatch and should be interpreted with caution.

Results
-------
results/ptq_study/
  ptq_baseline_s{seed}/        training results for each baseline
    train_metrics.csv
    eval_history.json
    final_summary.json
  ptq_eval_results.json        all PTQ evaluation results (raw)
  ptq_comparison.csv           per-seed, per-precision comparison table
  ptq_aggregate.json           mean ± std across seeds

checkpoints/ptq_baselines/
  ptq_baseline_s{seed}/
    step_0010000.npz           final checkpoint (only one saved)

Usage
-----
  python scripts/ptq_study.py                        # full study
  python scripts/ptq_study.py --skip-training        # PTQ eval only (needs checkpoints)
  python scripts/ptq_study.py --eval-only            # analysis only (needs ptq_eval_results.json)
  python scripts/ptq_study.py --dry-run              # print plan, train nothing
  python scripts/ptq_study.py --include-ternary      # also evaluate optional ternary level
"""

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

# Running this file as ``python scripts/ptq_study.py`` puts ``scripts/`` rather
# than the project root on sys.path.  Phase 2 imports ``src.*`` lazily, so make
# the project package available explicitly before those imports happen.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Constants ────────────────────────────────────────────────────────────────

SEEDS = [42, 123, 7]

# Main PTQ evaluation matrix.  bits=3 = true 3-bit (8 levels, no zero).
PTQ_BITS = [1, 2, 3, 4, 16]

# Optional ternary level (3 levels, ~1.585 eff. bits); not in main matrix.
# Evaluated when --include-ternary is passed.
PTQ_BITS_OPTIONAL = [0]

# Effective bits per level (mirrors src/quantization.py EFFECTIVE_BITS).
EFFECTIVE_BITS = {0: math.log2(3), 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 16: 16.0}

BITS_LABEL = {
    0:  "ternary / 3-state (~1.585 bits, 3 levels)",
    1:  "Q1 (binary, 2 levels)",
    2:  "Q2 (true 2-bit, 4 levels)",
    3:  "Q3 (true 3-bit, 8 levels)",
    4:  "Q4 (true 4-bit, 16 levels)",
    16: "FP32 (unquantized)",
}

# Ablation full-phase native results to reuse (no recomputation)
# Results verified from results/ablation_full/*/final_summary.json
ABLATION_NATIVE = {
    "baseline": {
        42:  {"best_val_loss": 7.419442},
        123: {"best_val_loss": 7.462691},
        7:   {"best_val_loss": 7.448140},
    },
    "const_1bit": {
        42:  {"best_val_loss": 7.458024},
        123: {"best_val_loss": 7.409461},
        7:   {"best_val_loss": 7.433085},
    },
    "const_2bit": {
        42:  {"best_val_loss": 7.445632},
        123: {"best_val_loss": 7.462482},
        7:   {"best_val_loss": 7.467700},
    },
    "const_4bit": {
        42:  {"best_val_loss": 7.426039},
        123: {"best_val_loss": 7.463434},
        7:   {"best_val_loss": 7.445718},
    },
}

RESULTS_DIR = Path("results/ptq_study")
CKPT_DIR = Path("checkpoints/ptq_baselines")
CONFIG_DIR = Path("configs/ptq")


# ── Phase 1: Train baseline checkpoints ──────────────────────────────────────

def baseline_ckpt_path(seed: int) -> Path:
    return CKPT_DIR / f"ptq_baseline_s{seed}" / "step_0010000.npz"


def baseline_result_path(seed: int) -> Path:
    return RESULTS_DIR / f"ptq_baseline_s{seed}" / "final_summary.json"


def baseline_config_path(seed: int) -> Path:
    return CONFIG_DIR / f"ptq_baseline_s{seed}.json"


def train_baseline(seed: int, dry_run: bool = False) -> bool:
    cfg_path = baseline_config_path(seed)
    exp_name = f"ptq_baseline_s{seed}"
    print(f"\n{'─'*60}")
    print(f"  TRAIN BASELINE: {exp_name}  (seed={seed})")
    print(f"  config: {cfg_path}")
    print(f"  checkpoint → {baseline_ckpt_path(seed)}")
    print(f"{'─'*60}")

    if not cfg_path.exists():
        print(f"  [ERROR] Config not found: {cfg_path}")
        return False

    if dry_run:
        print("  [DRY RUN] skipping")
        return True

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "src.train", "--config", str(cfg_path)],
        capture_output=False,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  [ERROR] Training failed (exit {result.returncode})")
        return False
    print(f"\n  [DONE] {exp_name} completed in {elapsed:.0f}s")
    return True


# ── Phase 2: PTQ evaluation ───────────────────────────────────────────────────

def run_ptq_eval(seed: int, bits: int, eval_steps: int = 100) -> dict:
    """
    Direct/Naive PTQ evaluation: load FP32 checkpoint, apply quantize_weights()
    at inference time via model.set_bits(bits), evaluate on val set.

    bits ∈ {1,2,3,4,16} — main matrix; bits=0 — optional ternary.
    No weights are modified in-place; the FP32 master weights are unchanged.
    """
    # Import here so the script can be imported without mlx at the top level
    import mlx.core as mx
    import mlx.nn as nn

    from src.config import ExperimentConfig, ModelConfig
    from src.model import DiffusionLM
    from src.data import build_and_cache_dataset, BatchIterator
    from src.diffusion import corrupt_tokens, mask_rate_to_step
    from src.train import load_checkpoint

    ckpt_path = baseline_ckpt_path(seed)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load config — we evaluate as "progressive" with a constant schedule
    # so that model.set_bits(bits) is called uniformly for every step.
    cfg = ExperimentConfig.from_json(str(baseline_config_path(seed)))
    cfg.model.model_type = "progressive"
    cfg.model.precision_schedule = [bits] * cfg.model.n_diffusion_steps

    # Build model and load FP32 weights
    model = DiffusionLM(cfg.model)
    load_checkpoint(model, str(ckpt_path))

    # Set bits (redundant for bits=16, but explicit for clarity)
    model.set_bits(bits)

    # Validation data — same cache as ablation study, same seed+1 for val iter
    train_data, val_data = build_and_cache_dataset(
        tokenizer_path=cfg.data.tokenizer_path,
        cache_dir=cfg.data.data_cache_dir,
        seq_len=cfg.data.seq_len,
        max_articles=cfg.data.max_articles,
        max_text_bytes=cfg.data.max_text_bytes,
    )
    val_iter = BatchIterator(val_data, cfg.train.batch_size, seed=seed + 1)

    mask_token_id = cfg.model.mask_token_id()
    precision_schedule = [bits] * cfg.model.n_diffusion_steps

    mx.random.seed(seed + 1000)  # fixed eval seed for reproducibility

    total_loss = 0.0
    total_correct = 0
    total_masked = 0

    t0 = time.time()
    for _ in range(eval_steps):
        batch_np = next(val_iter)
        x0 = mx.array(batch_np, dtype=mx.int32)

        mask_rates = mx.random.uniform(low=0.1, high=1.0, shape=(x0.shape[0],))
        x_t, mask = corrupt_tokens(x0, mask_rates, mask_token_id)
        mean_rate = float(mask_rates.mean())
        step_idx = mask_rate_to_step(mean_rate, cfg.model.n_diffusion_steps)
        step_bits = precision_schedule[step_idx]
        model.set_bits(step_bits)

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

    elapsed = time.time() - t0

    return {
        "seed": seed,
        "bits": bits,
        "bits_label": BITS_LABEL[bits],
        "eff_bits": EFFECTIVE_BITS[bits],
        "val_loss": total_loss / eval_steps,
        "val_accuracy": total_correct / max(total_masked, 1),
        "eval_steps": eval_steps,
        "eval_seconds": round(elapsed, 1),
        "ckpt_path": str(ckpt_path),
    }


# ── Phase 3: Analysis and reporting ──────────────────────────────────────────

def load_ablation_natives() -> dict:
    """
    Load best_val_loss for native low-bit models from completed ablation study.
    Returns dict keyed by (variant, seed).
    Falls back to hard-coded ABLATION_NATIVE values if files are missing.
    """
    variants = {
        "baseline":   "abl_baseline_s{seed}_full",
        "const_1bit": "abl_const_1bit_s{seed}_full",
        "const_2bit": "abl_const_2bit_s{seed}_full",
        "const_4bit": "abl_const_4bit_s{seed}_full",
    }
    out = {}
    ablation_dir = Path("results/ablation_full")
    for vname, pattern in variants.items():
        for seed in SEEDS:
            exp = pattern.format(seed=seed)
            fp = ablation_dir / exp / "final_summary.json"
            if fp.exists():
                d = json.load(open(fp))
                out[(vname, seed)] = d["best_val_loss"]
            else:
                # Fall back to hard-coded values
                fallback = ABLATION_NATIVE.get(vname, {}).get(seed)
                if fallback:
                    out[(vname, seed)] = fallback["best_val_loss"]
    return out


def _native_variant_for_bits(bits: int):
    """Return the ablation variant name whose native training matches `bits`."""
    return {16: "baseline", 1: "const_1bit", 2: "const_2bit", 4: "const_4bit"}.get(bits)
    # bits=3 (Q3) and bits=0 (ternary) have no native ablation counterpart.


def _degradation_label(delta: float) -> str:
    if abs(delta) < 0.005:
        return "Negligible"
    elif abs(delta) < 0.02:
        return "Mild"
    elif abs(delta) < 0.05:
        return "Moderate"
    else:
        return "Severe / collapse"


def analyze(ptq_results: list[dict], native: dict, include_ternary: bool = False) -> None:
    print("\n" + "=" * 72)
    print("  PTQ STUDY — Direct/Naive PTQ vs Native Low-Bit Training")
    print()
    print("  Method: Direct/Naive PTQ — same quantize_weights() function as")
    print("  native QAT, applied to FP32-trained weights at eval time only.")
    print("  No calibration, GPTQ, AWQ, reconstruction, or fine-tuning.")
    print("=" * 72)

    main_results = [r for r in ptq_results if r["bits"] in PTQ_BITS]
    ternary_results = [r for r in ptq_results if r["bits"] == 0]

    # bits=16 FP32 baseline val_loss per seed
    ptq_fp = {r["seed"]: r["val_loss"] for r in main_results if r["bits"] == 16}

    # ── Per-seed per-precision table (main matrix) ────────────────────────
    print("\n── PER-SEED RESULTS (main matrix: Q1–Q4 + FP32) ────────────────────")
    print(f"  {'Seed':>4}  {'Bits':>4}  {'Naive-PTQ VL':>12}  {'Δ vs FP32':>10}  "
          f"{'Native VL':>9}  {'Δ P−N':>7}  Label")
    print("  " + "─" * 72)

    rows = []
    for seed in SEEDS:
        fp_loss = ptq_fp.get(seed)
        for bits in PTQ_BITS:
            ptq_r = next((r for r in main_results if r["seed"] == seed and r["bits"] == bits), None)
            if ptq_r is None:
                continue
            ptq_loss = ptq_r["val_loss"]
            delta_fp = ptq_loss - fp_loss if fp_loss else None

            native_variant = _native_variant_for_bits(bits)
            native_loss = native.get((native_variant, seed)) if native_variant else None
            # Flag Q4 scheme mismatch in the raw delta but still report it
            q4_caveat = bits == 4
            delta_nvp = ptq_loss - native_loss if native_loss is not None else None

            row = {
                "seed": seed,
                "bits": bits,
                "bits_label": BITS_LABEL[bits],
                "eff_bits": EFFECTIVE_BITS[bits],
                "ptq_val_loss": round(ptq_loss, 4),
                "fp_val_loss": round(fp_loss, 4) if fp_loss else None,
                "delta_vs_fp": round(delta_fp, 4) if delta_fp is not None else None,
                "native_variant": native_variant,
                "native_val_loss": round(native_loss, 4) if native_loss else None,
                "delta_ptq_minus_native": round(delta_nvp, 4) if delta_nvp is not None else None,
                "q4_scheme_caveat": q4_caveat,
            }
            rows.append(row)

            d_fp_str = f"{delta_fp:+.4f}" if delta_fp is not None else "       N/A"
            nat_str  = f"{native_loss:.4f}" if native_loss is not None else "    N/A"
            d_nvp_str = (f"{delta_nvp:+.4f}" + ("*" if q4_caveat else " ")
                         if delta_nvp is not None else "      N/A")
            print(f"  {seed:>4}  {bits:>4}  {ptq_loss:>12.4f}  {d_fp_str:>10}  "
                  f"{nat_str:>9}  {d_nvp_str:>7}  {BITS_LABEL[bits]}")
        print()

    print("  * Q4 Δ P−N marked with * — native const_4bit used OLD 15-level scheme;")
    print("    PTQ Q4 uses new 16-level scheme.  Comparison is approximate.")

    # ── Aggregate across seeds ────────────────────────────────────────────
    print("\n── AGGREGATE ACROSS SEEDS (mean ± std, direct/naive PTQ) ────────────")
    print(f"  {'Bits':>4}  {'Naive-PTQ mean±std':>20}  {'Δ vs FP32':>10}  "
          f"{'Native mean':>11}  {'Δ P−N':>7}  Label")
    print("  " + "─" * 75)

    agg_rows = []
    for bits in PTQ_BITS:
        bit_rows = [r for r in rows if r["bits"] == bits]
        if not bit_rows:
            continue
        ptq_losses = [r["ptq_val_loss"] for r in bit_rows]
        ptq_mean = statistics.mean(ptq_losses)
        ptq_std = statistics.stdev(ptq_losses) if len(ptq_losses) > 1 else 0.0

        fp_losses = [r["fp_val_loss"] for r in bit_rows if r["fp_val_loss"]]
        delta_fp = ptq_mean - statistics.mean(fp_losses) if fp_losses else None

        nat_losses = [r["native_val_loss"] for r in bit_rows if r["native_val_loss"] is not None]
        nat_mean = statistics.mean(nat_losses) if nat_losses else None
        delta_nvp = ptq_mean - nat_mean if nat_mean is not None else None
        q4_caveat = bits == 4

        d_fp_str  = f"{delta_fp:+.4f}" if delta_fp is not None else "       N/A"
        nat_str   = f"{nat_mean:.4f}" if nat_mean is not None else "      N/A"
        d_nvp_str = (f"{delta_nvp:+.4f}" + ("*" if q4_caveat else " ")
                     if delta_nvp is not None else "      N/A")
        print(f"  {bits:>4}  {ptq_mean:.4f} ± {ptq_std:.4f}        {d_fp_str:>10}  "
              f"{nat_str:>11}  {d_nvp_str:>7}  {BITS_LABEL[bits]}")

        agg_rows.append({
            "bits": bits,
            "bits_label": BITS_LABEL[bits],
            "eff_bits": EFFECTIVE_BITS[bits],
            "ptq_mean": round(ptq_mean, 4),
            "ptq_std": round(ptq_std, 4),
            "delta_vs_fp": round(delta_fp, 4) if delta_fp is not None else None,
            "native_mean": round(nat_mean, 4) if nat_mean is not None else None,
            "delta_ptq_minus_native": round(delta_nvp, 4) if delta_nvp is not None else None,
            "q4_scheme_caveat": q4_caveat,
        })

    # ── Optional ternary section ──────────────────────────────────────────
    ternary_agg_rows = []
    if ternary_results:
        print("\n── OPTIONAL: Ternary / 3-state (bits=0, ~1.585 eff. bits) ──────────")
        print("  NOT part of the main Q1–Q4 matrix.  No native ablation counterpart.")
        print(f"  {'Seed':>4}  {'Ternary VL':>10}  {'Δ vs FP32':>10}")
        print("  " + "─" * 35)
        ternary_by_seed = {}
        for r in ternary_results:
            seed = r["seed"]
            fp_loss = ptq_fp.get(seed)
            delta_fp = r["val_loss"] - fp_loss if fp_loss else None
            d_str = f"{delta_fp:+.4f}" if delta_fp is not None else "       N/A"
            print(f"  {seed:>4}  {r['val_loss']:>10.4f}  {d_str:>10}")
            ternary_by_seed[seed] = r["val_loss"]
        if len(ternary_by_seed) > 1:
            t_losses = list(ternary_by_seed.values())
            t_mean = statistics.mean(t_losses)
            t_std  = statistics.stdev(t_losses)
            fp_mean_all = statistics.mean(ptq_fp.values()) if ptq_fp else None
            d_fp = t_mean - fp_mean_all if fp_mean_all else None
            d_str = f"{d_fp:+.4f}" if d_fp is not None else "N/A"
            print(f"  {'mean':>4}  {t_mean:>10.4f}  {d_str:>10}  (std={t_std:.4f})")
            ternary_agg_rows.append({
                "bits": 0,
                "bits_label": BITS_LABEL[0],
                "eff_bits": EFFECTIVE_BITS[0],
                "ptq_mean": round(t_mean, 4),
                "ptq_std": round(t_std, 4),
                "delta_vs_fp": round(d_fp, 4) if d_fp is not None else None,
                "native_mean": None,
                "delta_ptq_minus_native": None,
            })

    # ── Research questions ────────────────────────────────────────────────
    print("\n── RESEARCH QUESTIONS ───────────────────────────────────────────────")
    fp_mean = next((r["ptq_mean"] for r in agg_rows if r["bits"] == 16), None)

    print("\n  RQ1: How much quality lost from FP32 → direct/naive PTQ?")
    for bits, label in [(1, "Q1"), (2, "Q2"), (3, "Q3"), (4, "Q4")]:
        agg = next((r for r in agg_rows if r["bits"] == bits), None)
        if agg is None or fp_mean is None:
            continue
        d = agg["ptq_mean"] - fp_mean
        caveat = "  [Q4: scheme mismatch caveat — see * above]" if bits == 4 else ""
        print(f"      FP32={fp_mean:.4f}  PTQ-{label}={agg['ptq_mean']:.4f}  "
              f"Δ={d:+.4f}  → {_degradation_label(d)}{caveat}")

    print("\n  RQ2/RQ3: Native low-bit vs direct/naive PTQ at same precision?")
    print("  (Positive Δ = PTQ worse than native; negative = PTQ better)")
    for bits, label in [(1, "Q1"), (2, "Q2"), (4, "Q4")]:
        ptq_agg = next((r for r in agg_rows if r["bits"] == bits), None)
        if ptq_agg is None or ptq_agg["native_mean"] is None:
            continue
        d = ptq_agg["delta_ptq_minus_native"]
        if d is None:
            continue
        winner = "Native BETTER" if d > 0.001 else ("PTQ BETTER" if d < -0.001 else "TIED")
        caveat = "  [scheme mismatch — approximate]" if bits == 4 else ""
        print(f"      {label}: native={ptq_agg['native_mean']:.4f}  "
              f"ptq={ptq_agg['ptq_mean']:.4f}  Δ={d:+.4f}  → {winner}{caveat}")
    print("  RQ3 note: Q3 (true 3-bit) has no native ablation counterpart.")

    print("\n  RQ4: Does quality gap between native and PTQ widen at lower bits?")
    print("  (Reported where native ablation data exists: Q1, Q2, Q4)")
    for bits, label in [(4, "Q4"), (2, "Q2"), (1, "Q1")]:
        ptq_agg = next((r for r in agg_rows if r["bits"] == bits), None)
        if ptq_agg is None or ptq_agg["delta_ptq_minus_native"] is None:
            continue
        print(f"      {label}: Δ(PTQ−native)={ptq_agg['delta_ptq_minus_native']:+.4f}")

    print("\n  RQ5: Precision collapse threshold (direct/naive PTQ)?")
    for bits, label in [(4, "Q4"), (3, "Q3"), (2, "Q2"), (1, "Q1")]:
        agg = next((r for r in agg_rows if r["bits"] == bits), None)
        if agg is None or agg["delta_vs_fp"] is None:
            continue
        d = agg["delta_vs_fp"]
        flag = "  *** COLLAPSE" if d > 0.10 else ("  ** large" if d > 0.03 else "")
        print(f"      PTQ-{label}: Δ vs FP32={d:+.4f}{flag}")

    # ── Save outputs ──────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Per-run CSV (main matrix + ternary if present)
    all_rows_for_csv = rows[:]
    if ternary_results:
        for r in ternary_results:
            fp_loss = ptq_fp.get(r["seed"])
            delta_fp = r["val_loss"] - fp_loss if fp_loss else None
            all_rows_for_csv.append({
                "seed": r["seed"],
                "bits": 0,
                "bits_label": BITS_LABEL[0],
                "eff_bits": EFFECTIVE_BITS[0],
                "ptq_val_loss": round(r["val_loss"], 4),
                "fp_val_loss": round(fp_loss, 4) if fp_loss else None,
                "delta_vs_fp": round(delta_fp, 4) if delta_fp is not None else None,
                "native_variant": None,
                "native_val_loss": None,
                "delta_ptq_minus_native": None,
                "q4_scheme_caveat": False,
            })

    csv_path = RESULTS_DIR / "ptq_comparison.csv"
    fields = ["seed", "bits", "bits_label", "eff_bits", "ptq_val_loss",
              "fp_val_loss", "delta_vs_fp", "native_variant",
              "native_val_loss", "delta_ptq_minus_native", "q4_scheme_caveat"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows_for_csv)
    print(f"\n  Saved per-run CSV    → {csv_path}")

    agg_path = RESULTS_DIR / "ptq_aggregate.json"
    with open(agg_path, "w") as f:
        json.dump({
            "description": "PTQ study aggregate results — direct/naive PTQ vs native low-bit training",
            "method": "direct_naive_ptq",
            "seeds": SEEDS,
            "ptq_bits_main": PTQ_BITS,
            "ptq_bits_optional": PTQ_BITS_OPTIONAL,
            "aggregate_main": agg_rows,
            "aggregate_ternary": ternary_agg_rows,
            "caveats": {
                "q4_scheme_mismatch": (
                    "PTQ Q4 uses 16-level no-zero scheme. "
                    "Native const_4bit was trained under the prior 15-level with-zero scheme."
                ),
                "q3_no_native": "No native ablation run exists for Q3 (true 3-bit, 8 levels).",
                "ternary_no_native": "No native ablation run exists for ternary (3 levels).",
            },
        }, f, indent=2)
    print(f"  Saved aggregate JSON → {agg_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PTQ study: direct/naive PTQ vs native low-bit training")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip Phase 1 (baseline training); use existing checkpoints")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training + PTQ eval; load ptq_eval_results.json and analyze")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without running anything")
    parser.add_argument("--eval-steps", type=int, default=100,
                        help="Number of validation batches per PTQ evaluation (default: 100)")
    parser.add_argument("--include-ternary", action="store_true",
                        help="Also evaluate optional ternary (bits=0, 3-state) in Phase 2")
    args = parser.parse_args()

    eval_bits = PTQ_BITS + (PTQ_BITS_OPTIONAL if args.include_ternary else [])

    print("=" * 72)
    print("  PTQ STUDY: direct/naive PTQ vs native low-bit training")
    print(f"  Seeds: {SEEDS}  |  PTQ bits: {PTQ_BITS}  |  eval_steps: {args.eval_steps}")
    if args.include_ternary:
        print(f"  Optional: ternary (bits=0) also evaluated")
    print("=" * 72)

    ptq_results_path = RESULTS_DIR / "ptq_eval_results.json"

    # ── Phase 1: Train baselines ─────────────────────────────────────────
    if not args.skip_training and not args.eval_only:
        print("\n── PHASE 1: Train high-precision baselines ──────────────────────────")
        for seed in SEEDS:
            ckpt = baseline_ckpt_path(seed)
            if ckpt.exists():
                print(f"  [SKIP] Checkpoint already exists: {ckpt}")
                continue
            ok = train_baseline(seed, dry_run=args.dry_run)
            if not ok and not args.dry_run:
                print(f"  [FATAL] Training failed for seed {seed}. Aborting.")
                sys.exit(1)
    else:
        print("\n── PHASE 1: SKIPPED (--skip-training or --eval-only) ────────────────")
        for seed in SEEDS:
            ckpt = baseline_ckpt_path(seed)
            status = "✓ found" if ckpt.exists() else "✗ MISSING"
            print(f"  seed {seed}: {ckpt}  [{status}]")

    # ── Phase 2: PTQ evaluations ─────────────────────────────────────────
    if not args.eval_only:
        print("\n── PHASE 2: PTQ evaluations ─────────────────────────────────────────")
        ptq_results = []
        ptq_errors = []
        for seed in SEEDS:
            ckpt = baseline_ckpt_path(seed)
            if not ckpt.exists():
                print(f"  [SKIP] No checkpoint for seed {seed} — skipping PTQ eval")
                continue
            for bits in eval_bits:
                label = BITS_LABEL[bits]
                tag = "  [optional ternary]" if bits == 0 else ""
                print(f"  Evaluating seed={seed} bits={bits} ({label}){tag} ...", end=" ", flush=True)
                if args.dry_run:
                    print("[DRY RUN]")
                    continue
                try:
                    r = run_ptq_eval(seed, bits, eval_steps=args.eval_steps)
                    ptq_results.append(r)
                    print(f"val_loss={r['val_loss']:.4f}  ({r['eval_seconds']:.0f}s)")
                except Exception as e:
                    print(f"[ERROR] {e}")
                    ptq_errors.append({"seed": seed, "bits": bits, "error": str(e)})

        if not args.dry_run and ptq_errors:
            print(f"\n  [FATAL] {len(ptq_errors)} PTQ evaluations failed; refusing a successful exit.")
            for item in ptq_errors:
                print(f"    seed={item['seed']} bits={item['bits']}: {item['error']}")
            sys.exit(2)

        if not args.dry_run and ptq_results:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(ptq_results_path, "w") as f:
                json.dump(ptq_results, f, indent=2)
            print(f"\n  Saved PTQ eval results → {ptq_results_path}")
    else:
        print("\n── PHASE 2: SKIPPED (--eval-only) ──────────────────────────────────")
        if not ptq_results_path.exists():
            print(f"  [ERROR] {ptq_results_path} not found. Run without --eval-only first.")
            sys.exit(1)
        ptq_results = json.load(open(ptq_results_path))
        print(f"  Loaded {len(ptq_results)} PTQ results from {ptq_results_path}")

    # ── Phase 3: Analysis ─────────────────────────────────────────────────
    if not args.dry_run and ptq_results:
        print("\n── PHASE 3: Analysis ────────────────────────────────────────────────")
        native = load_ablation_natives()
        analyze(ptq_results, native, include_ternary=args.include_ternary)
    elif args.dry_run:
        print("\n── DRY RUN SUMMARY ──────────────────────────────────────────────────")
        print("  Phase 1: Train 3 baselines (~90 min each = ~4.5h total)")
        for seed in SEEDS:
            ckpt = baseline_ckpt_path(seed)
            status = "exists, will SKIP" if ckpt.exists() else "will TRAIN"
            print(f"    seed {seed}: {status}")
        n_evals = len(SEEDS) * len(eval_bits)
        print(f"  Phase 2: {n_evals} direct/naive PTQ evals (~5 min each = ~{n_evals*5//60}h total)")
        for seed in SEEDS:
            for bits in eval_bits:
                tag = "  [optional ternary]" if bits == 0 else ""
                print(f"    seed={seed} bits={bits} ({BITS_LABEL[bits]}){tag}")
        print("  Phase 3: Analysis from ablation_full results (no computation)")
        print(f"  Estimated total: ~6-7h (dominated by 3 training runs)")


if __name__ == "__main__":
    main()
