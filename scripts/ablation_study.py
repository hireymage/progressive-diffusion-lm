#!/usr/bin/env python3
"""
Ablation study: isolate the effect of progressive precision scheduling.

Phase 1 (screening): 6 variants × 3 seeds = 18 runs at 3,000 steps each.
Phase 2 (full):      selected variants × 3 seeds at 10,000 steps.

Variants
--------
  baseline    : all bits=16  (no quantisation)
  const_1bit  : all bits=1   (binary, constant)
  const_2bit  : all bits=2   (true 2-bit, constant)
  const_4bit  : all bits=4   (4-bit, constant)
  prog_1_2_4  : [1,1,1,1,2,2,4,4]  (coarse-to-fine, progressive)
  prog_4_2_1  : [4,4,2,2,1,1,1,1]  (fine-to-coarse, reversed)

Research questions
------------------
  1. Does progressive precision consistently outperform/match baseline?
  2. Does it outperform constant 1-bit, 2-bit, 4-bit alternatives?
  3. Does direction (1→2→4 vs 4→2→1) matter?
  4. Is the improvement from progressive structure or low-bit regularisation?

Usage
-----
  python scripts/ablation_study.py                   # full study
  python scripts/ablation_study.py --dry-run         # print plan, don't train
  python scripts/ablation_study.py --resume          # skip completed runs
  python scripts/ablation_study.py --analyze-only    # report from existing results
  python scripts/ablation_study.py --phase full      # 10k-step full runs only
"""

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

# ── Study parameters ──────────────────────────────────────────────────────────

VARIANTS = {
    "baseline":   {"model_type": "baseline",     "schedule": [16, 16, 16, 16, 16, 16, 16, 16]},
    "const_1bit": {"model_type": "progressive",  "schedule": [1,  1,  1,  1,  1,  1,  1,  1 ]},
    "const_2bit": {"model_type": "progressive",  "schedule": [2,  2,  2,  2,  2,  2,  2,  2 ]},
    "const_4bit": {"model_type": "progressive",  "schedule": [4,  4,  4,  4,  4,  4,  4,  4 ]},
    "prog_1_2_4": {"model_type": "progressive",  "schedule": [1,  1,  1,  1,  2,  2,  4,  4 ]},
    "prog_4_2_1": {"model_type": "progressive",  "schedule": [4,  4,  2,  2,  1,  1,  1,  1 ]},
}

SEEDS = [42, 123, 7]

# Shared architecture — identical to both full runs
BASE_MODEL = {
    "vocab_size": 16000, "pad_token_id": 0,
    "d_model": 512, "n_layers": 6, "n_heads": 8, "d_ff": 2048,
    "max_seq_len": 256, "dropout": 0.0,
    "tie_word_embeddings": True, "n_diffusion_steps": 8,
}

BASE_DATA = {
    "dataset_name": "wikimedia/wikipedia", "dataset_config": "20231101.en",
    "dataset_revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    "max_articles": 50000, "max_text_bytes": 500000000,
    "seq_len": 256, "tokenizer_path": "tokenizer/wiki_bpe",
    "data_cache_dir": "data/cache", "train_split": 0.95,
}

SCREEN_TRAIN = {   # 3,000-step screening
    "batch_size": 8, "learning_rate": 3e-4, "weight_decay": 0.0,
    "max_steps": 3000, "warmup_steps": 300, "grad_clip": 1.0,
    "checkpoint_dir": "checkpoints/ablation",
    "checkpoint_every": 999999,   # never fires — no disk waste
    "resume_from": None,
    "eval_every": 500, "eval_steps": 50,
    "log_every": 10, "results_dir": "results/ablation",
    "save_checkpoints": False,    # metrics only
}

FULL_TRAIN = {     # 10,000-step full training
    "batch_size": 8, "learning_rate": 3e-4, "weight_decay": 0.0,
    "max_steps": 10000, "warmup_steps": 500, "grad_clip": 1.0,
    "checkpoint_dir": "checkpoints/ablation_full",
    "checkpoint_every": 999999,
    "resume_from": None,
    "eval_every": 500, "eval_steps": 100,
    "log_every": 10, "results_dir": "results/ablation_full",
    "save_checkpoints": False,
}

# Mirrors src.quantization: bits=3 is true Q3; bits=0 is optional ternary.
EFFECTIVE_BITS = {0: math.log2(3), 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 16: 32.0}

# ── Config generation ─────────────────────────────────────────────────────────

def make_config(variant_name: str, seed: int, phase: str = "screen") -> dict:
    v = VARIANTS[variant_name]
    train_cfg = FULL_TRAIN.copy() if phase == "full" else SCREEN_TRAIN.copy()
    train_cfg["seed"] = seed
    exp_suffix = "full" if phase == "full" else "scr"
    return {
        "experiment_name": f"abl_{variant_name}_s{seed}_{exp_suffix}",
        "model": {**BASE_MODEL, "model_type": v["model_type"],
                  "precision_schedule": v["schedule"]},
        "data": BASE_DATA.copy(),
        "train": train_cfg,
    }


def config_path(variant_name: str, seed: int, phase: str = "screen") -> Path:
    d = Path("configs/ablation")
    d.mkdir(parents=True, exist_ok=True)
    return d / f"ablation_{variant_name}_s{seed}_{phase}.json"


def results_dir(cfg: dict) -> Path:
    return Path(cfg["train"]["results_dir"]) / cfg["experiment_name"]


def is_complete(cfg: dict) -> bool:
    """True if final_summary.json exists for this run."""
    return (results_dir(cfg) / "final_summary.json").exists()


# ── Run a single training job ─────────────────────────────────────────────────

def run_one(cfg: dict, cfg_file: Path, dry_run: bool = False) -> bool:
    exp = cfg["experiment_name"]
    print(f"\n{'─'*60}")
    print(f"  RUN: {exp}")
    print(f"  variant={exp.split('_s')[0].replace('abl_','')}  "
          f"seed={cfg['train']['seed']}  steps={cfg['train']['max_steps']}")
    print(f"  schedule={cfg['model']['precision_schedule']}")
    print(f"{'─'*60}")

    if dry_run:
        print("  [DRY RUN] skipping")
        return True

    with open(cfg_file, "w") as f:
        json.dump(cfg, f, indent=2)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "src.train", "--config", str(cfg_file)],
        capture_output=False,   # stream to terminal
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  [ERROR] {exp} failed (exit {result.returncode}) after {elapsed:.0f}s")
        return False

    print(f"\n  [DONE] {exp} completed in {elapsed:.0f}s ({elapsed/cfg['train']['max_steps']:.2f}s/step)")
    return True


# ── Analysis ──────────────────────────────────────────────────────────────────

def load_run_metrics(cfg: dict) -> dict | None:
    """Load eval_history and final_summary for one run. Returns None if missing."""
    rd = results_dir(cfg)
    eval_f = rd / "eval_history.json"
    sum_f  = rd / "final_summary.json"
    if not eval_f.exists() or not sum_f.exists():
        return None
    evals = json.load(open(eval_f))
    summary = json.load(open(sum_f))
    if not evals:
        return None

    val_losses = [e["val_loss"] for e in evals]
    val_accs   = [e["val_accuracy"] for e in evals]
    steps      = [e["step"] for e in evals]

    # Convergence speed: step at which val_loss first drops below 7.50
    THRESHOLD = 7.50
    conv_step = next((e["step"] for e in evals if e["val_loss"] < THRESHOLD), None)

    return {
        "experiment_name": cfg["experiment_name"],
        "variant":         cfg["experiment_name"].split("_s")[0].replace("abl_", ""),
        "seed":            cfg["train"]["seed"],
        "max_steps":       cfg["train"]["max_steps"],
        "schedule":        cfg["model"]["precision_schedule"],
        "model_type":      cfg["model"]["model_type"],
        "avg_eff_bits":    sum(EFFECTIVE_BITS.get(b, float(b)) for b in cfg["model"]["precision_schedule"])
                           / len(cfg["model"]["precision_schedule"]),
        "best_val_loss":   summary["best_val_loss"],
        "final_val_loss":  evals[-1]["val_loss"],
        "best_val_acc":    max(val_accs),
        "final_val_acc":   evals[-1]["val_accuracy"],
        "conv_step_7_50":  conv_step,
        "training_seconds": summary["total_training_seconds"],
        "steps_per_sec":   cfg["train"]["max_steps"] / summary["total_training_seconds"],
        "eval_history":    evals,
    }


def aggregate(runs: list[dict]) -> dict:
    """Mean/std/min/max of key metrics across a list of runs."""
    def stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
        return {
            "mean": statistics.mean(vals),
            "std":  statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min":  min(vals),
            "max":  max(vals),
            "n":    len(vals),
        }
    return {
        "best_val_loss":   stats([r["best_val_loss"]  for r in runs]),
        "final_val_loss":  stats([r["final_val_loss"] for r in runs]),
        "best_val_acc":    stats([r["best_val_acc"]   for r in runs]),
        "conv_step_7_50":  stats([r["conv_step_7_50"] for r in runs]),
        "training_seconds": stats([r["training_seconds"] for r in runs]),
    }


def analyze(phase: str = "screen") -> dict:
    """Load all completed runs and produce the full comparison report."""
    suffix = "full" if phase == "full" else "scr"
    all_runs = []
    missing  = []

    for vname in VARIANTS:
        for seed in SEEDS:
            cfg = make_config(vname, seed, phase)
            m = load_run_metrics(cfg)
            if m:
                all_runs.append(m)
            else:
                missing.append(cfg["experiment_name"])

    if not all_runs:
        print("No completed runs found.")
        return {}

    # Group by variant
    by_variant = {}
    for r in all_runs:
        by_variant.setdefault(r["variant"], []).append(r)

    print(f"\n{'='*72}")
    print(f"  ABLATION STUDY RESULTS  ({phase.upper()} PHASE — {next(iter(all_runs))['max_steps']}-step runs)")
    print(f"  {len(all_runs)}/{len(VARIANTS)*len(SEEDS)} runs completed")
    if missing:
        print(f"  Missing: {', '.join(missing)}")
    print(f"{'='*72}")

    # ── Table 1: per-run results ──────────────────────────────────────────
    print(f"\n── PER-RUN RESULTS ──────────────────────────────────────────────────")
    print(f"  {'Experiment':<35} {'Bits':>5} {'Best VL':>8} {'Final VL':>9} {'BestAcc':>8} {'Time':>7}")
    print("  " + "─" * 68)
    for r in sorted(all_runs, key=lambda x: (x["variant"], x["seed"])):
        print(f"  {r['experiment_name']:<35} {r['avg_eff_bits']:>5.2f} "
              f"{r['best_val_loss']:>8.4f} {r['final_val_loss']:>9.4f} "
              f"{r['best_val_acc']:>8.4f} {r['training_seconds']:>6.0f}s")

    # ── Table 2: aggregate across seeds ──────────────────────────────────
    print(f"\n── AGGREGATE ACROSS SEEDS (mean ± std / min / max) ─────────────────")
    print(f"  {'Variant':<14} {'Bits':>5}  {'Best VL mean±std':>20}  {'Final VL mean':>14}  {'BestAcc mean':>12}")
    print("  " + "─" * 72)

    agg_results = {}
    for vname in VARIANTS:
        runs = by_variant.get(vname, [])
        if not runs:
            continue
        ag = aggregate(runs)
        agg_results[vname] = ag
        bits = runs[0]["avg_eff_bits"]
        bvl = ag["best_val_loss"]
        fvl = ag["final_val_loss"]
        bac = ag["best_val_acc"]
        bvl_str = f"{bvl['mean']:.4f} ± {bvl['std']:.4f}" if bvl["n"] > 1 else f"{bvl['mean']:.4f}"
        fvl_str = f"{fvl['mean']:.4f}" if fvl["mean"] else "—"
        bac_str = f"{bac['mean']:.4f}" if bac["mean"] else "—"
        print(f"  {vname:<14} {bits:>5.2f}  {bvl_str:>20}  {fvl_str:>14}  {bac_str:>12}")

    # ── Ranking ───────────────────────────────────────────────────────────
    print(f"\n── RANKING BY MEAN BEST VAL_LOSS (lower = better) ──────────────────")
    ranked = sorted(
        [(vname, agg_results[vname]) for vname in agg_results],
        key=lambda x: x[1]["best_val_loss"]["mean"] or 999,
    )
    for rank, (vname, ag) in enumerate(ranked, 1):
        runs = by_variant.get(vname, [])
        bits = runs[0]["avg_eff_bits"] if runs else 0
        bvl = ag["best_val_loss"]
        delta_vs_base = None
        if "baseline" in agg_results:
            base_mean = agg_results["baseline"]["best_val_loss"]["mean"]
            delta_vs_base = bvl["mean"] - base_mean
        delta_str = f"  ({delta_vs_base:+.4f} vs baseline)" if delta_vs_base is not None else ""
        wins = sum(1 for r in runs if r["best_val_loss"] < (
            agg_results.get("baseline", {}).get("best_val_loss", {}).get("mean", 999) or 999))
        print(f"  #{rank}  {vname:<14}  {bvl['mean']:.4f} ± {bvl['std']:.4f}{delta_str}  "
              f"beats_base: {wins}/{ag['best_val_loss']['n']}")

    # ── Research questions ────────────────────────────────────────────────
    print(f"\n── RESEARCH QUESTIONS ──────────────────────────────────────────────")

    def mean_bvl(vname):
        return agg_results.get(vname, {}).get("best_val_loss", {}).get("mean")

    base_m    = mean_bvl("baseline")
    p124_m    = mean_bvl("prog_1_2_4")
    p421_m    = mean_bvl("prog_4_2_1")
    c1_m      = mean_bvl("const_1bit")
    c2_m      = mean_bvl("const_2bit")
    c4_m      = mean_bvl("const_4bit")

    # Q1: prog vs baseline
    if base_m and p124_m:
        d = p124_m - base_m
        wins = sum(1 for r in by_variant.get("prog_1_2_4", [])
                   if r["best_val_loss"] < base_m)
        n = len(by_variant.get("prog_1_2_4", []))
        verdict = "PROG WINS" if d < -0.001 else ("TIED" if abs(d) < 0.001 else "BASE WINS")
        print(f"\n  Q1: Does progressive outperform/match baseline?")
        print(f"      prog_1_2_4 mean={p124_m:.4f} vs baseline mean={base_m:.4f}  delta={d:+.4f}")
        print(f"      prog beats baseline in {wins}/{n} seeds  → {verdict}")

    # Q2: prog vs constant-bit alternatives
    if p124_m:
        print(f"\n  Q2: Does progressive outperform constant-bit alternatives?")
        for vname, m in [("const_1bit", c1_m), ("const_2bit", c2_m), ("const_4bit", c4_m)]:
            if m:
                d = p124_m - m
                verdict = "PROG WINS" if d < -0.001 else ("TIED" if abs(d) < 0.001 else f"{vname} WINS")
                print(f"      prog_1_2_4 vs {vname}: {p124_m:.4f} vs {m:.4f}  delta={d:+.4f}  → {verdict}")

    # Q3: direction matters?
    if p124_m and p421_m:
        d = p124_m - p421_m
        verdict = "1→2→4 WINS" if d < -0.001 else ("TIED" if abs(d) < 0.001 else "4→2→1 WINS")
        print(f"\n  Q3: Does direction (1→2→4 vs 4→2→1) matter?")
        print(f"      prog_1_2_4={p124_m:.4f}  prog_4_2_1={p421_m:.4f}  delta={d:+.4f}  → {verdict}")

    # Q4: progressive structure vs low-bit regularisation
    if p124_m and c2_m:
        d = p124_m - c2_m
        if d < -0.002:
            verdict = "Structure MATTERS: progressive better than same-avg-bits constant"
        elif abs(d) < 0.002:
            verdict = "INCONCLUSIVE: schedule structure vs regularisation indistinguishable at this scale"
        else:
            verdict = "Regularisation only: constant 2-bit matches or beats progressive schedule"
        print(f"\n  Q4: Progressive structure vs low-bit regularisation?")
        print(f"      prog_1_2_4 vs const_2bit (same avg 2.0 bits):")
        print(f"      {p124_m:.4f} vs {c2_m:.4f}  delta={d:+.4f}")
        print(f"      → {verdict}")

    # ── Save to CSV and JSON ──────────────────────────────────────────────
    out_dir = Path("results/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-run CSV
    import csv
    csv_path = out_dir / f"per_run_{phase}.csv"
    fields = ["experiment_name", "variant", "seed", "avg_eff_bits",
              "best_val_loss", "final_val_loss", "best_val_acc",
              "final_val_acc", "conv_step_7_50", "training_seconds", "steps_per_sec"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(all_runs, key=lambda r: (r["variant"], r["seed"])))
    print(f"\n  Saved per-run CSV to {csv_path}")

    # Aggregate JSON
    json_path = out_dir / f"aggregate_{phase}.json"
    out = {
        "phase": phase,
        "n_completed": len(all_runs),
        "variants": {},
    }
    for vname, ag in agg_results.items():
        runs = by_variant.get(vname, [])
        out["variants"][vname] = {
            "avg_eff_bits": runs[0]["avg_eff_bits"] if runs else None,
            "schedule": runs[0]["schedule"] if runs else None,
            "n_seeds": len(runs),
            "per_seed": {r["seed"]: {
                "best_val_loss":  r["best_val_loss"],
                "final_val_loss": r["final_val_loss"],
                "best_val_acc":   r["best_val_acc"],
            } for r in runs},
            **ag,
        }
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved aggregate JSON to {json_path}")

    return agg_results


# ── Full-phase selection ──────────────────────────────────────────────────────

def recommend_full_phase(agg_results: dict) -> list[str]:
    """Based on screening, pick variants worth full 10k training."""
    # Always include baseline + prog_1_2_4 as the core comparison
    must_include = ["baseline", "prog_1_2_4"]
    # Add the best-performing constant-bit variant (to test regularisation)
    const_variants = ["const_1bit", "const_2bit", "const_4bit"]
    best_const = min(
        (v for v in const_variants if v in agg_results),
        key=lambda v: agg_results[v]["best_val_loss"]["mean"] or 999,
        default=None,
    )
    if best_const:
        must_include.append(best_const)
    # Add prog_4_2_1 if close to prog_1_2_4 (to test direction)
    if "prog_4_2_1" in agg_results and "prog_1_2_4" in agg_results:
        d = abs(agg_results["prog_4_2_1"]["best_val_loss"]["mean"] -
                agg_results["prog_1_2_4"]["best_val_loss"]["mean"])
        if d < 0.005:
            must_include.append("prog_4_2_1")
    return list(dict.fromkeys(must_include))  # deduplicate, preserve order


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--resume",       action="store_true",
                        help="Skip runs that already have final_summary.json")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--phase",        choices=["screen", "full", "both"], default="screen")
    args = parser.parse_args()

    phases = (["screen", "full"] if args.phase == "both"
              else [args.phase])

    for phase in phases:
        print(f"\n{'='*72}")
        print(f"  ABLATION STUDY — {phase.upper()} PHASE")
        print(f"  {len(VARIANTS)} variants × {len(SEEDS)} seeds = {len(VARIANTS)*len(SEEDS)} runs")
        steps = FULL_TRAIN["max_steps"] if phase == "full" else SCREEN_TRAIN["max_steps"]
        est_h = len(VARIANTS) * len(SEEDS) * steps * 0.55 / 3600
        print(f"  {steps} steps/run  est. total time: {est_h:.1f}h")
        print(f"{'='*72}")

        if not args.analyze_only:
            total = failed = skipped = 0
            for vname in VARIANTS:
                for seed in SEEDS:
                    cfg = make_config(vname, seed, phase)
                    cpath = config_path(vname, seed, phase)
                    total += 1

                    if args.resume and is_complete(cfg):
                        print(f"  [SKIP] {cfg['experiment_name']} already complete")
                        skipped += 1
                        continue

                    ok = run_one(cfg, cpath, dry_run=args.dry_run)
                    if not ok:
                        failed += 1

            if not args.dry_run:
                print(f"\n  Training summary: {total-failed-skipped} ran, "
                      f"{skipped} skipped, {failed} failed")

        if not args.dry_run:
            agg = analyze(phase)
            if phase == "screen" and agg:
                recommended = recommend_full_phase(agg)
                print(f"\n── RECOMMENDATION FOR FULL 10K TRAINING ────────────────────────")
                print(f"  Variants to run: {recommended}")
                print(f"  Estimated time:  "
                      f"{len(recommended) * len(SEEDS) * 10000 * 0.55 / 3600:.1f}h")
                print(f"  To run full phase: python scripts/ablation_study.py --phase full --resume")


if __name__ == "__main__":
    main()
