#!/usr/bin/env python3
"""M0 precision-oracle analysis on deterministic masked-token fixtures.

This is an *evaluation-only* tool.  It loads one checkpoint once, replays the
same clean tokens, mask rates, and mask locations at Q1/Q2/Q4/Q8/FP32, and
writes summary.json plus per_token.csv.  The CSV deliberately contains only
top-2-derived statistics, never the full vocabulary logits.

Example
-------
python scripts/oracle_m0.py --config configs/baseline.json \
  --checkpoint checkpoints/example/step_0001000.npz --output-dir results/m0
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_BITS = (1, 2, 4, 8, 16)
# ``16`` is an internal compatibility identifier for the identity path.  That
# path retains FP32 master weights and executes in FP32, so it must not be
# treated as a 16-bit operation in cost accounting.
PRECISION_PROXY_COSTS = {1: 1, 2: 2, 4: 4, 8: 8, 16: 32}
FP32_PROXY_COST = 32


def proxy_cost_for_precision(precision: int) -> int:
    """Return the compute proxy for an internal precision identifier."""
    try:
        return PRECISION_PROXY_COSTS[precision]
    except KeyError as error:
        raise ValueError(f"No proxy cost configured for precision {precision}") from error


def proxy_cost_mapping(ordered_bits: Iterable[int]) -> dict[str, int]:
    """Serialize explicit proxy costs alongside compatible internal IDs."""
    return {str(precision): proxy_cost_for_precision(precision) for precision in ordered_bits}


@dataclass(frozen=True)
class MaskedFixture:
    """One fixed masked batch; all precision levels receive this exact input."""

    targets: np.ndarray
    inputs: np.ndarray
    mask: np.ndarray
    mask_rates: np.ndarray


def sha256_file(path: str | Path) -> str:
    """Return a file's SHA-256 while reading it only once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit() -> str | None:
    """Best-effort repository revision; unavailable git is not an M0 failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def run_provenance(checkpoint_path: str | Path,
                   validation_path: str | Path | None = None) -> dict:
    """Collect stable run identity and hash each provided input exactly once."""
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    validation = Path(validation_path).expanduser().resolve() if validation_path else None
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "utc_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "validation_data_sha256": sha256_file(validation) if validation else None,
    }


def make_fixtures(data: np.ndarray, batch_size: int, n_batches: int,
                  mask_token_id: int, seed: int) -> list[MaskedFixture]:
    """Create deterministic batches and independent deterministic masks in NumPy."""
    if len(data) == 0:
        raise ValueError("Cannot create fixtures from an empty dataset")
    rng = np.random.RandomState(seed)
    fixtures = []
    for _ in range(n_batches):
        indices = rng.choice(len(data), size=batch_size, replace=len(data) < batch_size)
        targets = np.asarray(data[indices], dtype=np.int32).copy()
        mask_rates = rng.uniform(0.1, 1.0, size=batch_size).astype(np.float32)
        mask = rng.uniform(size=targets.shape) < mask_rates[:, None]
        # A fixture with zero masked positions has no loss or escalation signal.
        if not mask.any():
            mask[0, 0] = True
        inputs = np.where(mask, mask_token_id, targets).astype(np.int32)
        fixtures.append(MaskedFixture(targets, inputs, mask, mask_rates))
    return fixtures


def load_validation_array(path: str | Path, seq_len: int, vocab_size: int) -> np.ndarray:
    """Load a prebuilt validation array without invoking the dataset builder."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Validation data file not found: {resolved}")
    data = np.load(resolved, allow_pickle=False)
    if data.ndim != 2:
        raise ValueError(f"Validation data must be rank 2, got shape {data.shape}")
    if data.shape[0] == 0:
        raise ValueError("Validation data must contain at least one sequence")
    if data.shape[1] != seq_len:
        raise ValueError(
            f"Validation sequence length {data.shape[1]} does not match config seq_len {seq_len}"
        )
    if not np.issubdtype(data.dtype, np.integer):
        raise ValueError(f"Validation token IDs must use an integer dtype, got {data.dtype}")
    if data.min() < 0 or data.max() >= vocab_size:
        raise ValueError(f"Validation token IDs must be within [0, {vocab_size})")
    return np.asarray(data, dtype=np.int32)


def _softmax_stats(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return prediction, top-1 confidence, entropy, and top-1/top-2 margin."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    predictions = probabilities.argmax(axis=-1)
    top_two = np.partition(probabilities, -2, axis=-1)[..., -2:]
    confidence = top_two[..., 1]
    margin = top_two[..., 1] - top_two[..., 0]
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=-1)
    return predictions, confidence, entropy, margin


def _metrics_for_precision(logits: Iterable[np.ndarray], fixtures: list[MaskedFixture]) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    loss_sum = correct_sum = confidence_sum = entropy_sum = margin_sum = 0.0
    masked_total = 0
    for fixture_index, (batch_logits, fixture) in enumerate(zip(logits, fixtures)):
        pred, confidence, entropy, margin = _softmax_stats(batch_logits)
        log_probs = batch_logits.astype(np.float64) - np.logaddexp.reduce(batch_logits.astype(np.float64), axis=-1, keepdims=True)
        for batch_index, position in np.argwhere(fixture.mask):
            target = int(fixture.targets[batch_index, position])
            is_correct = bool(pred[batch_index, position] == target)
            row = {
                "fixture_index": fixture_index,
                "batch_index": int(batch_index),
                "position": int(position),
                "target": target,
                "prediction": int(pred[batch_index, position]),
                "correct": is_correct,
                "loss": float(-log_probs[batch_index, position, target]),
                "confidence": float(confidence[batch_index, position]),
                "entropy": float(entropy[batch_index, position]),
                "margin": float(margin[batch_index, position]),
            }
            rows.append(row)
            loss_sum += row["loss"]
            correct_sum += is_correct
            confidence_sum += row["confidence"]
            entropy_sum += row["entropy"]
            margin_sum += row["margin"]
            masked_total += 1
    if not masked_total:
        raise ValueError("Fixtures contain no masked tokens")
    return {
        "masked_tokens": masked_total,
        "masked_loss": loss_sum / masked_total,
        "masked_accuracy": correct_sum / masked_total,
        "mean_confidence": confidence_sum / masked_total,
        "mean_entropy": entropy_sum / masked_total,
        "mean_top1_top2_margin": margin_sum / masked_total,
    }, rows


def analyze_precision_logits(logits_by_bits: Mapping[int, Iterable[np.ndarray]],
                             fixtures: list[MaskedFixture], bits: Iterable[int] = DEFAULT_BITS) -> tuple[dict, list[dict]]:
    """Analyze precomputed logits; kept NumPy-only to enable cheap unit tests."""
    ordered_bits = list(bits)
    if not ordered_bits or ordered_bits[-1] != 16:
        raise ValueError("bits must be ordered and end with FP32 identity bits=16")
    per_precision, rows_by_bits = {}, {}
    for precision in ordered_bits:
        if precision not in logits_by_bits:
            raise ValueError(f"Missing logits for precision {precision}")
        precision_logits = list(logits_by_bits[precision])
        _validate_logits(precision, precision_logits, fixtures)
        metrics, rows = _metrics_for_precision(precision_logits, fixtures)
        per_precision[str(precision)] = metrics
        rows_by_bits[precision] = rows

    token_rows = []
    transitions = {}
    for index in range(len(rows_by_bits[ordered_bits[0]])):
        combined = dict(rows_by_bits[ordered_bits[0]][index])
        combined.pop("prediction")
        combined.pop("correct")
        combined.pop("loss")
        combined.pop("confidence")
        combined.pop("entropy")
        combined.pop("margin")
        for precision in ordered_bits:
            row = rows_by_bits[precision][index]
            for key in ("prediction", "correct", "loss", "confidence", "entropy", "margin"):
                combined[f"q{precision}_{key}"] = row[key]
        # Oracle uses target labels only for this offline upper bound: select the
        # lowest precision which is correct, otherwise the finest result.
        selected = next((p for p in ordered_bits if rows_by_bits[p][index]["correct"]), ordered_bits[-1])
        combined["oracle_bits"] = selected
        combined["oracle_correct"] = rows_by_bits[selected][index]["correct"]
        selected_index = ordered_bits.index(selected)
        # Sequential escalation executes every earlier level as well.  This is
        # intentionally a proxy only: the M0 implementation evaluates all
        # levels independently and does not yet reuse residual computation.
        combined["oracle_cumulative_proxy_bits"] = sum(
            proxy_cost_for_precision(precision) for precision in ordered_bits[:selected_index + 1]
        )
        for lower, higher in zip(ordered_bits, ordered_bits[1:]):
            lo, hi = rows_by_bits[lower][index], rows_by_bits[higher][index]
            prefix = f"q{lower}_to_q{higher}"
            corrected = not lo["correct"] and hi["correct"]
            regressed = lo["correct"] and not hi["correct"]
            combined[f"{prefix}_corrected"] = corrected
            combined[f"{prefix}_regressed"] = regressed
            combined[f"{prefix}_prediction_changed"] = lo["prediction"] != hi["prediction"]
            # Candidate deployable signals available before escalation.
            combined[f"{prefix}_signal_low_confidence"] = 1.0 - lo["confidence"]
            combined[f"{prefix}_signal_entropy"] = lo["entropy"]
            combined[f"{prefix}_signal_small_margin"] = 1.0 - lo["margin"]
        token_rows.append(combined)

    for lower, higher in zip(ordered_bits, ordered_bits[1:]):
        prefix = f"q{lower}_to_q{higher}"
        corrected = sum(row[f"{prefix}_corrected"] for row in token_rows)
        regressed = sum(row[f"{prefix}_regressed"] for row in token_rows)
        changed = sum(row[f"{prefix}_prediction_changed"] for row in token_rows)
        transitions[prefix] = {
            "lower_bits": lower, "higher_bits": higher,
            "corrected": corrected, "regressed": regressed,
            "correctness_unchanged": len(token_rows) - corrected - regressed,
            "prediction_changed": changed, "prediction_unchanged": len(token_rows) - changed,
            "mean_low_confidence_when_corrected": _conditional_mean(token_rows, f"{prefix}_signal_low_confidence", f"{prefix}_corrected"),
            "mean_entropy_when_corrected": _conditional_mean(token_rows, f"{prefix}_signal_entropy", f"{prefix}_corrected"),
            "mean_small_margin_when_corrected": _conditional_mean(token_rows, f"{prefix}_signal_small_margin", f"{prefix}_corrected"),
        }

    oracle_accuracy = sum(row["oracle_correct"] for row in token_rows) / len(token_rows)
    mean_selected_bits = sum(row["oracle_bits"] for row in token_rows) / len(token_rows)
    mean_cumulative_cost = sum(row["oracle_cumulative_proxy_bits"] for row in token_rows) / len(token_rows)
    full_ladder_cost = sum(proxy_cost_for_precision(precision) for precision in ordered_bits)
    return {
        "precision_order": ordered_bits,
        "precision_proxy_costs": proxy_cost_mapping(ordered_bits),
        "per_precision": per_precision,
        "transitions": transitions,
        "oracle": {
            "quality_masked_accuracy": oracle_accuracy,
            "mean_terminal_selected_bits": mean_selected_bits,
            "mean_cumulative_proxy_bits": mean_cumulative_cost,
            "always_full_ladder_proxy_bits_per_token": full_ladder_cost,
            "cumulative_proxy_cost_vs_full_ladder": mean_cumulative_cost / full_ladder_cost,
            "cumulative_proxy_savings_vs_full_ladder": 1.0 - mean_cumulative_cost / full_ladder_cost,
            "single_fp32_proxy_bits_per_token": FP32_PROXY_COST,
            "cumulative_proxy_cost_vs_single_fp32": mean_cumulative_cost / FP32_PROXY_COST,
            "cumulative_proxy_savings_vs_single_fp32": 1.0 - mean_cumulative_cost / FP32_PROXY_COST,
            "note": "Internal precision 16 is the FP32 identity path, not FP16, and has proxy cost 32. Terminal selected bits are a precision outcome, not compute. Cumulative proxy cost sums explicit costs for each visited level. M0 currently full-recomputes every level independently, so these are not measured wall-clock savings or residual-reuse costs.",
        },
    }, token_rows


def _conditional_mean(rows: list[dict], value: str, condition: str) -> float | None:
    values = [row[value] for row in rows if row[condition]]
    return float(np.mean(values)) if values else None


def _validate_logits(precision: int, logits_batches: list[np.ndarray], fixtures: list[MaskedFixture]) -> None:
    """Fail closed rather than silently truncating mismatched fixture runs."""
    if len(logits_batches) != len(fixtures):
        raise ValueError(
            f"Precision {precision} has {len(logits_batches)} logits batches for "
            f"{len(fixtures)} fixtures"
        )
    for fixture_index, (batch_logits, fixture) in enumerate(zip(logits_batches, fixtures)):
        expected_prefix = fixture.targets.shape
        shape = np.asarray(batch_logits).shape
        if len(shape) != 3 or shape[:2] != expected_prefix or shape[2] < 2:
            raise ValueError(
                f"Precision {precision} fixture {fixture_index} logits shape {shape} "
                f"does not match expected (batch, sequence, vocab) with prefix "
                f"{expected_prefix} and vocab >= 2"
            )


def aggregate_fixture_analyses(analyses: Iterable[tuple[dict, list[dict]]]) -> tuple[dict, list[dict]]:
    """Exactly combine compact per-fixture analyses without retaining logits."""
    analyses = list(analyses)
    if not analyses:
        raise ValueError("At least one fixture analysis is required")
    precision_order = analyses[0][0]["precision_order"]
    token_rows = [row for _, rows in analyses for row in rows]
    if not token_rows:
        raise ValueError("Fixture analyses contain no masked tokens")
    total_tokens = len(token_rows)
    per_precision = {}
    for precision in precision_order:
        key = str(precision)
        per_precision[key] = {"masked_tokens": total_tokens}
        for metric in ("masked_loss", "masked_accuracy", "mean_confidence", "mean_entropy", "mean_top1_top2_margin"):
            per_precision[key][metric] = sum(
                summary["per_precision"][key][metric] * summary["per_precision"][key]["masked_tokens"]
                for summary, _ in analyses
            ) / total_tokens
    transitions = {}
    for lower, higher in zip(precision_order, precision_order[1:]):
        prefix = f"q{lower}_to_q{higher}"
        corrected = sum(row[f"{prefix}_corrected"] for row in token_rows)
        regressed = sum(row[f"{prefix}_regressed"] for row in token_rows)
        changed = sum(row[f"{prefix}_prediction_changed"] for row in token_rows)
        transitions[prefix] = {
            "lower_bits": lower, "higher_bits": higher,
            "corrected": corrected, "regressed": regressed,
            "correctness_unchanged": total_tokens - corrected - regressed,
            "prediction_changed": changed, "prediction_unchanged": total_tokens - changed,
            "mean_low_confidence_when_corrected": _conditional_mean(token_rows, f"{prefix}_signal_low_confidence", f"{prefix}_corrected"),
            "mean_entropy_when_corrected": _conditional_mean(token_rows, f"{prefix}_signal_entropy", f"{prefix}_corrected"),
            "mean_small_margin_when_corrected": _conditional_mean(token_rows, f"{prefix}_signal_small_margin", f"{prefix}_corrected"),
        }
    full_ladder_cost = sum(proxy_cost_for_precision(precision) for precision in precision_order)
    mean_cumulative_cost = sum(row["oracle_cumulative_proxy_bits"] for row in token_rows) / total_tokens
    oracle = {
        "quality_masked_accuracy": sum(row["oracle_correct"] for row in token_rows) / total_tokens,
        "mean_terminal_selected_bits": sum(row["oracle_bits"] for row in token_rows) / total_tokens,
        "mean_cumulative_proxy_bits": mean_cumulative_cost,
        "always_full_ladder_proxy_bits_per_token": full_ladder_cost,
        "cumulative_proxy_cost_vs_full_ladder": mean_cumulative_cost / full_ladder_cost,
        "cumulative_proxy_savings_vs_full_ladder": 1.0 - mean_cumulative_cost / full_ladder_cost,
        "single_fp32_proxy_bits_per_token": FP32_PROXY_COST,
        "cumulative_proxy_cost_vs_single_fp32": mean_cumulative_cost / FP32_PROXY_COST,
        "cumulative_proxy_savings_vs_single_fp32": 1.0 - mean_cumulative_cost / FP32_PROXY_COST,
        "note": analyses[0][0]["oracle"]["note"],
    }
    return {"precision_order": precision_order, "precision_proxy_costs": proxy_cost_mapping(precision_order), "per_precision": per_precision,
            "transitions": transitions, "oracle": oracle}, token_rows


def analyze_fixture_stream(fixtures: Iterable[MaskedFixture], bits: Iterable[int], runner) -> tuple[dict, list[dict]]:
    """Run and reduce one fixture at a time; ``runner(bits, fixture)`` returns logits.

    Raw vocabulary logits live only for the duration of a single fixture
    analysis.  The returned token rows are compact scalar statistics.
    """
    ordered_bits = list(bits)
    compact_analyses = []
    for fixture_index, fixture in enumerate(fixtures):
        fixture_logits = {precision: [runner(precision, fixture)] for precision in ordered_bits}
        summary, token_rows = analyze_precision_logits(fixture_logits, [fixture], ordered_bits)
        # ``analyze_precision_logits`` sees one local fixture; restore the
        # original fixture index for the combined CSV provenance.
        for row in token_rows:
            row["fixture_index"] = fixture_index
        compact_analyses.append((summary, token_rows))
        del fixture_logits
    return aggregate_fixture_analyses(compact_analyses)


def run_checkpoint_analysis(config_path: str, checkpoint_path: str, output_dir: str,
                            eval_steps: int, fixture_seed: int, bits: Iterable[int],
                            validation_data: str | None = None) -> dict:
    """Load real project components and write compact analysis artifacts."""
    import mlx.core as mx
    from src.config import ExperimentConfig
    from src.data import build_and_cache_dataset
    from src.model import DiffusionLM
    from src.train import load_checkpoint

    cfg = ExperimentConfig.from_json(config_path)
    model = DiffusionLM(cfg.model)
    checkpoint_step = load_checkpoint(model, checkpoint_path)
    model.eval()
    if validation_data is not None:
        validation_path = Path(validation_data).expanduser().resolve()
        val_data = load_validation_array(validation_path, cfg.data.seq_len, cfg.model.vocab_size)
        validation_source = "explicit_npy"
    else:
        _, val_data = build_and_cache_dataset(
            tokenizer_path=cfg.data.tokenizer_path, cache_dir=cfg.data.data_cache_dir,
            seq_len=cfg.data.seq_len, max_articles=cfg.data.max_articles,
            max_text_bytes=cfg.data.max_text_bytes, dataset_name=cfg.data.dataset_name,
            dataset_config=cfg.data.dataset_config, dataset_revision=cfg.data.dataset_revision,
            train_split=cfg.data.train_split, seed=cfg.train.seed,
        )
        validation_path = None
        validation_source = "dataset_builder"
    fixtures = make_fixtures(val_data, cfg.train.batch_size, eval_steps,
                             cfg.model.mask_token_id(), fixture_seed)
    ordered_bits = list(bits)
    timings = {precision: 0.0 for precision in ordered_bits}

    def runner(precision: int, fixture: MaskedFixture) -> np.ndarray:
        model.set_bits(precision)
        started = time.perf_counter()
        logits = model(mx.array(fixture.inputs), mx.array(fixture.mask_rates))
        mx.eval(logits)
        result = np.array(logits)
        timings[precision] += time.perf_counter() - started
        del logits
        return result

    summary, token_rows = analyze_fixture_stream(fixtures, ordered_bits, runner)
    summary["run"] = {
        "config_path": str(config_path), "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": checkpoint_step, "fixture_seed": fixture_seed,
        "fixture_batches": eval_steps, "batch_size": cfg.train.batch_size,
        "full_logits_saved": False,
        "validation_data_source": validation_source,
        "validation_data_path": str(validation_path) if validation_path is not None else None,
        **run_provenance(checkpoint_path, validation_path),
        "simulated_wall_clock": {
            str(precision): {
                "elapsed_seconds": timings[precision],
                "mean_batch_seconds": timings[precision] / eval_steps,
            }
            for precision in ordered_bits
        },
        "wall_clock_note": "Simulated MLX wall clock for independent full recomputes; it is not a low-bit hardware or sequential-escalation speed measurement.",
    }
    write_artifacts(Path(output_dir), summary, token_rows)
    return summary


def write_artifacts(output_dir: Path, summary: dict, token_rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (output_dir / "per_token.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(token_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(token_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--fixture-seed", type=int, default=20260804)
    parser.add_argument("--bits", type=int, nargs="+", default=list(DEFAULT_BITS))
    parser.add_argument("--validation-data", help="Existing rank-2 validation .npy; skips dataset building/downloads")
    args = parser.parse_args()
    if args.eval_steps <= 0:
        parser.error("--eval-steps must be positive")
    summary = run_checkpoint_analysis(args.config, args.checkpoint, args.output_dir,
                                      args.eval_steps, args.fixture_seed, args.bits,
                                      args.validation_data)
    print(json.dumps(summary["oracle"], indent=2))
    print(f"Wrote {Path(args.output_dir) / 'summary.json'} and per_token.csv")


if __name__ == "__main__":
    main()
