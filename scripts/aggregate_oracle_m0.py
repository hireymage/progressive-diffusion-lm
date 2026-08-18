#!/usr/bin/env python3
"""Aggregate provenance-compatible M0 ``summary.json`` artifacts.

Usage: python scripts/aggregate_oracle_m0.py results/m0/node-a results/m0/node-b
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.oracle_m0 import FP32_PROXY_COST, proxy_cost_mapping


COMPATIBILITY_FIELDS = (
    "precision_order", "git_commit", "checkpoint_sha256", "validation_data_sha256",
    "checkpoint_step", "config_path",
)
PER_PRECISION_MEANS = (
    "masked_loss", "masked_accuracy", "mean_confidence", "mean_entropy",
    "mean_top1_top2_margin",
)
CONDITIONAL_TRANSITION_MEANS = (
    "mean_low_confidence_when_corrected", "mean_entropy_when_corrected",
    "mean_small_margin_when_corrected",
)


def resolve_summary_path(value: str | Path) -> Path:
    path = Path(value)
    return path / "summary.json" if path.is_dir() else path


def load_summaries(values: Iterable[str | Path]) -> list[dict]:
    summaries = []
    for value in values:
        path = resolve_summary_path(value)
        with path.open() as handle:
            summary = json.load(handle)
        summary["_source_path"] = str(path.resolve())
        summaries.append(summary)
    if not summaries:
        raise ValueError("At least one summary.json path or run directory is required")
    return summaries


def _compatibility_value(summary: dict, field: str):
    return summary.get(field) if field == "precision_order" else summary.get("run", {}).get(field)


def verify_compatible(summaries: list[dict]) -> None:
    """Fail closed if artifacts are not evaluations of the same experiment."""
    reference = summaries[0]
    for field in COMPATIBILITY_FIELDS:
        expected = _compatibility_value(reference, field)
        if expected is None and field != "validation_data_sha256":
            raise ValueError(f"Reference summary lacks required compatibility field {field}")
        for summary in summaries[1:]:
            actual = _compatibility_value(summary, field)
            if actual != expected:
                raise ValueError(
                    f"Incompatible {field}: {summary['_source_path']} has {actual!r}; expected {expected!r}"
                )


def _weighted_mean(records: list[tuple[float, float]]) -> float | None:
    denominator = sum(weight for _, weight in records)
    return sum(value * weight for value, weight in records) / denominator if denominator else None


def aggregate_summaries(summaries: list[dict]) -> dict:
    verify_compatible(summaries)
    order = summaries[0]["precision_order"]
    expected_proxy_costs = proxy_cost_mapping(order)
    proxy_costs_present = [summary.get("precision_proxy_costs") for summary in summaries]
    # Pre-correction artifacts omit this field. Their already-recorded
    # cumulative means used internal ID 16 as a cost of 16 and cannot be
    # reconstructed from a compact summary, so never publish invalid ratios.
    if any(costs is None for costs in proxy_costs_present):
        raise ValueError(
            "Legacy proxy accounting detected: every input must include "
            "precision_proxy_costs; rerun oracle_m0.py before aggregation"
        )
    if any(costs is not None and costs != expected_proxy_costs for costs in proxy_costs_present):
        raise ValueError("Incompatible precision_proxy_costs")
    token_counts = [summary["per_precision"][str(order[0])]["masked_tokens"] for summary in summaries]
    if any(count <= 0 for count in token_counts):
        raise ValueError("Every run must have a positive masked-token count")
    total_tokens = sum(token_counts)

    per_precision = {}
    for bits in order:
        key = str(bits)
        per_precision[key] = {"masked_tokens": total_tokens}
        for field in PER_PRECISION_MEANS:
            per_precision[key][field] = _weighted_mean([
                (summary["per_precision"][key][field], summary["per_precision"][key]["masked_tokens"])
                for summary in summaries
            ])

    transitions = {}
    for lower, higher in zip(order, order[1:]):
        key = f"q{lower}_to_q{higher}"
        counts = {field: sum(summary["transitions"][key][field] for summary in summaries)
                  for field in ("corrected", "regressed", "correctness_unchanged", "prediction_changed", "prediction_unchanged")}
        transitions[key] = {"lower_bits": lower, "higher_bits": higher, **counts}
        for field in CONDITIONAL_TRANSITION_MEANS:
            transitions[key][field] = _weighted_mean([
                (summary["transitions"][key][field], summary["transitions"][key]["corrected"])
                for summary in summaries
                if summary["transitions"][key][field] is not None
            ])

    oracle_rows = [summary["oracle"] for summary in summaries]
    reconstructed_correct = sum(row["quality_masked_accuracy"] * count for row, count in zip(oracle_rows, token_counts))
    terminal_sum = sum(row["mean_terminal_selected_bits"] * count for row, count in zip(oracle_rows, token_counts))
    cumulative_sum = sum(row["mean_cumulative_proxy_bits"] * count for row, count in zip(oracle_rows, token_counts))
    ladder_cost = oracle_rows[0]["always_full_ladder_proxy_bits_per_token"]
    if any(row["always_full_ladder_proxy_bits_per_token"] != ladder_cost for row in oracle_rows):
        raise ValueError("Incompatible always_full_ladder_proxy_bits_per_token")
    mean_cumulative = cumulative_sum / total_tokens
    oracle = {
        "quality_masked_accuracy": reconstructed_correct / total_tokens,
        "reconstructed_oracle_correct_tokens": reconstructed_correct,
        "mean_terminal_selected_bits": terminal_sum / total_tokens,
        "reconstructed_terminal_selected_bits_sum": terminal_sum,
        "mean_cumulative_proxy_bits": mean_cumulative,
        "reconstructed_cumulative_proxy_bits_sum": cumulative_sum,
        "always_full_ladder_proxy_bits_per_token": ladder_cost,
        "cumulative_proxy_cost_vs_full_ladder": mean_cumulative / ladder_cost,
        "cumulative_proxy_savings_vs_full_ladder": 1.0 - mean_cumulative / ladder_cost,
        "single_fp32_proxy_bits_per_token": FP32_PROXY_COST,
        "cumulative_proxy_cost_vs_single_fp32": mean_cumulative / FP32_PROXY_COST,
        "cumulative_proxy_savings_vs_single_fp32": 1.0 - mean_cumulative / FP32_PROXY_COST,
        "reconstruction_note": "Counts and sums are reconstructed as reported per-run mean × masked-token count; floating-point roundoff is possible.",
    }
    ranges = {
        "oracle_quality_masked_accuracy": _min_max(row["quality_masked_accuracy"] for row in oracle_rows),
        "oracle_cumulative_proxy_savings_vs_full_ladder": _min_max(row["cumulative_proxy_savings_vs_full_ladder"] for row in oracle_rows),
        "oracle_cumulative_proxy_savings_vs_single_fp32": _min_max(row["cumulative_proxy_savings_vs_single_fp32"] for row in oracle_rows),
        "per_precision_masked_accuracy": {
            str(bits): _min_max(summary["per_precision"][str(bits)]["masked_accuracy"] for summary in summaries)
            for bits in order
        },
    }
    return {
        "aggregate_type": "oracle_m0_distributed",
        "total_runs": len(summaries), "total_masked_tokens": total_tokens,
        "precision_order": order,
        "precision_proxy_costs": expected_proxy_costs,
        "compatibility": {field: _compatibility_value(summaries[0], field) for field in COMPATIBILITY_FIELDS},
        "seeds": sorted({summary["run"].get("fixture_seed") for summary in summaries}),
        "hosts": sorted({summary["run"].get("hostname") for summary in summaries}),
        "sources": [summary["_source_path"] for summary in summaries],
        "per_precision": per_precision, "transitions": transitions, "oracle": oracle,
        "across_run_ranges": ranges,
        "timing_by_host": _timing_by_host(summaries),
    }


def _min_max(values) -> dict:
    values = list(values)
    return {"min": min(values), "max": max(values)}


def _timing_by_host(summaries: list[dict]) -> dict:
    """Keep timings host-scoped; no cross-hardware speed average is claimed."""
    grouped = defaultdict(lambda: defaultdict(list))
    for summary in summaries:
        host = summary["run"].get("hostname", "unknown")
        for bits, timing in summary["run"].get("simulated_wall_clock", {}).items():
            grouped[host][bits].append(timing)
    return {
        host: {
            bits: {
                "runs": len(values),
                "total_elapsed_seconds": sum(item["elapsed_seconds"] for item in values),
                "min_batch_seconds": min(item["mean_batch_seconds"] for item in values),
                "max_batch_seconds": max(item["mean_batch_seconds"] for item in values),
            }
            for bits, values in by_bits.items()
        }
        for host, by_bits in grouped.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="summary.json paths or run directories")
    parser.add_argument("--output", default="aggregate_summary.json")
    args = parser.parse_args()
    result = aggregate_summaries(load_summaries(args.inputs))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
