#!/usr/bin/env python3
"""Calibrate deployable M0 escalation policies and evaluate their Pareto tradeoff.

Policies are *fit only* from feature quantiles in the calibration CSV.  Target,
loss, correctness, and oracle columns are deliberately unavailable to the
selection function; they are read only after a terminal precision is selected
to score accuracy.  The default command uses the corrected three-node pilot.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


STAGES = (1, 2, 4, 8, 16)  # Internal 16 is the FP32 identity path.
CUMULATIVE_COSTS = {1: 1, 2: 3, 4: 7, 8: 15, 16: 47}
DIRECT_COSTS = {1: 1, 2: 2, 4: 4, 8: 8, 16: 32}
DEFAULT_BOOTSTRAP_ITERATIONS = 2000
DEFAULT_BOOTSTRAP_SEED = 20260807
PROXY_COSTS = {"1": 1, "2": 2, "4": 4, "8": 8, "16": 32}
CALIBRATION_SEED = 20260804
POLICY_FEATURE_SUFFIXES = ("prediction", "confidence", "entropy", "margin")
FORBIDDEN_POLICY_TERMS = ("target", "correct", "loss", "oracle")


@dataclass(frozen=True)
class TokenFeatures:
    """Only values allowed to reach the deployable stage-selection code."""

    predictions: Mapping[int, int]
    confidence: Mapping[int, float]
    entropy: Mapping[int, float]
    margin: Mapping[int, float]


@dataclass(frozen=True)
class Policy:
    name: str
    kind: str
    family: str
    cost_mode: str
    threshold: float | None = None


def policy_feature_columns(stages: Sequence[int] = STAGES) -> tuple[str, ...]:
    return tuple(f"q{stage}_{suffix}" for stage in stages for suffix in POLICY_FEATURE_SUFFIXES)


def _require_schema(fieldnames: Iterable[str] | None) -> None:
    present = set(fieldnames or ())
    required = set(policy_feature_columns()) | {f"q{stage}_correct" for stage in STAGES}
    missing = sorted(required - present)
    if missing:
        raise ValueError(f"per_token.csv lacks required columns: {', '.join(missing)}")


def features_from_row(row: Mapping[str, str]) -> TokenFeatures:
    """Extract a whitelist, rather than passing a full CSV row to a policy."""
    return TokenFeatures(
        predictions={stage: int(row[f"q{stage}_prediction"]) for stage in STAGES},
        confidence={stage: float(row[f"q{stage}_confidence"]) for stage in STAGES},
        entropy={stage: float(row[f"q{stage}_entropy"]) for stage in STAGES},
        margin={stage: float(row[f"q{stage}_margin"]) for stage in STAGES},
    )


def choose_stage(features: TokenFeatures, policy: Policy) -> int:
    """Select using only ``TokenFeatures``; FP32 is the mandatory fallback."""
    if policy.kind == "fixed_stop":
        assert policy.threshold is not None
        return int(policy.threshold)
    if policy.kind == "oracle":
        raise ValueError("oracle is non-deployable and has no feature-only selection rule")
    assert policy.threshold is not None
    for index, stage in enumerate(STAGES[:-1]):
        if policy.kind in ("confidence", "confidence_agreement"):
            pass_signal = features.confidence[stage] >= policy.threshold
        elif policy.kind == "entropy":
            pass_signal = features.entropy[stage] <= policy.threshold
        elif policy.kind == "margin":
            pass_signal = features.margin[stage] >= policy.threshold
        else:
            raise ValueError(f"Unknown policy kind {policy.kind!r}")
        # Agreement is available only after the current stage was executed.
        stable = index == 0 or features.predictions[stage] == features.predictions[STAGES[index - 1]]
        if pass_signal and (policy.kind != "confidence_agreement" or stable):
            return stage
    return 16


def _quantile_grid(values: Iterable[float]) -> list[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot construct thresholds from an empty calibration CSV")
    return sorted({ordered[round((len(ordered) - 1) * q / 10)] for q in range(11)})


def policies_from_calibration(rows: Sequence[Mapping[str, str]]) -> list[Policy]:
    """Build a frozen grid solely from calibration features, never outcomes."""
    grids = {
        "confidence": _quantile_grid(float(row[f"q{stage}_confidence"]) for row in rows for stage in STAGES[:-1]),
        "entropy": _quantile_grid(float(row[f"q{stage}_entropy"]) for row in rows for stage in STAGES[:-1]),
        "margin": _quantile_grid(float(row[f"q{stage}_margin"]) for row in rows for stage in STAGES[:-1]),
    }
    policies = [
        Policy(f"ladder_stop_q{stage}", "fixed_stop", "ladder_stop", "cumulative_ladder", float(stage))
        for stage in STAGES
    ] + [
        Policy(f"direct_q{'fp32' if stage == 16 else stage}", "fixed_stop", "direct_baseline", "direct_precision", float(stage))
        for stage in STAGES
    ]
    for kind, thresholds in grids.items():
        policies.extend(Policy(f"{kind}_ge_or_le_{threshold:.12g}", kind, "adaptive", "cumulative_ladder", threshold)
                        for threshold in thresholds)
    # A conservative combined rule: confident at Q1, then confident and stable.
    policies.extend(Policy(f"confidence_agreement_{threshold:.12g}", "confidence_agreement", "adaptive", "cumulative_ladder", threshold)
                    for threshold in grids["confidence"])
    return policies


def _score_selected(rows: Sequence[Mapping[str, str]], selected: Sequence[int], costs: Mapping[int, int]) -> dict:
    if not rows:
        raise ValueError("Cannot score an empty CSV")
    correct = sum(row[f"q{stage}_correct"].strip().lower() == "true" for row, stage in zip(rows, selected))
    mean_cost = sum(costs[stage] for stage in selected) / len(rows)
    terminal_counts = {str(stage): sum(value == stage for value in selected) for stage in STAGES}
    return {
        "tokens": len(rows), "accuracy": correct / len(rows), "correct_tokens": correct,
        "mean_cumulative_proxy_cost": mean_cost, "terminal_stage_counts": terminal_counts,
    }


def score_policy(rows: Sequence[Mapping[str, str]], policy: Policy) -> dict:
    selected = [choose_stage(features_from_row(row), policy) for row in rows]
    costs = DIRECT_COSTS if policy.cost_mode == "direct_precision" else CUMULATIVE_COSTS
    return _score_selected(rows, selected, costs)


def policy_outcomes(rows: Sequence[Mapping[str, str]], policy: Policy) -> list[dict]:
    """Compact held-out outcomes, retaining fixture clusters for bootstrap."""
    costs = DIRECT_COSTS if policy.cost_mode == "direct_precision" else CUMULATIVE_COSTS
    outcomes = []
    for row in rows:
        stage = choose_stage(features_from_row(row), policy)
        outcomes.append({"fixture_index": row["fixture_index"],
                         "correct": int(row[f"q{stage}_correct"].lower() == "true"),
                         "cost": costs[stage]})
    return outcomes


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = int(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_cluster_bootstrap(policy_outcomes_by_source: Mapping[str, Sequence[dict]],
                             baseline_outcomes_by_source: Mapping[str, Sequence[dict]],
                             *, iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
                             seed: int = DEFAULT_BOOTSTRAP_SEED) -> dict:
    """Paired percentile bootstrap, resampling whole ``(source, fixture)`` clusters."""
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    clusters: dict[tuple[str, str], list[tuple[int, int, int, int]]] = {}
    for source, policy_outcomes in policy_outcomes_by_source.items():
        baseline_outcomes = baseline_outcomes_by_source.get(source)
        if baseline_outcomes is None or len(policy_outcomes) != len(baseline_outcomes):
            raise ValueError("Paired bootstrap requires equally aligned policy and baseline outcomes")
        for policy, baseline in zip(policy_outcomes, baseline_outcomes):
            if policy["fixture_index"] != baseline["fixture_index"]:
                raise ValueError("Paired bootstrap fixture indices are not aligned")
            clusters.setdefault((source, str(policy["fixture_index"])), []).append(
                (policy["correct"], baseline["correct"], policy["cost"], baseline["cost"])
            )
    if not clusters:
        raise ValueError("Paired bootstrap requires at least one cluster")
    aggregates = []
    for values in clusters.values():
        aggregates.append((sum(item[0] for item in values), sum(item[1] for item in values),
                           sum(item[2] for item in values), sum(item[3] for item in values), len(values)))
    def deltas(sample):
        token_count = sum(item[4] for item in sample)
        return ((sum(item[0] - item[1] for item in sample) / token_count),
                (sum(item[2] - item[3] for item in sample) / token_count))
    point_accuracy, point_cost = deltas(aggregates)
    rng = random.Random(seed)
    accuracy_samples, cost_samples = [], []
    for _ in range(iterations):
        accuracy, cost = deltas([aggregates[rng.randrange(len(aggregates))] for _ in aggregates])
        accuracy_samples.append(accuracy); cost_samples.append(cost)
    accuracy_ci = [_percentile(accuracy_samples, 0.025), _percentile(accuracy_samples, 0.975)]
    return {
        "cluster_definition": "(heldout_source, fixture_index)", "cluster_count": len(aggregates),
        "bootstrap_iterations": iterations, "bootstrap_seed": seed,
        "point_accuracy_delta": point_accuracy, "point_mean_cost_delta": point_cost,
        "accuracy_delta_ci_95": accuracy_ci,
        "mean_cost_delta_ci_95": [_percentile(cost_samples, 0.025), _percentile(cost_samples, 0.975)],
        "accuracy_ci_excludes_zero": accuracy_ci[0] > 0 or accuracy_ci[1] < 0,
    }


def score_oracle(rows: Sequence[Mapping[str, str]]) -> dict:
    """Upper bound only: labels decide the first correct stage."""
    selected = [next((stage for stage in STAGES if row[f"q{stage}_correct"].lower() == "true"), 16)
                for row in rows]
    return _score_selected(rows, selected, CUMULATIVE_COSTS)


def nondominated(records: Sequence[dict]) -> list[dict]:
    """Maximize accuracy while minimizing cost, collapsing identical outcomes."""
    frontier = []
    for record in records:
        dominated = any(
            other["accuracy"] >= record["accuracy"] and
            other["mean_cumulative_proxy_cost"] <= record["mean_cumulative_proxy_cost"] and
            (other["accuracy"] > record["accuracy"] or other["mean_cumulative_proxy_cost"] < record["mean_cumulative_proxy_cost"])
            for other in records
        )
        if not dominated:
            frontier.append(record)
    unique = {}
    for record in frontier:
        # Records arrive with fixed-stop baselines first, so they are the
        # canonical representative of an otherwise identical policy outcome.
        unique.setdefault((record["accuracy"], record["mean_cumulative_proxy_cost"]), record)
    return sorted(unique.values(), key=lambda item: (item["mean_cumulative_proxy_cost"], -item["accuracy"], item["name"]))


def _read_run(run_dir: str | Path, *, expected_seed: int | None = None) -> tuple[list[dict[str, str]], dict, str]:
    directory = Path(run_dir)
    csv_path, summary_path = directory / "per_token.csv", directory / "summary.json"
    if not csv_path.is_file() or not summary_path.is_file():
        raise ValueError(f"Run must contain per_token.csv and summary.json: {directory}")
    summary = json.loads(summary_path.read_text())
    if summary.get("precision_order") != list(STAGES):
        raise ValueError(f"Unexpected precision_order in {summary_path}")
    if summary.get("precision_proxy_costs") != PROXY_COSTS:
        raise ValueError(f"Unexpected precision_proxy_costs in {summary_path}")
    run = summary.get("run")
    if not isinstance(run, dict):
        raise ValueError(f"Summary lacks run provenance: {summary_path}")
    if expected_seed is not None and run.get("fixture_seed") != expected_seed:
        raise ValueError(f"Calibration seed must be {expected_seed}, got {run.get('fixture_seed')}")
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        _require_schema(reader.fieldnames)
        rows = list(reader)
    if not rows:
        raise ValueError(f"per_token.csv is empty: {csv_path}")
    return rows, summary, str(directory.resolve())


def _aggregate(scores: Sequence[dict]) -> dict:
    total = sum(item["tokens"] for item in scores)
    return {
        "tokens": total,
        "accuracy": sum(item["correct_tokens"] for item in scores) / total,
        "correct_tokens": sum(item["correct_tokens"] for item in scores),
        "mean_cumulative_proxy_cost": sum(item["mean_cumulative_proxy_cost"] * item["tokens"] for item in scores) / total,
        "terminal_stage_counts": {str(stage): sum(item["terminal_stage_counts"][str(stage)] for item in scores) for stage in STAGES},
    }


def run_analysis(calibration_dir: str | Path, heldout_dirs: Sequence[str | Path], *,
                 bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
                 bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED) -> dict:
    calibration_rows, calibration_summary, calibration_source = _read_run(calibration_dir, expected_seed=CALIBRATION_SEED)
    heldout = [_read_run(path) for path in heldout_dirs]
    if not heldout:
        raise ValueError("At least one held-out run is required")
    heldout_seeds = [summary["run"].get("fixture_seed") for _, summary, _ in heldout]
    if any(seed == CALIBRATION_SEED for seed in heldout_seeds):
        raise ValueError("Held-out runs must not use the calibration fixture seed")
    if len(set(heldout_seeds)) != len(heldout_seeds):
        raise ValueError("Held-out fixture seeds must be unique")
    # When provenance exists, every held-out artifact must be the same model/data.
    for _, summary, source in heldout:
        for field in ("checkpoint_sha256", "validation_data_sha256", "checkpoint_step", "config_path", "git_commit"):
            calibration_value = calibration_summary["run"].get(field)
            heldout_value = summary["run"].get(field)
            if (field != "git_commit" or calibration_value is not None or heldout_value is not None) and heldout_value != calibration_value:
                raise ValueError(f"Incompatible provenance {field} in {source}")
    policies = policies_from_calibration(calibration_rows)
    records = []
    for policy in policies:
        calibration = score_policy(calibration_rows, policy)
        heldout_by_run = {source: score_policy(rows, policy) for rows, _, source in heldout}
        aggregate = _aggregate(list(heldout_by_run.values()))
        records.append({"name": policy.name, "kind": policy.kind, "family": policy.family, "cost_mode": policy.cost_mode, "threshold": policy.threshold,
                        "deployable": True, "calibration": calibration,
                        "heldout_by_run": heldout_by_run, "heldout_aggregate": aggregate,
                        **aggregate})
    oracle_by_run = {source: score_oracle(rows) for rows, _, source in heldout}
    oracle = {"name": "oracle_upper_bound", "kind": "oracle", "family": "oracle", "cost_mode": "cumulative_ladder", "threshold": None, "deployable": False,
              "calibration": score_oracle(calibration_rows), "heldout_by_run": oracle_by_run,
              "heldout_aggregate": _aggregate(list(oracle_by_run.values())), **_aggregate(list(oracle_by_run.values()))}
    # Policy selection is performed here, on calibration labels only.  Its names
    # are frozen before the held-out rows are considered.
    def calibration_candidates(subset):
        return [{"name": record["name"], "family": record["family"], "cost_mode": record["cost_mode"],
                 "accuracy": record["calibration"]["accuracy"],
                 "mean_cumulative_proxy_cost": record["calibration"]["mean_cumulative_proxy_cost"]}
                for record in subset]

    adaptive_records = [record for record in records if record["family"] == "adaptive"]
    adaptive_calibration_frontier = nondominated(calibration_candidates(adaptive_records))
    all_calibration_frontier = nondominated(calibration_candidates(records))
    def selected_heldout(frontier):
        selected_names = {record["name"] for record in frontier}
        return [record for record in records if record["name"] in selected_names]

    adaptive_selected_heldout = selected_heldout(adaptive_calibration_frontier)
    all_selected_heldout = selected_heldout(all_calibration_frontier)
    # This is descriptive only: it intentionally does not feed policy selection.
    observed_heldout_frontier = nondominated(records)
    policies_by_name = {policy.name: policy for policy in policies}
    heldout_rows_by_source = {source: rows for rows, _, source in heldout}
    outcome_cache: dict[str, dict[str, list[dict]]] = {}
    def outcomes_for(policy_name: str) -> dict[str, list[dict]]:
        if policy_name not in outcome_cache:
            outcome_cache[policy_name] = {
                source: policy_outcomes(rows, policies_by_name[policy_name])
                for source, rows in heldout_rows_by_source.items()
            }
        return outcome_cache[policy_name]

    direct_baselines = ["direct_q1", "direct_q2", "direct_q4", "direct_q8", "direct_qfp32"]
    paired_bootstrap = {
        record["name"]: {
            baseline: paired_cluster_bootstrap(outcomes_for(record["name"]), outcomes_for(baseline),
                                               iterations=bootstrap_iterations, seed=bootstrap_seed)
            for baseline in direct_baselines
        }
        for record in adaptive_selected_heldout
    }
    return {
        "analysis_type": "m0_offline_adaptive_policy_pareto",
        "selection_guard": {"policy_feature_columns": list(policy_feature_columns()),
                            "forbidden_policy_terms": list(FORBIDDEN_POLICY_TERMS),
                            "note": "Ground truth is used only after feature-only terminal-stage selection; oracle is separately marked non-deployable."},
        "methodology": {
            "threshold_candidates": "Feature quantiles from the calibration run only.",
            "policy_selection": "Calibration labels choose frozen adaptive-only and all-deployable calibration frontiers; the all-deployable comparison includes direct baselines.",
            "heldout_labels": "Evaluation-only: held-out outcomes do not select policies or alter either calibration-selected held-out report.",
            "bootstrap": "Deterministic paired cluster bootstrap is held-out evaluation-only and does not alter policy selection.",
        },
        "stages": list(STAGES), "cumulative_proxy_costs": {str(k): v for k, v in CUMULATIVE_COSTS.items()},
        "direct_proxy_costs": {str(k): v for k, v in DIRECT_COSTS.items()},
        "calibration": {"source": calibration_source, "fixture_seed": CALIBRATION_SEED, "tokens": len(calibration_rows),
                        "threshold_source": "feature quantiles from calibration run only"},
        "heldout_sources": [source for _, _, source in heldout],
        "policies": records,
        "adaptive_calibration_pareto_frontier": adaptive_calibration_frontier,
        "adaptive_calibration_selected_heldout": adaptive_selected_heldout,
        "all_deployable_calibration_pareto_frontier": all_calibration_frontier,
        "all_deployable_calibration_selected_heldout": all_selected_heldout,
        "observed_heldout_frontier": observed_heldout_frontier,
        "observed_heldout_frontier_note": "Diagnostic only; never used for policy selection.",
        "heldout_paired_cluster_bootstrap": paired_bootstrap,
        "oracle_upper_bound": oracle,
    }


def write_artifacts(output_dir: str | Path, result: dict) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "pareto_m0.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    fields = ("name", "kind", "family", "cost_mode", "deployable", "threshold", "accuracy", "mean_cumulative_proxy_cost", "tokens", "correct_tokens", "is_adaptive_calibration_selected", "is_all_deployable_calibration_selected")
    rows = []
    adaptive_names = {record["name"] for record in result["adaptive_calibration_pareto_frontier"]}
    all_names = {record["name"] for record in result["all_deployable_calibration_pareto_frontier"]}
    for record in [*result["policies"], result["oracle_upper_bound"]]:
        rows.append({field: record.get(field) for field in fields} | {
            "is_adaptive_calibration_selected": record["name"] in adaptive_names,
            "is_all_deployable_calibration_selected": record["name"] in all_names,
        })
    with (destination / "pareto_m0_policies.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", default="results/m0/pilot_m1-256_s20260804")
    parser.add_argument("--heldout-dir", action="append", default=None,
                        help="Repeat for each held-out run (defaults to m1-512 and m4-air pilots).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()
    heldout = args.heldout_dir or ["results/m0/pilot_m1-512_s20260805", "results/m0/pilot_m4-air_s20260806"]
    result = run_analysis(args.calibration_dir, heldout, bootstrap_iterations=args.bootstrap_iterations,
                          bootstrap_seed=args.bootstrap_seed)
    write_artifacts(args.output_dir, result)
    print(f"Wrote {Path(args.output_dir) / 'pareto_m0.json'} and pareto_m0_policies.csv")


if __name__ == "__main__":
    main()
