import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.pareto_m0 import (
    CALIBRATION_SEED,
    FORBIDDEN_POLICY_TERMS,
    Policy,
    choose_stage,
    features_from_row,
    policy_feature_columns,
    paired_cluster_bootstrap,
    run_analysis,
    score_policy,
)


def _row(index, *, q1_correct=False, q2_correct=True, confidence=0.9, prediction=3):
    row = {"fixture_index": str(index), "target": "9", "oracle_bits": "2", "oracle_correct": "True"}
    for stage in (1, 2, 4, 8, 16):
        row[f"q{stage}_prediction"] = str(prediction if stage != 2 else prediction + 1)
        row[f"q{stage}_confidence"] = str(confidence if stage == 1 else 0.8)
        row[f"q{stage}_entropy"] = str(0.1 if stage == 1 else 0.2)
        row[f"q{stage}_margin"] = str(0.7 if stage == 1 else 0.6)
        row[f"q{stage}_correct"] = str(q1_correct if stage == 1 else q2_correct if stage == 2 else True)
        row[f"q{stage}_loss"] = "123.0"
    return row


def _write_run(directory, rows, seed):
    directory.mkdir()
    with (directory / "per_token.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "precision_order": [1, 2, 4, 8, 16],
        "precision_proxy_costs": {"1": 1, "2": 2, "4": 4, "8": 8, "16": 32},
        "run": {"fixture_seed": seed, "checkpoint_sha256": "checkpoint", "validation_data_sha256": "validation",
                "checkpoint_step": 1, "config_path": "configs/test.json"},
    }
    (directory / "summary.json").write_text(json.dumps(summary))


def test_policy_selection_cannot_receive_ground_truth_and_uses_current_stage_cost():
    row = _row(0, q1_correct=False, confidence=0.95)
    features = features_from_row(row)
    assert not any(any(term in column for term in FORBIDDEN_POLICY_TERMS) for column in policy_feature_columns())
    # The features retain no target/correct/loss/oracle values, so changing each
    # outcome field cannot affect the feature-only terminal decision.
    changed = dict(row)
    changed.update({"target": "0", "oracle_bits": "16", "oracle_correct": "False"})
    for stage in (1, 2, 4, 8, 16):
        changed[f"q{stage}_correct"] = "True"
        changed[f"q{stage}_loss"] = "-999"
    policy = Policy("confident", "confidence", "adaptive", "cumulative_ladder", 0.9)
    assert choose_stage(features, policy) == choose_stage(features_from_row(changed), policy) == 1


def test_run_analysis_calibrates_only_first_run_and_marks_oracle_non_deployable(tmp_path):
    calibration = tmp_path / "calibration"
    heldout_a, heldout_b = tmp_path / "heldout_a", tmp_path / "heldout_b"
    _write_run(calibration, [_row(0, confidence=0.95), _row(1, q1_correct=True, confidence=0.2)], CALIBRATION_SEED)
    _write_run(heldout_a, [_row(0, confidence=0.95), _row(1, confidence=0.2)], CALIBRATION_SEED + 1)
    _write_run(heldout_b, [_row(0, confidence=0.4), _row(1, q1_correct=True, confidence=0.3)], CALIBRATION_SEED + 2)
    result = run_analysis(calibration, [heldout_a, heldout_b])
    assert result["calibration"]["fixture_seed"] == CALIBRATION_SEED
    assert result["cumulative_proxy_costs"] == {"1": 1, "2": 3, "4": 7, "8": 15, "16": 47}
    assert result["oracle_upper_bound"]["deployable"] is False
    assert result["adaptive_calibration_pareto_frontier"]
    assert result["all_deployable_calibration_pareto_frontier"]
    assert all(record["deployable"] for record in result["all_deployable_calibration_selected_heldout"])
    assert all(record["heldout_aggregate"]["tokens"] == 4 for record in result["policies"])
    assert "evaluation-only" in result["methodology"]["heldout_labels"].lower()


def test_run_analysis_rejects_bad_proxy_metadata_and_calibration_seed(tmp_path):
    calibration, heldout = tmp_path / "calibration", tmp_path / "heldout"
    _write_run(calibration, [_row(0)], 123)
    _write_run(heldout, [_row(0)], CALIBRATION_SEED + 1)
    with pytest.raises(ValueError, match="Calibration seed"):
        run_analysis(calibration, [heldout])
    summary_path = calibration / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["run"]["fixture_seed"] = CALIBRATION_SEED
    summary["precision_proxy_costs"]["16"] = 16
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="precision_proxy_costs"):
        run_analysis(calibration, [heldout])


def test_run_analysis_rejects_heldout_provenance_mismatch(tmp_path):
    calibration, heldout = tmp_path / "calibration", tmp_path / "heldout"
    _write_run(calibration, [_row(0)], CALIBRATION_SEED)
    _write_run(heldout, [_row(0)], CALIBRATION_SEED + 1)
    summary_path = heldout / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["run"]["checkpoint_sha256"] = "different"
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        run_analysis(calibration, [heldout])


def test_run_analysis_compares_git_commit_when_present(tmp_path):
    calibration, heldout = tmp_path / "calibration", tmp_path / "heldout"
    _write_run(calibration, [_row(0)], CALIBRATION_SEED)
    _write_run(heldout, [_row(0)], CALIBRATION_SEED + 1)
    for directory, commit in ((calibration, "left"), (heldout, "right")):
        path = directory / "summary.json"
        summary = json.loads(path.read_text())
        summary["run"]["git_commit"] = commit
        path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="git_commit"):
        run_analysis(calibration, [heldout])


def test_heldout_labels_change_diagnostic_frontier_but_not_calibration_selection(tmp_path):
    calibration, heldout = tmp_path / "calibration", tmp_path / "heldout"
    _write_run(calibration, [_row(0, confidence=0.9), _row(1, q1_correct=True, confidence=0.1)], CALIBRATION_SEED)
    initial = [_row(0, q1_correct=False, q2_correct=True, confidence=0.9), _row(1, q1_correct=False, q2_correct=True, confidence=0.1)]
    _write_run(heldout, initial, CALIBRATION_SEED + 1)
    first = run_analysis(calibration, [heldout])
    # Retain all feature values, then replace only held-out labels/outcomes.
    altered = [dict(row) for row in initial]
    for row in altered:
        for stage in (1, 2, 4, 8, 16):
            row[f"q{stage}_correct"] = str(stage == 1)
    with (heldout / "per_token.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(altered[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(altered)
    second = run_analysis(calibration, [heldout])
    selected = lambda result: [record["name"] for record in result["all_deployable_calibration_selected_heldout"]]
    observed = lambda result: [(record["name"], record["accuracy"]) for record in result["observed_heldout_frontier"]]
    assert selected(first) == selected(second)
    assert observed(first) != observed(second)


def test_direct_and_ladder_stop_baselines_have_distinct_cost_accounting():
    rows = [_row(0)]
    direct_q4 = Policy("direct_q4", "fixed_stop", "direct_baseline", "direct_precision", 4)
    ladder_q4 = Policy("ladder_stop_q4", "fixed_stop", "ladder_stop", "cumulative_ladder", 4)
    direct_fp32 = Policy("direct_qfp32", "fixed_stop", "direct_baseline", "direct_precision", 16)
    assert score_policy(rows, direct_q4)["mean_cumulative_proxy_cost"] == 4
    assert score_policy(rows, ladder_q4)["mean_cumulative_proxy_cost"] == 7
    assert score_policy(rows, direct_fp32)["mean_cumulative_proxy_cost"] == 32


def test_paired_bootstrap_is_deterministic_and_resamples_fixture_clusters_not_tokens():
    # Four tokens but two fixture batches: token-level resampling would report 4 clusters.
    policy = {"node": [
        {"fixture_index": "0", "correct": 1, "cost": 3}, {"fixture_index": "0", "correct": 1, "cost": 3},
        {"fixture_index": "1", "correct": 0, "cost": 7}, {"fixture_index": "1", "correct": 0, "cost": 7},
    ]}
    baseline = {"node": [
        {"fixture_index": "0", "correct": 0, "cost": 4}, {"fixture_index": "0", "correct": 0, "cost": 4},
        {"fixture_index": "1", "correct": 1, "cost": 4}, {"fixture_index": "1", "correct": 1, "cost": 4},
    ]}
    first = paired_cluster_bootstrap(policy, baseline, iterations=100, seed=17)
    second = paired_cluster_bootstrap(policy, baseline, iterations=100, seed=17)
    assert first == second
    assert first["cluster_definition"] == "(heldout_source, fixture_index)"
    assert first["cluster_count"] == 2
    assert first["point_accuracy_delta"] == 0
    assert first["point_mean_cost_delta"] == 1


def test_run_analysis_rejects_calibration_or_duplicate_heldout_seeds(tmp_path):
    calibration, left, right = tmp_path / "calibration", tmp_path / "left", tmp_path / "right"
    _write_run(calibration, [_row(0)], CALIBRATION_SEED)
    _write_run(left, [_row(0)], CALIBRATION_SEED)
    with pytest.raises(ValueError, match="calibration fixture seed"):
        run_analysis(calibration, [left])
    # Rewrite only the summary seed; rows are irrelevant to this validation.
    for directory in (left, right):
        if not directory.exists():
            _write_run(directory, [_row(0)], CALIBRATION_SEED + 1)
        else:
            summary = json.loads((directory / "summary.json").read_text())
            summary["run"]["fixture_seed"] = CALIBRATION_SEED + 1
            (directory / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="must be unique"):
        run_analysis(calibration, [left, right])


def test_pareto_cli_writes_machine_readable_json_and_csv(tmp_path):
    calibration = tmp_path / "calibration"
    heldout = tmp_path / "heldout"
    _write_run(calibration, [_row(0), _row(1, q1_correct=True, confidence=0.1)], CALIBRATION_SEED)
    _write_run(heldout, [_row(0), _row(1)], CALIBRATION_SEED + 1)
    output = tmp_path / "out"
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/pareto_m0.py", "--calibration-dir", str(calibration),
         "--heldout-dir", str(heldout), "--output-dir", str(output)],
        cwd=root, capture_output=True, text=True, check=True,
    )
    assert "Wrote" in completed.stdout
    result = json.loads((output / "pareto_m0.json").read_text())
    assert result["policies"]
    assert list(csv.DictReader((output / "pareto_m0_policies.csv").open()))
