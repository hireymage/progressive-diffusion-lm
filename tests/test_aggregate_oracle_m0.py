import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.aggregate_oracle_m0 import aggregate_summaries


def _summary(host, seed, tokens, accuracy, corrected, confidence=0.5):
    return {
        "precision_order": [1, 16],
        "precision_proxy_costs": {"1": 1, "16": 32},
        "per_precision": {
            "1": {"masked_tokens": tokens, "masked_loss": 2.0, "masked_accuracy": accuracy,
                  "mean_confidence": confidence, "mean_entropy": 1.0, "mean_top1_top2_margin": 0.2},
            "16": {"masked_tokens": tokens, "masked_loss": 1.0, "masked_accuracy": 0.9,
                   "mean_confidence": 0.8, "mean_entropy": 0.5, "mean_top1_top2_margin": 0.6},
        },
        "transitions": {"q1_to_q16": {"lower_bits": 1, "higher_bits": 16, "corrected": corrected,
            "regressed": 1, "correctness_unchanged": tokens - corrected - 1,
            "prediction_changed": corrected + 1, "prediction_unchanged": tokens - corrected - 1,
            "mean_low_confidence_when_corrected": confidence,
            "mean_entropy_when_corrected": 1.0, "mean_small_margin_when_corrected": 0.8}},
        "oracle": {"quality_masked_accuracy": 0.95, "mean_terminal_selected_bits": 2.0,
            "mean_cumulative_proxy_bits": 3.0, "always_full_ladder_proxy_bits_per_token": 33,
            "cumulative_proxy_savings_vs_full_ladder": 1 - 3 / 33,
            "single_fp32_proxy_bits_per_token": 32,
            "cumulative_proxy_savings_vs_single_fp32": 1 - 3 / 32},
        "run": {"git_commit": "abc", "checkpoint_sha256": "ckpt", "validation_data_sha256": "val",
            "checkpoint_step": 7, "config_path": "configs/a.json", "fixture_seed": seed, "hostname": host,
            "simulated_wall_clock": {"1": {"elapsed_seconds": 2.0, "mean_batch_seconds": 1.0}}},
        "_source_path": f"/{host}.json",
    }


def test_aggregate_oracle_weights_metrics_reconstructs_counts_and_keeps_host_timing():
    result = aggregate_summaries([_summary("a", 1, 10, 0.2, 2, 0.2), _summary("b", 2, 30, 0.6, 6, 0.8)])
    assert result["total_masked_tokens"] == 40
    assert result["per_precision"]["1"]["masked_accuracy"] == 0.5
    assert result["transitions"]["q1_to_q16"]["corrected"] == 8
    assert result["transitions"]["q1_to_q16"]["mean_low_confidence_when_corrected"] == pytest.approx(0.65)
    assert result["oracle"]["reconstructed_oracle_correct_tokens"] == 38.0
    assert result["oracle"]["reconstructed_cumulative_proxy_bits_sum"] == 120.0
    assert result["oracle"]["single_fp32_proxy_bits_per_token"] == 32
    assert result["precision_proxy_costs"] == {"1": 1, "16": 32}
    assert set(result["timing_by_host"]) == {"a", "b"}


def test_aggregate_oracle_rejects_provenance_mismatch():
    left, right = _summary("a", 1, 10, 0.2, 2), _summary("b", 2, 10, 0.2, 2)
    right["run"]["checkpoint_sha256"] = "other"
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        aggregate_summaries([left, right])


def test_aggregate_oracle_rejects_legacy_proxy_accounting():
    legacy = _summary("a", 1, 10, 0.2, 2)
    del legacy["precision_proxy_costs"]
    with pytest.raises(ValueError, match="rerun oracle_m0.py"):
        aggregate_summaries([legacy])


def test_aggregate_oracle_cli_runs_from_repository_root(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_summary("a", 1, 10, 0.2, 2)))
    output_path = tmp_path / "aggregate.json"
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/aggregate_oracle_m0.py", str(summary_path), "--output", str(output_path)],
        cwd=root, capture_output=True, text=True, check=True,
    )
    assert f"Wrote {output_path}" in result.stdout
    assert json.loads(output_path.read_text())["total_runs"] == 1
