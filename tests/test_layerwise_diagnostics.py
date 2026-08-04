import numpy as np

import json

import pytest

from scripts.layerwise_diagnostics import (
    aggregate_masked_metrics, atomic_json_write, count_tokens, deterministic_mask,
    frequency_summary, gate_streak, parse_mask_schedule, schedule_rate,
)


class TinyTokenizer:
    def decode(self, ids): return "/".join(map(str, ids))


def test_deterministic_mask_is_reproducible_and_nonempty_per_row():
    first = deterministic_mask((3, 8), .15, 7)
    np.testing.assert_array_equal(first, deterministic_mask((3, 8), .15, 7))
    assert first[:, 0].all()


def test_frequency_helpers_count_and_decode_top_token():
    counts = count_tokens(np.array([[1, 2, 2], [2, 3, 1]], dtype=np.int32), 4)
    assert counts.tolist() == [0, 2, 3, 1]
    info = frequency_summary(counts, TinyTokenizer(), top_k=2)
    assert info["top1_id"] == 2
    assert info["top_tokens"][0]["token"] == "2"


def test_masked_metric_aggregation_weights_uneven_batches_by_masked_tokens():
    combined = aggregate_masked_metrics([
        {"loss": 2.0, "accuracy": .5, "masked_tokens": 2},
        {"loss": 1.0, "accuracy": 1.0, "masked_tokens": 6},
    ])
    assert combined == {"loss": 1.25, "accuracy": .875, "masked_tokens": 8}


def test_curriculum_parser_and_step_schedule_are_deterministic():
    schedule = parse_mask_schedule("0.15:2,0.30:3,0.50:4")
    assert schedule_rate(schedule, 1) == .15
    assert schedule_rate(schedule, 2) == .15
    assert schedule_rate(schedule, 3) == .30
    assert schedule_rate(schedule, 5) == .30
    assert schedule_rate(schedule, 6) == .50
    assert schedule_rate(schedule, 99) == .50
    with pytest.raises(ValueError):
        parse_mask_schedule("0.0:5")


def test_gate_streak_early_stop_helper_requires_consecutive_reports():
    streak = gate_streak(0, .96, .95)
    assert streak == 1
    assert gate_streak(streak, .94, .95) == 0
    assert gate_streak(2, .95, .95) == 3


def test_atomic_report_is_complete_json_for_resume_monitoring(tmp_path):
    report = tmp_path / "report.json"
    atomic_json_write(report, {"step": 7, "schedule": [[.5, 10]], "gate_streak": 2})
    assert json.loads(report.read_text()) == {"step": 7, "schedule": [[.5, 10]], "gate_streak": 2}
