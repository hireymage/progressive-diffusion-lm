import numpy as np

import json
from types import SimpleNamespace

import pytest
import scripts.layerwise_diagnostics as diagnostics

from scripts.layerwise_diagnostics import (
    aggregate_masked_metrics, atomic_json_write, count_tokens, deterministic_mask,
    frequency_summary, gate_streak, parse_mask_schedule, schedule_rate,
    build_exit_sweep_model, build_model, parser, run_exit_sweep, run_overfit,
    simulate_oracle_earliest_correct, simulate_stable_confidence_policy,
    summarize_routing, validate_mode_args, validate_resume_metadata,
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


def test_run_overfit_smoke_exercises_disk_estimate_and_writes_checkpoint(tmp_path):
    """A one-step CPU/MLX smoke test catches missing checkpoint utility imports."""
    args = SimpleNamespace(
        overfit_sequences=1, seed=11, vocab_size=16, d_model=8, d_ff=16, model_variant="fp32",
        n_heads=2, n_layers=1, lr=1e-3, auxiliary_loss="final-only",
        milestone_weights=(), checkpoint_dir=tmp_path / "checkpoints",
        output=tmp_path / "report.json", min_free_gb=0.0, resume=False,
        steps=1, batch_size=1, mask_schedule=((.5, 1),), report_every=1,
        gate_mask_rate=.5, gate_accuracy=2.0, gate_reports=1, eval_batch_size=1,
    )
    result = run_overfit(args, np.ones((1, 256), dtype=np.int32), TinyTokenizer())
    assert result["steps"] == 1
    assert (args.checkpoint_dir / "latest.npz").exists()
    assert json.loads(args.output.read_text())["status"] == "running"


def test_model_variant_builds_fp32_or_full_25_layer_progressive_schedule():
    fp32 = build_model(SimpleNamespace(model_variant="fp32", d_model=8, d_ff=16, n_heads=2, n_layers=3,
                                       auxiliary_loss="final-only", milestone_weights=()), 16)
    assert fp32.cfg.layer_precisions == ["fp32"] * 3
    progressive = build_model(SimpleNamespace(model_variant="progressive", d_model=8, d_ff=16, n_heads=2, n_layers=25,
                                              auxiliary_loss="final-only", milestone_weights=()), 16)
    assert progressive.cfg.layer_precisions == ["q1"] * 5 + ["q2"] * 5 + ["q4"] * 5 + ["q8"] * 5 + ["fp16"] * 5
    with pytest.raises(ValueError, match="requires --n-layers 25"):
        build_model(SimpleNamespace(model_variant="progressive", d_model=8, d_ff=16, n_heads=2, n_layers=24,
                                    auxiliary_loss="final-only", milestone_weights=()), 16)


def test_resume_metadata_rejects_model_variant_mismatch():
    args = SimpleNamespace(mask_schedule=((.5, 10),), model_variant="fp32")
    validate_resume_metadata({"schedule": [[.5, 10]], "model_variant": "fp32"}, args)
    with pytest.raises(ValueError, match="model variant"):
        validate_resume_metadata({"schedule": [[.5, 10]], "model_variant": "progressive"}, args)


def test_weighted_progressive_overfit_smoke_has_eligible_milestone_exits(tmp_path):
    args = SimpleNamespace(
        overfit_sequences=1, seed=12, vocab_size=16, d_model=8, d_ff=16,
        n_heads=2, n_layers=25, model_variant="progressive", lr=1e-3,
        auxiliary_loss="weighted-milestones", milestone_weights=((5, .1), (10, .2), (15, .3), (20, .4), (25, 1.0)),
        checkpoint_dir=tmp_path / "checkpoints", output=tmp_path / "report.json",
        min_free_gb=0.0, resume=False, steps=1, batch_size=1,
        mask_schedule=((.5, 1),), report_every=1, gate_mask_rate=.5,
        gate_accuracy=2.0, gate_reports=1, eval_batch_size=1,
    )
    result = run_overfit(args, np.ones((1, 256), dtype=np.int32), TinyTokenizer())
    assert result["architecture"]["layer_precisions"][:5] == ["q1"] * 5
    assert result["architecture"]["n_layers"] == 25
    assert (args.checkpoint_dir / "latest.npz").exists()


def test_exit_sweep_uses_one_deterministic_mask_for_all_exits_and_default_costs(monkeypatch, tmp_path):
    args = SimpleNamespace(
        eval_sequences=2, vocab_size=16, d_model=8, d_ff=16, n_heads=2, n_layers=25,
        model_variant="progressive", auxiliary_loss="final-only", seed=21,
        gate_mask_rate=.5, eval_batch_size=1, checkpoint=tmp_path / "final-only.npz",
        milestone_weights=((5, .1), (10, .2), (15, .3), (20, .4), (25, 1.0)),
    )
    masks, exits = [], []
    monkeypatch.setattr(diagnostics, "load_weights_only", lambda model, checkpoint: None)
    def fake_evaluate(model, targets, mask, batch_size, capture_reconstructions=0, exit_layer=None):
        masks.append(mask.copy())
        exits.append(exit_layer)
        return ({"loss": float(exit_layer), "accuracy": .5, "masked_tokens": int(mask.sum())}, [])
    monkeypatch.setattr(diagnostics, "evaluate_in_chunks", fake_evaluate)
    result = run_exit_sweep(args, np.ones((2, 256), dtype=np.int32))
    assert exits == [5, 10, 15, 20, 25]
    assert all(np.array_equal(masks[0], mask) for mask in masks[1:])
    assert result["masking"] == {"rate": .5, "seed": 900021}
    assert [row["proxy_cost"] for row in result["rows"]] == [5, 15, 35, 75, 155]
    assert [row["precision"] for row in result["rows"]] == ["q1", "q2", "q4", "q8", "fp16"]


def test_exit_sweep_builder_enables_milestones_even_for_final_only_checkpoints():
    args = SimpleNamespace(model_variant="progressive", n_layers=25, vocab_size=16,
        d_model=8, d_ff=16, n_heads=2, auxiliary_loss="final-only",
        milestone_weights=((5, .1), (10, .2), (15, .3), (20, .4), (25, 1.0)))
    assert build_exit_sweep_model(args, 16).cfg.min_exit_layer == 5


def test_exit_sweep_parser_and_checkpoint_requirement():
    assert "exit-sweep" in parser()._option_string_actions["--mode"].choices
    with pytest.raises(ValueError, match="--checkpoint is required for exit-sweep"):
        validate_mode_args(SimpleNamespace(mode="exit-sweep", checkpoint=None))


def test_policy_routing_requires_stability_and_falls_back_to_final_layer():
    predictions = [np.array([1, 2, 3, 4]), np.array([1, 8, 3, 9]), np.array([1, 8, 7, 9])]
    confidences = [np.array([.99, .99, .99, .99]), np.array([.9, .9, .6, .9]), np.array([.9, .9, .9, .4])]
    result = simulate_stable_confidence_policy(predictions, confidences, np.array([1, 8, 7, 4]),
                                                (5, 10, 25), (5.0, 15.0, 155.0), .8)
    # Token 0 exits at 10; token 1 at 25; 2 is unstable until 25; 3 falls back.
    assert result["exit_distribution"] == [{"layer": 5, "count": 0}, {"layer": 10, "count": 1}, {"layer": 25, "count": 3}]
    assert result["accuracy"] == .75
    assert result["mean_proxy_cost"] == 120.0
    assert result["savings_vs_full"] == pytest.approx(1 - 120 / 155)
    assert sum(row["count"] for row in result["exit_distribution"]) == result["masked_tokens"]


def test_policy_oracle_is_explicitly_non_deployable_and_uses_earliest_correct():
    predictions = [np.array([1, 2]), np.array([3, 4]), np.array([3, 5])]
    result = simulate_oracle_earliest_correct(predictions, np.array([3, 5]), (5, 10, 25), (5., 15., 155.))
    assert result["label"] == "ground-truth oracle earliest-correct upper bound (non-deployable)"
    assert result["accuracy"] == 1.0
    assert result["exit_distribution"] == [{"layer": 5, "count": 0}, {"layer": 10, "count": 1}, {"layer": 25, "count": 1}]


def test_policy_sweep_parser_and_checkpoint_requirement():
    assert "policy-sweep" in parser()._option_string_actions["--mode"].choices
    with pytest.raises(ValueError, match="--checkpoint is required for policy-sweep"):
        validate_mode_args(SimpleNamespace(mode="policy-sweep", checkpoint=None))
