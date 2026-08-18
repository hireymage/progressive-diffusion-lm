import csv
import hashlib
import json

import numpy as np
import pytest

from scripts.oracle_m0 import (
    MaskedFixture,
    analyze_fixture_stream,
    analyze_precision_logits,
    load_validation_array,
    make_fixtures,
    run_provenance,
    sha256_file,
    write_artifacts,
)


def _logits_for_predictions(predictions, vocab=4, batch_size=1):
    logits = np.full((batch_size, len(predictions), vocab), -4.0)
    for batch_index in range(batch_size):
        for position, prediction in enumerate(predictions):
            logits[batch_index, position, prediction] = 4.0
            logits[batch_index, position, (prediction + 1) % vocab] = 2.0
    return logits


def test_m0_oracle_counts_transitions_and_selects_lowest_correct_precision():
    fixture = MaskedFixture(
        targets=np.array([[0, 1, 2]], dtype=np.int32),
        inputs=np.array([[4, 4, 4]], dtype=np.int32),
        mask=np.array([[True, True, True]]),
        mask_rates=np.array([0.5], dtype=np.float32),
    )
    logits = {
        1: [_logits_for_predictions([1, 1, 2])],  # wrong, right, right
        2: [_logits_for_predictions([0, 2, 2])],  # corrected, regressed, right
        16: [_logits_for_predictions([0, 1, 3])], # right, corrected, regressed
    }
    summary, rows = analyze_precision_logits(logits, [fixture], bits=[1, 2, 16])
    first = summary["transitions"]["q1_to_q2"]
    assert first["corrected"] == 1
    assert first["regressed"] == 1
    assert first["correctness_unchanged"] == 1
    assert [row["oracle_bits"] for row in rows] == [2, 1, 1]
    assert summary["oracle"]["quality_masked_accuracy"] == 1.0
    assert summary["oracle"]["mean_terminal_selected_bits"] == 4 / 3
    assert summary["oracle"]["mean_cumulative_proxy_bits"] == 5 / 3
    assert summary["oracle"]["always_full_ladder_proxy_bits_per_token"] == 35
    assert summary["oracle"]["cumulative_proxy_cost_vs_full_ladder"] == (5 / 3) / 35
    assert summary["oracle"]["single_fp32_proxy_bits_per_token"] == 32
    assert summary["oracle"]["cumulative_proxy_cost_vs_single_fp32"] == (5 / 3) / 32
    assert "not FP16" in summary["oracle"]["note"]
    assert "mean_entropy_when_corrected" in summary["transitions"]["q1_to_q2"]
    assert "q1_to_q2_signal_entropy" in rows[0]


def test_m0_proxy_costs_distinguish_internal_fp32_identity_from_fp16_target():
    fixture = MaskedFixture(
        targets=np.array([[0]], dtype=np.int32), inputs=np.array([[4]], dtype=np.int32),
        mask=np.array([[True]]), mask_rates=np.array([0.5], dtype=np.float32),
    )
    # Q4 is the first correct result, therefore the sequential cost is 1+2+4.
    logits = {
        1: [_logits_for_predictions([1])], 2: [_logits_for_predictions([1])],
        4: [_logits_for_predictions([0])], 8: [_logits_for_predictions([0])],
        16: [_logits_for_predictions([0])],
    }
    summary, rows = analyze_precision_logits(logits, [fixture])
    assert rows[0]["oracle_bits"] == 4
    assert rows[0]["oracle_cumulative_proxy_bits"] == 7
    assert summary["precision_proxy_costs"] == {"1": 1, "2": 2, "4": 4, "8": 8, "16": 32}
    assert summary["oracle"]["always_full_ladder_proxy_bits_per_token"] == 47
    assert summary["oracle"]["single_fp32_proxy_bits_per_token"] == 32


def test_m0_fixtures_are_deterministic_and_artifacts_avoid_logits(tmp_path):
    data = np.arange(24, dtype=np.int32).reshape(6, 4) % 4
    left = make_fixtures(data, batch_size=2, n_batches=2, mask_token_id=4, seed=7)
    right = make_fixtures(data, batch_size=2, n_batches=2, mask_token_id=4, seed=7)
    assert all(np.array_equal(a.inputs, b.inputs) and np.array_equal(a.mask, b.mask)
               for a, b in zip(left, right))
    logits = {bit: [_logits_for_predictions([0, 1, 2, 3], batch_size=2), _logits_for_predictions([0, 1, 2, 3], batch_size=2)]
              for bit in [1, 2, 16]}
    summary, rows = analyze_precision_logits(logits, left, bits=[1, 2, 16])
    write_artifacts(tmp_path, summary, rows)
    assert json.loads((tmp_path / "summary.json").read_text())["oracle"]["cumulative_proxy_cost_vs_single_fp32"] > 0
    csv_text = (tmp_path / "per_token.csv").read_text()
    assert "logit" not in csv_text.lower()
    assert "\r\n" not in csv_text
    assert list(csv.DictReader(csv_text.splitlines()))


def test_m0_rejects_missing_or_misaligned_logits_batches():
    fixture = MaskedFixture(
        targets=np.array([[0, 1]], dtype=np.int32),
        inputs=np.array([[4, 4]], dtype=np.int32),
        mask=np.array([[True, True]]),
        mask_rates=np.array([0.5], dtype=np.float32),
    )
    valid = _logits_for_predictions([0, 1])
    with pytest.raises(ValueError, match="logits batches"):
        analyze_precision_logits({1: [], 16: [valid]}, [fixture], bits=[1, 16])
    with pytest.raises(ValueError, match="does not match expected"):
        analyze_precision_logits({1: [valid[:, :1]], 16: [valid]}, [fixture], bits=[1, 16])


def test_m0_streaming_runner_aggregates_multiple_fixtures_without_logits_collection():
    fixtures = [
        MaskedFixture(np.array([[0, 1]], dtype=np.int32), np.array([[4, 4]], dtype=np.int32),
                      np.array([[True, True]]), np.array([0.5], dtype=np.float32)),
        MaskedFixture(np.array([[2, 3]], dtype=np.int32), np.array([[4, 4]], dtype=np.int32),
                      np.array([[True, True]]), np.array([0.5], dtype=np.float32)),
    ]
    precomputed = {
        1: [_logits_for_predictions([1, 1]), _logits_for_predictions([2, 0])],
        16: [_logits_for_predictions([0, 1]), _logits_for_predictions([2, 3])],
    }
    calls = []

    def runner(bits, fixture):
        fixture_index = next(index for index, known in enumerate(fixtures) if known is fixture)
        calls.append((bits, fixture_index))
        # The stream API receives and returns one batch, not the full matrix.
        return precomputed[bits][fixture_index].copy()

    streamed_summary, streamed_rows = analyze_fixture_stream(fixtures, [1, 16], runner)
    full_summary, full_rows = analyze_precision_logits(precomputed, fixtures, bits=[1, 16])
    assert calls == [(1, 0), (16, 0), (1, 1), (16, 1)]
    assert streamed_summary == full_summary
    assert streamed_rows == full_rows


def test_m0_loads_explicit_validation_array_and_rejects_invalid_inputs(tmp_path):
    valid = tmp_path / "validation.npy"
    np.save(valid, np.array([[0, 1, 2], [3, 0, 1]], dtype=np.int64))
    loaded = load_validation_array(valid, seq_len=3, vocab_size=4)
    assert loaded.dtype == np.int32
    assert np.array_equal(loaded, np.load(valid))

    wrong_shape = tmp_path / "wrong_shape.npy"
    np.save(wrong_shape, np.array([0, 1, 2], dtype=np.int32))
    with pytest.raises(ValueError, match="rank 2"):
        load_validation_array(wrong_shape, seq_len=3, vocab_size=4)
    wrong_length = tmp_path / "wrong_length.npy"
    np.save(wrong_length, np.array([[0, 1]], dtype=np.int32))
    with pytest.raises(ValueError, match="sequence length"):
        load_validation_array(wrong_length, seq_len=3, vocab_size=4)
    out_of_range = tmp_path / "out_of_range.npy"
    np.save(out_of_range, np.array([[0, 4, 1]], dtype=np.int32))
    with pytest.raises(ValueError, match=r"\[0, 4\)"):
        load_validation_array(out_of_range, seq_len=3, vocab_size=4)


def test_m0_provenance_hashes_inputs_once_and_has_utc_metadata(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.npz"
    validation = tmp_path / "validation.npy"
    checkpoint.write_bytes(b"checkpoint")
    validation.write_bytes(b"validation")
    assert sha256_file(checkpoint) == hashlib.sha256(b"checkpoint").hexdigest()
    monkeypatch.setattr("scripts.oracle_m0.current_git_commit", lambda: "abc123")
    metadata = run_provenance(checkpoint, validation)
    assert metadata["git_commit"] == "abc123"
    assert metadata["checkpoint_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    assert metadata["validation_data_sha256"] == hashlib.sha256(b"validation").hexdigest()
    assert metadata["utc_timestamp"].endswith("+00:00")
    assert metadata["hostname"]
    assert metadata["platform"]
