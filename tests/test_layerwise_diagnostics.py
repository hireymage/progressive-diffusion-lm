import numpy as np

from scripts.layerwise_diagnostics import aggregate_masked_metrics, deterministic_mask, frequency_summary, count_tokens


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
