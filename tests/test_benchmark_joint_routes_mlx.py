import pytest

from scripts.benchmark_joint_routes_mlx import average_gradient_trees, checkpoint_contract


def test_average_gradient_trees_preserves_structure_and_averages_values():
    import mlx.core as mx

    result = average_gradient_trees([
        {"layer": {"weight": mx.array([1.0, 3.0])}},
        {"layer": {"weight": mx.array([3.0, 5.0])}},
    ])
    assert result["layer"]["weight"].tolist() == [2.0, 4.0]


def test_average_gradient_trees_rejects_empty_or_different_structures():
    import mlx.core as mx

    with pytest.raises(ValueError, match="at least one"):
        average_gradient_trees([])
    with pytest.raises(ValueError, match="identical structure"):
        average_gradient_trees([{"a": mx.array(1.0)}, {"b": mx.array(1.0)}])


def test_checkpoint_contract_requires_every_compatibility_field():
    metadata = {
        "cache_train_sha256": "train",
        "cache_val_sha256": "val",
        "route_pool": {},
        "strategy": "A-constant-50pct",
        "architecture": [25, 64, 256, 4, 256],
    }
    assert checkpoint_contract(metadata) == metadata
    del metadata["architecture"]
    with pytest.raises(ValueError, match="architecture"):
        checkpoint_contract(metadata)
