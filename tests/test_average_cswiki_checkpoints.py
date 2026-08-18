import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from scripts.average_cswiki_checkpoints import merge_checkpoints


def write_checkpoint(path: Path, weight: float, moment: float, step: int = 1000) -> None:
    mx.savez(str(path), weight=mx.array([weight]), opt_weight_m=mx.array([moment]), opt_step=mx.array(step))
    path.with_suffix(".json").write_text(json.dumps({
        "step": step, "best_loss": 5.0, "history": [{"step": step, "loss": 5.0}],
        "cache_train_sha256": "a", "cache_val_sha256": "b", "route_pool": {},
        "strategy": "A-constant-50pct", "architecture": [25, 64, 256, 4, 256],
    }))


def test_merge_averages_float_state_and_preserves_step(tmp_path: Path):
    paths = [tmp_path / f"in-{index}.npz" for index in range(3)]
    for path, weight, moment in zip(paths, (1.0, 2.0, 3.0), (3.0, 6.0, 9.0)):
        write_checkpoint(path, weight, moment)
    output = tmp_path / "latest.npz"
    metadata = merge_checkpoints(paths, output)
    merged = mx.load(str(output))
    assert np.allclose(np.asarray(merged["weight"]), [2.0])
    assert np.allclose(np.asarray(merged["opt_weight_m"]), [6.0])
    assert int(np.asarray(merged["opt_step"])) == 1000
    assert metadata["step"] == 1000
