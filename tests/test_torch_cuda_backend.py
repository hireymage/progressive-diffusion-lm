import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

torch = pytest.importorskip("torch")

from src.config import LayerwiseModelConfig
from src.layerwise_model import LayerwiseProgressiveLM
from src.torch_layerwise_model import quantize_weight
from src.torch_mlx_checkpoint import MLXCompatibleAdamW, convert_mlx_checkpoint


def test_torch_q2_and_q8_match_mlx_levels():
    from src.quantization import quantize_weights
    values = np.array([[-3.1, -1.9, -0.1, 0.0, 0.2, 2.2, 3.0]], dtype=np.float32)
    for bits in (2, 8):
        expected = np.asarray(quantize_weights(mx.array(values), bits))
        actual = quantize_weight(torch.from_numpy(values), bits).numpy()
        assert np.allclose(actual, expected, atol=1e-7)


def make_mlx_checkpoint(path: Path):
    cfg = LayerwiseModelConfig(vocab_size=16, d_model=8, d_ff=16, n_heads=2,
                               n_layers=2, max_seq_len=8, min_exit_layer=1,
                               layer_precisions=["q8", "q8"])
    mx.random.seed(7)
    model = LayerwiseProgressiveLM(cfg)
    weights = dict(tree_flatten(model.parameters()))
    payload = dict(weights)
    for name, value in weights.items():
        payload[f"opt_{name}.m"] = mx.zeros_like(value)
        payload[f"opt_{name}.v"] = mx.zeros_like(value)
    payload["opt_step"] = mx.array(123, dtype=mx.uint64)
    mx.savez(str(path), **payload)
    path.with_suffix(".json").write_text(json.dumps({
        "step": 123, "best_loss": 5.0, "history": [],
        "architecture": [2, 8, 16, 2, 8],
    }))
    return model


def test_mlx_conversion_preserves_weights_optimizer_and_q8_logits(tmp_path: Path):
    path = tmp_path / "latest.npz"
    mlx_model = make_mlx_checkpoint(path)
    torch_model, optimizer, metadata = convert_mlx_checkpoint(path, torch.device("cpu"))
    assert metadata["step"] == 123
    for parameter in torch_model.parameters():
        assert optimizer.state[parameter]["step"] == 123
        assert torch.count_nonzero(optimizer.state[parameter]["exp_avg"]) == 0
    tokens_np = np.array([[1, 2, 16, 4, 5, 6, 7, 8]], dtype=np.int32)
    mlx_logits = np.asarray(mlx_model(mx.array(tokens_np), exit_layer=2))
    torch_model.set_layer_precisions(["q8", "q8"])
    with torch.no_grad():
        torch_logits = torch_model(torch.from_numpy(tokens_np.astype(np.int64)), exit_layer=2).numpy()
    assert np.allclose(torch_logits, mlx_logits, atol=5e-4, rtol=5e-4)


def test_mlx_compatible_adamw_has_no_bias_correction():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = MLXCompatibleAdamW([parameter], lr=0.1, weight_decay=0.01)
    parameter.grad = torch.tensor([2.0])
    optimizer.step()
    m = 0.2
    v = 0.004
    expected = 1.0 * (1.0 - 0.1 * 0.01) - 0.1 * m / (v ** 0.5 + 1e-8)
    assert parameter.item() == pytest.approx(expected, abs=1e-6)
