"""Tests for the offline layer-wise pilot helpers (no network access)."""
import json
import numpy as np
import mlx.core as mx
import mlx.optimizers as optim
import mlx.utils

from scripts.layerwise_pilot import select_cache, variant_schedule, save_checkpoint, load_checkpoint
from src.config import LayerwiseModelConfig
from src.layerwise_model import LayerwiseProgressiveLM


def test_variants_are_full_25_layer_controls():
    assert variant_schedule("fp32") == ["fp32"] * 25
    assert variant_schedule("progressive") == ["q1"] * 5 + ["q2"] * 5 + ["q4"] * 5 + ["q8"] * 5 + ["fp16"] * 5


def test_cache_selection_never_builds_or_uses_network(tmp_path):
    meta = {"seq_len": 256, "n_train_chunks": 3, "n_val_chunks": 1, "total_tokens": 1024,
            "tokenizer_sha256": "tokenizer", "train_sha256": "train", "val_sha256": "val"}
    (tmp_path / "meta_seq256_local.json").write_text(json.dumps(meta))
    np.save(tmp_path / "train_seq256_local.npy", np.ones((3, 256), dtype=np.int32))
    np.save(tmp_path / "val_seq256_local.npy", np.ones((1, 256), dtype=np.int32))
    train, val, selected, path = select_cache(tmp_path)
    assert train.shape == (3, 256) and val.shape == (1, 256)
    assert selected == meta and path.name == "meta_seq256_local.json"


def test_latest_checkpoint_restores_model_optimizer_and_step(tmp_path):
    cfg = LayerwiseModelConfig(vocab_size=16, d_model=8, n_heads=2, d_ff=16, n_layers=5, min_exit_layer=5, max_seq_len=8, layer_precisions=["fp32"]*5)
    model, opt = LayerwiseProgressiveLM(cfg), optim.Adam(learning_rate=1e-3)
    # Initialise optimizer state before serialisation.
    gradients = mlx.utils.tree_map(mx.zeros_like, model.parameters())
    opt.update(model, gradients)
    mx.eval(model.parameters(), opt.state)
    path = save_checkpoint(model, opt, tmp_path, "latest", 7, 1.25)
    restored, restored_opt = LayerwiseProgressiveLM(cfg), optim.Adam(learning_rate=1e-3)
    assert load_checkpoint(restored, restored_opt, path) == 7
    before = dict(mlx.utils.tree_flatten(model.parameters()))
    after = dict(mlx.utils.tree_flatten(restored.parameters()))
    np.testing.assert_allclose(np.array(before["token_embed.weight"]), np.array(after["token_embed.weight"]))
