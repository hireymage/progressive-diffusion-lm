"""Offline contract tests for the real Czech flexible trainer."""
import hashlib
import json

import numpy as np
import pytest

from scripts.train_cswiki_flexible import (
    N_LAYERS, build_model, corrupt_50, ensure_outside_icloud, fixed_batch,
    load_checkpoint, route_pool, select_verified_cswiki_cache, run,
)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_cache(root, *, czech=True, valid_hashes=True):
    suffix = "fixture"
    train, val = root / f"train_seq256_{suffix}.npy", root / f"val_seq256_{suffix}.npy"
    np.save(train, np.ones((2, 256), dtype=np.int32)); np.save(val, np.ones((1, 256), dtype=np.int32))
    tokenizer = root / "cs-tokenizer"; tokenizer.mkdir(); token_file = tokenizer / "tokenizer.json"; token_file.write_text("{}")
    meta = {"format": "cswiki-cache-v1" if czech else "english-cache-v1", "seq_len": 256,
            "source": {"dump_filename": "cswiki-20260801-pages-articles.xml.bz2" if czech else "enwiki-latest.xml",
                       "sha1": "a" * 40},
            "tokenizer": str(tokenizer), "tokenizer_sha256": _hash(token_file), "n_train_chunks": 2, "n_val_chunks": 1,
            "total_tokens": 768, "train_sha256": _hash(train), "val_sha256": _hash(val)}
    if not valid_hashes: meta["train_sha256"] = "0" * 64
    (root / f"meta_seq256_{suffix}.json").write_text(json.dumps(meta))


def test_cswiki_cache_loader_rehashes_and_rejects_non_czech_or_tampered(tmp_path):
    _write_cache(tmp_path)
    train, val, meta, path = select_verified_cswiki_cache(tmp_path)
    assert train.shape == (2, 256) and val.shape == (1, 256)
    assert meta["format"] == "cswiki-cache-v1" and path.name.startswith("meta_seq256")
    other = tmp_path / "english"; other.mkdir(); _write_cache(other, czech=False)
    with pytest.raises(FileNotFoundError): select_verified_cswiki_cache(other)
    bad = tmp_path / "bad"; bad.mkdir(); _write_cache(bad, valid_hashes=False)
    with pytest.raises(ValueError, match="checksum"): select_verified_cswiki_cache(bad)


def test_strategy_a_batches_and_masks_are_deterministic_and_50_percent_contract():
    data = np.arange(4 * 256, dtype=np.int32).reshape(4, 256)
    first = fixed_batch(data, 2, 7)
    np.testing.assert_array_equal(first, fixed_batch(data, 2, 7))
    x, targets, mask = corrupt_50(first, 999, 17)
    x2, targets2, mask2 = corrupt_50(first, 999, 17)
    np.testing.assert_array_equal(np.array(x), np.array(x2))
    np.testing.assert_array_equal(np.array(targets), np.array(targets2))
    np.testing.assert_array_equal(np.array(mask), np.array(mask2))
    assert np.array(mask)[:, 0].all() and N_LAYERS == 25


def test_explicit_outputs_reject_icloud_locations(tmp_path):
    ensure_outside_icloud(tmp_path / "result.json")
    with pytest.raises(ValueError, match="outside iCloud"):
        ensure_outside_icloud("/Users/a/Library/Mobile Documents/iCloud~md~obsidian/out.json")


def test_new_run_refuses_historical_report_before_loading_cache(tmp_path):
    report = tmp_path / "report.json"
    report.write_text("{}")
    args = type("Args", (), {"steps": 1, "batch_size": 1, "eval_steps": 1,
        "eval_every": 1, "resume": False, "output": report,
        "checkpoint_dir": tmp_path / "checkpoints", "cache_dir": tmp_path / "missing"})()
    with pytest.raises(FileExistsError, match="historical report"):
        run(args)


def test_real_trainer_hard_cap_allows_100000_and_rejects_over_one_million(tmp_path):
    accepted = type("Args", (), {"steps": 100000, "output": tmp_path / "report.json"})()
    rejected = type("Args", (), {"steps": 1_000_001, "output": tmp_path / "report.json"})()
    # 100k clears the cap check; the following missing batch field proves it
    # reached later validation without touching cache/model state.
    with pytest.raises(AttributeError, match="batch_size"):
        run(accepted)
    with pytest.raises(ValueError, match="1000000"):
        run(rejected)


def test_custom_128_512_architecture_and_routes_use_requested_values():
    model = build_model(32, d_model=128, d_ff=512, n_heads=8,
                        n_layers=6, seq_len=128)
    assert (model.cfg.d_model, model.cfg.d_ff, model.cfg.n_heads) == (128, 512, 8)
    assert (model.cfg.n_layers, model.cfg.max_seq_len) == (6, 128)
    assert model.cfg.layer_precisions == ["q8"] * 6
    assert all(len(schedule) == 6 for schedule in route_pool(6).values())


def test_resume_rejects_architecturally_incompatible_checkpoint_before_weights(tmp_path):
    path = tmp_path / "latest.npz"
    path.with_suffix(".json").write_text(json.dumps({"architecture": [25, 64, 256, 4, 256]}))
    with pytest.raises(ValueError, match="architecture"):
        load_checkpoint(None, None, path, {"architecture": [25, 128, 512, 4, 256]})


def test_reset_optimizer_requires_resume_before_loading_cache(tmp_path):
    args = type("Args", (), {"steps": 1, "batch_size": 1, "eval_steps": 1,
        "eval_every": 1, "resume": False, "reset_optimizer": True, "grad_clip": 1.0,
        "output": tmp_path / "report.json", "checkpoint_dir": tmp_path / "checkpoints",
        "cache_dir": tmp_path / "missing"})()
    with pytest.raises(ValueError, match="reset-optimizer requires"):
        run(args)


def test_negative_gradient_clip_is_rejected_before_loading_cache(tmp_path):
    args = type("Args", (), {"steps": 1, "batch_size": 1, "eval_steps": 1,
        "eval_every": 1, "resume": True, "reset_optimizer": False, "grad_clip": -1.0,
        "output": tmp_path / "report.json", "checkpoint_dir": tmp_path / "checkpoints",
        "cache_dir": tmp_path / "missing"})()
    with pytest.raises(ValueError, match="grad-clip"):
        run(args)
