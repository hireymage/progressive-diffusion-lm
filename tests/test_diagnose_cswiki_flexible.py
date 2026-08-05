from types import SimpleNamespace

import numpy as np
import pytest

import scripts.diagnose_cswiki_flexible as diag
from scripts.layerwise_diagnostics import flexible_route_pool


def test_diagnostic_is_validation_only_and_reports_worst_route(monkeypatch, tmp_path):
    val = np.ones((3, 256), dtype=np.int32)
    meta = {"tokenizer": str(tmp_path / "tok")}
    masks, schedules, loads = [], [], []
    monkeypatch.setattr(diag, "select_verified_cswiki_cache", lambda _: (np.zeros_like(val), val, meta, tmp_path / "meta.json"))
    monkeypatch.setattr(diag, "load_tokenizer", lambda _: SimpleNamespace(get_vocab_size=lambda: 16, decode=lambda ids: "/".join(map(str, ids))))
    class Model:
        cfg = SimpleNamespace(mask_token_id=lambda: 15)
        def eval(self): pass
        def set_layer_precisions(self, schedule): schedules.append(schedule)
    monkeypatch.setattr(diag, "build_model", lambda _: Model())
    monkeypatch.setattr(diag, "load_weights_only", lambda *_: loads.append(True))
    def fake_eval(_model, targets, mask, _batch, capture_reconstructions=0, exit_layer=None):
        masks.append(mask.copy())
        return {"loss": float(exit_layer or 25), "accuracy": .5, "masked_tokens": int(mask.sum())}, [targets[0]] * capture_reconstructions
    monkeypatch.setattr(diag, "evaluate_in_chunks", fake_eval)
    monkeypatch.setattr(diag, "refinement_steps", lambda *_: ["česky"] * 4)
    a = SimpleNamespace(cache_dir=tmp_path, checkpoint=tmp_path / "best.npz", output=tmp_path / "out.json",
                        eval_sequences=2, eval_batch_size=1, examples=1, refinement_steps=4, seed=7)
    result = diag.run(a)
    assert loads == [True] and schedules == list(flexible_route_pool(25).values())
    assert all(np.array_equal(masks[0], mask) for mask in masks[1:])
    assert result["split"] == "held-out validation only"
    assert set(result["per_route"]) == set(flexible_route_pool(25))
    assert [row["layer"] for row in result["worst_route_by_exit"]] == [5, 10, 15, 20, 25]
    assert result["masking"]["seed"] == 900007


def test_diagnostic_refuses_overwrite_and_invalid_refinement_count(tmp_path):
    output = tmp_path / "historical.json"; output.write_text("{}")
    a = SimpleNamespace(output=output, eval_sequences=1, examples=1, refinement_steps=4)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        diag.run(a)
    a.output = tmp_path / "new.json"; a.refinement_steps = 3
    with pytest.raises(ValueError, match="exactly 4"):
        diag.run(a)


def test_prompt_continuation_never_changes_encoded_czech_prompt_prefix():
    class Tokenizer:
        def encode(self, text):
            assert text == "kočka leze dírou."
            return SimpleNamespace(ids=[3, 4, 5])
        def decode(self, ids): return "/".join(map(str, ids))
        def token_to_id(self, token): return 2 if token == "[MASK]" else None
    class Model:
        cfg = SimpleNamespace(mask_token_id=lambda: 15, max_seq_len=256)
        def __call__(self, tokens, exit_layer):
            # Highest-probability id deliberately differs from the prompt ids.
            shape = tuple(tokens.shape) + (20,)
            logits = np.zeros(shape, dtype=np.float32); logits[..., 9] = 10
            return diag.mx.array(logits)
    result = diag.prompt_continuation(Model(), Tokenizer(), max_new_tokens=24)
    assert result["passes"] == 4 and len(result["refinements"]) == 4
    assert result["continuation_token_ids"][:3] == [3, 4, 5]


def test_refinement_fills_remaining_positions_over_exactly_four_passes(monkeypatch):
    remaining_counts = []
    class Tokenizer:
        def token_to_id(self, token): return 2 if token == "[MASK]" else None
        def decode(self, ids): return ",".join(map(str, ids))
    class Model:
        cfg = SimpleNamespace(mask_token_id=lambda: 9)
        def __call__(self, current, exit_layer):
            current_np = np.asarray(current)
            remaining_counts.append(int(np.sum(current_np == 9)))
            batch, length = current_np.shape
            logits = np.zeros((batch, length, 9), dtype=np.float32)
            for position in range(length): logits[:, position, position % 9] = position + 1
            return diag.mx.array(logits)
    rows = diag.refinement_steps(Model(), Tokenizer(), np.zeros((1, 8), dtype=np.int32), 4)
    assert len(rows) == 4
    assert remaining_counts == [8, 6, 4, 2]
