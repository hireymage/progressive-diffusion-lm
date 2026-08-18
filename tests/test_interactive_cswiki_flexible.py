from types import SimpleNamespace

import numpy as np
import pytest

import scripts.interactive_cswiki_flexible as interactive
from scripts.layerwise_diagnostics import flexible_route_pool


def test_interactive_loads_once_reuses_route_and_stops_on_konec(monkeypatch, tmp_path):
    calls, schedules, prompts, outputs = [], [], [], []
    meta = {"tokenizer": str(tmp_path / "tokenizer")}
    monkeypatch.setattr(interactive, "select_verified_cswiki_cache",
                        lambda _path: (calls.append("cache") or np.zeros((1, 256)), np.ones((1, 256)), meta, tmp_path / "meta.json"))
    tokenizer = SimpleNamespace(get_vocab_size=lambda: 16)
    monkeypatch.setattr(interactive, "load_tokenizer", lambda _path: calls.append("tokenizer") or tokenizer)
    model = SimpleNamespace(eval=lambda: calls.append("eval"),
                            set_layer_precisions=lambda schedule: schedules.append(schedule))
    monkeypatch.setattr(interactive, "build_model", lambda _vocab: calls.append("model") or model)
    monkeypatch.setattr(interactive, "load_weights_only", lambda _model, _path: calls.append("checkpoint"))
    def fake_continuation(_model, _tokenizer, prompt, max_new_tokens, passes):
        prompts.append((prompt, max_new_tokens, passes))
        return {"refinements": ["a", "b", "c", f"výsledek:{prompt}"]}
    monkeypatch.setattr(interactive, "prompt_continuation", fake_continuation)
    entries = iter(["První český prompt,", "", "Druhý prompt,", "/konec", "nepřečíst"])
    args = SimpleNamespace(cache_dir=tmp_path, checkpoint=tmp_path / "best.npz",
                           route="q2_q8_fp16", max_new_tokens=7)
    interactive.run_interactive(args, input_fn=lambda _label: next(entries), output_fn=outputs.append)
    assert calls == ["cache", "tokenizer", "model", "checkpoint", "eval"]
    assert schedules == [flexible_route_pool(25)["q2_q8_fp16"]]
    assert prompts == [("První český prompt,", 7, 4), ("Druhý prompt,", 7, 4)]
    assert outputs[-2:] == ["výsledek:První český prompt,", "výsledek:Druhý prompt,"]


def test_interactive_rejects_nonpositive_generation_length_before_loading(monkeypatch, tmp_path):
    monkeypatch.setattr(interactive, "select_verified_cswiki_cache",
                        lambda _path: pytest.fail("cache must not load"))
    args = SimpleNamespace(cache_dir=tmp_path, checkpoint=tmp_path / "best.npz",
                           route="q8_only", max_new_tokens=0)
    with pytest.raises(ValueError, match="positive"):
        interactive.run_interactive(args)


def test_interactive_parser_exposes_all_flexible_routes():
    action = interactive.parser()._option_string_actions["--route"]
    assert tuple(action.choices) == tuple(flexible_route_pool(25))
