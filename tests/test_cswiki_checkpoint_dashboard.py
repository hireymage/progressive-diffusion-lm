import json
from pathlib import Path

import scripts.cswiki_checkpoint_dashboard as dash


def _write_batch(batch_dir: Path) -> None:
    batch_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "complete",
        "mode": "batch-prompt-continuation",
        "routes": ["q8_only", "q2_q8_fp16"],
        "prompt_count": 2,
    }
    (batch_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    rows = [
        {
            "model": "m1-512",
            "checkpoint_kind": "step_0052000",
            "checkpoint_step": 52000,
            "route": "q8_only",
            "prompt_index": 0,
            "prompt": "Praha je hlavní",
            "final_text": "Praha je hlavní město.",
            "generation_state": {"stop_reason": "eos"},
            "exit_state": {"token_count": 1, "early_exit_token_ratio": 0.5, "mean_exit_layer": 12.0, "mean_layers_saved": 8.0},
            "architecture": {"n_layers": 25, "d_model": 64, "d_ff": 256, "n_heads": 4, "seq_len": 256},
            "loss": 6.54,
            "accuracy": 0.0821,
        },
        {
            "model": "m1-512",
            "checkpoint_kind": "step_0052000",
            "checkpoint_step": 52000,
            "route": "q2_q8_fp16",
            "prompt_index": 1,
            "prompt": "Kočka leze dírou",
            "final_text": "Kočka leze dírou.",
            "generation_state": {"stop_reason": "max_new_tokens"},
            "exit_state": {"token_count": 2, "early_exit_token_ratio": 0.0, "mean_exit_layer": 25.0, "mean_layers_saved": 0.0},
            "architecture": {"n_layers": 25, "d_model": 64, "d_ff": 256, "n_heads": 4, "seq_len": 256},
            "loss": 6.50,
            "accuracy": 0.0912,
        },
    ]
    with (batch_dir / "generations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_dashboard_builds_html_from_watch_summary(tmp_path):
    watch_summary = tmp_path / "m1-512-watch-summary.json"
    batch_dir = tmp_path / "m1-512-latest-best-123"
    _write_batch(batch_dir)
    watch_summary.write_text(json.dumps({
        "status": "complete",
        "name": "m1-512",
        "checkpoint_dir": "/tmp/run/checkpoints",
        "batches": [{"batch_dir": str(batch_dir), "checkpoints": [["step_0052000", 52000]]}],
        "observed_checkpoints": [["step_0052000", 52000]],
    }, ensure_ascii=False, indent=2))
    output_dir = tmp_path / "dashboard"
    html_path = dash.build_dashboard(watch_summary, output_dir)
    text = html_path.read_text(encoding="utf-8")
    assert html_path.name == "index.html"
    assert "CSWiki checkpoint dashboard" in text
    assert "m1-512 · step_0052000 · krok 52,000" in text
    assert "Praha je hlavní" in text
    assert "Kočka leze dírou" in text
    assert "routes: q2_q8_fp16, q8_only" in text


def test_pick_free_port_returns_listenable_port():
    port = dash.pick_free_port("127.0.0.1", start=18000, limit=200)
    assert 18000 <= port < 18200


def test_resolve_watch_summary_prefers_latest_json_in_directory(tmp_path):
    old = tmp_path / "a-watch-summary.json"
    new = tmp_path / "b-watch-summary.json"
    old.write_text("{}")
    new.write_text("{}")
    assert dash.resolve_watch_summary(tmp_path) == new
