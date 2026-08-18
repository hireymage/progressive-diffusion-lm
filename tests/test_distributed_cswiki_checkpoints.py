import json
from pathlib import Path
from types import SimpleNamespace

import scripts.distributed_cswiki_checkpoints as dist


def _write_route_output(route_dir: Path, checkpoint_kind: str = "best") -> None:
    route_dir.mkdir(parents=True, exist_ok=True)
    (route_dir / "summary.json").write_text(json.dumps({
        "status": "complete",
        "mode": "batch-prompt-continuation",
        "routes": ["q8_only"],
        "prompt_count": 2,
        "checkpoints": [{"model": "m1-512", "checkpoint_kind": checkpoint_kind, "checkpoint_step": 52000, "rows": 2}],
        "rows": 2,
    }, ensure_ascii=False, indent=2))
    (route_dir / "generations.jsonl").write_text(
        "\n".join([
            json.dumps({
                "model": "m1-512",
                "checkpoint_kind": checkpoint_kind,
                "checkpoint_step": 52000,
                "route": "q8_only",
                "prompt_index": 0,
                "prompt": "Praha je hlavní",
                "final_text": "Praha je hlavní město.",
                "generation_state": {"stop_reason": "eos"},
                "exit_state": {"token_count": 1, "early_exit_token_ratio": 0.5, "mean_exit_layer": 12.0, "mean_layers_saved": 8.0},
                "loss": 6.54,
                "accuracy": 0.0821,
            }, ensure_ascii=False),
            json.dumps({
                "model": "m1-512",
                "checkpoint_kind": checkpoint_kind,
                "checkpoint_step": 52000,
                "route": "q8_only",
                "prompt_index": 1,
                "prompt": "Kočka leze dírou",
                "final_text": "Kočka leze dírou.",
                "generation_state": {"stop_reason": "max_new_tokens"},
                "exit_state": {"token_count": 2, "early_exit_token_ratio": 0.0, "mean_exit_layer": 25.0, "mean_layers_saved": 0.0},
                "loss": 6.50,
                "accuracy": 0.0912,
            }, ensure_ascii=False),
        ]) + "\n"
    )


def test_route_hosts_round_robin_uses_all_hosts():
    mapping = dist.route_hosts(["q8_only", "q8_fp16", "q2_q8_fp16"], ["m1-256", "m1-512"])
    assert mapping == {
        "q8_only": "m1-256",
        "q8_fp16": "m1-512",
        "q2_q8_fp16": "m1-256",
    }


def test_remote_command_uses_remote_python_not_local_interpreter(tmp_path):
    args = SimpleNamespace(
        cache_dir=Path("/cache"),
        models_json=Path("/models.json"),
        prompts=3,
        max_new_tokens=8,
        include_latest_best=True,
        measure_exits=False,
    )
    command = dist.remote_command(args, "q8_only", tmp_path, "/remote/python")
    assert "/remote/python" in command
    assert "batch_prompt_cswiki_checkpoints.py" in command
    assert "--routes q8_only" in command


def test_build_dashboard_focuses_on_current_output_dir(tmp_path):
    route_dir = tmp_path / "q8_only-m1-512-20260814-120000"
    _write_route_output(route_dir)
    html_path = dist.build_dashboard(tmp_path, tmp_path / "dashboard", {
        "status": "complete",
        "name": "distributed-eval",
        "checkpoint_dir": str(tmp_path),
        "batches": [{"batch_dir": str(route_dir), "checkpoints": [{"model": "m1-512"}]}],
        "observed_checkpoints": [["m1-512", 52000]],
        "route_results": [{"host": "m1-512", "route": "q8_only", "batch_dir": str(route_dir), "summary": json.loads((route_dir / "summary.json").read_text())}],
        "models_json": "/tmp/models.json",
        "routes": ["q8_only"],
        "hosts": ["m1-512"],
    }, [])
    text = html_path.read_text(encoding="utf-8")
    assert "CSWiki checkpoint dashboard" in text
    assert "distributed-eval" in text
    assert "q8_only-m1-512-20260814-120000" in text
    assert "Praha je hlavní" not in text

