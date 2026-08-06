import json
import math
import subprocess
from datetime import datetime

import scripts.monitor_cswiki_training as monitor


def _remote(report, latest, process="123 python scripts/train_cswiki_flexible.py"):
    return "__REPORT__\n" + json.dumps(report) + "\n__LATEST__\n" + json.dumps(latest) + "\n__PROCESS__\n" + process


def test_parse_and_render_czech_dashboard_with_worst_route_and_best_eta():
    history = [
        {"step": 40000, "elapsed_seconds": 7000, "loss": 5.4, "accuracy": .20, "perplexity": 220, "worst_route": "q2_q8_fp16"},
        {"step": 40500, "elapsed_seconds": 7090, "loss": 5.3, "accuracy": .21, "perplexity": 200, "worst_route": "q2_q8_fp16"},
    ]
    state = monitor.parse_remote_output(_remote({"status": "running", "history": history}, {"step": 40500, "best_loss": 5.3}))
    text = monitor.render_dashboard(state, 80000, datetime(2026, 8, 5, 12, 0, 0))
    assert "běží" in text and "40,500 / 80,000" in text
    assert "q2_q8_fp16" in text and "5.3000" in text and "ETA:" in text
    assert "Poznámka" not in text
    assert monitor.recent_speed(history) == 500 / 90


def test_live_process_target_replaces_old_default_and_restores_eta():
    history = [
        {"step": 80500, "elapsed_seconds": 80, "loss": 5.1},
        {"step": 81000, "elapsed_seconds": 160, "loss": 5.0},
    ]
    process = "123 python train_cswiki_flexible.py --steps 100000 --resume"
    state = monitor.parse_remote_output(_remote({"status": "running", "history": history}, {}, process))
    text = monitor.render_dashboard(state, now=datetime(2026, 8, 5, 12, 0, 0))
    assert "81,000 / 100,000" in text
    assert "ETA: 0:50:40" in text


def test_explicit_target_overrides_live_process_target():
    state = {"process": "python train.py --steps=100000"}
    assert monitor.inferred_target_steps(state) == 100000
    assert monitor.inferred_target_steps(state, 120000) == 120000


def test_temporary_bad_json_is_reported_not_misclassified_as_ssh_failure():
    state = monitor.parse_remote_output("__REPORT__\n{bad\n__LATEST__\n{}\n__PROCESS__\n")
    assert state["report"] is None and "dočasně nečitelný JSON" in state["error"]
    assert "Poznámka" in monitor.render_dashboard(state, 80000)


def test_fetch_uses_noninteractive_ssh_and_handles_remote_error(monkeypatch):
    def fake_run(command, **kwargs):
        assert command[:4] == ["ssh", "-o", "BatchMode=yes", "-o"]
        assert "ConnectTimeout=10" in command
        assert kwargs["timeout"] == 15
        return subprocess.CompletedProcess(command, 255, "", "Connection timed out")
    monkeypatch.setattr(monitor.subprocess, "run", fake_run)
    raw, error = monitor.fetch_remote("m4-air", "/tmp/run")
    assert raw == "" and "SSH nedostupné" in error


def test_remote_command_quotes_base_and_has_report_checkpoint_and_process_reads():
    command = monitor.remote_command("/tmp/a space")
    assert "report.json" in command and "checkpoints/latest.json" in command
    assert "pgrep" in command and "ps -ww" in command and "'" in command


def test_default_selection_contains_all_three_nodes_with_independent_runs():
    nodes = dict(monitor.selected_nodes(None, None))
    assert list(nodes) == ["m4-air", "m1-512", "m1-256"]
    assert "cswiki-real" in nodes["m4-air"]
    assert "cswiki-wide128" in nodes["m1-512"]
    assert monitor.DEFAULT_TARGETS["m1-512"] == 400000


def test_render_dashboard_uses_supplied_host_name():
    state = monitor.parse_remote_output(_remote({"status": "running", "history": []}, {}))
    text = monitor.render_dashboard(state, host="m1-512")
    assert "CSWiki flexible · m1-512" in text


def test_custom_remote_base_requires_exactly_one_host():
    assert monitor.selected_nodes(["custom"], "/tmp/run") == [("custom", "/tmp/run")]
    try:
        monitor.selected_nodes(["m4-air", "m1-512"], "/tmp/run")
    except ValueError as exc:
        assert "jedním --host" in str(exc)
    else:
        raise AssertionError("multiple hosts with one remote base must fail")


def test_nonfinite_metrics_are_shown_as_numerical_error_and_ignored_for_best():
    history = [
        {"step": 51000, "loss": 6.5, "accuracy": .08, "perplexity": 665.},
        {"step": 51500, "loss": math.nan, "accuracy": 0., "perplexity": math.nan},
    ]
    state = monitor.parse_remote_output(_remote({"status": "running", "history": history}, {}))
    text = monitor.render_dashboard(state, 400000, host="m1-512")
    assert "NUMERICKÁ CHYBA" in text
    assert "loss: CHYBA" in text
    assert "Best: loss 6.5000 @ krok 51000" in text
