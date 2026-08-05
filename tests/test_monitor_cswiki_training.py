import json
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
    text = monitor.render_dashboard(state, 60000, datetime(2026, 8, 5, 12, 0, 0))
    assert "běží" in text and "40,500 / 60,000" in text
    assert "q2_q8_fp16" in text and "5.3000" in text and "ETA:" in text
    assert "Poznámka" not in text
    assert monitor.recent_speed(history) == 500 / 90


def test_temporary_bad_json_is_reported_not_misclassified_as_ssh_failure():
    state = monitor.parse_remote_output("__REPORT__\n{bad\n__LATEST__\n{}\n__PROCESS__\n")
    assert state["report"] is None and "dočasně nečitelný JSON" in state["error"]
    assert "Poznámka" in monitor.render_dashboard(state, 60000)


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
    assert "pgrep" in command and "'" in command
