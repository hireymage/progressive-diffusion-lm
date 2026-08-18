#!/usr/bin/env python3
"""Lokální terminálový monitor pro běžící cswiki flexible trénink přes SSH."""
from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import PurePosixPath

REMOTE_ROOT = ("/Users/hozzy/Library/Application Support/ML-Experiments/"
               "progressive-diffusion-lm")
DEFAULT_REMOTE_BASE = f"{REMOTE_ROOT}/cswiki-real/run-20260804-a-40k"
DEFAULT_NODES = {
    "m4-air": DEFAULT_REMOTE_BASE,
    "m1-512": f"{REMOTE_ROOT}/cswiki-d64-m1-512/run-20260807-d64-800k",
    # Připraveno pro příští běh; chybějící report se zobrazí jako čekající stav.
    "m1-256": f"{REMOTE_ROOT}/cswiki-m1-256/run-current",
}
DEFAULT_SSH_TARGETS = {
    "m1-512": ["m1-512", "hozzy@10.68.119.206"],
}
DEFAULT_TARGETS = {"m4-air": 400000, "m1-512": 3000000, "m1-256": 400000}


def remote_command(remote_base: str) -> str:
    """Return one read-only POSIX command with unambiguous JSON delimiters."""
    base = shlex.quote(str(PurePosixPath(remote_base)))
    return (
        f"base={base}; "
        "printf '__REPORT__\\n'; cat \"$base/report.json\" 2>/dev/null || true; "
        "printf '\\n__LATEST__\\n'; cat \"$base/checkpoints/latest.json\" 2>/dev/null || true; "
        "printf '\\n__PROCESS__\\n'; "
        "for pid in $(pgrep -f '[t]rain_cswiki_flexible.py' 2>/dev/null); do "
        "ps -ww -p \"$pid\" -o pid=,command= 2>/dev/null; done; "
        "printf '\\n__LOGS__\\n'; "
        "ls -t \"$base\"/*.log 2>/dev/null | head -n 1 | while IFS= read -r log; do "
        "printf '--- %s ---\\n' \"$log\"; tail -n 12 \"$log\" 2>/dev/null; done"
    )


def fetch_remote(host: str, remote_base: str, *, timeout: int = 15) -> tuple[str, str]:
    """Fetch state without prompting for credentials; errors stay transient state."""
    targets = DEFAULT_SSH_TARGETS.get(host, [host])
    last_error = ""
    for target in targets:
        try:
            completed = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target, remote_command(remote_base)],
                capture_output=True, text=True, timeout=timeout, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            last_error = f"SSH nedostupné: {exc}"
            continue
        if completed.returncode == 0:
            return completed.stdout, ""
        last_error = f"SSH nedostupné: {(completed.stderr or 'neznámá chyba').strip()}"
    return "", last_error or "SSH nedostupné"


def _json_or_none(text: str) -> tuple[dict | None, str | None]:
    text = text.strip()
    if not text:
        return None, None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"dočasně nečitelný JSON: {exc.msg}"
    if isinstance(value, dict):
        return value, None
    return None, "JSON nemá objektový tvar"


def parse_remote_output(raw: str) -> dict:
    """Parse report/checkpoint/process sections; malformed current writes are nonfatal."""
    markers = ("__REPORT__\n", "__LATEST__\n", "__PROCESS__\n")
    if not all(marker in raw for marker in markers):
        return {"report": None, "latest": None, "process": "", "error": "neúplná odpověď SSH"}
    report_text = raw.split(markers[0], 1)[1].split(markers[1], 1)[0]
    latest_text = raw.split(markers[1], 1)[1].split(markers[2], 1)[0]
    tail = raw.split(markers[2], 1)[1]
    process, logs = (tail.split("\n__LOGS__\n", 1) + [""])[:2] if "\n__LOGS__\n" in tail else (tail, "")
    report, report_error = _json_or_none(report_text)
    latest, latest_error = _json_or_none(latest_text)
    return {"report": report, "latest": latest, "process": process.strip(), "logs": logs.strip(),
            "error": report_error or latest_error}


def log_failure(logs: str) -> str | None:
    """Return a short failure reason from recent training logs."""
    if "FloatingPointError" in logs or "non-finite training value" in logs:
        match = re.search(r"non-finite training value at step (\d+): ([^\n]+)", logs)
        return f"numerická chyba v tréninku @ krok {int(match.group(1)):,}: {match.group(2)}" if match else "numerická chyba v tréninku"
    if "Traceback (most recent call last)" in logs:
        return "proces skončil s Python tracebackem"
    return None


def history_from(state: dict) -> list[dict]:
    report, latest = state.get("report") or {}, state.get("latest") or {}
    # During a model-only recovery the old report can be ahead of the restored
    # healthy checkpoint until the first new evaluation is written. In that
    # window the checkpoint is the authoritative state.
    if "--reset-optimizer" in (state.get("process") or ""):
        history = latest.get("history") or report.get("history") or []
    else:
        history = report.get("history") or latest.get("history") or []
    return [row for row in history if isinstance(row, dict) and isinstance(row.get("step"), int)]


def inferred_target_steps(state: dict, override: int | None = None) -> int:
    """Prefer an explicit override, otherwise read --steps from the live trainer."""
    if override is not None:
        return override
    process = state.get("process") or ""
    match = re.search(r"(?:^|\s)--steps(?:=|\s+)(\d+)(?:\s|$)", process)
    if match:
        return int(match.group(1))
    report = state.get("report") or {}
    report_steps = report.get("steps")
    if isinstance(report_steps, int) and report_steps > 0:
        return report_steps
    return 80000


def progress(state: dict, target_steps: int | None) -> dict:
    history = history_from(state)
    latest = state.get("latest") or {}
    target_steps = inferred_target_steps(state, target_steps)
    current = history[-1]["step"] if history else int(latest.get("step", 0) or 0)
    best_row = min((row for row in history if isinstance(row.get("loss"), (int, float))
                    and math.isfinite(row["loss"])),
                   key=lambda row: row["loss"], default=None)
    return {"current": current, "target": target_steps, "percent": min(100., 100 * current / max(target_steps, 1)),
            "history": history, "best_loss": (best_row or {}).get("loss", latest.get("best_loss")),
            "best_step": (best_row or {}).get("step")}


def recent_speed(history: list[dict]) -> float | None:
    """Steps/s from recent same-resume history deltas; skips reset elapsed blocks."""
    rates = []
    recent = history[-6:]
    for older, newer in zip(recent, recent[1:]):
        ds = newer.get("step", 0) - older.get("step", 0)
        dt = newer.get("elapsed_seconds", 0) - older.get("elapsed_seconds", 0)
        if ds > 0 and dt > 0:
            rates.append(ds / dt)
    return sum(rates) / len(rates) if rates else None


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0: return "—"
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600); minutes, seconds = divmod(seconds, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def render_dashboard(state: dict, target_steps: int | None = None, now: datetime | None = None,
                     host: str = "m4-air") -> str:
    now = now or datetime.now()
    if state.get("ssh_error"):
        return f"CSWiki flexible · {host}\nSSH offline: {state['ssh_error']}\nKontrola: {now:%Y-%m-%d %H:%M:%S}"
    p = progress(state, target_steps); history = p["history"]
    latest_row = history[-1] if history else {}
    report = state.get("report") or {}
    running = bool(state.get("process"))
    failure = None if running else log_failure(state.get("logs") or "")
    nonfinite = any(isinstance(latest_row.get(key), (int, float))
                    and not math.isfinite(latest_row[key]) for key in ("loss", "accuracy", "perplexity"))
    status = ("SSH online · NUMERICKÁ CHYBA" if nonfinite or failure else
              "SSH online · běží" if running else
              ("SSH online · dokončeno" if report.get("status") == "complete"
               else "SSH online · plánováno / čeká"))
    speed = recent_speed(history)
    eta = (p["target"] - p["current"]) / speed if speed and p["current"] < p["target"] else 0 if p["current"] >= p["target"] else None
    loss = latest_row.get("loss"); acc = latest_row.get("accuracy"); ppl = latest_row.get("perplexity")
    def numeric(value, digits=4):
        if not isinstance(value, (int, float)):
            return "—"
        return f"{value:.{digits}f}" if math.isfinite(value) else "CHYBA"
    return "\n".join((
        f"CSWiki flexible · {host}",
        f"Stav: {status}  |  krok: {p['current']:,} / {p['target']:,} ({p['percent']:.1f} %)",
        f"Worst route: {latest_row.get('worst_route', '—')}  |  loss: {numeric(loss)}  |  accuracy: {numeric(acc * 100 if isinstance(acc, (int, float)) else None, 2)} %  |  ppl: {numeric(ppl, 2)}",
        f"Best: loss {numeric(p['best_loss'])} @ krok {p['best_step'] or '—'}",
        f"Blok: {format_duration(latest_row.get('elapsed_seconds'))}  |  rychlost: {numeric(speed, 3)} kroků/s  |  ETA: {format_duration(eta)}",
        f"Kontrola: {now:%Y-%m-%d %H:%M:%S}",
        (f"Poslední chyba: {failure}" if failure else ""),
        (f"Poznámka: {state['error']}" if state.get("error") else ""),
    )).rstrip()


def _format_metric_pair(row: dict | None) -> str:
    if not row:
        return "—"
    loss, accuracy = row.get("loss"), row.get("accuracy")
    if not isinstance(loss, (int, float)) or not math.isfinite(loss):
        return "CHYBA"
    if not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy):
        return f"{loss:.4f} / —"
    return f"{loss:.4f} / {accuracy * 100:.2f} %"


def _row_at_or_before(history: list[dict], step: int) -> dict | None:
    if not history or history[-1].get("step", 0) < step:
        return None
    rows = [row for row in history if row.get("step", 0) <= step]
    return rows[-1] if rows else None


def render_comparison(states: dict[str, dict], targets: dict[str, int | None], step_size: int = 10_000) -> str:
    """Render worst-route loss/accuracy comparison at regular step milestones."""
    if step_size < 1:
        raise ValueError("step_size must be positive")
    histories = {host: history_from(state) for host, state in states.items() if not state.get("ssh_error")}
    max_step = max((row["step"] for history in histories.values() for row in history), default=0)
    max_target = max((inferred_target_steps(states[host], targets.get(host))
                      for host in states if not states[host].get("ssh_error")), default=0)
    end_step = max(max_step, max_target)
    if end_step <= 0:
        return "Srovnání po 10k krocích\nzatím nejsou žádná data"
    hosts = list(states)
    header = "krok".ljust(9) + "  " + "  ".join(host.ljust(24) for host in hosts)
    lines = ["Srovnání po 10k krocích (loss / accuracy, worst-route)", header, "-" * len(header)]
    for step in range(step_size, end_step + step_size, step_size):
        cells = []
        for host in hosts:
            history = histories.get(host, [])
            cells.append(_format_metric_pair(_row_at_or_before(history, step)).ljust(24))
        lines.append(f"{step:,}".ljust(9) + "  " + "  ".join(cells))
    return "\n".join(lines)


def poll(host: str, remote_base: str) -> dict:
    raw, ssh_error = fetch_remote(host, remote_base)
    if ssh_error: return {"ssh_error": ssh_error}
    return parse_remote_output(raw)


def selected_nodes(hosts: list[str] | None, remote_base: str | None) -> list[tuple[str, str]]:
    """Resolve CLI selection while keeping the historical single-host options useful."""
    if not hosts:
        return list(DEFAULT_NODES.items())
    if remote_base is not None and len(hosts) != 1:
        raise ValueError("--remote-base lze použít jen s jedním --host")
    unknown = [host for host in hosts if host not in DEFAULT_NODES]
    if unknown and remote_base is None:
        raise ValueError(f"neznámý host bez --remote-base: {', '.join(unknown)}")
    return [(host, remote_base or DEFAULT_NODES[host]) for host in hosts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", action="append",
                        help="omezit výpis na host (lze zadat opakovaně); výchozí jsou všechny tři nody")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--remote-base", default=None,
                        help="vlastní adresář běhu; lze použít jen s jedním --host")
    parser.add_argument("--target-steps", type=int, default=None,
                        help="ruční cíl; bez něj se přečte --steps z běžícího trenéru")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 1 or (args.target_steps is not None and args.target_steps < 1):
        parser.error("interval a target-steps musí být kladné")
    try:
        nodes = selected_nodes(args.host, args.remote_base)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        while True:
            now = datetime.now()
            dashboards = []
            states = {}
            targets = {}
            for host, base in nodes:
                state = poll(host, base)
                states[host] = state
                # A live --steps argument is more accurate than the remembered
                # per-node target (for example during a short recovery run).
                target = args.target_steps or (None if state.get("process") else DEFAULT_TARGETS.get(host))
                targets[host] = target
                dashboards.append(render_dashboard(state, target, now, host))
            comparison = render_comparison(states, targets)
            print("\033[2J\033[H" + "\n\n".join(dashboards + [comparison]), flush=True)
            if args.once: return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor ukončen.")


if __name__ == "__main__":
    main()
