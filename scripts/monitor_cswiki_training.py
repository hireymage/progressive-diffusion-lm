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
    "m1-512": f"{REMOTE_ROOT}/cswiki-wide128/run-20260805-d128",
    # Připraveno pro příští běh; chybějící report se zobrazí jako čekající stav.
    "m1-256": f"{REMOTE_ROOT}/cswiki-real/run-20260804-a-40k",
}
DEFAULT_TARGETS = {"m4-air": 400000, "m1-512": 400000, "m1-256": 80000}


def remote_command(remote_base: str) -> str:
    """Return one read-only POSIX command with unambiguous JSON delimiters."""
    base = shlex.quote(str(PurePosixPath(remote_base)))
    return (
        f"base={base}; "
        "printf '__REPORT__\\n'; cat \"$base/report.json\" 2>/dev/null || true; "
        "printf '\\n__LATEST__\\n'; cat \"$base/checkpoints/latest.json\" 2>/dev/null || true; "
        "printf '\\n__PROCESS__\\n'; "
        "for pid in $(pgrep -f '[t]rain_cswiki_flexible.py' 2>/dev/null); do "
        "ps -ww -p \"$pid\" -o pid=,command= 2>/dev/null; done"
    )


def fetch_remote(host: str, remote_base: str, *, timeout: int = 15) -> tuple[str, str]:
    """Fetch state without prompting for credentials; errors stay transient state."""
    try:
        completed = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, remote_command(remote_base)],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "", f"SSH nedostupné: {exc}"
    if completed.returncode:
        return "", f"SSH nedostupné: {(completed.stderr or 'neznámá chyba').strip()}"
    return completed.stdout, ""


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
    process = raw.split(markers[2], 1)[1].strip()
    report, report_error = _json_or_none(report_text)
    latest, latest_error = _json_or_none(latest_text)
    return {"report": report, "latest": latest, "process": process,
            "error": report_error or latest_error}


def history_from(state: dict) -> list[dict]:
    report, latest = state.get("report") or {}, state.get("latest") or {}
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
    nonfinite = any(isinstance(latest_row.get(key), (int, float))
                    and not math.isfinite(latest_row[key]) for key in ("loss", "accuracy", "perplexity"))
    status = ("SSH online · NUMERICKÁ CHYBA" if nonfinite else
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
        (f"Poznámka: {state['error']}" if state.get("error") else ""),
    )).rstrip()


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
            dashboards = [render_dashboard(poll(host, base),
                                           args.target_steps or DEFAULT_TARGETS.get(host), now, host)
                          for host, base in nodes]
            print("\033[2J\033[H" + "\n\n".join(dashboards), flush=True)
            if args.once: return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor ukončen.")


if __name__ == "__main__":
    main()
