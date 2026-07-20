#!/usr/bin/env python3
import csv
import datetime as dt
import json
import re
import subprocess
import time
from pathlib import Path

RUN_ROOT = Path(__file__).resolve().parents[1]
STATE = RUN_ROOT / "state.json"
OUT = RUN_ROOT / "logs" / "memory-monitor.csv"


def command(args):
    return subprocess.run(args, text=True, capture_output=True, check=False).stdout


def sample():
    pressure = command(["memory_pressure"])
    m = re.search(r"System-wide memory free percentage:\s*(\d+)%", pressure)
    free_pct = int(m.group(1)) if m else None
    swap = command(["sysctl", "vm.swapusage"]).strip()
    processes = command(["ps", "-axo", "pid=,%cpu=,rss=,command="])
    rows = []
    for line in processes.splitlines():
        if "src.train" in line or "ptq_study.py" in line:
            parts = line.strip().split(None, 3)
            if len(parts) == 4:
                rows.append({"pid": parts[0], "cpu_pct": parts[1], "rss_kib": parts[2], "command": parts[3]})
    return free_pct, swap, rows


OUT.parent.mkdir(parents=True, exist_ok=True)
new = not OUT.exists()
with OUT.open("a", newline="") as handle:
    writer = csv.writer(handle)
    if new:
        writer.writerow(["timestamp_utc", "campaign_status", "memory_free_pct", "swapusage", "processes_json"])
        handle.flush()
    while True:
        try:
            state = json.loads(STATE.read_text())
            status = state.get("status", "unknown")
        except Exception:
            status = "unavailable"
        free_pct, swap, rows = sample()
        writer.writerow([dt.datetime.now(dt.timezone.utc).isoformat(), status, free_pct, swap, json.dumps(rows, separators=(",", ":"))])
        handle.flush()
        if status in {"completed", "failed"}:
            break
        time.sleep(60)
