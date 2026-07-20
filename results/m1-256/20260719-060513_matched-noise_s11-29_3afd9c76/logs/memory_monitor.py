#!/usr/bin/env python3
import csv
import datetime as dt
import json
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = HERE.parent / "state.json"
OUT = HERE / "memory-monitor.csv"

def free_percent():
    p = subprocess.run(["/usr/bin/memory_pressure", "-Q"], text=True, capture_output=True)
    m = re.search(r"System-wide memory free percentage:\s*(\d+)%", p.stdout)
    return int(m.group(1)) if m else None

def swap_used_mb():
    p = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"], text=True, capture_output=True)
    m = re.search(r"used\s*=\s*([0-9.]+)([MG])", p.stdout)
    if not m:
        return None
    value = float(m.group(1))
    return value * 1024 if m.group(2) == "G" else value

new = not OUT.exists()
with OUT.open("a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["timestamp_utc", "campaign_status", "memory_free_percent", "swap_used_mb"])
        f.flush()
    while True:
        try:
            status = json.loads(STATE.read_text()).get("status", "unknown")
        except Exception:
            status = "unknown"
        w.writerow([
            dt.datetime.now(dt.timezone.utc).isoformat(),
            status,
            free_percent(),
            swap_used_mb(),
        ])
        f.flush()
        if status in {"completed", "failed"}:
            break
        time.sleep(60)
