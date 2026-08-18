#!/usr/bin/env python3
"""Distribute cswiki checkpoint prompt evaluation across multiple hosts.

Each assigned host evaluates one route across the full checkpoint set, then the
local coordinator merges the resulting JSONL summaries into a single report and
HTML dashboard.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cswiki_checkpoint_dashboard import render_page, ensure_outside_icloud, load_json, load_jsonl, group_generations, summarize_checkpoint_group

DEFAULT_REMOTE_PYTHON = {
    "m1-256": "/Users/hozzy/Library/Application Support/ML-Experiments/progressive-diffusion-lm/m1-256/env/bin/python",
    "m1-512": "/Users/hozzy/Library/Application Support/ML-Experiments/progressive-diffusion-lm/m1-512/env/bin/python",
    "m4-air": "/Users/hozzy/Library/Application Support/ML-Experiments/progressive-diffusion-lm/m4-air/env/bin/python",
}


def read_models(models_json: Path) -> list[dict]:
    data = json.loads(models_json.read_text())
    if isinstance(data, dict):
        data = data.get("models", [])
    if not isinstance(data, list):
        raise ValueError("--models-json must contain a list or a {models:[...]} object")
    models = []
    for row in data:
        if not isinstance(row, dict):
            continue
        for key in ("name", "checkpoint_dir"):
            if key not in row:
                raise ValueError(f"model spec missing {key!r}")
        models.append(row)
    if not models:
        raise ValueError("no models found in --models-json")
    return models


def route_hosts(routes: list[str], hosts: list[str]) -> dict[str, str]:
    if not routes:
        raise ValueError("--routes must not be empty")
    if not hosts:
        raise ValueError("--hosts must not be empty")
    mapping = {}
    for index, route in enumerate(routes):
        mapping[route] = hosts[index % len(hosts)]
    return mapping


def remote_python_for_host(host: str, overrides: dict[str, str] | None = None) -> str:
    if overrides and host in overrides:
        return overrides[host]
    return DEFAULT_REMOTE_PYTHON.get(host, "python3")


def remote_command(args, route: str, output_dir: Path, remote_python: str) -> str:
    base = [
        remote_python,
        str(ROOT / "scripts" / "batch_prompt_cswiki_checkpoints.py"),
        "--cache-dir", str(args.cache_dir),
        "--models-json", str(args.models_json),
        "--output-dir", str(output_dir),
        "--prompts", str(args.prompts),
        "--max-new-tokens", str(args.max_new_tokens),
        "--routes", route,
    ]
    if args.include_latest_best:
        base.append("--include-latest-best")
    else:
        base.append("--no-include-latest-best")
    if args.measure_exits:
        base.append("--measure-exits")
    else:
        base.append("--no-measure-exits")
    return " ".join(shlex.quote(part) for part in base)


def run_remote(host: str, route: str, output_dir: Path, args, remote_python: str) -> None:
    command = remote_command(args, route, output_dir, remote_python)
    ssh_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command]
    subprocess.run(ssh_cmd, check=True)


def merge_outputs(run_dir: Path, route_dirs: dict[str, Path]) -> dict:
    merged = {
        "status": "complete",
        "name": run_dir.name,
        "checkpoint_dir": str(run_dir),
        "batches": [],
        "observed_checkpoints": [],
    }
    cards = []
    for route, route_dir in route_dirs.items():
        summary = load_json(route_dir / "summary.json")
        if not summary:
            continue
        merged["batches"].append({
            "batch_dir": str(route_dir),
            "checkpoints": summary.get("checkpoints", []),
            "route": route,
        })
        merged["observed_checkpoints"].extend(summary.get("observed_checkpoints", []))
        rows = load_jsonl(route_dir / "generations.jsonl")
        for group in group_generations(rows).values():
            cards.append(summarize_checkpoint_group(group))
    return merged, cards


def build_dashboard(root: Path, output_dir: Path, merged: dict, cards: list[dict]) -> Path:
    ensure_outside_icloud(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "merged-watch-summary.json"
    summary_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Render only the current distributed run so the UI stays focused on the
    # route/host split for this job rather than the whole project history.
    page = render_page([
        {
            "kind": "watch-summary",
            "name": merged.get("name", "distributed-eval"),
            "checkpoint_dir": merged.get("checkpoint_dir", "—"),
            "batches": merged.get("batches", []),
            "observed_checkpoints": merged.get("observed_checkpoints", []),
            "cards": cards,
        },
        *cards,
    ], root)
    html_path = output_dir / "index.html"
    html_path.write_text(page, encoding="utf-8")
    return html_path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT / "results")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--models-json", type=Path, required=True)
    p.add_argument("--routes", nargs="+", choices=("q8_only", "q8_fp16", "q2_q8_fp16"), default=("q8_only", "q8_fp16", "q2_q8_fp16"))
    p.add_argument("--hosts", nargs="+", default=("m1-256", "m1-512", "m4-air"))
    p.add_argument("--remote-python", nargs="*", help="optional host=python overrides for SSH execution")
    p.add_argument("--prompts", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--measure-exits", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-latest-best", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no-run", action="store_true", help="only build the dashboard from existing route outputs")
    return p


def main() -> None:
    args = parser().parse_args()
    models = read_models(args.models_json)
    mapping = route_hosts(list(args.routes), list(args.hosts))
    remote_python_overrides = {}
    if args.remote_python:
        for entry in args.remote_python:
            if "=" not in entry:
                raise ValueError("--remote-python entries must be host=python")
            host, python = entry.split("=", 1)
            remote_python_overrides[host] = python
    route_dirs = {}
    route_results = []
    ts = time.strftime("%Y%m%d-%H%M%S")
    for route, host in mapping.items():
        route_dir = args.output_dir / f"{route}-{host}-{ts}"
        route_dir.mkdir(parents=True, exist_ok=True)
        route_dirs[route] = route_dir
        if not args.no_run:
            remote_python = remote_python_for_host(host, remote_python_overrides)
            run_remote(host, route, route_dir, args, remote_python)
        summary_path = route_dir / "summary.json"
        route_results.append({
            "host": host,
            "route": route,
            "batch_dir": str(route_dir),
            "summary": load_json(summary_path),
        })
    merged, cards = merge_outputs(args.root, route_dirs)
    merged["models_json"] = str(args.models_json)
    merged["routes"] = list(args.routes)
    merged["hosts"] = list(mapping.values())
    merged["route_results"] = route_results
    html_path = build_dashboard(args.output_dir, args.output_dir, merged, cards)
    print(json.dumps({"html": str(html_path), "summary": str(args.output_dir / "merged-watch-summary.json"), "routes": mapping}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
