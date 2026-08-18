#!/usr/bin/env python3
"""Watch flexible checkpoints and run prompt diagnostics on new snapshots."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.batch_prompt_cswiki_checkpoints import discover_checkpoints
from scripts.train_cswiki_flexible import ensure_outside_icloud

STEP_RE = re.compile(r"step_(\d{7})\.npz$")


def checkpoint_ids(checkpoint_dir: Path, include_latest_best: bool) -> list[tuple[str, int]]:
    items = []
    for row in discover_checkpoints(checkpoint_dir, include_latest_best):
        items.append((row["kind"], int(row["step"])))
    return items


def run_batch(args, checkpoint_dir: Path, output_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "batch_prompt_cswiki_checkpoints.py"),
        "--cache-dir", str(args.cache_dir),
        "--models-json", str(args.models_json),
        "--output-dir", str(output_dir),
        "--prompts", str(args.prompts),
        "--max-new-tokens", str(args.max_new_tokens),
        "--include-latest-best",
    ]
    if args.routes:
        cmd.append("--routes")
        cmd.extend(args.routes)
    if args.measure_exits:
        cmd.append("--measure-exits")
    else:
        cmd.append("--no-measure-exits")
    subprocess.run(cmd, check=True)


def watch(args) -> dict:
    ensure_outside_icloud(args.output_root)
    ensure_outside_icloud(args.checkpoint_dir)
    seen: set[tuple[str, int]] = set()
    batches = []
    poll_interval = max(1, args.poll_seconds)
    while True:
        current = checkpoint_ids(args.checkpoint_dir, args.include_latest_best)
        new_items = [item for item in current if item not in seen]
        if new_items:
            marker = "latest-best" if any(kind in {"latest", "best"} for kind, _ in new_items) else "step"
            batch_dir = args.output_root / f"{args.name}-{marker}-{int(time.time())}"
            run_batch(args, args.checkpoint_dir, batch_dir)
            batches.append({"batch_dir": str(batch_dir), "checkpoints": [[kind, step] for kind, step in new_items]})
            seen.update(new_items)
            if args.once:
                break
        elif args.once:
            break
        time.sleep(poll_interval)
    summary = {
        "status": "complete",
        "name": args.name,
        "checkpoint_dir": str(args.checkpoint_dir),
        "batches": batches,
        "observed_checkpoints": [[kind, step] for kind, step in sorted(seen, key=lambda x: (x[1], x[0]))],
    }
    output = args.output_root / f"{args.name}-watch-summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--models-json", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--prompts", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--routes", nargs="+", choices=("q8_only", "q8_fp16", "q2_q8_fp16"))
    p.add_argument("--measure-exits", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-latest-best", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--once", action="store_true")
    return p


def main() -> None:
    print(json.dumps(watch(parser().parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
