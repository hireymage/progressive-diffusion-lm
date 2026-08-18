#!/usr/bin/env python3
"""Run route-local SGD blocks and merge them on a coordinator host.

This is an experimental local-SGD/FedAvg-style trainer. It is intentionally
bounded and never modifies the source checkpoint directory.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path


ASSIGNMENTS = {
    "q8_only": ("m1-256", "m1-256"),
    "q8_fp16": ("m1-512", "m1-512"),
    "q2_q8_fp16": ("m4-air", "m4-air"),
}
REMOTE_ROOT = "/Users/hozzy/Library/Application Support/ML-Experiments/progressive-diffusion-lm"
REPO = "/Users/hozzy/Documents/Projekty/progressive-diffusion-lm"


def ssh(host: str, command: str, *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, command], check=check, text=True)


def stream_files(src_host: str, src_dir: str, files: list[str], dst_host: str, dst_dir: str) -> None:
    source = subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", src_host,
         f"tar -C {shlex.quote(src_dir)} -cf - " + " ".join(map(shlex.quote, files))],
        stdout=subprocess.PIPE,
    )
    destination = subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", dst_host,
         f"mkdir -p {shlex.quote(dst_dir)} && tar -C {shlex.quote(dst_dir)} -xf -"],
        stdin=source.stdout,
    )
    assert source.stdout is not None
    source.stdout.close()
    destination_code = destination.wait()
    source_code = source.wait()
    if source_code or destination_code:
        raise RuntimeError(f"checkpoint transfer failed: {src_host}:{src_dir} -> {dst_host}:{dst_dir}")


def train_command(env_name: str, run_dir: str, route: str, target_step: int) -> str:
    python = f"{REMOTE_ROOT}/{env_name}/env/bin/python"
    return (
        f"cd {shlex.quote(REPO)} && "
        f"{shlex.quote(python)} scripts/train_cswiki_flexible.py "
        f"--cache-dir {shlex.quote(REMOTE_ROOT + '/cswiki/cache')} "
        f"--output {shlex.quote(run_dir + '/report.json')} "
        f"--checkpoint-dir {shlex.quote(run_dir + '/checkpoints')} "
        "--d-model 64 --d-ff 256 --n-heads 4 --n-layers 25 --seq-len 256 "
        f"--steps {target_step} --batch-size 4 --eval-steps 32 --eval-every 500 "
        f"--seed 20260804 --lr 0.001 --archive-every 0 --training-routes {route} --resume "
        f"> {shlex.quote(run_dir + '/train.log')} 2>&1"
    )


def write_status(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--coordinator", default="m4-air")
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--total-steps", type=int, default=20_000)
    parser.add_argument("--sync-every", type=int, default=500)
    args = parser.parse_args()
    if args.total_steps <= 0 or args.sync_every <= 0 or args.total_steps % args.sync_every:
        raise ValueError("total steps must be a positive multiple of sync interval")

    args.output_root.mkdir(parents=True, exist_ok=False)
    status_path = args.output_root / "status.json"
    remote_run = f"{REMOTE_ROOT}/cswiki-d64-distributed/{args.output_root.name}"
    common_dir = f"{remote_run}/common/checkpoints"
    coordinator_code = f"{remote_run}/code"
    routes = list(ASSIGNMENTS)
    started = time.time()
    write_status(status_path, {"status": "preparing", "start_step": args.start_step})

    # Put the immutable source on the coordinator, then fan it out unchanged.
    stream_files("m1-512", args.source_dir, ["latest.npz", "latest.json"],
                 args.coordinator, common_dir)
    stream_files("m1-256", str(Path(__file__).parent), ["average_cswiki_checkpoints.py"],
                 args.coordinator, coordinator_code)
    for route, (host, _) in ASSIGNMENTS.items():
        branch = f"{remote_run}/branches/{route}/checkpoints"
        stream_files(args.coordinator, common_dir, ["latest.npz", "latest.json"], host, branch)

    current = args.start_step
    blocks = []
    while current < args.start_step + args.total_steps:
        target = current + args.sync_every
        block_started = time.time()
        write_status(status_path, {"status": "training", "step": current, "target_step": target,
                                  "blocks": blocks, "elapsed_seconds": time.time() - started})
        jobs = []
        for route, (host, env_name) in ASSIGNMENTS.items():
            branch = f"{remote_run}/branches/{route}"
            jobs.append((route, host, subprocess.Popen(
                ["ssh", "-o", "BatchMode=yes", host, train_command(env_name, branch, route, target)]
            )))
        failures = [(route, host, job.wait()) for route, host, job in jobs if job.wait() != 0]
        if failures:
            raise RuntimeError(f"route training failed: {failures}")

        merge_dir = f"{remote_run}/merge/step-{target:07d}"
        inputs = []
        for route, (host, _) in ASSIGNMENTS.items():
            destination = f"{merge_dir}/{route}"
            branch_checkpoints = f"{remote_run}/branches/{route}/checkpoints"
            stream_files(host, branch_checkpoints, ["latest.npz", "latest.json"],
                         args.coordinator, destination)
            inputs.append(f"{destination}/latest.npz")
        merge_python = f"{REMOTE_ROOT}/m4-air/env/bin/python"
        merge_command = (
            f"{shlex.quote(merge_python)} {shlex.quote(coordinator_code + '/average_cswiki_checkpoints.py')} "
            + " ".join(f"--input {shlex.quote(path)}" for path in inputs)
            + f" --output {shlex.quote(common_dir + '/latest.npz')}"
        )
        ssh(args.coordinator, merge_command)
        for route, (host, _) in ASSIGNMENTS.items():
            branch = f"{remote_run}/branches/{route}/checkpoints"
            stream_files(args.coordinator, common_dir, ["latest.npz", "latest.json"], host, branch)
        blocks.append({"step": target, "seconds": time.time() - block_started})
        current = target

    write_status(status_path, {"status": "complete", "step": current, "blocks": blocks,
                              "elapsed_seconds": time.time() - started, "remote_run": remote_run})


if __name__ == "__main__":
    main()
