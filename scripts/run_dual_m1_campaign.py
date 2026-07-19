#!/usr/bin/env python3
"""Safe, sequential dual-M1 campaign orchestrator.

Runtime work happens in a node-local snapshot outside iCloud.  A completed
campaign is bundled locally and then published, under a global lock, to the
unique immutable destination results/<node>/<run-id>.  Dry-run is read-only.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ALLOWED_NODES = {"m1-256", "m1-512"}
RUN_ID_RE = re.compile(r"^\d{8}-\d{6}_[a-z0-9][a-z0-9-]*_s[a-z0-9-]+_[0-9a-f]{8}$")
DEFAULT_LOCAL_ROOT = Path.home() / "Library/Application Support/ML-Experiments/progressive-diffusion-lm"
SOURCE_EXCLUDES = {".git", ".venv", "checkpoints", "data", "results", "results.zip", "__pycache__", ".pytest_cache"}


DEFAULT_TRAIN_METRIC_COLUMNS = [
    "step", "train_loss", "gradient_norm", "q1_residual_rms",
    "injected_noise_rms", "bits_used", "lr", "elapsed_s",
]
DEFAULT_EVAL_METRIC_FIELDS = [
    "step", "val_loss", "val_perplexity", "val_accuracy",
    "generalization_gap", "train_loss", "bits_used",
]
DEFAULT_FINITE_SUMMARY_FIELDS = ["best_val_loss", "total_training_seconds"]


class CampaignError(RuntimeError):
    pass


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def ensure_safe_paths(source: Path, local_root: Path, shared_results: Path) -> None:
    source = source.resolve()
    local_root = local_root.resolve()
    shared_results = shared_results.resolve()
    expected_results = (source / "results").resolve()
    if not source.is_dir() or not (source / "src").is_dir():
        raise CampaignError(f"invalid project source: {source}")
    if shared_results != expected_results:
        raise CampaignError(f"shared results must be exactly {expected_results}")
    if is_relative_to(local_root, source) or is_relative_to(source, local_root):
        raise CampaignError("local runtime root must be outside the synchronized source tree")
    if "Mobile Documents" in str(local_root) or "iCloud" in str(local_root):
        raise CampaignError("local runtime root must not be on iCloud")


def validate_node(node: str, campaign: dict[str, Any]) -> None:
    if node not in ALLOWED_NODES:
        raise CampaignError(f"unsupported node {node!r}; expected one of {sorted(ALLOWED_NODES)}")
    configured = campaign.get("node")
    if configured != node:
        raise CampaignError(f"campaign is for node {configured!r}, not {node!r}")
    if campaign.get("max_parallel_tasks") != 1:
        raise CampaignError("max_parallel_tasks must be exactly 1 on 8GB M1 nodes")


def new_run_id(campaign: dict[str, Any], now: dt.datetime | None = None, nonce: str | None = None) -> str:
    now = now or dt.datetime.now(dt.timezone.utc)
    slug = campaign["campaign_slug"]
    seeds = "-".join(str(s) for s in campaign["seeds"])
    value = f"{now:%Y%m%d-%H%M%S}_{slug}_s{seeds}_{nonce or uuid.uuid4().hex[:8]}"
    if not RUN_ID_RE.fullmatch(value):
        raise CampaignError(f"generated unsafe run id: {value}")
    return value


def validate_run_id(run_id: str) -> None:
    if not RUN_ID_RE.fullmatch(run_id):
        raise CampaignError(f"invalid run id: {run_id!r}")


def deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise CampaignError(f"expected JSON object in {path}")
    return value


def git_head(source: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def memory_bytes() -> int | None:
    if sys.platform != "darwin":
        return None
    result = subprocess.run(["sysctl", "-n", "hw.memsize"], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def environment_record(source: Path, node: str) -> dict[str, Any]:
    mem = memory_bytes()
    return {
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "node": node,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "executable": sys.executable,
        "physical_memory_bytes": mem,
        "physical_memory_gib": round(mem / 2**30, 2) if mem else None,
        "source_git_head": git_head(source),
        "environment": {k: os.environ[k] for k in ("MLX_METAL_CACHE_DIR", "TMPDIR") if k in os.environ},
    }


def validate_memory(campaign: dict[str, Any], record: dict[str, Any]) -> None:
    measured = record["physical_memory_gib"]
    minimum = float(campaign.get("minimum_memory_gib", 7.0))
    if measured is not None and measured < minimum:
        raise CampaignError(f"node has {measured} GiB RAM; campaign requires at least {minimum} GiB")


def copy_source_snapshot(source: Path, repo: Path) -> None:
    if repo.exists():
        return
    repo.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, repo, ignore=shutil.ignore_patterns(*SOURCE_EXCLUDES), symlinks=False)


def prepare_local_data(repo: Path, local_root: Path, node: str) -> None:
    cache = local_root / node / "shared-cache" / "data-cache"
    cache.mkdir(parents=True, exist_ok=True)
    data_dir = repo / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_link = data_dir / "cache"
    if not cache_link.exists():
        cache_link.symlink_to(cache, target_is_directory=True)


def materialize_task_config(repo: Path, task: dict[str, Any]) -> Path | None:
    base_rel = task.get("base_config")
    if not base_rel:
        return None
    base_path = repo / base_rel
    cfg = deep_merge(load_json(base_path), task.get("overrides", {}))
    generated = repo / "configs/campaign/generated" / f"{task['id']}.json"
    generated.parent.mkdir(parents=True, exist_ok=True)
    with generated.open("w") as handle:
        json.dump(cfg, handle, indent=2)
        handle.write("\n")
    return generated.relative_to(repo)


def command_for_task(task: dict[str, Any], python: str, config_path: Path | None) -> list[str]:
    substitutions = {"python": python, "config": str(config_path) if config_path else ""}
    command = [part.format(**substitutions) for part in task["command"]]
    if not command or any(not part for part in command):
        raise CampaignError(f"invalid command for task {task['id']}")
    return command


def initial_state(campaign: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "campaign": campaign["campaign_slug"],
        "node": campaign["node"],
        "status": "pending",
        "tasks": {task["id"]: {"status": "pending", "exit_code": None} for task in campaign["tasks"]},
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def task_is_complete(state: dict[str, Any], task_id: str) -> bool:
    task = state["tasks"].get(task_id, {})
    return task.get("status") == "completed" and task.get("exit_code") == 0


def _require_finite_record(
    record: dict[str, Any], required_fields: list[str], artifact: Path
) -> None:
    missing = [field for field in required_fields if field not in record]
    if missing:
        raise CampaignError(f"missing fields in {artifact}: {', '.join(missing)}")
    for field in required_fields:
        try:
            value = float(record[field])
        except (TypeError, ValueError) as exc:
            raise CampaignError(f"nonnumeric field {field!r} in {artifact}") from exc
        if not math.isfinite(value):
            raise CampaignError(f"nonfinite field {field!r} in {artifact}")


def validate_task_completion(repo: Path, task: dict[str, Any], exit_code: int) -> None:
    """Enforce opt-in semantic metrics artifacts after process success."""
    if exit_code != 0:
        raise CampaignError(f"task {task['id']} exited with code {exit_code}")
    contract = task.get("expected_metrics_contract")
    if contract is None:
        return
    if not isinstance(contract, dict):
        raise CampaignError(f"invalid expected_metrics_contract for task {task['id']}")

    artifact_dir = (repo / contract["artifact_dir"]).resolve()
    if not is_relative_to(artifact_dir, repo):
        raise CampaignError(f"artifact contract escapes runtime repo for task {task['id']}")
    csv_path = artifact_dir / "train_metrics.csv"
    eval_path = artifact_dir / "eval_history.json"
    summary_path = artifact_dir / "final_summary.json"
    for path in (csv_path, eval_path, summary_path):
        if not path.is_file():
            raise CampaignError(f"missing required artifact for task {task['id']}: {path}")

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        required_columns = contract.get(
            "train_metrics_columns", DEFAULT_TRAIN_METRIC_COLUMNS
        )
        missing_columns = [field for field in required_columns if field not in columns]
        if missing_columns:
            raise CampaignError(
                f"missing columns in {csv_path}: {', '.join(missing_columns)}"
            )
        train_rows = list(reader)
    if not train_rows:
        raise CampaignError(f"missing metric rows in {csv_path}")
    for row in train_rows:
        _require_finite_record(row, required_columns, csv_path)

    with eval_path.open() as handle:
        eval_history = json.load(handle)
    if not isinstance(eval_history, list) or not eval_history:
        raise CampaignError(f"missing evaluation records in {eval_path}")
    eval_fields = contract.get("eval_history_fields", DEFAULT_EVAL_METRIC_FIELDS)
    for record in eval_history:
        if not isinstance(record, dict):
            raise CampaignError(f"invalid evaluation record in {eval_path}")
        _require_finite_record(record, eval_fields, eval_path)

    summary = load_json(summary_path)
    for field in contract.get(
        "finite_summary_fields", DEFAULT_FINITE_SUMMARY_FIELDS
    ):
        _require_finite_record(summary, [field], summary_path)
    for field, expected in contract.get("expected_summary", {}).items():
        if field not in summary:
            raise CampaignError(f"missing summary field {field!r} in {summary_path}")
        if summary[field] != expected:
            raise CampaignError(
                f"summary field {field!r} mismatch for task {task['id']}: "
                f"expected {expected!r}, got {summary[field]!r}"
            )


def tail_summary(log_path: Path, lines: int = 40) -> str:
    if not log_path.exists():
        return ""
    with log_path.open(errors="replace") as handle:
        content = handle.readlines()
    return "".join(content[-lines:])


def run_task(repo: Path, task: dict[str, Any], command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MLX_METAL_CACHE_DIR"] = str(repo.parent / "mlx-cache")
    (repo.parent / "mlx-cache").mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        log.flush()
        result = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    return result.returncode


def collect_bundle(run_root: Path, repo: Path, campaign: dict[str, Any], state: dict[str, Any]) -> Path:
    bundle = run_root / "bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir()
    for name in ("manifest.json", "environment.json", "state.json"):
        shutil.copy2(run_root / name, bundle / name)
    shutil.copytree(run_root / "logs", bundle / "logs")
    generated = repo / "configs/campaign/generated"
    if generated.exists():
        shutil.copytree(generated, bundle / "configs")
    artifact_summary: dict[str, list[str]] = {}
    for task in campaign["tasks"]:
        copied: list[str] = []
        for rel in task.get("artifacts", []):
            src = repo / rel
            if src.exists():
                dst = bundle / "artifacts" / task["id"] / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
                copied.append(rel)
        artifact_summary[task["id"]] = copied
    write_json(bundle / "artifact_summary.json", artifact_summary)
    summaries = {task["id"]: tail_summary(run_root / "logs" / f"{task['id']}.log") for task in campaign["tasks"]}
    write_json(bundle / "log_summaries.json", summaries)
    return bundle


def acquire_publish_lock(lock: Path, timeout: float = 120.0) -> int:
    deadline = time.monotonic() + timeout
    lock.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            return os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise CampaignError(f"timed out waiting for shared publish lock: {lock}")
            time.sleep(1.0)


def make_read_only(path: Path) -> None:
    for item in sorted(path.rglob("*"), reverse=True):
        item.chmod(0o555 if item.is_dir() else 0o444)
    path.chmod(0o555)


def publish_bundle(bundle: Path, shared_results: Path, node: str, run_id: str) -> Path:
    destination = shared_results / node / run_id
    lock = shared_results / ".dual-m1-publish.lock"
    fd = acquire_publish_lock(lock)
    try:
        os.write(fd, f"pid={os.getpid()} node={node} run_id={run_id}\n".encode())
        if destination.exists():
            raise CampaignError(f"immutable result destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.parent / f".{run_id}.staging-{uuid.uuid4().hex[:8]}"
        shutil.copytree(bundle, staging)
        make_read_only(staging)
        os.replace(staging, destination)
        return destination
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)


def describe_plan(campaign: dict[str, Any], run_id: str, local_root: Path, source: Path) -> dict[str, Any]:
    run_root = local_root / campaign["node"] / run_id
    planned_tasks = []
    for task in campaign["tasks"]:
        config_path = (
            Path("configs/campaign/generated") / f"{task['id']}.json"
            if task.get("base_config")
            else None
        )
        planned_tasks.append({
            "id": task["id"],
            "description": task["description"],
            "command": command_for_task(task, sys.executable, config_path),
        })
    return {
        "dry_run": True,
        "node": campaign["node"],
        "run_id": run_id,
        "runtime_repo": str(run_root / "repo"),
        "shared_destination": str(source / "results" / campaign["node"] / run_id),
        "sequential": True,
        "tasks": planned_tasks,
    }


def execute(args: argparse.Namespace) -> int:
    source = args.source.resolve()
    campaign_path = args.campaign.resolve()
    campaign = load_json(campaign_path)
    validate_node(args.node, campaign)
    ensure_safe_paths(source, args.local_root, args.shared_results_root)
    run_id = args.run_id or new_run_id(campaign)
    validate_run_id(run_id)

    if args.dry_run:
        print(json.dumps(describe_plan(campaign, run_id, args.local_root, source), indent=2))
        return 0

    run_root = args.local_root / args.node / run_id
    state_path = run_root / "state.json"
    published = args.shared_results_root / args.node / run_id
    if args.resume and published.exists():
        published_manifest = published / "manifest.json"
        if published_manifest.exists() and load_json(published_manifest).get("run_id") == run_id:
            print(f"[SKIP] campaign already published at {published}")
            return 0
        raise CampaignError(f"published destination exists but cannot be verified: {published}")
    if run_root.exists() and not args.resume:
        raise CampaignError(f"run already exists; pass --resume --run-id {run_id}")
    if args.resume and not state_path.exists():
        raise CampaignError(f"cannot resume; state not found: {state_path}")
    run_root.mkdir(parents=True, exist_ok=True)
    repo = run_root / "repo"
    copy_source_snapshot(source, repo)
    prepare_local_data(repo, args.local_root, args.node)

    environment = environment_record(source, args.node)
    validate_memory(campaign, environment)
    write_json(run_root / "environment.json", environment)
    manifest = {
        "run_id": run_id,
        "node": args.node,
        "campaign_config": str(campaign_path),
        "campaign": campaign,
        "source": str(source),
        "runtime_repo": str(repo),
        "shared_destination": str(args.shared_results_root / args.node / run_id),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    write_json(run_root / "manifest.json", manifest)
    state = load_json(state_path) if state_path.exists() else initial_state(campaign, run_id)
    state["status"] = "running"
    write_json(state_path, state)

    for task in campaign["tasks"]:
        task_id = task["id"]
        if task_is_complete(state, task_id):
            try:
                validate_task_completion(repo, task, 0)
            except CampaignError as exc:
                print(f"[RETRY] {task_id}: prior semantic completion invalid: {exc}")
            else:
                print(f"[SKIP] {task_id} already completed")
                continue
        config_path = materialize_task_config(repo, task)
        command = command_for_task(task, sys.executable, config_path)
        entry = state["tasks"][task_id]
        entry.update({"status": "running", "command": command, "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        write_json(state_path, state)
        print(f"[RUN] {task_id}: {' '.join(command)}")
        code = run_task(repo, task, command, run_root / "logs" / f"{task_id}.log")
        semantic_error = None
        if code == 0:
            try:
                validate_task_completion(repo, task, code)
            except CampaignError as exc:
                semantic_error = str(exc)
        succeeded = code == 0 and semantic_error is None
        entry.update({
            "exit_code": code,
            "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "completed" if succeeded else "failed",
        })
        if semantic_error is not None:
            entry["semantic_error"] = semantic_error
        else:
            entry.pop("semantic_error", None)
        write_json(state_path, state)
        if not succeeded:
            state["status"] = "failed"
            write_json(state_path, state)
            if semantic_error is not None:
                print(
                    f"[FAIL] {task_id} exit=0 but artifact contract failed: "
                    f"{semantic_error}; resume with --resume --run-id {run_id}",
                    file=sys.stderr,
                )
                return 2
            print(f"[FAIL] {task_id} exit={code}; resume with --resume --run-id {run_id}", file=sys.stderr)
            return code

    state["status"] = "completed"
    state["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(state_path, state)
    bundle = collect_bundle(run_root, repo, campaign, state)
    destination = publish_bundle(bundle, args.shared_results_root, args.node, run_id)
    print(f"[DONE] immutable bundle published to {destination}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=project)
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--shared-results-root", type=Path, default=project / "results")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    try:
        return execute(build_parser().parse_args())
    except CampaignError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
