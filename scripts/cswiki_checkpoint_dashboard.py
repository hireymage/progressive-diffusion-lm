#!/usr/bin/env python3
"""HTML dashboard for cswiki flexible runs and checkpoint diagnostics."""
from __future__ import annotations

import argparse
import html
import json
import socket
import sys
import time
from collections import defaultdict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_cswiki_flexible import ensure_outside_icloud


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def find_watch_summaries(root: Path) -> list[Path]:
    return sorted(root.rglob("*-watch-summary.json"), key=lambda p: p.stat().st_mtime)


def resolve_watch_summary(path: Path) -> Path:
    if path.is_dir():
        summaries = find_watch_summaries(path)
        if summaries:
            return summaries[-1]
    return path


def resolve_summary_source(path: Path) -> Path:
    if path.is_dir():
        summaries = find_watch_summaries(path)
        if summaries:
            return summaries[-1]
        # Fallback to direct artifact scan below.
    return path


def escape_text(text: object) -> str:
    return html.escape("" if text is None else str(text))


def fmt_float(value, digits=4):
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return "—"


def is_training_report(data: dict) -> bool:
    return isinstance(data, dict) and isinstance(data.get("history"), list) and "final" in data


def is_eval_report(data: dict) -> bool:
    return isinstance(data, dict) and "result" in data


def summarize_training(report_path: Path, data: dict) -> dict:
    history = [row for row in data.get("history", []) if isinstance(row, dict)]
    tail = history[-8:]
    final = data.get("final") or {}
    return {
        "kind": "train-report",
        "title": report_path.parent.name,
        "path": str(report_path),
        "status": data.get("status", "—"),
        "steps": data.get("steps", "—"),
        "history_len": len(history),
        "history_tail": tail,
        "final": final,
    }


def summarize_eval(report_path: Path, data: dict) -> dict:
    result = data.get("result") or {}
    return {
        "kind": "eval-report",
        "title": report_path.parent.name,
        "path": str(report_path),
        "mode": result.get("mode", data.get("mode", "—")),
        "result": result,
    }


def summarize_watch(watch_summary: Path) -> dict:
    data = load_json(watch_summary)
    batches = data.get("batches", [])
    cards = []
    for batch in batches:
        batch_dir = Path(batch["batch_dir"])
        generations = load_jsonl(batch_dir / "generations.jsonl")
        if not generations:
            continue
        cards.extend(summarize_checkpoint_group(group) for group in group_generations(generations).values())
    return {
        "kind": "watch-summary",
        "title": watch_summary.parent.name,
        "path": str(watch_summary),
        "data": data,
        "cards": cards,
    }


def group_generations(rows: list[dict]) -> dict[tuple[str, int, str], dict]:
    groups: dict[tuple[str, int, str], dict] = {}
    for row in rows:
        key = (
            str(row.get("model", "model")),
            int(row.get("checkpoint_step", -1)),
            str(row.get("checkpoint_kind", "step")),
        )
        group = groups.setdefault(key, {"model": key[0], "step": key[1], "kind": key[2], "rows": []})
        group["rows"].append(row)
    return groups


def summarize_checkpoint_group(group: dict) -> dict:
    rows = group["rows"]
    best_row = min(rows, key=lambda r: (float(r.get("loss", float("inf"))) if isinstance(r.get("loss"), (int, float)) else float("inf")))
    worst_route = best_row.get("route", "—")
    rows_sorted = sorted(rows, key=lambda r: (str(r.get("route", "")), int(r.get("prompt_index", 0))))
    return {
        "kind": "checkpoint-group",
        "model": group["model"],
        "step": group["step"],
        "checkpoint_kind": group["kind"],
        "rows": rows_sorted,
        "worst_route": worst_route,
        "best_row": best_row,
    }


def collect_artifacts(root: Path) -> list[dict]:
    artifacts: list[dict] = []
    seen_dirs: set[Path] = set()

    for report in sorted(root.rglob("report.json")):
        if report.parent in seen_dirs:
            continue
        data = load_json(report)
        if is_training_report(data):
            artifacts.append(summarize_training(report, data))
            seen_dirs.add(report.parent)
        elif is_eval_report(data):
            artifacts.append(summarize_eval(report, data))
            seen_dirs.add(report.parent)

    for summary in find_watch_summaries(root):
        artifacts.append(summarize_watch(summary))

    return artifacts


def render_training_card(item: dict) -> str:
    final = item["final"]
    history_tail = item["history_tail"]
    history_rows = []
    for row in history_tail:
        per_route = row.get("per_route") or {}
        worst = row.get("worst_route", "—")
        route_data = per_route.get(worst, {})
        history_rows.append(
            f"<tr><td>{row.get('step', '—'):,}</td>"
            f"<td>{fmt_float(row.get('loss'))}</td>"
            f"<td>{fmt_float((row.get('accuracy') or 0) * 100, 2) if isinstance(row.get('accuracy'), (int,float)) else '—'}</td>"
            f"<td>{escape_text(worst)}</td>"
            f"<td>{fmt_float(route_data.get('loss'))}</td>"
            f"<td>{fmt_float(route_data.get('accuracy', 0) * 100, 2) if isinstance(route_data.get('accuracy'), (int,float)) else '—'}</td></tr>"
        )
    final_route = final.get("worst_route", "—")
    final_route_data = (final.get("per_route") or {}).get(final_route, {})
    return f"""
    <section class="card">
      <div class="card-head">
        <div>
          <h2>{escape_text(item["title"])}</h2>
          <div class="meta">Soubor: <code>{escape_text(item["path"])}</code></div>
        </div>
        <div class="scorebox">
          <div><span>status</span><strong>{escape_text(item["status"])}</strong></div>
          <div><span>kroků</span><strong>{escape_text(item["steps"])}</strong></div>
          <div><span>záznamů historie</span><strong>{item["history_len"]}</strong></div>
        </div>
      </div>
      <div class="grid">
        <div class="panel">
          <h3>Finální stav</h3>
          <ul>
            <li>worst route: {escape_text(final_route)}</li>
            <li>loss: {fmt_float(final.get('loss'))}</li>
            <li>accuracy: {fmt_float((final.get('accuracy') or 0) * 100, 2) if isinstance(final.get('accuracy'), (int, float)) else '—'} %</li>
            <li>perplexity: {fmt_float(final.get('perplexity'), 2)}</li>
            <li>route loss: {fmt_float(final_route_data.get('loss'))}</li>
            <li>route accuracy: {fmt_float(final_route_data.get('accuracy', 0) * 100, 2) if isinstance(final_route_data.get('accuracy'), (int, float)) else '—'} %</li>
          </ul>
        </div>
        <div class="panel">
          <h3>Posledních několik kontrol</h3>
          <table class="history">
            <thead><tr><th>krok</th><th>loss</th><th>acc %</th><th>worst route</th><th>route loss</th><th>route acc %</th></tr></thead>
            <tbody>{''.join(history_rows)}</tbody>
          </table>
        </div>
        <div class="panel">
          <h3>Raw final JSON</h3>
          <pre>{escape_text(json.dumps(final, ensure_ascii=False, indent=2))}</pre>
        </div>
      </div>
    </section>
    """


def render_eval_card(item: dict) -> str:
    result = item["result"]
    rows = result.get("history") or []
    tail = rows[-5:] if isinstance(rows, list) else []
    final = result.get("final") or {}
    examples = result.get("examples") or []
    examples_html = "".join(
        f"<details class='row'><summary>{escape_text(ex.get('checkpoint', '—'))} · {escape_text(ex.get('route', '—'))} · {escape_text(ex.get('prompt', ''))}</summary>"
        f"<div class='row-grid'><div><h4>Final text</h4><pre>{escape_text(ex.get('final_text', ''))}</pre></div>"
        f"<div><h4>Diagnostics</h4><pre>{escape_text(json.dumps(ex, ensure_ascii=False, indent=2))}</pre></div></div></details>"
        for ex in examples[:12]
    )
    tail_rows = "".join(
        f"<tr><td>{escape_text(row.get('step', '—'))}</td><td>{fmt_float(row.get('loss'))}</td>"
        f"<td>{fmt_float((row.get('accuracy') or 0) * 100, 2) if isinstance(row.get('accuracy'), (int,float)) else '—'}</td>"
        f"<td>{escape_text(row.get('worst_route', '—'))}</td></tr>"
        for row in tail if isinstance(row, dict)
    )
    return f"""
    <section class="card">
      <div class="card-head">
        <div>
          <h2>{escape_text(item["title"])}</h2>
          <div class="meta">Soubor: <code>{escape_text(item["path"])}</code> · režim: {escape_text(item["mode"])}</div>
        </div>
      </div>
      <div class="grid">
        <div class="panel">
          <h3>Souhrn výsledku</h3>
          <pre>{escape_text(json.dumps(final if final else result, ensure_ascii=False, indent=2))}</pre>
        </div>
        <div class="panel">
          <h3>Posledních několik bodů historie</h3>
          <table class="history">
            <thead><tr><th>krok</th><th>loss</th><th>acc %</th><th>worst route</th></tr></thead>
            <tbody>{tail_rows}</tbody>
          </table>
        </div>
        <div class="panel">
          <h3>Ukázky</h3>
          {examples_html or "<div class='empty'>Žádné ukázky.</div>"}
        </div>
      </div>
    </section>
    """


def render_checkpoint_card(item: dict) -> str:
    rows = item["rows"]
    route_names = sorted({str(r.get("route", "—")) for r in rows})
    prompt_count = len({str(r.get("prompt", "")) for r in rows})
    best = item["best_row"]
    rows_html = "".join(
        f"<details class='row'><summary><span class='badge'>{escape_text(row.get('route', '—'))}</span>"
        f"<span class='muted'>prompt {row.get('prompt_index', '—')}</span>"
        f"<span class='pill'>loss {fmt_float(row.get('loss'))}</span>"
        f"<span class='pill'>acc {fmt_float((row.get('accuracy') or 0) * 100, 2) if isinstance(row.get('accuracy'), (int, float)) else '—'} %</span>"
        f"<span class='pill'>stop {escape_text((row.get('generation_state') or {}).get('stop_reason', '—'))}</span></summary>"
        f"<div class='row-grid'><div><h4>Prompt</h4><pre>{escape_text(row.get('prompt', ''))}</pre></div>"
        f"<div><h4>Výstup</h4><pre>{escape_text(row.get('final_text', ''))}</pre></div>"
        f"<div><h4>Diagnostika</h4><pre>{escape_text(json.dumps(row.get('exit_state') or {}, ensure_ascii=False, indent=2))}</pre></div></div></details>"
        for row in rows
    )
    return f"""
    <section class="card">
      <div class="card-head">
        <div>
          <h2>{escape_text(item['model'])} · {escape_text(item['checkpoint_kind'])} · krok {item['step']:,}</h2>
          <div class="meta">routes: {escape_text(', '.join(route_names))} · worst route: {escape_text(item['worst_route'])} · soubor batch generací</div>
        </div>
        <div class="scorebox">
          <div><span>routes</span><strong>{escape_text(', '.join(route_names))}</strong></div>
          <div><span>prompts</span><strong>{prompt_count}</strong></div>
          <div><span>nejlepší loss</span><strong>{fmt_float(best.get('loss'))}</strong></div>
          <div><span>nejlepší acc</span><strong>{fmt_float((best.get('accuracy') or 0) * 100, 2) if isinstance(best.get('accuracy'), (int, float)) else '—'} %</strong></div>
        </div>
      </div>
      <div class="rows">{rows_html}</div>
    </section>
    """


def render_summary_card(summary: dict) -> str:
    observed = summary.get("observed_checkpoints", [])
    batches = summary.get("batches", [])
    route_results = summary.get("route_results", [])
    def _checkpoint_cell(item: dict) -> str:
        checkpoints = (item.get("summary") or {}).get("checkpoints", [])
        if not checkpoints:
            return "—"
        parts = []
        for ckpt in checkpoints:
            if isinstance(ckpt, dict):
                parts.append(f"{ckpt.get('model', '—')} · {ckpt.get('checkpoint_kind', '—')} · {ckpt.get('checkpoint_step', '—')}")
            else:
                parts.append(str(ckpt))
        return " | ".join(parts)

    route_rows = "".join(
        "<tr>"
        f"<td><code>{escape_text(item.get('host', '—'))}</code></td>"
        f"<td><span class='badge'>{escape_text(item.get('route', '—'))}</span></td>"
        f"<td><code>{escape_text(item.get('batch_dir', ''))}</code></td>"
        f"<td><code>{escape_text(_checkpoint_cell(item))}</code></td>"
        "</tr>"
        for item in route_results
    )
    return f"""
    <section class="card">
      <div class="card-head">
        <div>
          <h2>Watch summary · {escape_text(summary.get('name', '—'))}</h2>
          <div class="meta">Checkpoint dir: <code>{escape_text(summary.get('checkpoint_dir', '—'))}</code></div>
        </div>
        <div class="scorebox">
          <div><span>batches</span><strong>{len(batches)}</strong></div>
          <div><span>observed checkpoints</span><strong>{len(observed)}</strong></div>
        </div>
      </div>
      <div class="rows">
        <ul>
          {''.join(f"<li><code>{escape_text(b.get('batch_dir', ''))}</code> — {escape_text(b.get('checkpoints', []))}</li>" for b in batches)}
        </ul>
        {'<h3>Route assignments</h3><table class="history"><thead><tr><th>host</th><th>route</th><th>batch</th><th>checkpoints</th></tr></thead><tbody>' + route_rows + '</tbody></table>' if route_results else "<div class='empty'>Žádné route assignmenty.</div>"}
      </div>
    </section>
    """


def render_page(artifacts: list[dict], root: Path) -> str:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    cards = []
    for item in artifacts:
        if item["kind"] == "train-report":
            cards.append(render_training_card(item))
        elif item["kind"] == "eval-report":
            cards.append(render_eval_card(item))
        elif item["kind"] == "watch-summary":
            cards.append(render_summary_card(item))
            cards.extend(render_checkpoint_card(card) for card in item.get("cards", []))
        elif item["kind"] == "checkpoint-group":
            cards.append(render_checkpoint_card(item))
    cards_html = "\n".join(cards) if cards else "<div class='empty'>Žádné artefakty nebyly nalezeny.</div>"
    return f"""<!doctype html>
<html lang="cs">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CSWiki checkpoint dashboard</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b1020;
      --panel: #121a2f;
      --panel-2: #101726;
      --text: #e5eef8;
      --muted: #9bb0c5;
      --line: #23314f;
      --accent: #7dd3fc;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: radial-gradient(circle at top, #18203a 0, var(--bg) 60%); color: var(--text); line-height: 1.45; }}
    .wrap {{ max-width: 1600px; margin: 0 auto; padding: 24px; }}
    header {{ display: flex; justify-content: space-between; gap: 20px; align-items: start; flex-wrap: wrap; padding: 20px; border: 1px solid var(--line); border-radius: 20px; background: rgba(10,16,32,.82); margin-bottom: 22px; }}
    h1, h2, h3, h4 {{ margin: 0 0 10px 0; }}
    h1 {{ font-size: 2rem; }}
    .muted, .meta, .small {{ color: var(--muted); }}
    .statline {{ display: grid; gap: 6px; }}
    .pill, .badge {{ display: inline-flex; align-items: center; padding: 4px 10px; border-radius: 999px; border: 1px solid var(--line); background: rgba(148,163,184,.08); margin-left: 8px; font-size: .88rem; }}
    .badge {{ background: rgba(125,211,252,.12); border-color: rgba(125,211,252,.35); }}
    .card {{ border: 1px solid var(--line); border-radius: 20px; background: rgba(15, 23, 42, .84); margin: 18px 0; overflow: hidden; }}
    .card-head {{ display: flex; justify-content: space-between; gap: 20px; align-items: start; flex-wrap: wrap; padding: 20px; border-bottom: 1px solid var(--line); background: rgba(255,255,255,.02); }}
    .scorebox {{ display: grid; grid-template-columns: repeat(2, minmax(120px, 1fr)); gap: 10px; }}
    .scorebox div, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 14px; }}
    .scorebox span {{ display: block; color: var(--muted); font-size: .85rem; }}
    .scorebox strong {{ font-size: 1.05rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; padding: 16px 20px; }}
    .panel dl {{ display: grid; grid-template-columns: auto 1fr; gap: 6px 14px; margin: 0; }}
    .panel dt {{ color: var(--muted); }}
    .panel dd {{ margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
    .rows {{ padding: 0 20px 18px; }}
    details.row {{ border-top: 1px solid var(--line); padding: 10px 0; }}
    summary {{ cursor: pointer; list-style: none; display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }}
    summary::-webkit-details-marker {{ display:none; }}
    .row-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 12px; }}
    pre {{ white-space: pre-wrap; word-wrap: break-word; background: var(--panel-2); border: 1px solid var(--line); border-radius: 12px; padding: 12px; margin: 0; min-height: 120px; }}
    table.history {{ width: 100%; border-collapse: collapse; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
    .history th, .history td {{ border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; }}
    ul {{ margin: 0; padding-left: 20px; }}
    .empty {{ padding: 12px; border: 1px dashed var(--line); border-radius: 12px; color: var(--muted); }}
    code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: .92rem; }}
    @media (max-width: 1100px) {{ .grid, .row-grid {{ grid-template-columns: 1fr; }} header {{ flex-direction: column; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <h1>CSWiki checkpoint dashboard</h1>
        <div class="statline">
          <div><strong>Root:</strong> <code>{escape_text(root)}</code></div>
          <div><strong>Generated:</strong> {generated_at}</div>
          <div><strong>Tips:</strong> najdi zde běhy, checkpointy i starší testy v jedné historii.</div>
        </div>
      </div>
      <div class="small">
        <div><strong>Spuštění:</strong></div>
        <div><code>python scripts/cswiki_checkpoint_dashboard.py --root ... --output-dir ... --serve --host 0.0.0.0 --port 8000</code></div>
      </div>
    </header>
    {cards_html}
  </div>
</body>
</html>
"""


def build_dashboard(root: Path, output_dir: Path) -> Path:
    ensure_outside_icloud(output_dir)
    root = Path(root)
    if root.is_file():
        data = load_json(root)
        if root.name.endswith("-watch-summary.json"):
            summary = summarize_watch(root)
            artifacts = [summary, *summary.get("cards", [])]
        elif is_eval_report(data):
            artifacts = [summarize_eval(root, data)]
        else:
            artifacts = [summarize_training(root, data)] if is_training_report(data) else []
    else:
        artifacts = []
        summary_source = resolve_summary_source(root)
        if summary_source.is_file() and summary_source.name.endswith("-watch-summary.json"):
            artifacts.append(summarize_watch(summary_source))
        # Current and historical reports/checkpoints anywhere under root.
        for report in sorted(root.rglob("report.json")):
            data = load_json(report)
            if is_training_report(data):
                artifacts.append(summarize_training(report, data))
            elif is_eval_report(data):
                artifacts.append(summarize_eval(report, data))
        for latest in sorted(root.rglob("latest.json")):
            data = load_json(latest)
            if data:
                artifacts.append({
                    "kind": "checkpoint-group",
                    "model": latest.parent.parent.name,
                    "step": int(data.get("step", -1)),
                    "checkpoint_kind": "latest",
                    "rows": [],
                    "worst_route": "—",
                    "best_row": {"loss": data.get("best_loss")},
                })
        # Fold in generation-eval batches.
        for summary in sorted(root.rglob("summary.json")):
            if summary.parent == root:
                continue
            gens = summary.parent / "generations.jsonl"
            if gens.exists():
                rows = load_jsonl(gens)
                for group in group_generations(rows).values():
                    artifacts.append(summarize_checkpoint_group(group))
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "index.html"
    html_path.write_text(render_page(artifacts, root), encoding="utf-8")
    return html_path


def pick_free_port(host: str, start: int = 8000, limit: int = 2000) -> int:
    for port in range(start, start + limit):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"nenalezen volný port v rozsahu {start}-{start + limit - 1}")


def serve_directory(directory: Path, host: str, port: int) -> None:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, format, *args):
            return

    httpd = ThreadingHTTPServer((host, port), Handler)
    actual = httpd.server_address[1]
    print(f"Serving {directory} on http://{host}:{actual}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT / "results", help="directory to scan for reports and checkpoints")
    p.add_argument("--output-dir", type=Path, required=True, help="directory for generated HTML")
    p.add_argument("--serve", action="store_true", help="serve the generated HTML over HTTP")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    return p


def main() -> None:
    args = parser().parse_args()
    html_path = build_dashboard(args.root, args.output_dir)
    print(str(html_path))
    if args.serve:
        port = args.port or pick_free_port(args.host)
        serve_directory(args.output_dir, args.host, port)


if __name__ == "__main__":
    main()
