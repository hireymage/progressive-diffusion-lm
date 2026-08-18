#!/usr/bin/env python3
"""Verify bilingual coverage, lifecycle status, and local Markdown links."""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", ".pytest_cache", ".venv", "results"}
LINK_RE = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
STATUS_RE = re.compile(
    r"<!-- doc-status: (living|snapshot|proposal|historical); "
    r"verified: \d{4}-\d{2}-\d{2} -->"
)


def public_markdown_files(root: Path = ROOT) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.md")
        if not any(part in EXCLUDED_PARTS for part in path.relative_to(root).parts)
    )


def language_peer(path: Path) -> Path | None:
    if path.name == "README.md":
        return path.with_name("README.cs.md")
    english = path.with_name(f"{path.stem}.en.md")
    czech = path.with_name(f"{path.stem}.cs.md")
    if english.exists():
        return english
    if czech.exists():
        return czech
    return None


def check_bilingual_docs(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    for path in public_markdown_files(root):
        if path.name == "README.cs.md" or path.name.endswith((".en.md", ".cs.md")):
            continue
        peer = language_peer(path)
        relative = path.relative_to(root)
        if peer is None:
            errors.append(f"missing language peer: {relative}")
            continue
        for document in (path, peer):
            head = "\n".join(document.read_text().splitlines()[:12])
            if "[English](" not in head or "[Čeština](" not in head:
                errors.append(f"missing language switch: {document.relative_to(root)}")
        statuses = []
        for document in (path, peer):
            match = STATUS_RE.search("\n".join(document.read_text().splitlines()[:12]))
            statuses.append(match.group(1) if match else None)
        if all(statuses) and statuses[0] != statuses[1]:
            errors.append(
                f"language peers have different document status: {relative}"
            )

    for path in public_markdown_files(root):
        head = "\n".join(path.read_text().splitlines()[:12])
        if not STATUS_RE.search(head):
            errors.append(f"missing or invalid document status: {path.relative_to(root)}")

        for target in LINK_RE.findall(path.read_text(errors="replace")):
            local_target = target.split("#", 1)[0]
            if not local_target or "://" in local_target or local_target.startswith("mailto:"):
                continue
            if not (path.parent / local_target).resolve().exists():
                errors.append(f"broken link in {path.relative_to(root)}: {target}")
    return errors


def main() -> int:
    errors = check_bilingual_docs()
    if errors:
        print("\n".join(errors))
        return 1
    print("Bilingual documentation check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
