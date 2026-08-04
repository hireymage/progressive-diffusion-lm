#!/usr/bin/env python3
"""Offline, reproducible preparation of a Czech Wikipedia dump for the pilot.

All inputs and outputs are explicit local paths.  In particular this module
does not download a corpus, tokenizer, or model.
"""
from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[MASK]", "[BOS]", "[EOS]"]
DEFAULT_MAX_ARTICLES = 50_000
DEFAULT_MAX_TEXT_BYTES = 500_000_000
CSWIKI_DUMP_RE = re.compile(r"^cswiki-\d{8}-pages-articles\.xml\.bz2$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_manifest(dump: Path, manifest: Path | None = None) -> Path:
    """Find a dated SHA1 manifest near *dump*, without accepting a guess."""
    if manifest:
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        return manifest
    candidates = sorted(dump.parent.glob("*sha1*")) + sorted(dump.parent.glob("*SHA1*"))
    for candidate in candidates:
        if dump.name in candidate.read_text(errors="replace"):
            return candidate
    raise FileNotFoundError("No SHA1 manifest containing " + dump.name)


def verify_dump(dump: Path, manifest: Path | None = None) -> dict:
    dump, manifest = Path(dump), resolve_manifest(Path(dump), Path(manifest) if manifest else None)
    if not CSWIKI_DUMP_RE.fullmatch(dump.name):
        raise ValueError("Expected dated Czech dump filename cswiki-YYYYMMDD-pages-articles.xml.bz2")
    expected = None
    for line in manifest.read_text(errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[-1].lstrip("*") == dump.name:
            expected = parts[0].lower()
            break
    if not expected or not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError(f"Manifest has no exact SHA1 entry for {dump.name}")
    actual = sha1_file(dump)
    if actual != expected:
        raise ValueError(f"SHA1 mismatch for {dump.name}: expected {expected}, got {actual}")
    return {"dump": str(dump.resolve()), "dump_filename": dump.name, "sha1": actual,
            "manifest": str(manifest.resolve()), "manifest_sha256": sha256_file(manifest)}


def clean_wikitext(text: str) -> str:
    """Conservative readable-text conversion; malformed markup never aborts extraction."""
    try:
        import mwparserfromhell
        code = mwparserfromhell.parse(text)
        for tag in code.filter_tags(recursive=True):
            code.remove(tag)
        for template in code.filter_templates(recursive=True):
            code.remove(template)
        text = code.strip_code(normalize=True, collapse=True)
    except Exception:
        text = re.sub(r"<ref[^>/]*?>.*?</ref>|<[^>]+>", " ", text, flags=re.I | re.S)
        text = re.sub(r"\{\{.*?\}\}", " ", text, flags=re.S)
        text = re.sub(r"\[\[([^]|]+)\|?([^]]*)\]\]", lambda m: m.group(2) or m.group(1), text)
    text = re.sub(r"(?m)^\s*[=*#;:].*?$", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _child(el: ET.Element, name: str) -> str:
    found = el.find("{*}" + name)
    return found.text if found is not None and found.text else ""


def iter_articles(dump: Path) -> Iterator[dict]:
    """Stream only article-namespace, non-redirect revisions, with bounded XML memory."""
    with bz2.open(dump, "rb") as fh:
        for _, page in ET.iterparse(fh, events=("end",)):
            if not page.tag.endswith("}page"):
                continue
            try:
                if _child(page, "ns") != "0" or page.find("{*}redirect") is not None:
                    continue
                revision = page.find("{*}revision")
                raw = _child(revision, "text") if revision is not None else ""
                text = clean_wikitext(raw)
                if text:
                    yield {"id": _child(page, "id"), "title": _child(page, "title"), "text": text}
            finally:
                page.clear()


def _atomic_replace(part: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(part, final)


def _atomic_json(path: Path, value: dict) -> None:
    """Write a sidecar as a single publish operation, never replacing history."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite metadata: {path}")
    part = path.with_name(path.name + ".part")
    if part.exists():
        raise FileExistsError(f"Refusing stale partial metadata: {part}")
    try:
        part.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        _atomic_replace(part, path)
    except Exception:
        part.unlink(missing_ok=True)
        raise


def validated_corpus_metadata(corpus: Path) -> dict:
    """Accept only a self-checking corpus extracted from the dated cswiki dump."""
    sidecar = Path(corpus).with_suffix(Path(corpus).suffix + ".meta.json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Required corpus provenance sidecar missing: {sidecar}")
    try:
        meta = json.loads(sidecar.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid corpus provenance sidecar: {sidecar}") from exc
    source = meta.get("source", {})
    if (meta.get("format") != "cswiki-jsonl-v1" or meta.get("corpus_sha256") != sha256_file(Path(corpus))
            or not CSWIKI_DUMP_RE.fullmatch(str(source.get("dump_filename", "")))
            or not re.fullmatch(r"[0-9a-f]{40}", str(source.get("sha1", "")))):
        raise ValueError("Corpus provenance is not a verified dated cswiki extraction")
    return meta


def validated_tokenizer_metadata(tokenizer_dir: Path, *, test_only_allow_nonstandard_vocab: bool = False) -> dict:
    path = Path(tokenizer_dir) / "metadata.json"
    tok = Path(tokenizer_dir) / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"Required tokenizer provenance sidecar missing: {path}")
    meta = json.loads(path.read_text())
    source = meta.get("source", {})
    if (meta.get("format") != "cswiki-byte-bpe-v1" or meta.get("tokenizer_sha256") != sha256_file(tok)
            or (meta.get("vocab_size_actual") != 16000 and not test_only_allow_nonstandard_vocab)
            or not CSWIKI_DUMP_RE.fullmatch(str(source.get("dump_filename", "")))
            or not re.fullmatch(r"[0-9a-f]{40}", str(source.get("sha1", "")))):
        raise ValueError("Tokenizer provenance is not a verified 16000-vocabulary cswiki tokenizer")
    return meta


def extract(dump: Path, output: Path, manifest: Path | None = None, max_articles: int = DEFAULT_MAX_ARTICLES,
            max_text_bytes: int = DEFAULT_MAX_TEXT_BYTES) -> dict:
    if max_articles <= 0 or max_text_bytes <= 0:
        raise ValueError("limits must be positive")
    source = verify_dump(dump, manifest)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite corpus: {output}")
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    if meta_path.exists():
        raise FileExistsError(f"Refusing to overwrite corpus metadata: {meta_path}")
    part = output.with_name(output.name + ".part")
    if part.exists():
        raise FileExistsError(f"Refusing stale partial output: {part}")
    n = total = 0
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with part.open("x", encoding="utf-8") as f:
            for article in iter_articles(Path(dump)):
                size = len(article["text"].encode("utf-8"))
                if total + size > max_text_bytes:
                    break
                f.write(json.dumps(article, ensure_ascii=False, separators=(",", ":")) + "\n")
                n += 1; total += size
                if n >= max_articles:
                    break
        _atomic_replace(part, output)
    except Exception:
        part.unlink(missing_ok=True)
        raise
    meta = {"format": "cswiki-jsonl-v1", "source": source, "corpus": str(output.resolve()),
            "corpus_sha256": sha256_file(output), "articles": n, "text_bytes": total,
            "limits": {"max_articles": max_articles, "max_text_bytes": max_text_bytes}}
    _atomic_json(meta_path, meta)
    return meta


def iter_jsonl(corpus: Path) -> Iterator[dict]:
    with Path(corpus).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if isinstance(row.get("text"), str) and row["text"]:
                yield row


def train_tokenizer(corpus: Path, output: Path, vocab_size: int = 16_000, *, test_only_allow_nonstandard_vocab: bool = False) -> dict:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    corpus, output = Path(corpus), Path(output)
    if vocab_size != 16_000 and not test_only_allow_nonstandard_vocab:
        raise ValueError("Czech pilot tokenizer vocabulary must be exactly 16000")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite tokenizer: {output}")
    corpus_meta = validated_corpus_metadata(corpus)
    corpus_hash = corpus_meta["corpus_sha256"]
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator((row["text"] for row in iter_jsonl(corpus)),
                                  trainer=trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                                              special_tokens=SPECIAL_TOKENS, show_progress=False))
    ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    if list(ids.values()) != list(range(len(SPECIAL_TOKENS))):
        raise RuntimeError("tokenizers did not preserve required special-token IDs")
    tokenizer.post_processor = processors.TemplateProcessing(single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", ids["[BOS]"]), ("[EOS]", ids["[EOS]"])])
    if tokenizer.get_vocab_size() != vocab_size:
        raise ValueError(f"Corpus cannot train required Czech BPE vocabulary of {vocab_size}; got {tokenizer.get_vocab_size()}")
    output.mkdir(parents=True)
    tok_path = output / "tokenizer.json"; tokenizer.save(str(tok_path))
    meta = {"format": "cswiki-byte-bpe-v1", "corpus": str(corpus.resolve()), "corpus_sha256": corpus_hash,
            "source": corpus_meta["source"], "corpus_metadata_sha256": sha256_file(corpus.with_suffix(corpus.suffix + ".meta.json")),
            "vocab_size_requested": vocab_size, "vocab_size_actual": tokenizer.get_vocab_size(),
            "special_tokens": ids, "tokenizer_sha256": sha256_file(tok_path)}
    (output / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    return meta


def _is_val(article: dict) -> bool:
    key = (str(article.get("id", "")) + "\0" + article.get("title", "")).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 20 == 0


def _count_chunks(corpus: Path, tokenizer, seq_len: int, want_val: bool) -> int:
    count = 0
    for row in iter_jsonl(corpus):
        if _is_val(row) != want_val: continue
        count += len(tokenizer.encode(row["text"]).ids) // seq_len
    return count


def _write_chunks(corpus: Path, tokenizer, target: Path, rows: int, seq_len: int, want_val: bool) -> None:
    arr = np.lib.format.open_memmap(target, mode="w+", dtype=np.int32, shape=(rows, seq_len))
    at = 0
    for row in iter_jsonl(corpus):
        if _is_val(row) != want_val: continue
        ids = tokenizer.encode(row["text"]).ids
        for offset in range(0, len(ids) - seq_len + 1, seq_len):
            arr[at] = ids[offset:offset + seq_len]; at += 1
    if at != rows: raise RuntimeError("cache count changed between passes")
    arr.flush(); del arr


def build_cache(corpus: Path, tokenizer_dir: Path, cache_dir: Path, seq_len: int = 256,
                *, test_only_allow_nonstandard_vocab: bool = False) -> dict:
    from tokenizers import Tokenizer
    if seq_len != 256: raise ValueError("Czech pilot cache sequence length must be 256")
    corpus, tokenizer_dir, cache_dir = Path(corpus), Path(tokenizer_dir), Path(cache_dir)
    corpus_meta = validated_corpus_metadata(corpus)
    tokenizer_meta = validated_tokenizer_metadata(
        tokenizer_dir, test_only_allow_nonstandard_vocab=test_only_allow_nonstandard_vocab)
    if (tokenizer_meta.get("corpus_sha256") != corpus_meta["corpus_sha256"]
            or tokenizer_meta.get("source") != corpus_meta.get("source")):
        raise ValueError("Tokenizer and corpus provenance do not match")
    tok_file = tokenizer_dir / "tokenizer.json"
    identity = {"format": "cswiki-cache-v1", "corpus_sha256": corpus_meta["corpus_sha256"], "source": corpus_meta["source"],
                "corpus_metadata_sha256": sha256_file(corpus.with_suffix(corpus.suffix + ".meta.json")),
                "tokenizer_sha256": sha256_file(tok_file), "tokenizer_metadata_sha256": sha256_file(tokenizer_dir / "metadata.json"),
                "seq_len": seq_len, "split": "article-sha256-mod20-v1"}
    suffix = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    train, val = cache_dir / f"train_seq{seq_len}_{suffix}.npy", cache_dir / f"val_seq{seq_len}_{suffix}.npy"
    meta_path = cache_dir / f"meta_seq{seq_len}_{suffix}.json"
    if meta_path.exists() and train.exists() and val.exists():
        existing = json.loads(meta_path.read_text())
        if (existing.get("train_sha256") == sha256_file(train)
                and existing.get("val_sha256") == sha256_file(val)):
            return existing
        raise ValueError("Existing cache failed checksum validation; refusing to overwrite history")
    if train.exists() or val.exists() or meta_path.exists():
        raise FileExistsError("Incomplete cache identity exists; refusing to overwrite history")
    tokenizer = Tokenizer.from_file(str(tok_file))
    n_train, n_val = _count_chunks(corpus, tokenizer, seq_len, False), _count_chunks(corpus, tokenizer, seq_len, True)
    if not n_train or not n_val: raise ValueError("split produced an empty cache; corpus needs more articles")
    temp_train, temp_val = train.with_suffix(".npy.part"), val.with_suffix(".npy.part")
    try:
        _write_chunks(corpus, tokenizer, temp_train, n_train, seq_len, False)
        _write_chunks(corpus, tokenizer, temp_val, n_val, seq_len, True)
        _atomic_replace(temp_train, train); _atomic_replace(temp_val, val)
        meta = identity | {"corpus": str(corpus.resolve()), "tokenizer": str(tokenizer_dir.resolve()),
            "n_train_chunks": n_train, "n_val_chunks": n_val, "total_tokens": (n_train+n_val)*seq_len,
            "train_sha256": sha256_file(train), "val_sha256": sha256_file(val)}
        tmp_meta = meta_path.with_suffix(".json.part"); tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2)+"\n")
        _atomic_replace(tmp_meta, meta_path)
        return meta
    except Exception:
        temp_train.unlink(missing_ok=True); temp_val.unlink(missing_ok=True)
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verify"); v.add_argument("--dump", type=Path, required=True); v.add_argument("--manifest", type=Path)
    e = sub.add_parser("extract"); e.add_argument("--dump", type=Path, required=True); e.add_argument("--manifest", type=Path); e.add_argument("--output", type=Path, required=True); e.add_argument("--max-articles", type=int, default=DEFAULT_MAX_ARTICLES); e.add_argument("--max-text-bytes", type=int, default=DEFAULT_MAX_TEXT_BYTES)
    t = sub.add_parser("train-tokenizer"); t.add_argument("--corpus", type=Path, required=True); t.add_argument("--output", type=Path, required=True); t.add_argument("--vocab-size", type=int, default=16000)
    c = sub.add_parser("build-cache"); c.add_argument("--corpus", type=Path, required=True); c.add_argument("--tokenizer", type=Path, required=True); c.add_argument("--cache-dir", type=Path, required=True); c.add_argument("--seq-len", type=int, default=256)
    a = p.parse_args()
    if a.command == "verify": result = verify_dump(a.dump, a.manifest)
    elif a.command == "extract": result = extract(a.dump, a.output, a.manifest, a.max_articles, a.max_text_bytes)
    elif a.command == "train-tokenizer": result = train_tokenizer(a.corpus, a.output, a.vocab_size)
    else: result = build_cache(a.corpus, a.tokenizer, a.cache_dir, a.seq_len)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
