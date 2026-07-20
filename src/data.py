"""
Memory-efficient Wikipedia data loading.

The full English Wikipedia is ~22 GB of text — far too large to load into
16 GB unified memory.  This module streams articles from the Hugging Face
Hub using the `datasets` library with streaming=True so only one shard is
fetched at a time.

Preprocessing pipeline
-----------------------
1. Stream articles from wikimedia/wikipedia (English, 20231101).
2. Tokenise each article's 'text' field with the trained BPE tokenizer.
3. Concatenate all token IDs into one long sequence (language-model style).
4. Split the long sequence into non-overlapping chunks of `seq_len` tokens.
5. Yield batches of chunks as numpy int32 arrays.

Limits
------
  --max-articles N    : stop after N articles
  --max-text-bytes B  : stop after accumulating B bytes of text

These limits are critical on memory-constrained hardware.  The default
smoke-test values are deliberately tiny.
"""

import os
import json
import random
import hashlib
import numpy as np
from pathlib import Path
from typing import Iterator, Optional


def load_tokenizer(tokenizer_path: str):
    """Load a HuggingFace BPE tokenizer from disk."""
    from tokenizers import Tokenizer
    tok_file = Path(tokenizer_path) / "tokenizer.json"
    if not tok_file.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tok_file}.\n"
            "Run: python scripts/train_tokenizer.py --help"
        )
    return Tokenizer.from_file(str(tok_file))


def stream_wikipedia_tokens(
    tokenizer_path: str,
    max_articles: int = 1000,
    max_text_bytes: int = 10_000_000,
    dataset_name: str = "wikimedia/wikipedia",
    dataset_config: str = "20231101.en",
    dataset_revision: str | None = None,
    seed: int = 42,
) -> list[list[int]]:
    """
    Return a list of tokenised Wikipedia articles.

    Each element is a flat list of integer token IDs for one article.

    The corpus is materialised as a list (rather than yielded from a
    generator) so that the HuggingFace streaming iterable is fully closed
    before the caller processes the data, preventing pyarrow
    ThreadPool::Shutdown from hanging on process exit on macOS.
    """
    from datasets import load_dataset

    tokenizer = load_tokenizer(tokenizer_path)
    eos_id = tokenizer.token_to_id("[EOS]") or 2

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split="train",
        streaming=True,
        revision=dataset_revision,
    )

    articles: list[list[int]] = []
    try:
        n_articles = 0
        n_bytes = 0

        for example in dataset:
            text = example.get("text", "")
            if not text:
                continue

            n_bytes += len(text.encode("utf-8"))
            if n_bytes > max_text_bytes:
                break

            encoding = tokenizer.encode(text)
            ids = encoding.ids + [eos_id]
            articles.append(ids)

            n_articles += 1
            if n_articles >= max_articles:
                break
    finally:
        # Explicitly close the HuggingFace streaming iterable to release
        # pyarrow ThreadPool resources.  Without this, the process can
        # hang on exit on macOS because pyarrow::ThreadPool::Shutdown blocks.
        _close_iterable_dataset(dataset)

    return articles


def _close_iterable_dataset(dataset) -> None:
    """Best-effort cleanup of a HuggingFace IterableDataset."""
    for target in (dataset, getattr(dataset, "_ex_iterable", None)):
        if target is None:
            continue
        close = getattr(target, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def build_chunks(
    token_stream,
    seq_len: int,
) -> list[list[int]]:
    """
    Concatenate all token IDs and split into fixed-length chunks.
    Incomplete final chunk is discarded.
    """
    buffer: list[int] = []
    chunks: list[list[int]] = []

    for ids in token_stream:
        buffer.extend(ids)
        while len(buffer) >= seq_len:
            chunks.append(buffer[:seq_len])
            buffer = buffer[seq_len:]

    return chunks


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_and_cache_dataset(
    tokenizer_path: str,
    cache_dir: str,
    seq_len: int,
    max_articles: int = 1000,
    max_text_bytes: int = 10_000_000,
    dataset_name: str = "wikimedia/wikipedia",
    dataset_config: str = "20231101.en",
    dataset_revision: str | None = None,
    train_split: float = 0.95,
    seed: int = 42,
    force_rebuild: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build or load cached numpy arrays of token chunks.

    Returns
    -------
    train_data : (N_train, seq_len) int32 numpy array
    val_data   : (N_val, seq_len) int32 numpy array
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    tokenizer_file = Path(tokenizer_path) / "tokenizer.json"
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_file}")
    tokenizer_sha256 = _sha256_file(tokenizer_file)
    provenance = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "dataset_revision": dataset_revision,
        "tokenizer_sha256": tokenizer_sha256,
        "seq_len": seq_len,
        "max_articles": max_articles,
        "max_text_bytes": max_text_bytes,
        "train_split": train_split,
        "seed": seed,
    }
    identity = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    key = f"seq{seq_len}_{identity}"
    train_file = cache_path / f"train_{key}.npy"
    val_file = cache_path / f"val_{key}.npy"
    meta_file = cache_path / f"meta_{key}.json"

    if train_file.exists() and val_file.exists() and meta_file.exists() and not force_rebuild:
        with open(meta_file) as f:
            meta = json.load(f)
        metadata_matches = all(meta.get(k) == v for k, v in provenance.items())
        checksums_match = (
            meta.get("train_sha256") == _sha256_file(train_file)
            and meta.get("val_sha256") == _sha256_file(val_file)
        )
        if metadata_matches and checksums_match:
            print(f"Loading cached dataset from {cache_dir}")
            train_data = np.load(str(train_file))
            val_data = np.load(str(val_file))
            print(f"  Train chunks: {len(train_data):,}  Val chunks: {len(val_data):,}")
            print(f"  Total tokens: {meta['total_tokens']:,}")
            return train_data, val_data
        print(f"Cache metadata/checksum mismatch for {key}; rebuilding")

    print(f"Building dataset (max_articles={max_articles}, max_text_bytes={max_text_bytes:,})")
    token_stream = stream_wikipedia_tokens(
        tokenizer_path=tokenizer_path,
        max_articles=max_articles,
        max_text_bytes=max_text_bytes,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_revision=dataset_revision,
        seed=seed,
    )
    chunks = build_chunks(token_stream, seq_len)
    print(f"  Built {len(chunks):,} chunks of {seq_len} tokens each")

    if len(chunks) == 0:
        raise RuntimeError(
            "No chunks built — the tokenizer may not be trained yet or the "
            "dataset download failed."
        )

    # Shuffle and split
    rng = random.Random(seed)
    rng.shuffle(chunks)
    n_train = int(len(chunks) * train_split)
    train_chunks = chunks[:n_train]
    val_chunks = chunks[n_train:]

    train_data = np.array(train_chunks, dtype=np.int32)
    val_data = np.array(val_chunks, dtype=np.int32) if val_chunks else train_data[:1]

    np.save(str(train_file), train_data)
    np.save(str(val_file), val_data)

    meta = {
        **provenance,
        "n_train_chunks": len(train_data),
        "n_val_chunks": len(val_data),
        "total_tokens": (len(train_data) + len(val_data)) * seq_len,
        "train_sha256": _sha256_file(train_file),
        "val_sha256": _sha256_file(val_file),
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Train: {len(train_data):,}  Val: {len(val_data):,}  "
          f"Total tokens: {meta['total_tokens']:,}")
    return train_data, val_data


class BatchIterator:
    """
    Infinite shuffling batch iterator over numpy token chunks.
    Memory-efficient: operates on numpy arrays, converts to MLX only in batches.
    """

    def __init__(self, data: np.ndarray, batch_size: int, seed: int = 42):
        self.data = data
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self._indices = np.arange(len(data))
        self._pos = len(data)  # force immediate shuffle

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if self._pos + self.batch_size > len(self._indices):
            self.rng.shuffle(self._indices)
            self._pos = 0
        batch_idx = self._indices[self._pos : self._pos + self.batch_size]
        self._pos += self.batch_size
        return self.data[batch_idx]  # (B, L) numpy int32

    def __len__(self) -> int:
        return len(self.data) // self.batch_size
