"""
Download and preprocess Wikipedia data for training.

Usage
-----
# Small smoke-test dataset (fast)
python scripts/prepare_data.py --max-articles 100 --max-bytes 1000000

# Medium dataset
python scripts/prepare_data.py --max-articles 10000 --max-bytes 100000000

# Full dataset (WARNING: very slow, many GB)
python scripts/prepare_data.py --max-articles 999999999

The script requires a trained tokenizer at tokenizer/wiki_bpe/tokenizer.json.
Run scripts/train_tokenizer.py first.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data import build_and_cache_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare Wikipedia dataset")
    parser.add_argument("--max-articles", type=int, default=1000)
    parser.add_argument("--max-bytes", type=int, default=10_000_000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/wiki_bpe")
    parser.add_argument("--cache-dir", type=str, default="data/cache")
    parser.add_argument("--dataset-name", type=str, default="wikimedia/wikipedia")
    parser.add_argument("--dataset-config", type=str, default="20231101.en")
    parser.add_argument(
        "--dataset-revision",
        type=str,
        default="b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    )
    parser.add_argument("--train-split", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Rebuild even if cached")
    args = parser.parse_args()

    train, val = build_and_cache_dataset(
        tokenizer_path=args.tokenizer_path,
        cache_dir=args.cache_dir,
        seq_len=args.seq_len,
        max_articles=args.max_articles,
        max_text_bytes=args.max_bytes,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_revision=args.dataset_revision,
        train_split=args.train_split,
        seed=args.seed,
        force_rebuild=args.force,
    )
    print(f"\nDataset ready:")
    print(f"  Train: {len(train):,} chunks × {args.seq_len} tokens")
    print(f"  Val:   {len(val):,} chunks × {args.seq_len} tokens")

    # Force-exit to avoid pyarrow ThreadPool::Shutdown hang on macOS.
    import os as _os
    _os._exit(0)


if __name__ == "__main__":
    main()
