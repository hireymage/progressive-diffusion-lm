"""
Train a BPE tokenizer on English Wikipedia.

Usage
-----
python scripts/train_tokenizer.py \\
    --vocab-size 16000 \\
    --max-articles 5000 \\
    --max-bytes 50000000 \\
    --output tokenizer/wiki_bpe

The tokenizer is saved as tokenizer/wiki_bpe/tokenizer.json and can be
loaded with tokenizers.Tokenizer.from_file().

Special tokens
--------------
[PAD]  → id 0   padding
[UNK]  → id 1   unknown byte sequences
[MASK] → id 2   diffusion mask token (the core "noise" token)
[BOS]  → id 3   beginning of sequence
[EOS]  → id 4   end of sequence
"""

import os
import sys
import argparse
from pathlib import Path


def stream_text(max_articles: int, max_bytes: int, dataset_config: str):
    """Yield text strings from Wikipedia for tokenizer training."""
    from datasets import load_dataset

    print(f"Streaming Wikipedia for tokenizer training "
          f"(max_articles={max_articles}, max_bytes={max_bytes:,})")
    dataset = load_dataset(
        "wikimedia/wikipedia",
        dataset_config,
        split="train",
        streaming=True,
    )

    n = 0
    total_bytes = 0
    for ex in dataset:
        text = ex.get("text", "")
        if not text:
            continue
        total_bytes += len(text.encode("utf-8"))
        yield text
        n += 1
        if n % 500 == 0:
            print(f"  {n} articles / {total_bytes/1e6:.1f} MB streamed...")
        if n >= max_articles or total_bytes >= max_bytes:
            break
    print(f"Tokenizer training corpus: {n} articles, {total_bytes/1e6:.1f} MB")


def train_tokenizer(
    vocab_size: int,
    max_articles: int,
    max_bytes: int,
    output_dir: str,
    dataset_config: str = "20231101.en",
):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tok_file = output_path / "tokenizer.json"
    if tok_file.exists():
        print(f"Tokenizer already exists at {tok_file}")
        print("Delete it to retrain.  Loading existing tokenizer.")
        return

    print(f"Training BPE tokenizer with vocab_size={vocab_size}")

    # Build BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # Byte-level pre-tokeniser — handles any Unicode text robustly
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["[PAD]", "[UNK]", "[MASK]", "[BOS]", "[EOS]"]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
    )

    # Train from iterator to avoid loading all text into RAM
    text_iter = stream_text(max_articles, max_bytes, dataset_config)
    tokenizer.train_from_iterator(text_iter, trainer=trainer)

    # Post-processor: add BOS/EOS
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[BOS]:0 $A:0 [EOS]:0",
        pair=f"[BOS]:0 $A:0 [EOS]:0 [BOS]:1 $B:1 [EOS]:1",
        special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id)],
    )

    tokenizer.save(str(tok_file))
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"\nTokenizer trained and saved to {tok_file}")
    print(f"Actual vocabulary size: {actual_vocab_size:,}")
    print(f"Special tokens: {special_tokens}")
    print(f"  [PAD]  → {tokenizer.token_to_id('[PAD]')}")
    print(f"  [UNK]  → {tokenizer.token_to_id('[UNK]')}")
    print(f"  [MASK] → {tokenizer.token_to_id('[MASK]')}")
    print(f"  [BOS]  → {tokenizer.token_to_id('[BOS]')}")
    print(f"  [EOS]  → {tokenizer.token_to_id('[EOS]')}")

    # Save vocab info
    info = {
        "vocab_size": actual_vocab_size,
        "model_vocab_size": actual_vocab_size,
        "mask_token_id_in_model": actual_vocab_size,  # one past end
        "special_tokens": {tok: tokenizer.token_to_id(tok) for tok in special_tokens},
    }
    import json
    with open(output_path / "vocab_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nVocab info saved to {output_path / 'vocab_info.json'}")
    print("NOTE: The model's MASK_TOKEN is vocab_size (one past the tokenizer vocab).")


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on Wikipedia")
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--max-articles", type=int, default=5000,
                        help="Max Wikipedia articles for training")
    parser.add_argument("--max-bytes", type=int, default=50_000_000,
                        help="Max bytes of text for training")
    parser.add_argument("--output", type=str, default="tokenizer/wiki_bpe")
    parser.add_argument("--dataset-config", type=str, default="20231101.en")
    args = parser.parse_args()

    train_tokenizer(
        vocab_size=args.vocab_size,
        max_articles=args.max_articles,
        max_bytes=args.max_bytes,
        output_dir=args.output,
        dataset_config=args.dataset_config,
    )


if __name__ == "__main__":
    main()
