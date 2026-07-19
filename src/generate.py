"""
Text generation via progressive masked-diffusion refinement.

Usage
-----
python -m src.generate \\
    --checkpoint checkpoints/progressive/step_0010000.npz \\
    --config configs/progressive_1_2_4.json \\
    --n-sequences 4 \\
    --seq-len 64

Prints generated text decoded with the trained BPE tokenizer.
"""

import argparse
import time
from pathlib import Path

import mlx.core as mx

from .config import ExperimentConfig
from .model import DiffusionLM
from .diffusion import generate
from .data import load_tokenizer
from .train import load_checkpoint


def decode_sequence(token_ids: list[int], tokenizer) -> str:
    """Decode a list of token IDs to a string, skipping special tokens."""
    from tokenizers import Tokenizer
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return text


def main():
    parser = argparse.ArgumentParser(description="Generate text with trained diffusion LM")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n-sequences", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--show-steps", action="store_true",
                        help="Print intermediate refinement steps")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_json(args.config)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = DiffusionLM(cfg.model)
    step = load_checkpoint(model, args.checkpoint)
    print(f"  Loaded checkpoint at step {step}")
    print(f"  Model type: {cfg.model.model_type}")
    print(f"  Precision schedule: {cfg.model.precision_schedule}")

    # Load tokenizer
    tokenizer = load_tokenizer(cfg.data.tokenizer_path)
    mask_token_id = cfg.model.mask_token_id()

    precision_schedule = (
        cfg.model.precision_schedule
        if cfg.model.model_type == "progressive"
        else [16] * cfg.model.n_diffusion_steps
    )

    print(f"\nGenerating {args.n_sequences} sequences of length {args.seq_len}...")
    print(f"Diffusion steps: {len(precision_schedule)}")
    print(f"Temperature: {args.temperature}")
    print()

    t0 = time.time()
    token_ids = generate(
        model,
        seq_len=args.seq_len,
        mask_token_id=mask_token_id,
        precision_schedule=precision_schedule,
        temperature=args.temperature,
        batch_size=args.n_sequences,
    )
    mx.eval(token_ids)
    elapsed = time.time() - t0

    print(f"Generated in {elapsed:.2f}s  "
          f"({args.n_sequences * args.seq_len / elapsed:.0f} tokens/s)")
    print()

    # Decode and print
    for i in range(args.n_sequences):
        ids = token_ids[i].tolist()
        # Filter out mask tokens that might remain
        ids = [t for t in ids if t != mask_token_id]
        text = decode_sequence(ids, tokenizer)
        print(f"━━━ Sample {i+1} ━━━")
        print(text)
        print()


if __name__ == "__main__":
    main()
