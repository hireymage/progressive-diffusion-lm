#!/usr/bin/env bash
# run_smoke_test.sh — End-to-end smoke test for progressive diffusion LM
#
# This script runs the full pipeline on a tiny Wikipedia subset to verify
# everything works before launching a longer training run.
#
# Expected runtime: 2–5 minutes on Apple Silicon M4.
# Uses only ~100 Wikipedia articles and 50 training steps per model.
#
# Usage:
#   ./run_smoke_test.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv not found. Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

echo "============================================================"
echo "  Progressive Diffusion LM — Smoke Test"
echo "============================================================"
echo ""

# ── Step 1: Train tokenizer (if not already trained) ─────────────────────────
echo "[1/6] Training BPE tokenizer (if needed)..."
if [ -f "tokenizer/wiki_bpe/tokenizer.json" ]; then
    echo "  Tokenizer already exists — skipping."
else
    python scripts/train_tokenizer.py \
        --vocab-size 16000 \
        --max-articles 500 \
        --max-bytes 5000000 \
        --output tokenizer/wiki_bpe
fi
echo ""

# ── Step 2: Prepare tiny dataset ─────────────────────────────────────────────
echo "[2/6] Preparing small Wikipedia dataset (100 articles)..."
python scripts/prepare_data.py \
    --max-articles 100 \
    --max-bytes 1000000 \
    --seq-len 64 \
    --tokenizer-path tokenizer/wiki_bpe \
    --cache-dir data/cache
echo ""

# ── Step 3: Run unit tests ────────────────────────────────────────────────────
echo "[3/6] Running unit tests..."
python tests/test_quantization.py
python tests/test_model.py
python tests/test_diffusion.py
python tests/test_training.py
echo ""

# ── Step 4: Train baseline (50 steps) ────────────────────────────────────────
echo "[4/6] Training baseline model (50 steps)..."
python -m src.train --config configs/smoke_test_baseline.json
echo ""

# ── Step 5: Train progressive model (50 steps) ───────────────────────────────
echo "[5/6] Training progressive 1→2→4-bit model (50 steps)..."
python -m src.train --config configs/smoke_test.json
echo ""

# ── Step 6: Evaluate & compare ───────────────────────────────────────────────
echo "[6/6] Evaluating and comparing models..."
python -m src.evaluate \
    --baseline  checkpoints/smoke_test_baseline/step_0000050.npz \
    --progressive checkpoints/smoke_test_progressive/step_0000050.npz \
    --config    configs/smoke_test.json \
    --eval-steps 20

echo ""
echo "── Text generation from progressive model ──"
python -m src.generate \
    --checkpoint checkpoints/smoke_test_progressive/step_0000050.npz \
    --config     configs/smoke_test.json \
    --n-sequences 2 \
    --seq-len 32

echo ""
echo "============================================================"
echo "  Smoke test PASSED"
echo "  All pipeline stages completed successfully."
echo ""
echo "  Next steps:"
echo "    Full progressive training:  python -m src.train --config configs/progressive_1_2_4.json"
echo "    Full baseline training:     python -m src.train --config configs/baseline.json"
echo "============================================================"
