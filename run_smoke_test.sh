#!/usr/bin/env bash
# End-to-end smoke test for the progressive-precision diffusion LM.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON_BIN="${PYTHON:-python}"

if ! "$PYTHON_BIN" -c 'import mlx, numpy, tokenizers, datasets, pytest' >/dev/null 2>&1; then
    echo "ERROR: required dependencies are unavailable to $PYTHON_BIN"
    echo "Create/activate a venv and run: pip install -r requirements-dev.txt"
    exit 1
fi

echo "============================================================"
echo "  Progressive Diffusion LM — Smoke Test"
echo "============================================================"

echo "[1/6] Training BPE tokenizer (if needed)..."
if [ -f "tokenizer/wiki_bpe/tokenizer.json" ]; then
    echo "  Tokenizer already exists — skipping."
else
    "$PYTHON_BIN" scripts/train_tokenizer.py \
        --vocab-size 16000 \
        --max-articles 500 \
        --max-bytes 5000000 \
        --output tokenizer/wiki_bpe
fi

echo "[2/6] Preparing small Wikipedia dataset..."
"$PYTHON_BIN" scripts/prepare_data.py \
    --max-articles 100 \
    --max-bytes 1000000 \
    --seq-len 64 \
    --tokenizer-path tokenizer/wiki_bpe \
    --cache-dir data/cache \
    --dataset-revision b04c8d1ceb2f5cd4588862100d08de323dccfbaa \
    --train-split 0.9 \
    --seed 42

echo "[3/6] Running the complete test suite..."
"$PYTHON_BIN" -m pytest -q

echo "[4/6] Training baseline model (50 steps)..."
"$PYTHON_BIN" -m src.train --config configs/smoke_test_baseline.json

echo "[5/6] Training progressive model (50 steps)..."
"$PYTHON_BIN" -m src.train --config configs/smoke_test.json

echo "[6/6] Evaluating matched baseline/progressive fixtures..."
"$PYTHON_BIN" -m src.evaluate \
    --baseline checkpoints/smoke_test_baseline/step_0000050.npz \
    --progressive checkpoints/smoke_test_progressive/step_0000050.npz \
    --config configs/smoke_test.json \
    --eval-steps 20

echo "── Text generation from progressive model ──"
"$PYTHON_BIN" -m src.generate \
    --checkpoint checkpoints/smoke_test_progressive/step_0000050.npz \
    --config configs/smoke_test.json \
    --n-sequences 2 \
    --seq-len 32

echo "============================================================"
echo "  Smoke test PASSED"
echo "============================================================"
