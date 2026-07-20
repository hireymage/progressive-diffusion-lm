# Progressive-Precision Diffusion Language Model

> **Experimental proof-of-concept** on Apple Silicon. Not a production chatbot.

A research project investigating whether a masked diffusion language model can train effectively with extremely low-bit weight representations, and whether assigning lower precision to high-noise denoising steps and higher precision to fine-grained steps (a "progressive precision schedule") provides any benefit.

---

## Research Hypothesis

Early denoising steps (high noise, coarse structure) may only need binary (1-bit) weights. Late refinement steps (low noise, token disambiguation) benefit from higher precision (4-bit). A single set of FP32 master weights can be evaluated at different precisions across diffusion steps via runtime switching — no separate models needed.

**Key finding so far** (from 18 completed ablation runs, 6 variants × 3 seeds × 10k steps):
- Binary (const_1bit) ranks first with mean best_val_loss 7.4336 vs. baseline 7.4434 (0.01 nats better)
- Progressive schedule [1,1,1,1,2,2,4,4] is statistically tied with the FP32 baseline
- All differences are small (range 0.001–0.025 nats) and seed variance is large — no definitive ranking at 3 seeds
- **Critical**: all low-bit operations are SIMULATED in FP32 via Straight-Through Estimation — no real memory or speed benefit at present

---

## Architecture

Bidirectional Transformer (28.3M parameters):
- d_model=512, n_layers=6, n_heads=8, d_ff=2048, max_seq_len=256
- Every linear projection is a `QuantizedLinear` layer supporting runtime precision switching
- Embeddings and LayerNorm remain float32; only linear weights are quantized
- Weight tying: LM head shares token embedding matrix
- Noise-level conditioning: sinusoidal embedding of mask rate added to all positions

Quantization schemes (all simulated in float32 via STE):

| bits | Scheme | Levels | Eff. bits |
|---|---|---|---|
| 1 | Binary | 2: {-1, +1} × mean(\|w\|) | 1.0 |
| 2 | True 2-bit | 4: {-3,-1,+1,+3} × step | 2.0 |
| 3 | True 3-bit | 8: {-7,...,+7} × step | 3.0 |
| 4 | True 4-bit | 16: {-15,...,+15} × step | 4.0 |
| 16 | FP32 | identity | 16.0 |
| 0 | Ternary (optional) | 3: {-1,0,+1} × max(\|w\|) | ~1.585 |

---

## Quick Start

```bash
# 1. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. End-to-end smoke test (2–5 minutes, verifies everything works)
./run_smoke_test.sh

# 3. Full training (requires prepared data — see below)
python -m src.train --config configs/full_baseline.json
python -m src.train --config configs/full_progressive_1_2_4.json
```

---

## Step-by-Step Commands

### Prepare data

```bash
# Train BPE tokenizer (run once)
python scripts/train_tokenizer.py --vocab-size 16000 --max-articles 500 --max-bytes 5000000

# Prepare dataset (50k articles, ~69M tokens, caches to data/cache/)
python scripts/prepare_data.py --max-articles 50000 --max-bytes 500000000 --seq-len 256
```

### Train models

```bash
# Baseline (FP32, no quantization)
python -m src.train --config configs/full_baseline.json

# Progressive precision [1-bit → 2-bit → 4-bit]
python -m src.train --config configs/full_progressive_1_2_4.json

# Any custom config
python -m src.train --config configs/<your_config>.json
```

### Run the full ablation study

```bash
# Screening (3k steps × 18 runs, ~9h)
python scripts/ablation_study.py --phase screen

# Full ablation (10k steps × 18 runs, ~45h)
python scripts/ablation_study.py --phase full --resume

# Analysis only (requires existing results)
python scripts/ablation_study.py --analyze-only --phase full
```

### Reproduce the PTQ study (completed; commands below rerun it)

```bash
# Full study: train 3 baselines + run PTQ evals (~6-7h)
python scripts/ptq_study.py

# Dry run: print plan without executing
python scripts/ptq_study.py --dry-run

# Skip training if baselines already trained
python scripts/ptq_study.py --skip-training
```

### Evaluate and generate

```bash
# Compare baseline vs. progressive
python -m src.evaluate \
    --baseline checkpoints/full_baseline/step_0010000.npz \
    --progressive checkpoints/full_progressive_1_2_4/step_0010000.npz \
    --config configs/full_progressive_1_2_4.json --eval-steps 100

# Generate text
python -m src.generate \
    --checkpoint checkpoints/full_progressive_1_2_4/step_0010000.npz \
    --config configs/full_progressive_1_2_4.json \
    --n-sequences 4 --seq-len 128
```

---

## Project Status

| Experiment | Status | Runs |
|---|---|---|
| Smoke tests | DONE | 2 variants, 50 steps |
| Short experiments (500 steps) | DONE | 3 variants, seed=42 |
| Full initial comparison (10k steps) | DONE | 2 variants, seed=42 |
| Ablation screening (3k steps) | DONE | 6 variants × 3 seeds = 18 runs |
| **Full ablation (10k steps)** | **DONE** | **6 variants × 3 seeds = 18 runs** |
| PTQ study | **DONE** | 18/18 Q1/Q2/Q3/Q4/FP32/ternary evaluations across 3 seeds |

See `PROJECT_DOCUMENTATION.md` for full technical documentation including all numerical results, quantization scheme details, methodological limitations, and research roadmap.

---

## Requirements and Hardware

- macOS 13.5+ with Apple Silicon (M1/M2/M3/M4)
- 16 GB unified memory (tested on M4 16 GB)
- No CUDA, no NVIDIA GPU required
- Python dependencies: `mlx>=0.21.0`, `tokenizers`, `datasets`, `numpy`, `tqdm`

---

## Known Limitations

**Most important**: All 1-bit, 2-bit, 3-bit, and 4-bit operations are **simulated in float32** via Straight-Through Estimation. No packed integer arithmetic is used. A packed Q1 `QuantizedLinear` weight tensor alone would be 32× smaller than FP32, but embeddings, normalization, biases, and other non-quantized parameters prevent that ratio from applying to the whole model. For the current progressive schedule, the storage report estimates only ~2.67× whole-model compression vs. FP32. None of this compression is realized by the current implementation, and wall-clock speed does NOT reflect real low-bit hardware performance.

Other limitations:
- Small model (~28M params) and dataset (~69M tokens) — findings may not generalize to production scale
- The original full ablation has only 3 seeds; two later paired replications strengthen the baseline/Q1/progressive comparison but do not cover every precision variant
- Apple Silicon non-determinism: same seed in different sessions may produce slightly different val_loss
- const_4bit ablation used an old 15-level Q4 scheme; PTQ study uses the new 16-level scheme — Q4 comparisons carry a caveat

---

*Experimental research software. No warranty expressed or implied.*
