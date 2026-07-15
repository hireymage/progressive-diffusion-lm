# Progressive-Precision Diffusion Language Model

> **Experimental proof-of-concept** — not a production chatbot.

## Research Hypothesis

A diffusion language model may not need every refinement step to use
high-precision weights. Early denoising steps only need to get the coarse
token layout right (cheap with 1-bit weights); later refinement steps
need to disambiguate between similar tokens (requires higher precision).

**Progressive precision schedule:**

| Refinement step | Weight precision |
|:-:|:-:|
| 1 – 4 (coarse) | 1-bit (binary) |
| 5 – 6 (middle) | 2-bit |
| 7 – 8 (fine)   | 4-bit |

The schedule is fully configurable without editing model source code.

**Key question:** Can progressive precision approach the quality of a full-precision
baseline while dramatically reducing theoretical memory and compute requirements?

---

## Architecture

```
Input tokens (with [MASK] at noised positions)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Token Embedding  (vocab_size+1, d_model)  [full prec]  │
│  Position Embedding (max_seq_len, d_model) [full prec]  │
│  Step Embedding  sinusoidal(mask_rate) → MLP  [f.p.]    │
│         ADD ───────────────────────────────────────────  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  TransformerBlock × N  (bidirectional attention)   │  │
│  │                                                    │  │
│  │  LayerNorm → MultiHeadAttention → residual         │  │
│  │             ├─ Q_proj  (QuantizedLinear)           │  │
│  │             ├─ K_proj  (QuantizedLinear)           │  │
│  │             ├─ V_proj  (QuantizedLinear)           │  │
│  │             └─ out_proj (QuantizedLinear)          │  │
│  │                                                    │  │
│  │  LayerNorm → MLP → residual                        │  │
│  │             ├─ ff1  (QuantizedLinear)              │  │
│  │             └─ ff2  (QuantizedLinear)              │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  LayerNorm → LM head (Linear, full prec)                 │
└─────────────────────────────────────────────────────────┘
         │
         ▼
   Logits (B, L, vocab_size)
```

All `QuantizedLinear` layers share one set of **full-precision master weights**
(float32). During the forward pass, weights are quantised to the current
precision level using the **Straight-Through Estimator (STE)**, so gradients
flow back to the master weights unchanged.

---

## How Progressive Precision Works

### Weight Quantisation

| Precision | Representation |
|:-:|:-:|
| **1-bit (binary)** | `sign(W)` → `{-1, +1}` × per-row `mean(|W|)` scale |
| **2-bit** | Symmetric 3-level: `{-1, 0, +1}` × `max(|W|)` per row |
| **4-bit** | Symmetric 15-level: codes `{-7…7}` × `max(|W|)/7` per row |
| **16-bit** | Full-precision pass-through (baseline mode) |

The 1-bit representation is **true binary {-1, +1}** — not ternary.
Zero weights map to +1 by convention.

### Straight-Through Estimator (STE)

```python
w_ste = w + stop_gradient(quantize(w) - w)
# Forward:  w_ste = quantize(w)   ← uses quantised weights
# Backward: dL/dw = dL/dw_ste    ← identity, gradient to latent weights
```

### Training

For each batch:
1. Sample mask rate `m ~ Uniform(0.1, 1.0)`
2. Mask each token position with probability `m`
3. Map `m` to a step index → look up precision from schedule
4. Set all `QuantizedLinear` layers to that precision
5. Run forward pass with quantised weights (STE active)
6. Compute cross-entropy loss only at masked positions
7. Backpropagate to master weights

### Inference (Generation)

Start from a fully masked sequence.  For step `i = 1 … T`:
1. Set model precision to `schedule[i-1]`
2. Predict token distributions at all masked positions
3. Unmask the top-`k` highest-confidence positions (ordered by `max softmax`)
4. Fix those tokens; repeat

---

## How Diffusion Language Modelling Works

This project uses **absorbing / masked diffusion** (similar to MDLM, BERT-style).

- **Noise**: randomly replace tokens with `[MASK]` at rate `t ∈ [0,1]`
- **Denoise**: train the model to predict the original token at every masked position
- **Generation**: iteratively reveal tokens from fully masked → clean

Unlike autoregressive models, ALL token positions are updated in parallel at each refinement step, enabling true parallel generation.

---

## Why MLX

- Native Apple Silicon / Metal acceleration
- Unified memory — no CPU↔GPU copy overhead
- Lazy evaluation graph enables efficient gradient computation
- Runs on MacBook / Mac Mini with 16 GB memory without CUDA

---

## Hardware Assumptions

- **Tested on**: Apple Silicon M4, 16 GB unified memory, macOS
- **Requirements**: macOS 13.5+ with Apple Silicon (M1/M2/M3/M4)
- **Does not require**: CUDA, NVIDIA GPU, GGUF, llama.cpp

---

## Installation

```bash
git clone <repo>
cd progressive-diffusion-lm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Start — Smoke Test

Runs the full pipeline end-to-end on a tiny dataset (~2–5 minutes):

```bash
./run_smoke_test.sh
```

This will:
1. Train a BPE tokenizer on 500 Wikipedia articles
2. Prepare a dataset from 100 articles
3. Run all unit tests
4. Train a baseline model (50 steps)
5. Train the progressive model (50 steps)
6. Compare their validation metrics
7. Generate sample text

---

## Step-by-Step Commands

### 1. Train tokenizer

```bash
# Small (smoke test)
python scripts/train_tokenizer.py \
    --vocab-size 16000 \
    --max-articles 500 \
    --max-bytes 5000000

# Larger (better coverage)
python scripts/train_tokenizer.py \
    --vocab-size 16000 \
    --max-articles 10000 \
    --max-bytes 100000000
```

### 2. Prepare dataset

```bash
# Tiny (fast validation)
python scripts/prepare_data.py --max-articles 100 --max-bytes 1000000

# Medium (10k articles)
python scripts/prepare_data.py --max-articles 10000 --max-bytes 100000000

# Large (requires patience and disk space)
python scripts/prepare_data.py --max-articles 100000 --max-bytes 1000000000
```

### 3. Run unit tests

```bash
python tests/test_quantization.py
python tests/test_model.py
python tests/test_diffusion.py
python tests/test_training.py
```

### 4. Train baseline model

```bash
# Smoke test (50 steps)
python -m src.train --config configs/smoke_test_baseline.json

# Full training
python -m src.train --config configs/baseline.json
```

### 5. Train progressive model

```bash
# Smoke test (50 steps)
python -m src.train --config configs/smoke_test.json

# Full training
python -m src.train --config configs/progressive_1_2_4.json
```

### 6. Compare models

```bash
python -m src.evaluate \
    --baseline   checkpoints/baseline/step_0010000.npz \
    --progressive checkpoints/progressive_1_2_4/step_0010000.npz \
    --config configs/progressive_1_2_4.json \
    --eval-steps 100 \
    --measure-speed
```

### 7. Generate text

```bash
python -m src.generate \
    --checkpoint checkpoints/progressive_1_2_4/step_0010000.npz \
    --config     configs/progressive_1_2_4.json \
    --n-sequences 4 \
    --seq-len 128
```

---

## Configuration

All configuration is in JSON files under `configs/`. Key parameters:

```json
{
  "model": {
    "d_model": 512,
    "n_layers": 6,
    "n_heads": 8,
    "d_ff": 2048,
    "max_seq_len": 256,
    "n_diffusion_steps": 8,
    "precision_schedule": [1, 1, 1, 1, 2, 2, 4, 4],
    "model_type": "progressive"
  },
  "data": {
    "max_articles": 50000,
    "max_text_bytes": 500000000,
    "seq_len": 256
  },
  "train": {
    "batch_size": 8,
    "learning_rate": 3e-4,
    "max_steps": 10000
  }
}
```

To change the precision schedule, edit `precision_schedule` in the config JSON — no code changes needed.

---

## Project Structure

```
progressive-diffusion-lm/
├── src/
│   ├── config.py          Configuration dataclasses
│   ├── quantization.py    QAT layers (1-bit/2-bit/4-bit + STE)
│   ├── model.py           DiffusionLM Transformer architecture
│   ├── diffusion.py       Masking, loss, and generation
│   ├── data.py            Wikipedia streaming + cached dataset
│   ├── train.py           Training loop with checkpointing
│   ├── evaluate.py        Evaluation and model comparison
│   └── generate.py        Text generation CLI
├── scripts/
│   ├── train_tokenizer.py BPE tokenizer training
│   └── prepare_data.py    Dataset download/preprocessing
├── configs/
│   ├── smoke_test.json           Tiny progressive (50 steps)
│   ├── smoke_test_baseline.json  Tiny baseline (50 steps)
│   ├── progressive_1_2_4.json    Full progressive run
│   └── baseline.json             Full baseline run
├── tests/
│   ├── test_quantization.py
│   ├── test_model.py
│   ├── test_diffusion.py
│   └── test_training.py
├── tokenizer/wiki_bpe/    Trained tokenizer (gitignored)
├── data/cache/            Cached numpy token chunks (gitignored)
├── checkpoints/           Model checkpoints (gitignored)
├── run_smoke_test.sh      One-command end-to-end test
├── requirements.txt
└── README.md
```

---

## Known Limitations

### Critical (simulation vs. real hardware)

> **The most important limitation in this project:**
>
> All 1-bit, 2-bit, and 4-bit operations are **SIMULATED using float32 MLX
> operations** via the Straight-Through Estimator. Quantised weights are
> computed as float32 approximations; no integer arithmetic is used.
>
> This means:
> - Theoretical storage savings (8× for 1-bit vs float32) are real if weights
>   were stored as packed integers.
> - Wall-clock speed does **NOT** reflect real 1-bit hardware performance.
>   In fact, simulating 1-bit in float32 is *slower* than native float32.
> - Real speedups require:
>   - Custom low-bit Metal shaders for Apple GPU
>   - Apple Neural Engine kernels supporting 1-bit MAC operations
>   - Or dedicated 1-bit hardware (e.g., 1-bit ASIC or NPU)

### Model quality

- 50-step smoke-test models are far from converged; generated text is incoherent.
- Full training (10k+ steps on 50k+ articles) is needed for meaningful quality comparison.
- The architecture is a research scaffold, not optimised for quality.

### 2-bit has 3 levels (not 4)

Due to symmetric signed integer rounding, the 2-bit mode produces 3 distinct
levels `{-1, 0, +1} × scale` (≈ 1.58 effective bits). A 4-level 2-bit scheme
without zero would require asymmetric quantisation, which is left as future work.

### No ternary mode for 1-bit

The project explicitly uses binary `{-1, +1}` 1-bit (not ternary). Per the
research brief, ternary was intentionally not substituted as the primary mode.

### Tokenizer MASK token is out-of-vocabulary

The model's `[MASK]` token ID is `vocab_size` (one position past the tokenizer
vocabulary). This is intentional: the tokenizer's `[MASK]` token (ID 2) is
used in the tokenizer but the diffusion model uses a separate mask sentinel to
avoid ambiguity with regular text.

### Future extensions (designed but not implemented)

The codebase is structured to support:
- Confidence-based routing between precision levels
- Early exit for certain token positions
- Token freezing (fix high-confidence tokens early)
- Adaptive 1-bit → 2-bit → 4-bit within a single refinement step
- Temperature annealing during generation

---

## Theoretical Compression Summary

| Model | Avg bits/step | Theoretical compression vs fp32 |
|:-:|:-:|:-:|
| Baseline | 16 | 1× (fp32 = fp32) |
| Progressive 1→2→4 | 2.0 | **16×** |
| Progressive 4→4→4 | 4.0 | 8× |

*Compression applies to weight storage, not activations or master weights.*

---

*This is experimental research software. No warranty is expressed or implied.*
