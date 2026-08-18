# Progressive-Precision Diffusion Language Model — Technical Research Documentation

[English](PROJECT_DOCUMENTATION.md) | [Čeština](PROJECT_DOCUMENTATION.cs.md)

*Generated 2026-07-18. All numbers read from actual result files in the repository.*

---

## 1. PROJECT OVERVIEW

### What it is

Progressive-Precision Diffusion LM is a research project investigating whether a masked diffusion language model can be trained effectively with extremely low-bit weight representations, and whether assigning different precisions to different denoising steps (a "progressive precision schedule") provides any benefit over constant-precision alternatives.

The model is a bidirectional Transformer trained on the masked-diffusion objective (absorbing-state diffusion over discrete tokens). It is distinct from autoregressive models: all token positions are predicted in parallel at each denoising step. Progressive precision means that coarse early denoising steps use low-bit weights (e.g., binary 1-bit) while late fine-grained steps use higher precision (e.g., 4-bit), on the hypothesis that coarse steps do not need high arithmetic precision.

### Core motivation and hypothesis

The central hypothesis is that the precision required for denoising correlates with the noise level: high-noise (coarse) steps only need to establish a rough token layout and may tolerate extremely compressed weights, while low-noise (fine) steps need to distinguish between similar tokens and benefit from higher precision. A single model can be evaluated at different precisions across steps, because all QuantizedLinear layers are controlled at runtime via a per-step precision schedule.

A secondary hypothesis is that native low-bit training (training from scratch with low-bit weights via STE) may produce better models than applying the same quantization post-hoc to a high-precision checkpoint. The direct/naive PTQ campaign is complete: 18/18 evaluations across seeds 42/123/7 and Q1/Q2/Q3/Q4/FP32/optional ternary were recovered and verified.

### Critical distinction: simulated vs. real low-bit

**This distinction is fundamental to interpreting all results.**

- **FP32 master weights**: All QuantizedLinear layers store their weights as float32 at all times. These are the parameters that the Adam optimizer updates.
- **Simulated quantization (STE forward pass)**: During the forward pass, weights are passed through `quantize_weights(w, bits)` which returns a float32 approximation of the quantized values. No integer arithmetic is used. This is fully simulated in float32 on Apple Silicon.
- **Fake/simulated quantization**: The STE trick (`w_ste = w + stop_gradient(quantize(w) - w)`) means the backward pass receives identity gradients as if the quantization did not happen. This is standard quantization-aware training (QAT).
- **No real packed low-bit kernels**: There are no custom Metal shaders, no integer MACs, no packed weight storage. All operations execute in float32. Simulating 1-bit in float32 is actually slower than native float32 on current hardware.
- **Theoretical compression is real**: If weights were actually stored as packed integers (not the case here), the compression ratios reported would apply to inference-time weight storage.

### Long-term goal

The long-term research trajectory aims to:
1. Confirm that native low-bit training works (completed with ablation study)
2. Quantify the quality gap between native QAT and post-training quantization (direct/naive PTQ campaign completed; broader replicated native Q3/Q4/ternary comparison remains open)
3. Potentially implement real packed low-bit kernels in MLX to measure actual memory and speed benefits
4. Test binary decomposition hypotheses at larger scale

All results to date are on a small model (~28M parameters) with a limited dataset (~69M tokens). Generalization to larger models is a hypothesis, not a demonstrated result.

---

## 2. RESEARCH QUESTIONS

The following questions were explicitly stated in the ablation study script (`scripts/ablation_study.py`) and PTQ study script (`scripts/ptq_study.py`). Questions with results are marked [HAS DATA]; questions still open are marked [OPEN] or [HYPOTHETICAL].

**Q1: Can a diffusion LM be trained successfully with extremely low-bit weights?** [HAS DATA]
Yes. The const_1bit variant (binary weights, bits=1 at every step) converges and achieves mean best_val_loss of 7.4336 across 3 seeds, which is lower (better) than the baseline mean of 7.4434. Training does not diverge at 1-bit precision.

**Q2: Can native low-bit training match or outperform a high-precision baseline?** [HAS DATA]
The const_1bit variant outperforms the baseline on 2 out of 3 seeds (seeds 7 and 123 beat the baseline per-seed mean; seed 42 does not). The mean difference is 0.0098 nats in favor of const_1bit, but seed variance is large (std ~0.024). The result is suggestive but not conclusive at 3 seeds.

**Q3: Does progressive precision scheduling provide an advantage?** [HAS DATA — INCONCLUSIVE]
prog_1_2_4 (1→2→4) achieves mean best_val_loss 7.4428, which is within 0.0006 of the baseline (7.4434). At 3 seeds with this variance level, the result is indistinguishable from noise.

**Q4: Does precision direction matter (1→2→4 vs 4→2→1)?** [HAS DATA — INCONCLUSIVE]
prog_1_2_4 mean = 7.4428, prog_4_2_1 mean = 7.4571. The coarse-to-fine direction appears slightly better (by 0.014 nats mean), but at 3 seeds this is not statistically established.

**Q5: Is improvement from progressive structure or low-bit regularization?** [HAS DATA — INCONCLUSIVE]
prog_1_2_4 (mean 7.4428) vs const_2bit (mean 7.4586, same average 2.0 effective bits). The progressive structure appears slightly better by 0.016 nats, but the ablation study script classifies differences <0.002 as inconclusive and those <0.02 as "indistinguishable at this scale." The result is in the borderline range.

**Q6: Native low-bit training vs post-training quantization (PTQ)?** [HAS DATA — DIRECT/NAIVE PTQ COMPLETED]
The recovery campaign produced and verified all 18 requested `(seed, bits)` evaluations. Interpretation remains limited by the legacy-Q4/current-Q4 scheme mismatch and by only one native seed each for current true Q3 and ternary.

**Q7: Does native-vs-PTQ quality gap increase at lower bits?** [HAS DATA — INTERPRET WITH CAVEATS]
The completed direct/naive PTQ aggregate addresses this question for Q1/Q2 and approximately for Q4. Current-Q4 lacks a scheme-matched native baseline, while native Q3 and ternary each have only one later seed.

**Q8: Is there a precision threshold where PTQ collapses but native low-bit remains stable?** [HAS DATA — PRELIMINARY]
The completed PTQ aggregate reports the collapse-threshold analysis. Strong inference is still limited because current Q3/ternary native evidence is single-seed and current Q4 lacks a scheme-matched native baseline.

**Q9 (HYPOTHETICAL): Can multiple 1-bit components replace wider precision arithmetic?** [HYPOTHETICAL]
This is a future research hypothesis (binary decomposition). Not implemented or tested.

**Q10 (HYPOTHETICAL): Can progressive precision enable adaptive compute/memory at inference?** [HYPOTHETICAL]
The architecture supports this (model.set_bits() can be called per step), but no adaptive inference system has been built or evaluated.

---

## 3. MODEL ARCHITECTURE

### Hyperparameters (full/ablation model)

| Parameter | Value |
|---|---|
| vocab_size | 16,000 |
| d_model | 512 |
| n_layers | 6 |
| n_heads | 8 |
| head_dim | 64 (= d_model / n_heads) |
| d_ff | 2,048 |
| max_seq_len | 256 |
| dropout | 0.1 |
| n_diffusion_steps | 8 |
| tie_word_embeddings | True |

Source: `configs/full_baseline.json`, `configs/ablation/ablation_baseline_s42_full.json`, confirmed by `results/full_baseline/final_summary.json`.

Note: smoke-test and short-exp configs use smaller architectures (see Section 6).

### Parameter count

Total parameters: **28,295,808** (28.3M).

Breakdown from `results/full_baseline/final_summary.json` (storage section):
- Total params: 28,295,808
- QuantizedLinear weight params (quantized during training/inference): 18,874,368
- Non-quantized params (embeddings, LayerNorm, biases, lm_head_bias): 9,421,440

With weight tying (tie_word_embeddings=True), the LM head shares the token embedding table (vocab_size × d_model = 16,000 × 512 = 8,192,000 parameters), saving ~8M parameters compared to a separate LM head matrix. A small learned bias vector (vocab_size = 16,000 values) is still allocated.

FP32 storage: 113.18 MB (28.3M × 4 bytes).

Training memory estimate (master weights + gradients + Adam m and v states, all FP32): ~452.7 MB.

### Diffusion-specific components

**SinusoidalEmbedding (step_embed)**: Maps the mask rate (a scalar in [0,1], where 1.0 = fully noisy, 0.0 = clean) to a d_model-dimensional conditioning vector. Implementation: sinusoidal encoding → 2-layer MLP (d_model → d_model*2 → d_model) with SiLU activation. This embedding is broadcast to all token positions and added to the token+position embedding before the Transformer blocks.

**Forward diffusion (masking)**: Each token is independently replaced by MASK_TOKEN (= vocab_size = 16,000) with probability equal to the mask rate sampled from Uniform(0.1, 1.0). This is the standard absorbing-state masked diffusion.

**Training objective**: Cross-entropy loss computed only at masked positions. The model predicts the original token given the partially masked sequence and the mask rate as conditioning.

**Precision schedule during training**: Per batch, a mean mask rate is computed across the batch. This is mapped to a step index (step = floor(mean_rate * n_steps), clamped to [0, T-1]). The precision for that step index is looked up from precision_schedule and applied to all QuantizedLinear layers via model.set_bits(bits).

**Inference / generation**: Starting from a fully masked sequence, T denoising steps are executed. At step i, the model uses precision_schedule[i] bits, predicts all positions, and unmasks the top-k highest-confidence (max softmax probability) masked positions. This continues until all positions are filled.

### Attention mechanism

Bidirectional (non-causal) multi-head self-attention. No attention mask is applied during training on non-padded inputs. The attention score softmax is computed in float32 (cast up from input dtype) to avoid numerical instability, then cast back. The rationale for bidirectional attention is that masked diffusion is a fill-in-the-blank task: the model sees all unmasked tokens and must predict masked ones, which requires attending to context in both directions.

### QuantizedLinear

Every linear projection in attention (Q, K, V, out_proj) and every feed-forward layer (ff1, ff2) is a QuantizedLinear. Embeddings (token_embed, pos_embed, lm_head_bias) and LayerNorm layers remain in float32 at all times.

The QuantizedLinear layer:
1. Stores full-precision float32 weights (the master weights updated by Adam)
2. At forward time, calls `ste_quantize(self.weight, self.bits)` to get quantized weights
3. Computes the matrix multiply: `x @ w_quantized.T`
4. Adds float32 bias (if bias=True)

STE implementation: `w_ste = w + stop_gradient(quantize(w) - w)`. The forward pass sees `quantize(w)` (the low-bit approximation). The backward pass sees the identity (gradient flows to `w` unchanged). MLX implements this via `mx.stop_gradient`.

`bits` is a mutable runtime attribute on each QuantizedLinear. Calling `model.set_bits(bits)` updates all QuantizedLinear layers simultaneously. This allows a single model checkpoint to be evaluated at any precision level without reloading weights.

### model_type behavioral difference

- `model_type="baseline"`: The model always uses bits=16 (identity pass-through, no quantization). The precision_schedule stored in config is overridden with [16]*n_diffusion_steps.
- `model_type="progressive"`: The model uses precision_schedule[step_idx] for the current step, which may include bits=1, 2, 3, 4, or 16.

Note: const_1bit through const_4bit variants use model_type="progressive" with a constant schedule (all entries equal). The "baseline" model_type is structurally identical but bypasses all quantization.

### Weight tying

When tie_word_embeddings=True (default and used in all full experiments), the output LM head reuses the token embedding matrix. The projection at forward time is: `logits = x @ token_embed.weight[:vocab_size].T + lm_head_bias`. This saves 8,192,000 parameters compared to a separate Linear layer.

---

## 4. DATASET

### Source

Dataset: `wikimedia/wikipedia`, configuration `20231101.en` (English Wikipedia snapshot, November 2023). Loaded via the Hugging Face `datasets` library with `streaming=True` to avoid loading the full ~22 GB into memory.

### Limits used in main experiments (ablation and full runs)

- `max_articles`: 50,000
- `max_text_bytes`: 500,000,000 (500 MB of raw text)
- `seq_len`: 256

### Tokenizer

- Type: Byte-Pair Encoding (BPE), trained using the HuggingFace `tokenizers` library
- Vocabulary size: 16,000 subword tokens
- Location: `tokenizer/wiki_bpe/tokenizer.json`
- Special tokens: `[PAD]=0, [UNK]=1, [MASK]=2, [BOS]=3, [EOS]=4`
- Note: The diffusion model uses a separate MASK_TOKEN at ID 16,000 (= vocab_size), distinct from the tokenizer's `[MASK]` at ID 2. This avoids ambiguity between "this position was masked by diffusion" and regular text.

### Preprocessing pipeline

1. Stream articles from wikimedia/wikipedia one at a time (streaming=True)
2. Tokenize each article's `text` field and append `[EOS]` token
3. Concatenate all token IDs into one long buffer
4. Split into non-overlapping chunks of `seq_len` tokens (incomplete final chunk discarded)
5. Shuffle all chunks with a fixed random seed
6. Split into train (95%) and validation (5%)
7. Save as numpy int32 arrays to `data/cache/`

### Dataset statistics (main experiments)

From `data/cache/meta_seq256_art50000_bytes500000000.json`:
- Train chunks: **256,180**
- Val chunks: **13,484**
- Total tokens: **69,033,984** (approximately 69M tokens)
- Chunk shape: (256,) per chunk (seq_len=256)

### Train/val split

- Train: 95% (`train_split=0.95`)
- Validation: 5%

### Cache file locations

Main experiment cache:
- `data/cache/train_seq256_art50000_bytes500000000.npy` (~250 MB)
- `data/cache/val_seq256_art50000_bytes500000000.npy` (~13 MB)
- `data/cache/meta_seq256_art50000_bytes500000000.json`

Smoke test / short exp cache:
- `data/cache/train_seq64_art100_bytes1000000.npy` (~800 KB)
- `data/cache/val_seq64_art100_bytes1000000.npy` (~89 KB)
- `data/cache/train_seq128_art100_bytes1000000.npy` (~800 KB)
- `data/cache/val_seq128_art100_bytes1000000.npy` (~89 KB)

---

## 5. QUANTIZATION SCHEMES

All quantization is implemented in `src/quantization.py`. The key design choice is the "uniform no-zero symmetric" scheme for Q1–Q4: levels are always odd multiples of a step, so zero is never a representable value. This avoids the zero-collapse problem in binary-style schemes.

### Complete scheme table

| bits param | Scheme name | Levels | Level values | Scale/step formula | Zero representable | Eff. bits |
|---|---|---|---|---|---|---|
| 1 | Q1 / Binary | 2 | {-1, +1} × scale | scale = mean(|w|) per output-row | No (0 maps to +1) | 1.0 |
| 2 | Q2 / True 2-bit | 4 | {-3, -1, +1, +3} × step | step = max(|w|) / 3 per output-row | No | 2.0 |
| 3 | Q3 / True 3-bit | 8 | {-7, -5, -3, -1, +1, +3, +5, +7} × step | step = max(|w|) / 7 per output-row | No | 3.0 |
| 4 | Q4 / True 4-bit | 16 | {-15, -13, …, -1, +1, …, +15} × step | step = max(|w|) / 15 per output-row | No | 4.0 |
| 16 | FP32 pass-through | continuous | identity (w unchanged) | — | Yes | 16.0 (float32) |
| 0 | Ternary (optional) | 3 | {-1, 0, +1} × scale | scale = max(|w|) per output-row | Yes | ~1.585 (log2(3)) |

### General formula for Q2–Q4

For bits=n (n ∈ {2,3,4}): levels are {±1, ±3, ±5, …, ±(2^n−1)} × step. The magnitude of a quantized weight is computed as `mag = 2·floor(|w_norm|/2) + 1`, capped at `2^n−1`. Boundaries between consecutive levels are at ±2, ±4, … × step.

### bits=16 semantics

`bits=16` (and any value ≥16) is the pass-through case: `ste_quantize` returns `w` directly, and the baseline model effectively trains with full float32 precision. This is used for model_type="baseline" and for the FP32 reference point in the PTQ study.

### bits=0 (Ternary) — OPTIONAL / EXPERIMENTAL

Ternary uses 3 levels {-1, 0, +1} × max(|w|) with a boundary at ±0.5 × scale. It is intentionally separated from the main Q1–Q4 matrix (accessed via the sentinel value bits=0 rather than bits=3, which is now True 3-bit). Ternary is evaluated only when `--include-ternary` is passed to `ptq_study.py`. The original ablation matrix has no ternary variant; one later native ternary run exists at seed 31415 and must remain labelled single-seed evidence.

The `src/model.py` and `src/config.py` comments now mirror these semantics. The authoritative runtime remains `src/quantization.py`.

### CRITICAL: Q4 scheme change history

The `_quantize_4bit` function was updated from a 15-level with-zero scheme to the current 16-level no-zero scheme. The comment in `src/quantization.py` line 27–29 documents this explicitly:

> NOTE: prior to this scheme the 4-bit mode used 15 levels {-7,…,+7}×scale/7 (with zero). The ablation const_4bit variant was trained under the old scheme — see ptq_study.py for caveat.

**Consequence**: The `const_4bit` ablation variant (both screening and full 10k-step runs) was trained under the OLD 15-level with-zero Q4 scheme. The PTQ study's Q4 evaluation uses the NEW 16-level no-zero scheme. Any comparison between PTQ Q4 results and native const_4bit results is comparing two different quantization functions. The PTQ study script (`scripts/ptq_study.py`) marks all Q4 comparisons with a `*` caveat and includes a `q4_scheme_caveat: true` flag in output data.

The Q3 (True 3-bit) scheme was added for the PTQ study. The original ablation matrix has no Q3 variant; one later native Q3 run exists at seed 31415, so cross-seed native evidence is still missing.

### STE implementation

```python
def ste_quantize(w: mx.array, bits: int) -> mx.array:
    if bits >= 16:
        return w
    w_q = quantize_weights(w, bits)
    return w + mx.stop_gradient(w_q - w)
```

Forward: the expression evaluates to `w_q` (quantized weights).
Backward: `mx.stop_gradient(w_q - w)` has zero gradient, so the gradient of `w + stop_gradient(...)` with respect to `w` is 1 (identity). The full-precision master weights receive the full gradient.

### EFFECTIVE_BITS registry

```python
EFFECTIVE_BITS = {0: math.log2(3), 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 16: 16.0}
```

This is used for theoretical compression estimates. For the prog_1_2_4 schedule [1,1,1,1,2,2,4,4], the effective average bits = (1+1+1+1+2+2+4+4)/8 = 2.0.

---

## 6. EXPERIMENT HISTORY (CHRONOLOGICAL)

### Phase 0: Smoke tests

**Status: COMPLETED**

**Purpose**: Verify the full pipeline works end-to-end before committing to longer runs. Sanity check that training does not crash, that quantization operates correctly, and that the data pipeline functions.

**Config/setup**:
- Model: d_model=128, n_layers=2, n_heads=4, d_ff=512 (tiny, ~0.7M params based on storage report for short_exp)
- Data: 100 Wikipedia articles, 1 MB max text, seq_len=64
- Training: 50 steps, batch=4, LR=1e-3, warmup=5 steps
- Configs: `configs/smoke_test_baseline.json`, `configs/smoke_test.json`

**Results**: Checkpoints saved at `checkpoints/smoke_test_baseline/` and `checkpoints/smoke_test_progressive/` (each ~2 files at step_25 and step_50). Final result files not checked in detail (results not in main results tree for smoke_test).

**Conclusion**: Pipeline works. No meaningful quality conclusions from 50 steps.

---

### Phase 1: Short experiments (500 steps, small model)

**Status: COMPLETED**

**Purpose**: First comparison of baseline vs. progressive schedules at a slightly larger scale than smoke test, still quick enough to iterate.

**Config/setup**:
- Model: d_model=256, n_layers=4, n_heads=8, d_ff=1024 (~7.6M params from storage report)
- Data: 100 Wikipedia articles, 1 MB max text, seq_len=128
- Training: 500 steps, batch=8, LR=3e-4, warmup=50 steps, eval every 100 steps
- Three variants: short_exp_baseline, short_exp_progressive_2bit, short_exp_progressive_ternary
- Configs: `configs/short_exp_baseline.json`, `configs/short_exp_progressive_2bit.json`, `configs/short_exp_progressive_ternary.json`
- All single seed=42

**Results** (from `results/short_exp_*/final_summary.json`):

| Experiment | Precision schedule | best_val_loss | total_seconds |
|---|---|---|---|
| short_exp_baseline | [16]*8 | 7.529157 | 39.0 |
| short_exp_progressive_2bit | [1,1,1,1,2,2,4,4] | 7.530751 | 41.5 |
| short_exp_progressive_ternary | [1,1,1,1,3,3,4,4] | 7.533732 | 40.8 |

Note: The ternary schedule in short_exp_progressive_ternary uses the OLD bits=3 ternary interpretation (before the refactoring where bits=3 became True 3-bit and bits=0 became ternary). This is a historical artifact.

**Conclusion**: At 500 steps with a small model and tiny dataset, all three variants produce nearly identical validation loss (~7.53). No signal at this scale. Progressive does not clearly win or lose. Ternary is marginally worse, but the difference is within noise. The experiment justified scaling up.

---

### Phase 2: Full-scale initial comparison (10,000 steps)

**Status: COMPLETED**

**Purpose**: Full 10k-step training at the main model scale with 50k Wikipedia articles, comparing baseline vs. progressive_1_2_4. Single seed (seed=42) for each.

**Config/setup**:
- Model: d_model=512, n_layers=6, n_heads=8, d_ff=2048, tie_word_embeddings=True (28.3M params)
- Data: 50,000 articles, 500 MB max text, seq_len=256 (train: 256,180 chunks; val: 13,484 chunks)
- Training: 10,000 steps, batch=8, LR=3e-4, warmup=500 steps, eval every 500 steps (100 eval batches)
- Two variants: full_baseline (seed=42), full_progressive_1_2_4 (seed=42)
- Configs: `configs/full_baseline.json`, `configs/full_progressive_1_2_4.json`

**Results** (from `results/full_baseline/final_summary.json` and `results/full_progressive_1_2_4/final_summary.json`):

| Experiment | Model type | Precision schedule | best_val_loss | total_seconds |
|---|---|---|---|---|
| full_baseline | baseline | [16]*8 | 7.432665 | 4,990.3 (83.2 min) |
| full_progressive_1_2_4 | progressive | [1,1,1,1,2,2,4,4] | 7.414686 | 5,663.1 (94.4 min) |

**Checkpoints**: Both checkpoints are fully retained at `checkpoints/full_baseline/` and `checkpoints/full_progressive_1_2_4/`, each containing 17 checkpoint files at 324 MB each (step_500 through step_10000). The PTQ study was designed to use new training runs with `save_checkpoints` configured differently (one checkpoint at the final step only), so these are not the intended PTQ source checkpoints.

**Theoretical compression for progressive**:
- Effective avg bits: 2.0
- Theoretical Q storage: 42.4 MB (vs 113.2 MB FP32, vs 56.6 MB BF16)
- Compression vs FP32: 2.67×; vs BF16: 1.33×

**Observation**: The progressive model beat the baseline by 0.018 nats at seed=42. However, this is a single seed — Apple Silicon non-determinism across sessions means even the same seed does not guarantee reproducibility. The ablation study was designed to address this with 3 seeds.

---

### Phase 3: Ablation screen (3,000-step, 6 variants × 3 seeds)

**Status: COMPLETED (18/18 runs)**

**Purpose**: Systematic screening to identify which variants are worth full 10k-step training. Control for seed variance by running 3 seeds. Variants designed to decompose: (a) does any low-bit training work? (b) does progressive structure help beyond constant low-bit? (c) does direction matter?

**Variants**:
- baseline: [16]*8 (FP32, no quantization)
- const_1bit: [1]*8 (binary throughout)
- const_2bit: [2]*8 (true 2-bit throughout)
- const_4bit: [4]*8 (4-bit throughout)
- prog_1_2_4: [1,1,1,1,2,2,4,4] (coarse-to-fine)
- prog_4_2_1: [4,4,2,2,1,1,1,1] (fine-to-coarse, reversed)

**Seeds**: 42, 123, 7

**Config/setup** (from `configs/ablation/ablation_baseline_s42_screen.json`):
- Model: same as full (d_model=512, n_layers=6, n_heads=8, d_ff=2048, 28.3M params)
- Data: same (50k articles, 500 MB, seq_len=256)
- Training: 3,000 steps, batch=8, LR=3e-4, warmup=300 steps, eval every 500 steps
- Checkpoints disabled (save_checkpoints=False); only metrics saved
- Results saved to `results/ablation/`

**Screening results** (from `results/ablation/aggregate_screen.json`):

| Variant | Avg eff bits | Mean best_val_loss | Std | Min | Max |
|---|---|---|---|---|---|
| baseline | 16.0 | 7.489006 | 0.004098 | 7.485992 | 7.493672 |
| const_1bit | 1.0 | 7.487793 | 0.003420 | 7.485748 | 7.491741 |
| const_2bit | 2.0 | 7.489598 | 0.004093 | 7.485646 | 7.493819 |
| const_4bit | 4.0 | 7.488849 | 0.003643 | 7.485985 | 7.492949 |
| prog_1_2_4 | 2.0 | 7.491556 | 0.004255 | 7.488473 | 7.496411 |
| prog_4_2_1 | 2.0 | 7.489618 | 0.001666 | 7.487710 | 7.490788 |

At 3,000 steps all variants cluster extremely tightly (range: 7.4857–7.4964). No variant shows a clear advantage. All variants are still converging. The screen identified all 6 variants as worth running to full 10k steps (since the differences are all within noise).

Convergence speed (step at which val_loss first drops below 7.50, from aggregate_screen.json):
- baseline: mean 2833 steps
- const_1bit: mean 2500 steps (fastest)
- const_2bit: mean 2500 steps
- const_4bit: mean 2833 steps
- prog_1_2_4: mean 2833 steps
- prog_4_2_1: mean 2500 steps

Training times at 3,000 steps varied significantly across seeds (1314s to 3175s), reflecting Apple Silicon non-determinism and background load variation during runs.

---

### Phase 4: Full ablation (10,000-step, all 6 variants × 3 seeds)

**Status: COMPLETED (18/18 runs)**

This is the primary completed experiment. All 18 runs finished; no runs failed.

**Config/setup** (from `configs/ablation/ablation_baseline_s42_full.json`):
- Model: d_model=512, n_layers=6, n_heads=8, d_ff=2048, tie_word_embeddings=True (28.3M params)
- Data: 50k articles, 500 MB, seq_len=256
- Training: 10,000 steps, batch=8, LR=3e-4, warmup=500 steps, eval every 500 steps (100 eval batches)
- Checkpoints disabled (save_checkpoints=False); only metrics saved
- Results saved to `results/ablation_full/`

#### Per-run results table

All numbers read from `results/ablation_full/*/eval_history.json` and `results/ablation_full/*/final_summary.json`.

| Variant | Seed | best_val_loss | best_step | final_val_loss | best_val_acc | training_seconds |
|---|---|---|---|---|---|---|
| baseline | 42 | 7.419442 | 9500 | 7.435617 | 0.047905 | 13724.7 |
| baseline | 123 | 7.462691 | 8500 | 7.472883 | 0.039417 | 8529.9 |
| baseline | 7 | 7.448140 | 10000 | 7.448140 | 0.044462 | 9159.9 |
| const_1bit | 42 | 7.458013 | 5000 | 7.477719 | 0.039497 | 9312.6 |
| const_1bit | 123 | 7.409514 | 9500 | 7.410912 | 0.047712 | 9207.4 |
| const_1bit | 7 | 7.433141 | 10000 | 7.433141 | 0.048047 | 9378.9 |
| const_2bit | 42 | 7.445650 | 9500 | 7.458969 | 0.042673 | 9961.8 |
| const_2bit | 123 | 7.462503 | 8500 | 7.473033 | 0.039417 | 8646.6 |
| const_2bit | 7 | 7.467745 | 6500 | 7.468079 | 0.039473 | 8202.7 |
| const_4bit | 42 | 7.426016 | 9500 | 7.441531 | 0.046293 | 7471.4 |
| const_4bit | 123 | 7.463445 | 8500 | 7.473605 | 0.039417 | 10032.6 |
| const_4bit | 7 | 7.445710 | 10000 | 7.445710 | 0.043417 | 12879.6 |
| prog_1_2_4 | 42 | 7.412376 | 9500 | 7.428867 | 0.047496 | 12675.6 |
| prog_1_2_4 | 123 | 7.454207 | 9500 | 7.490064 | 0.043799 | 9972.0 |
| prog_1_2_4 | 7 | 7.461684 | 10000 | 7.461684 | 0.042976 | 7809.0 |
| prog_4_2_1 | 42 | 7.445405 | 7500 | 7.461251 | 0.043121 | 7932.5 |
| prog_4_2_1 | 123 | 7.459054 | 9500 | 7.466103 | 0.043628 | 8443.2 |
| prog_4_2_1 | 7 | 7.466833 | 10000 | 7.466833 | 0.040836 | 8060.2 |

#### Aggregate per-variant (from `results/ablation/aggregate_full.json`)

| Variant | Avg eff bits | Mean best_val_loss | Std | Min | Max |
|---|---|---|---|---|---|
| **const_1bit** | 1.0 | **7.433556** | 0.024252 | 7.409514 | 7.458013 |
| **prog_1_2_4** | 2.0 | **7.442756** | 0.026574 | 7.412376 | 7.461684 |
| **baseline** | 16.0 | **7.443424** | 0.022007 | 7.419442 | 7.462691 |
| **const_4bit** | 4.0 | **7.445057** | 0.018723 | 7.426016 | 7.463445 |
| **prog_4_2_1** | 2.0 | **7.457097** | 0.010847 | 7.445405 | 7.466833 |
| **const_2bit** | 2.0 | **7.458633** | 0.011545 | 7.445650 | 7.467745 |

Sorted by mean best_val_loss ascending (lower = better).

#### Ranking

1. const_1bit — mean 7.4336, beats baseline in 2/3 seeds (seeds 7 and 123)
2. prog_1_2_4 — mean 7.4428, beats baseline in 1/3 seeds (seed 42)
3. baseline — mean 7.4434 (reference)
4. const_4bit — mean 7.4451, beats baseline in 1/3 seeds (seed 42)
5. prog_4_2_1 — mean 7.4571, beats baseline in 0/3 seeds
6. const_2bit — mean 7.4586, beats baseline in 0/3 seeds

Delta prog_1_2_4 vs baseline: −0.0007 (better, but within noise).
Delta const_1bit vs baseline: −0.0099 (better, larger signal, still within 1σ given the std of ~0.022).

#### Answers to research questions (from full ablation data)

**Q1 — Does progressive precision consistently outperform/match baseline?**
prog_1_2_4 mean (7.4428) vs baseline mean (7.4434): delta = −0.0007 in favor of progressive. The ablation script's threshold for "PROG WINS" is delta < −0.001. At delta = −0.0007, this falls short of the threshold. Classification: TIED (within noise). prog_1_2_4 beats the baseline per-seed mean on 1 of 3 seeds.

**Q2 — Does progressive outperform constant-bit alternatives at same avg bits (const_2bit)?**
prog_1_2_4 (7.4428) vs const_2bit (7.4586): delta = −0.016. The ablation script classifies |delta| < 0.002 as inconclusive and 0.002–0.02 as "indistinguishable at this scale." At delta = −0.016, this is borderline. The script output would say "INCONCLUSIVE: schedule structure vs regularisation indistinguishable at this scale."

**Q3 — Does direction matter (1→2→4 vs 4→2→1)?**
prog_1_2_4 (7.4428) vs prog_4_2_1 (7.4571): delta = −0.014. Coarse-to-fine appears better, but at 3 seeds with std ~0.011–0.027, this is not statistically established.

**Q4 — const_1bit vs baseline:**
Largest signal in the dataset. const_1bit ranks first. Binary weights appear to act as regularization that can help generalization. However, seed variance is large (const_1bit std = 0.024 vs baseline std = 0.022), and the differences per-seed are not consistent across seeds.

#### Observations about variance

Seed variance is substantial across all variants. Standard deviations of 0.011–0.027 nats in best_val_loss, compared to differences between variants of 0.001–0.025 nats. This means overlapping confidence intervals are likely for most pairwise comparisons. Training time also varies substantially across seeds (e.g., const_4bit: 7471s to 12879s), likely reflecting Apple Silicon thermal throttling and background processes during long runs.

Many runs are still improving at step 10,000 (best_step = 10,000 for 8 of 18 runs), suggesting that longer training might reveal clearer differences.

---

## 7. METHODOLOGICAL LIMITATIONS

### 7.1 Small model

At 28.3M parameters, this is a toy-scale model. Behaviors observed at this scale may not transfer to models where quantization is practically relevant (e.g., 1B+ parameters where memory is a real constraint). The research questions are explored here as proofs of concept.

### 7.2 Limited dataset

~69M tokens from 50k Wikipedia articles. Modern small language models are trained on trillions of tokens. At 10k steps with batch_size=8 and seq_len=256, total training tokens processed = 10,000 × 8 × 256 × 0.5 (masked fraction) ≈ 10M unique masked token predictions. Many training sequences may be repeated. This limits the quality ceiling.

### 7.3 Statistical power: only 3 seeds

With n=3, the standard error of the mean is std/sqrt(3) ≈ 0.013 for const_1bit and ≈ 0.013 for baseline. Given the observed differences between variants of 0.001–0.016 nats, most pairwise comparisons do not reach statistical significance at conventional thresholds. The ablation study provides directional evidence but cannot definitively rank variants.

### 7.4 Val loss differences are small

All differences between variants fall in the range 0.001–0.025 nats. Whether these are practically meaningful (in terms of text quality, downstream task performance) is unknown; perplexity differences of this magnitude at this scale may not be detectable in generated text quality.

### 7.5 Simulated quantization only

The most critical limitation: all quantized operations are simulated in float32 via STE. The weight values used in the matmul are float32 approximations (e.g., sign(w) × mean(|w|) for binary), not packed integer codes. Consequences:
- No actual memory reduction at training time (all weights are float32 master weights)
- No actual inference speedup (float32 simulation is slower than native float32)
- Storage compression estimates (e.g., 2.67× for prog_1_2_4) are theoretical and would only apply if weights were actually packed into 1/2/4-bit integers
- The experiment tests whether the STE gradient signal during training affects model quality, not whether low-bit inference is fast

### 7.6 Apple Silicon non-determinism

Despite setting both `mx.random.seed(seed)` and `np.random.seed(seed)`, MLX on Apple Silicon does not guarantee bit-for-bit reproducibility across separate process runs. The same config with the same seed run twice in different sessions may produce slightly different val_loss curves. All 18 ablation full runs were separate processes. This is noted explicitly in the code and is an inherent limitation of the hardware/framework combination.

### 7.7 Memory estimates are theoretical

The `training_memory_estimate_mb` in storage reports (≈452 MB for the full model) is calculated as `total_params × 4 × 4 / 1e6` (4 bytes per param × 4 copies: master weights, gradients, Adam m, Adam v). This does not account for activation memory during the forward/backward pass, which scales with batch_size × seq_len × d_model. Actual peak memory on device may differ.

### 7.8 Old Q4 scheme in const_4bit

The const_4bit ablation variant was trained under the legacy 15-level with-zero Q4 scheme, while the PTQ study uses the new 16-level no-zero Q4 scheme. Any comparison between these two is comparing different quantization functions. This is documented in `src/quantization.py` and flagged in `scripts/ptq_study.py`.

---

## 8. PTQ STUDY — STATUS: PREPARED, NOT YET RUN

### Scientific question

Given the exact same quantization function (`quantize_weights(w, bits)`), is it better to train under STE gradient pressure at the target bit-width throughout training (native QAT), or to apply the same quantization post-hoc to a high-precision checkpoint (Direct/Naive PTQ)?

This is NOT a comparison against state-of-the-art PTQ methods (GPTQ, AWQ, calibration-based). It is a controlled comparison where the only variable is when quantization is applied (during training vs. at evaluation time).

### Why Direct/Naive PTQ

At evaluation time, both native QAT and direct PTQ call `model.set_bits(bits)` before each forward pass, which triggers `ste_quantize()` → `quantize_weights()`. Since there are no gradients during evaluation, the STE vs. direct quantize distinction collapses: both paths apply `quantize_weights(w, bits)` to the stored weights. The difference is entirely in what was done during training: native QAT optimized the master weights under the gradient pressure of STE at bits, while PTQ optimized at bits=16 (full precision).

### Experimental matrix

- Seeds: 42, 123, 7 (matching ablation study)
- PTQ bits (main matrix): 1, 2, 3, 4, 16
- PTQ bits (optional): 0 (ternary, accessed via `--include-ternary`)
- Total main evaluations: 3 seeds × 5 bits = 15 evaluations
- With ternary: 3 seeds × 6 bits = 18 evaluations

### Phases

**Phase 1 — Train 3 baseline checkpoints** (configs exist, checkpoints do not):
Each baseline is a full 10,000-step training run using `configs/ptq/ptq_baseline_s{42,123,7}.json`. These configs are identical to the ablation full-phase baseline configs, except `save_checkpoints=True` and `checkpoint_every=999999` (only one checkpoint saved: the final step). The target checkpoint for each is `checkpoints/ptq_baselines/ptq_baseline_s{seed}/step_0010000.npz`.

Note: The existing `checkpoints/full_baseline/` and `checkpoints/full_progressive_1_2_4/` checkpoints (from Phase 2, seed=42 only) are NOT used by the PTQ study. The PTQ study requires separate baseline runs at all 3 seeds, into a different checkpoint directory.

**Phase 2 — Apply Direct/Naive PTQ evaluations**:
Load the FP32 checkpoint, call `model.set_bits(bits)` uniformly for all steps, evaluate on the validation set (100 batches per evaluation). No retraining, no calibration.

**Phase 3 — Analysis**:
Compare PTQ results against native ablation results (loaded from `results/ablation_full/*/final_summary.json` — no recomputation needed). Build comparison table: Δ(PTQ vs FP32), Δ(native vs PTQ) at each bit level. Test whether native QAT provides a quality advantage over Direct PTQ, and whether this advantage increases at lower bit levels.

### Q4 scheme caveat

The PTQ Q4 evaluation uses the new 16-level no-zero scheme. The native const_4bit baseline was trained under the old 15-level with-zero scheme. The PTQ script explicitly flags this mismatch.

### Q3 note

Q3 (True 3-bit, 8 levels) was absent from the original native ablation suite. A later campaign trained one native Q3 counterpart at seed 31415 (best val loss 7.402252), which is useful single-seed evidence but not a replicated comparison. Additional paired seeds remain required.

### Configs that exist

- `configs/ptq/ptq_baseline_s42.json`
- `configs/ptq/ptq_baseline_s123.json`
- `configs/ptq/ptq_baseline_s7.json`

### Checkpoints that exist

No PTQ baseline checkpoints exist yet. The `checkpoints/ptq_baselines/` directory does not exist.

### Script

`scripts/ptq_study.py` — fully implemented, ready to run.

### Commands

```bash
# Full study (Phase 1 + Phase 2 + Phase 3, ~6-7h estimated)
python scripts/ptq_study.py

# Skip training if checkpoints already exist
python scripts/ptq_study.py --skip-training

# Skip training + PTQ eval; load saved ptq_eval_results.json and analyze
python scripts/ptq_study.py --eval-only

# Dry run: print plan, train nothing
python scripts/ptq_study.py --dry-run

# Include optional ternary evaluation
python scripts/ptq_study.py --include-ternary

# Reduce eval batches for faster (less accurate) evaluation
python scripts/ptq_study.py --eval-steps 50
```

### Estimated runtime

3 × ~1.5h training + 15 × ~5 min evaluation = ~4.5h + 1.25h ≈ 6–7h total.

---

## 9. RESULTS FILE MAP

Verified against actual repository contents.

```
results/
  ablation/                           # 3k-step screening phase (COMPLETED)
    per_run_screen.csv                # 18 rows: one per (variant, seed)
    per_run_full.csv                  # 18 rows: full-phase summary (written by ablation_study.py --analyze-only --phase full)
    aggregate_screen.json             # mean/std/min/max per variant across 3 seeds (screen phase)
    aggregate_full.json               # mean/std/min/max per variant across 3 seeds (full phase)
    abl_baseline_s7_scr/
      final_summary.json
      eval_history.json
      train_metrics.csv
    abl_baseline_s42_scr/             # (similar structure for all 18 screening runs)
    abl_baseline_s123_scr/
    abl_const_1bit_s7_scr/
    abl_const_1bit_s42_scr/
    abl_const_1bit_s123_scr/
    abl_const_2bit_s7_scr/
    abl_const_2bit_s42_scr/
    abl_const_2bit_s123_scr/
    abl_const_4bit_s7_scr/
    abl_const_4bit_s42_scr/
    abl_const_4bit_s123_scr/
    abl_prog_1_2_4_s7_scr/
    abl_prog_1_2_4_s42_scr/
    abl_prog_1_2_4_s123_scr/
    abl_prog_4_2_1_s7_scr/
    abl_prog_4_2_1_s42_scr/
    abl_prog_4_2_1_s123_scr/

  ablation_full/                      # 10k-step full ablation (COMPLETED)
    abl_baseline_s7_full/
      final_summary.json
      eval_history.json
      train_metrics.csv
    abl_baseline_s42_full/            # (same structure for all 18 full runs)
    abl_baseline_s123_full/
    abl_const_1bit_s7_full/
    abl_const_1bit_s42_full/
    abl_const_1bit_s123_full/
    abl_const_2bit_s7_full/
    abl_const_2bit_s42_full/
    abl_const_2bit_s123_full/
    abl_const_4bit_s7_full/
    abl_const_4bit_s42_full/
    abl_const_4bit_s123_full/
    abl_prog_1_2_4_s7_full/
    abl_prog_1_2_4_s42_full/
    abl_prog_1_2_4_s123_full/
    abl_prog_4_2_1_s7_full/
    abl_prog_4_2_1_s42_full/
    abl_prog_4_2_1_s123_full/

  full_baseline/                      # Phase 2 single-seed baseline (seed=42)
    final_summary.json                # best_val_loss: 7.432665
    eval_history.json
    train_metrics.csv

  full_progressive_1_2_4/             # Phase 2 single-seed progressive (seed=42)
    final_summary.json                # best_val_loss: 7.414686
    eval_history.json
    train_metrics.csv

  short_exp_baseline/                 # Phase 1 short experiment
    final_summary.json                # best_val_loss: 7.529157 (500 steps)
    eval_history.json
    train_metrics.csv

  short_exp_progressive_2bit/         # Phase 1 short experiment
    final_summary.json                # best_val_loss: 7.530751 (500 steps)
    eval_history.json
    train_metrics.csv

  short_exp_progressive_ternary/      # Phase 1 short experiment (old ternary scheme)
    final_summary.json                # best_val_loss: 7.533732 (500 steps)
    eval_history.json
    train_metrics.csv

checkpoints/
  full_baseline/                      # Phase 2 baseline: 17 checkpoints at 324 MB each
    latest_meta.json
    step_0000500.npz  ... step_0010000.npz

  full_progressive_1_2_4/             # Phase 2 progressive: 16 checkpoints at 324 MB each
    latest_meta.json
    step_0000500.npz  ... step_0010000.npz

  short_exp_baseline/                 # Phase 1 baseline: 4 checkpoints at ~87 MB each
    latest_meta.json
    step_0000100.npz, step_0000200.npz, step_0000400.npz, step_0000500.npz

  short_exp_progressive_2bit/
  short_exp_progressive_ternary/

  smoke_test_baseline/                # Smoke test: 2 checkpoints
    step_0000025.npz, step_0000050.npz

  smoke_test_progressive/
    step_0000025.npz, step_0000050.npz

  # NOTE: checkpoints/ptq_baselines/ does NOT exist yet (PTQ study not run)

configs/
  baseline.json
  full_baseline.json
  full_progressive_1_2_4.json
  progressive_1_2_4.json
  short_exp_baseline.json
  short_exp_progressive_2bit.json
  short_exp_progressive_ternary.json
  smoke_test.json                     # smoke test progressive
  smoke_test_baseline.json

  ablation/                           # 36 auto-generated configs (18 screen + 18 full)
    ablation_baseline_s42_screen.json
    ablation_baseline_s42_full.json
    ablation_baseline_s123_screen.json
    ablation_baseline_s123_full.json
    ablation_baseline_s7_screen.json
    ablation_baseline_s7_full.json
    ablation_const_1bit_s42_screen.json
    ... (similar pattern for all 6 variants × 3 seeds × 2 phases)

  ptq/                                # PTQ baseline configs (exist, not yet used)
    ptq_baseline_s42.json
    ptq_baseline_s123.json
    ptq_baseline_s7.json

scripts/
  ablation_study.py
  ptq_study.py
  prepare_data.py
  train_tokenizer.py

src/
  __init__.py
  config.py
  quantization.py
  model.py
  diffusion.py
  data.py
  train.py
  evaluate.py
  generate.py

tests/
  test_quantization.py
  test_model.py
  test_diffusion.py
  test_training.py

tokenizer/
  wiki_bpe/
    tokenizer.json                    # BPE tokenizer (1.1 MB)
    vocab_info.json

data/
  cache/
    train_seq256_art50000_bytes500000000.npy  (~250 MB)
    val_seq256_art50000_bytes500000000.npy    (~13 MB)
    meta_seq256_art50000_bytes500000000.json
    (plus smaller caches for smoke test / short exp)
```

---

## 10. REPRODUCIBILITY

### Environment

- Platform: Apple Silicon macOS (tested on M4, 16 GB unified memory)
- Framework: MLX >= 0.21.0
- Python dependencies: see `requirements.txt`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data preparation

```bash
# Step 1: Train BPE tokenizer (needed once)
python scripts/train_tokenizer.py \
    --vocab-size 16000 \
    --max-articles 500 \
    --max-bytes 5000000 \
    --output tokenizer/wiki_bpe

# Step 2: Prepare main dataset (will cache to data/cache/)
python scripts/prepare_data.py \
    --max-articles 50000 \
    --max-bytes 500000000 \
    --seq-len 256 \
    --tokenizer-path tokenizer/wiki_bpe \
    --cache-dir data/cache
```

### Smoke test (end-to-end pipeline check)

```bash
./run_smoke_test.sh
```

### Training a single model

```bash
# Baseline (FP32, no quantization)
python -m src.train --config configs/full_baseline.json

# Progressive [1,1,1,1,2,2,4,4]
python -m src.train --config configs/full_progressive_1_2_4.json

# Any custom config
python -m src.train --config configs/<your_config>.json
```

### Ablation study

```bash
# Screening phase only (3k steps × 18 runs, ~9h)
python scripts/ablation_study.py --phase screen

# Full phase only (10k steps × 18 runs, ~45h)
python scripts/ablation_study.py --phase full

# Full study (screen then full)
python scripts/ablation_study.py --phase both

# Resume skipping completed runs
python scripts/ablation_study.py --phase full --resume

# Dry run: print plan only
python scripts/ablation_study.py --dry-run

# Analysis only (requires existing results)
python scripts/ablation_study.py --analyze-only --phase full

# Analysis of both phases
python scripts/ablation_study.py --analyze-only --phase both
```

### PTQ study

```bash
# Full study (~6-7h: 3 training runs + 15 PTQ evals)
python scripts/ptq_study.py

# Skip training, run PTQ evals only (requires checkpoints/ptq_baselines/)
python scripts/ptq_study.py --skip-training

# Skip training + evals, analyze saved ptq_eval_results.json only
python scripts/ptq_study.py --eval-only

# Dry run: print plan without executing
python scripts/ptq_study.py --dry-run

# Include optional ternary (3-state) evaluation
python scripts/ptq_study.py --include-ternary

# Use fewer eval batches (faster, less accurate)
python scripts/ptq_study.py --eval-steps 50
```

### Evaluation and generation

```bash
# Compare two checkpoints
python -m src.evaluate \
    --baseline checkpoints/full_baseline/step_0010000.npz \
    --progressive checkpoints/full_progressive_1_2_4/step_0010000.npz \
    --config configs/full_progressive_1_2_4.json \
    --eval-steps 100

# Generate text
python -m src.generate \
    --checkpoint checkpoints/full_progressive_1_2_4/step_0010000.npz \
    --config configs/full_progressive_1_2_4.json \
    --n-sequences 4 \
    --seq-len 128
```

---

## 11. FUTURE EXPERIMENTS ROADMAP

### Status: NEXT (immediate priority)

**A. Complete the Direct/Naive PTQ study**

Run `scripts/ptq_study.py`. This requires:
1. Training 3 baseline checkpoints (seeds 42, 123, 7) into `checkpoints/ptq_baselines/`
2. Running PTQ evaluations at bits ∈ {1, 2, 3, 4, 16} for each checkpoint
3. Producing the comparison table against native ablation results

This will answer RQ6–RQ8 directly.

**B. Interpret PTQ vs. native results in context of Q4 scheme mismatch**

After the PTQ study, the Q4 comparison requires care. To get a clean Q4 comparison, a native ablation run under the new 16-level Q4 scheme would be needed (see item E below).

---

### Status: PLANNED (near-term follow-up)

**C. Add more seeds if results remain inconclusive**

With n=3, most pairwise comparisons have overlapping confidence intervals. Adding seeds 456, 789, 2024 would increase statistical power. Recommended if PTQ results are also inconclusive.

**D. Train native True-Q3 model (bits=3, 8 levels, no zero)**

Q3 was absent from the original native ablation suite, but one native Q3 run at seed 31415 is now complete. It must not be treated as replicated evidence; the next comparison should pair current true Q3 with current true Q4, ternary, and FP32 across multiple seeds.

**E. Train native True-Q4 model (bits=4, 16 levels, no zero)**

The existing const_4bit ablation used the old 15-level scheme. A new ablation run under the current 16-level no-zero Q4 scheme would:
- Give a clean baseline for Q4 PTQ comparison (without the scheme-mismatch caveat)
- Allow Q4 to be compared against Q3 under the same no-zero symmetric family

**F. Calibrated / advanced PTQ as a separate study**

This would compare the current Direct PTQ approach against calibrated methods (GPTQ-style weight reconstruction, AWQ-style activation-aware scaling, fine-tuning after PTQ). This is a substantially larger scope and should be treated as a separate project phase.

---

### Status: HYPOTHETICAL / FUTURE RESEARCH

These are ideas and hypotheses that have not been implemented or systematically investigated. They are listed as research directions, not established results.

**G. Scale model size**

All current experiments use ~28M parameters. Repeating key findings at 100M, 500M, or 1B+ parameters would test whether the observed behaviors scale. Low-bit training becomes more practically relevant at larger sizes where memory is a real constraint.

**H. Larger datasets / longer training**

The current ~69M token dataset with 10k training steps is modest. Training on 1B+ tokens with 100k+ steps might reveal qualitatively different behaviors (e.g., whether native 1-bit training continues to match or surpass FP32 baselines as training data increases).

**I. Real packed low-bit kernels**

Implementing actual integer arithmetic via custom MLX Metal shaders or exploiting Apple Neural Engine would convert theoretical compression estimates into real memory and speed measurements. This would allow measuring actual tokens/second, actual peak memory usage, and actual power consumption — none of which are currently measurable.

**J. Binary decomposition (HYPOTHETICAL)**

The hypothesis that multiple 1-bit matrix operations can be combined to approximate higher-precision computation. For example, two 1-bit matmuls with different scale factors could theoretically approximate a 2-bit operation. This is an unproven architectural idea, not explored in any current experiment.

**K. Progressive precision at inference time for adaptive compute (HYPOTHETICAL)**

The current model supports switching precision per step via `model.set_bits()`. A hypothetical adaptive inference system could dynamically choose precision per token position or per layer based on confidence, allowing high-precision computation only where needed. This has not been implemented or benchmarked.

**L. Transfer findings to substantially larger models**

All current findings are on a small research scaffold (28M params, 69M tokens). The hypothesis is that native low-bit training benefits will remain or increase at scale (since larger models may have more redundancy to exploit via quantization). This is a research hypothesis, not a demonstrated result.

---

## 12. CURRENT PROJECT STATUS

### COMPLETED

- Full software implementation: model, quantization, diffusion process, training loop, evaluation, generation, ablation framework, PTQ framework
- Tokenizer training (BPE, 16k vocab, stored at `tokenizer/wiki_bpe/`)
- Dataset preparation and caching (50k articles, ~69M tokens, cached at `data/cache/`)
- Smoke tests (50 steps, verify pipeline)
- Phase 1 short experiments (500 steps, 3 variants, single seed)
- Phase 2 full-scale initial comparison (10k steps, baseline vs. prog_1_2_4, seed=42)
- **Phase 3 ablation screening** (3k steps, 6 variants × 3 seeds = 18 runs, all completed)
- **Phase 4 full ablation** (10k steps, 6 variants × 3 seeds = 18 runs, all completed)
- Q4 scheme update (from 15-level with-zero to 16-level no-zero for consistency)
- Q3 (true 3-bit) implementation
- PTQ study script design and implementation
- **Direct/naive PTQ recovery**: 18/18 evaluations across 3 seeds and Q1/Q2/Q3/Q4/FP32/ternary, with aggregate JSON/CSV verified
- One native true-Q3 and one native ternary 10k run at seed 31415
- Two additional paired baseline/Q1/progressive replications (seeds 31415 and 27182)

### NEXT (proposed; not launched)

1. Run a scheme-matched multi-seed native comparison of FP32, current true Q3, current true Q4, and ternary.
2. Split six paired seeds evenly across m1-256 and m1-512, with four serial 10k variants per seed and immutable node-local outputs.
3. Pre-register paired deltas and confidence intervals; keep legacy-Q4 and old-bits=3 results historical and out of the primary matrix.

### FUTURE

- Add seeds (better statistical confidence)
- Native Q3 training
- Native Q4 under updated scheme
- Calibrated PTQ study
- Scale-up experiments
- Real low-bit kernel implementation (actual memory/speed measurements)

---

## 13. RESEARCH LOG

Chronological narrative reconstructed from file timestamps, config contents, and result dates.

**[2025-07-15] — Initial setup and smoke tests**

Repository created. Core source files written: `src/model.py`, `src/quantization.py`, `src/diffusion.py`, `src/train.py`, `src/data.py`, `src/config.py`, `src/evaluate.py`, `src/generate.py`. Unit tests written. BPE tokenizer trained. Initial smoke test configs created and run (50 steps, tiny model d_model=128). Checkpoints saved to `checkpoints/smoke_test_baseline/` and `checkpoints/smoke_test_progressive/`. At this stage, quantization.py used a different scheme for bits=3 (ternary) and bits=4 (15-level with zero) compared to the current implementation.

Short experiment configs created (d_model=256, 4 layers, 500 steps) and run: `short_exp_baseline`, `short_exp_progressive_2bit`, `short_exp_progressive_ternary`. Results: all variants ~7.53 val_loss, no signal at 500 steps with 100 articles. Decision: scale up to full model.

Full-scale configs written (`configs/baseline.json`, `configs/progressive_1_2_4.json`, `configs/full_baseline.json`, `configs/full_progressive_1_2_4.json`).

**[2025-07-15 / 07-16] — Full-scale initial comparison**

Full 10k-step training run for `full_baseline` (seed=42) and `full_progressive_1_2_4` (seed=42). Both used the updated configs with `tie_word_embeddings=True` and `results_dir="results"`. Results: baseline best_val_loss = 7.432665 (4990s), progressive best_val_loss = 7.414686 (5663s). Progressive beat baseline by 0.018 nats at seed=42. This motivated the ablation study to test with multiple seeds.

**[2025-07-16] — Ablation study design and screening phase**

`scripts/ablation_study.py` written. 6 variants defined: baseline, const_1bit, const_2bit, const_4bit, prog_1_2_4, prog_4_2_1. All 18 screening configs auto-generated into `configs/ablation/`. 18 screening runs (3k steps each) executed sequentially. All completed. Aggregate results show all variants within 0.007 nats of each other at 3k steps; decision made to run all 6 variants to full 10k steps.

**[2025-07-16 / 07-18] — Full ablation phase**

All 18 full 10k-step ablation runs executed. Runs spanned approximately Jul 16 18:00 through Jul 18 15:46 (inferred from config file timestamps). All 18 completed without failures. `aggregate_full.json` and `per_run_full.csv` written. Key finding: const_1bit achieves the best mean best_val_loss (7.4336), beating the baseline (7.4434) by 0.0099 nats, though with high seed variance. prog_1_2_4 (7.4428) is essentially tied with baseline (7.4434).

**[2025-07-18] — PTQ study design and Q4/Q3 scheme update**

`scripts/ptq_study.py` written. Q4 scheme updated from 15-level with-zero to 16-level no-zero to be consistent with Q1 and Q2. Q3 (True 3-bit) added as a new PTQ-only evaluation level. Ternary moved from bits=3 to bits=0 sentinel. PTQ baseline configs written to `configs/ptq/`. The caveat about const_4bit using the old scheme is documented in both `src/quantization.py` and `scripts/ptq_study.py`. PTQ study has not yet been run.

---

*End of technical documentation. All numerical results sourced directly from JSON/CSV files in the repository.*
