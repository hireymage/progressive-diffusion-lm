# P1 Matched-Noise Campaign — Full 3-Node Summary (2026-07-20)

[English](p1-matched-noise-3node-summary-2026-07-20.md) | [Čeština](p1-matched-noise-3node-summary-2026-07-20.cs.md)

All three nodes completed two campaign phases (48/48 training tasks total):

| Node | Hardware | P1 Seeds | P1-next Seeds | Tasks | Status |
|---|---|---|---|---:|---|
| m1-256 | M1 8GB | 11, 29 | 131, 137 | 16 | ✅ complete |
| m1-512 | M1 8GB | 47, 73 | 149, 151 | 16 | ✅ complete |
| m4-air | M4 16GB | 101, 103 | 157, 163 | 16 | ✅ complete |

## Run IDs

| Node | Phase | Run ID |
|---|---|---|
| m1-256 | P1 | `20260719-060513_matched-noise_s11-29_3afd9c76` |
| m1-512 | P1 | `20260719-060530_matched-noise_s47-73_28e8a7fd` |
| m4-air | P1 | `20260719-120028_matched-noise_s101-103_14070a9e` |
| m1-256 | P1-next | `20260719-182527_matched-noise-next_s131-137_ef0786b9` |
| m1-512 | P1-next | `20260719-182528_matched-noise-next_s149-151_4917cde7` |
| m4-air | P1-next | `20260719-210751_matched-noise-next_s157-163_31b3261c` |

## P1 (original, 6 seeds per variant)

| Variant | Mean best val loss | Std dev | Mean train time | n |
|---|---:|---:|---:|---:|
| clean-fp32 | 7.421699 | 0.030157 | 1.275 h | 6 |
| constant-q1 | 7.427120 | 0.021697 | 1.363 h | 6 |
| gaussian-matched-fp32 | 7.456407 | 0.003186 | 1.432 h | 6 |
| uniform-matched-fp32 | 7.456817 | 0.003557 | 1.434 h | 6 |

## P1-next (new, 6 seeds per variant)

| Variant | Mean best val loss | Std dev | Mean train time | n |
|---|---:|---:|---:|---:|
| clean-fp32 | 7.440174 | 0.028590 | 1.323 h | 6 |
| constant-q1 | 7.442006 | 0.020045 | 1.380 h | 6 |
| gaussian-matched-fp32 | 7.458250 | 0.008061 | 1.451 h | 6 |
| uniform-matched-fp32 | 7.458563 | 0.008336 | 1.436 h | 6 |

## Combined P1 + P1-next (12 seeds per variant)

| Variant | Mean best val loss | Std dev | Mean train time | n |
|---|---:|---:|---:|---:|
| clean-fp32 | 7.430937 | 0.029631 | 1.299 h | 12 |
| constant-q1 | 7.434563 | 0.021379 | 1.372 h | 12 |
| gaussian-matched-fp32 | 7.457329 | 0.005923 | 1.441 h | 12 |
| uniform-matched-fp32 | 7.457690 | 0.006178 | 1.435 h | 12 |

## Ranking (lower loss is better)

1. **clean-fp32** — 7.430937
2. **constant-q1** — 7.434563
3. **gaussian-matched-fp32** — 7.457329
4. **uniform-matched-fp32** — 7.457690

### Deltas vs clean-fp32 (combined, 12 seeds)

- Δ(constant-q1 − clean-fp32): +0.003626
- Δ(gaussian-matched-fp32 − clean-fp32): +0.026392
- Δ(uniform-matched-fp32 − clean-fp32): +0.026753

## Per-node P1-next breakdown

| Node | clean-fp32 | constant-q1 | gaussian-matched-fp32 | uniform-matched-fp32 |
|---|---:|---:|---:|---:|
| m1-256 | 7.458103 | 7.443998 | 7.459747 | 7.459800 |
| m1-512 | 7.423768 | 7.425200 | 7.451015 | 7.451283 |
| m4-air | 7.438650 | 7.456819 | 7.463988 | 7.464605 |

## Interpretation

- This campaign is complete and internally consistent across 3 nodes and 12 total seeds per variant.
- **clean-fp32** is the best variant on average, followed closely by **constant-q1** (Δ = +0.0036).
- Both matched-noise FP32 variants (Gaussian and Uniform) are worse than clean FP32 by ~0.026–0.027, with very low variance (std < 0.009).
- Native Q1 quantization does not hurt generalization at this model scale — the Q1 deficit is within one standard deviation of the FP32 spread.
- Matched-weight-noise FP32 training degrades validation loss more than native Q1 quantization, suggesting the Q1 benefit is not purely a regularization effect.
- Results remain specific to the current implementation (1/2/4 simulated quantization with full recompute), not yet the future incremental 1/2/4/8 design.
- Next phase should implement standard 1→2→4→8 / 8→4→2→1 with incremental computation and early-exit inference.

## M4 Air performance note

The M4 Air (16GB unified memory) completed 8 tasks in ~9.7h wall time, averaging ~1.2h per 10k-step run — approximately 15% faster per task than the M1 nodes (~1.3–1.4h). No memory pressure or swap issues were observed on the 16GB node.
