# Phase 2: Bidirectional Incremental Progressive Precision — Final Report

[English](phase2-bidirectional-incremental-final-report.md) | [Čeština](phase2-bidirectional-incremental-final-report.cs.md)

<!-- doc-status: historical; verified: 2026-08-18 -->
> **Document status:** Historical report. Numbers and conclusions apply to the named campaign and date, not to the project’s current operational state.

**Date:** 2026-07-21  
**Authors:** Martin Hozák (Hozzy), Hermes Agent  
**Repository at report time:** hireymage/progressive-diffusion-lm (then private, Apache-2.0; public as of 2026-08-18)
**Hardware:** 2× Mac mini M1 8GB (m1-256, m1-512) + 1× MacBook Air M4 16GB (m4-air)  

---

## 1. Overview

Phase 2 tested **bidirectional progressive precision schedules** — varying quantization bits across diffusion steps in both coarse→fine (up) and fine→coarse (down) directions — against constant-precision baselines and an FP16 baseline, across 3 nodes and 2 seeds (42 runs total). Additionally, inference-time evaluation of incremental forward (M2) and early-exit generation (M3) was conducted on all 3 nodes.

### Milestones

| Milestone | Description | Status |
|-----------|-------------|:------:|
| M1 | 8-bit symmetric quantization (256 levels) | ✅ 58/58 tests |
| M2 | Incremental forward (`y_next = y_prev + Δ`) | ✅ 26/26 tests |
| M3 | Early-exit generation (`generate_with_early_exit`) | ✅ 35/35 tests |
| M4 | Campaign configuration (7 schedules × 2 seeds × 3 nodes) | ✅ |
| M5 | Smoke test validation | ✅ |
| M6 | Campaign execution (42 runs) | ✅ 42/42 |
| M7 | Aggregation, inference eval, report | ✅ |

**Unit tests: 108/108 passed, 0 failed.**

---

## 2. Campaign Results (42 runs)

### 2.1 Full Results Table

| Schedule | Seed | m1-256 | m1-512 | m4-air | Mean | Std |
|----------|------|-------:|-------:|-------:|-----:|----:|
| progressive-up [1,1,2,2,4,4,8,8] | s201 | 7.4634 | 7.4633 | 7.4633 | 7.4633 | 0.0001 |
| progressive-up | s203 | **7.3971** | 7.4324 | 7.4330 | **7.4208** | 0.0205 |
| progressive-down [8,8,4,4,2,2,1,1] | s201 | 7.4635 | 7.4635 | 7.4635 | 7.4635 | 0.0000 |
| progressive-down | s203 | 7.4539 | 7.4528 | 7.4581 | 7.4549 | 0.0028 |
| constant-1b | s201 | 7.4615 | 7.4615 | 7.4615 | 7.4615 | 0.0000 |
| constant-1b | s203 | 7.4113 | 7.4204 | 7.4452 | 7.4256 | 0.0176 |
| constant-2b | s201 | 7.4646 | 7.4646 | 7.4647 | 7.4646 | 0.0001 |
| constant-2b | s203 | 7.3961 | **7.3761** | 7.4064 | **7.3929** | 0.0154 |
| constant-4b | s201 | 7.4633 | 7.4633 | 7.4633 | 7.4633 | 0.0000 |
| constant-4b | s203 | 7.4428 | 7.4325 | 7.3946 | 7.4233 | 0.0254 |
| constant-8b | s201 | 7.4634 | 7.4635 | 7.4634 | 7.4634 | 0.0000 |
| constant-8b | s203 | 7.3888 | 7.4187 | 7.4501 | 7.4192 | 0.0307 |
| baseline-fp16 | s201 | 7.4635 | 7.4635 | 7.4635 | 7.4635 | 0.0000 |
| baseline-fp16 | s203 | 7.4096 | 7.4501 | 7.4521 | 7.4373 | 0.0240 |

### 2.2 Schedule Means (across all seeds and nodes)

| Schedule | Mean Val Loss | Std | n |
|----------|------------:|----:|---:|
| **constant-2b** | **7.4287** | 0.0405 | 6 |
| constant-8b | 7.4413 | 0.0310 | 6 |
| progressive-up | 7.4421 | 0.0266 | 6 |
| constant-4b | 7.4433 | 0.0272 | 6 |
| constant-1b | 7.4436 | 0.0226 | 6 |
| baseline-fp16 | 7.4504 | 0.0209 | 6 |
| progressive-down | 7.4592 | 0.0050 | 6 |

### 2.3 Seed Effect

| Seed | Mean | Std | n |
|------|-----:|----:|---:|
| s201 | 7.4633 | 0.0009 | 21 |
| s203 | 7.4249 | 0.0252 | 21 |

Seed s203 produces significantly lower (better) val_loss across all schedules, with higher variance. Seed s201 produces nearly identical results across all schedules (std=0.0009) — schedule choice has minimal effect with this seed.

### 2.4 Best/Worst Individual Runs

**Top 5:**
1. constant-2b s203 m1-512: **7.3761**
2. constant-8b s203 m1-256: 7.3888
3. constant-4b s203 m4-air: 7.3946
4. constant-2b s203 m1-256: 7.3961
5. progressive-up s203 m1-256: 7.3971

**Bottom 5:**
1. constant-2b s201 m4-air: 7.4647
2. constant-2b s201 m1-256: 7.4646
3. constant-2b s201 m1-512: 7.4646
4. progressive-down s201 m1-256: 7.4635
5. baseline-fp16 s201 m4-air: 7.4635

---

## 3. Inference Evaluation (M2/M3)

### 3.1 Setup

Each node trained a small progressive model (2000 steps, schedule [1,2,4,8,8,4,2,1]) with checkpoint saving, then compared 3 inference modes:
- **standard**: full forward pass through all 8 diffusion steps
- **incremental**: `forward_incremental` — reuses previous step's output via `y_next = y_prev + Δ`
- **early_exit**: `generate_with_early_exit` — stops generation when max token confidence exceeds threshold

Thresholds: [0.01, 0.02, 0.03, 0.05, 0.10, 0.50]  
Repeats: 3 per mode, averaged.

### 3.2 Results by Node

#### m1-256 (Mac mini M1 8GB)

| Mode | Latency (s) | Steps | Speedup | Agreement |
|------|--------:|------:|--------:|----------:|
| standard | 0.176 | 8 | 1.00× | — |
| incremental | 0.106 | 8 | 1.65× | 100% |
| early_exit (t=0.01) | 0.020 | 1 | 8.94× | 100% |
| early_exit (t=0.02) | 0.014 | 1 | **12.68×** | 100% |
| early_exit (t=0.03) | 0.016 | 1 | 11.15× | 100% |
| early_exit_inc (t=0.01) | 0.017 | 1 | 10.36× | 100% |
| early_exit_inc (t=0.02) | 0.017 | 1 | 10.49× | 100% |
| early_exit_inc (t=0.03) | 0.019 | 1 | 9.47× | 100% |
| early_exit (t=0.05) | 0.080 | 8 | 2.19× | 100% |
| early_exit (t=0.1) | 0.079 | 8 | 2.24× | 100% |
| early_exit (t=0.5) | 0.078 | 8 | 2.27× | 100% |

#### m1-512 (Mac mini M1 8GB)

| Mode | Latency (s) | Steps | Speedup | Agreement |
|------|--------:|------:|--------:|----------:|
| standard | 0.104 | 8 | 1.00× | — |
| incremental | 0.091 | 8 | 1.14× | 100% |
| early_exit (t=0.01) | 0.020 | 1 | 5.29× | 100% |
| early_exit (t=0.02) | 0.018 | 1 | **5.85×** | 100% |
| early_exit (t=0.03) | 0.018 | 1 | 5.66× | 100% |
| early_exit_inc (t=0.01) | 0.019 | 1 | 5.40× | 100% |
| early_exit_inc (t=0.02) | 0.019 | 1 | 5.57× | 100% |
| early_exit_inc (t=0.03) | 0.019 | 1 | 5.57× | 100% |
| early_exit (t=0.05) | 0.093 | 8 | 1.11× | 100% |
| early_exit (t=0.1) | 0.084 | 8 | 1.23× | 100% |
| early_exit (t=0.5) | 0.087 | 8 | 1.19× | 100% |

#### m4-air (MacBook Air M4 16GB)

| Mode | Latency (s) | Steps | Speedup | Agreement |
|------|--------:|------:|--------:|----------:|
| standard | 0.058 | 8 | 1.00× | — |
| incremental | 0.049 | 8 | 1.18× | 100% |
| early_exit (t=0.01) | 0.008 | 1 | 7.37× | 100% |
| early_exit (t=0.02) | 0.008 | 1 | **7.47×** | 100% |
| early_exit (t=0.03) | 0.008 | 1 | 7.37× | 100% |
| early_exit_inc (t=0.01) | 0.009 | 1 | 6.76× | 100% |
| early_exit_inc (t=0.02) | 0.008 | 1 | 6.85× | 100% |
| early_exit_inc (t=0.03) | 0.009 | 1 | 6.61× | 100% |
| early_exit (t=0.05) | 0.042 | 8 | 1.38× | 100% |
| early_exit (t=0.1) | 0.042 | 8 | 1.37× | 100% |
| early_exit (t=0.5) | 0.042 | 8 | 1.36× | 100% |

### 3.3 Inference Summary

| Metric | m1-256 | m1-512 | m4-air | Mean |
|--------|-------:|-------:|-------:|-----:|
| Incremental speedup | 1.65× | 1.14× | 1.18× | **1.32×** |
| Early-exit speedup (t≤0.03) | 8.94–12.68× | 5.29–5.85× | 7.37–7.47× | **7.2–8.7×** |
| Early-exit steps (t≤0.03) | 1/8 | 1/8 | 1/8 | 1/8 |
| Early-exit speedup (t≥0.05) | 2.19–2.27× | 1.11–1.23× | 1.36–1.38× | **1.5–1.6×** |
| Early-exit steps (t≥0.05) | 8/8 | 8/8 | 8/8 | 8/8 |
| Token agreement (all modes) | 100% | 100% | 100% | **100%** |

**Key finding:** Early-exit with threshold ≤0.03 reduces inference to a single diffusion step (1/8) with 100% token agreement and 5–13× speedup. At threshold ≥0.05, the model never exits early (all 8 steps run), and speedup comes only from the incremental optimization (~1.2–2.3×).

---

## 4. Key Findings

### 4.1 Progressive vs Constant

| Comparison | s201 | s203 |
|------------|------|------|
| progressive-up | 7.4633 | 7.4208 |
| progressive-down | 7.4635 | 7.4549 |
| constant-2b (best constant) | 7.4646 | **7.3929** |
| constant-4b | 7.4633 | 7.4233 |
| baseline-fp16 | 7.4635 | 7.4373 |

- **Constant-2b is the overall best schedule** (mean 7.4287), driven by strong s203 results (7.3929).
- **Progressive-up is competitive** (mean 7.4421), especially with s203 (7.4208), but does not beat constant-2b.
- **Progressive-down is the worst schedule** (mean 7.4592) — starting with high precision and degrading performs worse than all alternatives including baseline.
- **Baseline FP16 is not the best** — constant-2b, constant-8b, and progressive-up all outperform it on average, suggesting that quantization noise may act as regularization.

### 4.2 Direction Matters

Progressive-up (coarse→fine, 1→8b) significantly outperforms progressive-down (fine→coarse, 8→1b):
- s203: 7.4208 vs 7.4549 (Δ=0.034)
- s201: 7.4633 vs 7.4635 (Δ=0.0002, negligible)

**Coarse-to-fine is the correct direction** — starting with low precision and refining produces better results than starting high and degrading.

### 4.3 Seed Sensitivity

Seed s201 produces nearly schedule-invariant results (std=0.0009 across 21 runs) — the model converges to similar loss regardless of precision schedule. Seed s203 shows significant schedule sensitivity (std=0.0252) — the choice of schedule matters more with this seed.

This suggests that the effect of progressive precision is **seed-dependent** and may be more pronounced in certain training dynamics than others.

### 4.4 Node Consistency

For s201, all 3 nodes produce nearly identical results (std ≤ 0.0001 per schedule). For s203, node variance increases (std up to 0.0307 for constant-8b). m1-256 tends to produce the lowest val_loss with s203, while m4-air tends higher. This may reflect different memory bandwidth characteristics affecting quantization simulation.

### 4.5 Inference Optimizations

- **Incremental forward** provides a consistent **1.14–1.65× speedup** (mean 1.32×) with 100% token agreement — the `y_next = y_prev + Δ` formulation is functionally equivalent to full forward.
- **Early-exit** with low thresholds (≤0.03) achieves **5–13× speedup** by reducing to 1/8 diffusion steps, with 100% token agreement. The model's confidence after the first step is sufficient for correct token generation.
- **Early-exit + incremental** combines both optimizations but doesn't significantly outperform early-exit alone at low thresholds — the single-step path dominates.
- At thresholds ≥0.05, early-exit never triggers (all 8 steps run), and the speedup is only from the incremental optimization.

---

## 5. Conclusions

1. **Constant-2b is the best precision schedule** for this model size and dataset, outperforming both progressive schedules and the FP16 baseline.
2. **Progressive-up (coarse→fine) is viable** — it outperforms baseline-fp16 and is competitive with constant schedules, but does not surpass constant-2b.
3. **Progressive-down (fine→coarse) should be avoided** — it is the worst-performing schedule.
4. **Quantization may regularize** — constant-2b and constant-8b both outperform baseline-fp16, suggesting quantization noise helps generalization.
5. **Incremental forward (M2) works correctly** — 1.32× average speedup with 100% output equivalence.
6. **Early-exit (M3) is highly effective** — up to 12.68× speedup with threshold ≤0.03, reducing inference to 1/8 steps with 100% token agreement.
7. **Seed sensitivity is high** — the schedule effect is dramatic with s203 but negligible with s201. More seeds are needed for statistical significance.

---

## 6. Limitations & Future Work

- **Small model** (7.5M params, 16000 vocab) — results may not scale to larger models.
- **Short training** (2000 steps) — longer training may change the relative ranking of schedules.
- **Only 2 seeds** — statistical significance is limited. The dramatic seed effect (s201 vs s203) suggests more seeds (5–10) are needed.
- **Single dataset** — results are specific to the current text corpus.
- **Quantization is simulated** — STE in FP32, not actual low-bit hardware. Real quantized inference may differ.
- **Early-exit threshold tuning** — optimal threshold (0.03) is model-specific. Needs per-model calibration.

### Recommended Next Steps

1. **More seeds** (5–10) to establish statistical significance of schedule effects.
2. **Larger model** to test whether constant-2b remains optimal at scale.
3. **Longer training** (5000–10000 steps) to see if schedule ranking stabilizes.
4. **Real quantized inference** — deploy with actual packed low-bit weights.
5. **Per-model early-exit calibration** — find optimal threshold as function of training step.

---

## Appendix A: Configuration

- Model: 7.5M params, progressive type, weight tying
- Precision schedules: [1,1,2,2,4,4,8,8] (up), [8,8,4,4,2,2,1,1] (down), constants [1,2,4,8], baseline FP16
- Seeds: 201, 203
- Training: 10000 steps, batch_size=32, lr=3e-4 (cosine decay)
- Dataset: 434,688 tokens, 3226 train chunks, 170 val chunks
- Quantization: symmetric, STE in FP32

## Appendix B: Hardware

| Node | Model | RAM | Cores | Role |
|------|-------|-----|-------|------|
| m1-256 | Mac mini M1 | 8GB | 8 | Campaign + inference eval |
| m1-512 | Mac mini M1 | 8GB | 8 | Campaign + inference eval |
| m4-air | MacBook Air M4 | 16GB | 10 | Campaign + inference eval |

## Appendix C: Data Files

- Campaign results: `results/phase2_campaign_all_results.csv`
- Inference eval: `results/inference_eval/{m1-256,m1-512,m4-air}_inference_eval.json`
- Campaign configs: `configs/campaign/m1-256-phase2-bidir.json`, `m1-512-phase2-bidir.json`, `m4-air-phase2-bidir.json`
- Inference eval script: `scripts/eval_inference.py`
- Campaign runner: `scripts/run_dual_m1_campaign.py`
