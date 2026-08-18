# TODO — Progressive Diffusion LM

[English](TODO.en.md) | [Čeština](TODO.md)

Last update: 2026-08-18

---

## 🎯 New project direction

- The goal is no longer just to compare progressive quantization to fixed schedule.
- The new direction is to develop a diffusion language model, which starts with a rough estimate and is gradually refined only where it is still needed.
- The model should be able to generate multiple tokens at once and add an additional computation step or higher precision when uncertainty remains.
- Practically, this means designing a new generation strategy, clarifying the necessary components, and then reassembling the experiments according to this goal.
- As a useful intermediate step, it makes sense to create a simple functional PD model that can actually generate text, even if it has not yet been finally optimized.
- The original experiments and results remain as the history of the project, but new developments will follow this refined goal.
- The target first model changes accuracy **between groups of Transformer layers**, not over the entire pass through the model: 5× Q1, 5× Q2, 5× Q4, 5× Q8, and 5× FP16.
- Starting at layer 5, computation may stop after any subsequent layer, including inside a precision group (for example, after layer 8).

## ✅ Done

### Infrastructure and pipeline
- [x] Basic pipeline: tokenizer, dataset, model, training loop, eval
- [x] BPE tokenizer training (`vocab_size=16 000`, wikimedia/wikipedia)
- [x] `QuantizedLinear` + Straight-Through Estimator
- [x] `model.set_bits()` — runtime precision switching
- [x] Campaign harness `scripts/run_dual_m1_campaign.py` (lock, artifact publish)
- [x] Multi-node SSH launch (m1-256, m1-512, m4-air via ZeroTier)
- [x] **PyTorch/CUDA backend for Windows and Linux**
  - Implemented conversion of MLX checkpoints including weights, AdamW moments and global step.
  - Implemented continuation of Czech flexible training through CUDA with the same architecture, tokenizer, cache and routes.
  - Both the transfer and the CUDA trainer are covered by the tests; the procedure is in `docs/cuda-training.md` and `docs/cuda-training.en.md`.
  - Q2/Q8 for now, QAT fake-quant calculations remain above FP32 master weights, not packed integer CUDA kernels.
- [x] Unit tests — 280/280 passed

### Quantization schemes (fixed)
- [x] Q1 binary (2 levels, bits=1)
- [x] Q2 true 2-bit (4 levels, bits=2)
- [x] Q3 true 3-bit (8 levels, bits=3) — **implemented in this session**
- [x] Q4 true 4-bit (16 non-zero levels, bits=4) — **fixed from 15 levels**
- [x] Q8 8-bit symmetric (bits=8, Phase 2)
- [x] Ternary/3-state moved to bits=0 (separated from Q3)
- [x] PTQ renamed to "Direct/Naive PTQ" (without kalibrace/GPTQ/AWQ)

### Experiments
- [x] Smoke tests (50 steps, tiny model)
- [x] Short experiments (500 steps, 3 variants)
- [x] Initial full comparison (10,000 steps, seed=42) — progressive by 0.018 better
- [x] Ablation screening (3000 steps, 18 runs = 6 variants × 3 seeds)
- [x] **Full ablation (10,000 steps, 18 runs)** — const_1bit best average (7.4336)
- [x] **Phase 1 — PTQ + native gaps** (m1-256, seeds 42/123/7/31415)
- [x] **Phase 1 — Paired replication** (m1-512, seeds 31415/27182)
- [x] **Phase 1 — Matched-noise campaign** (3 nodes, 12 seedů/varianta, 48 runs)
  - Conclusion: Q1 the advantage is NOT noise regularization; matched-noise FP32 worse by ~0.026
- [x] **Phase 2 — Bidirectional incremental** (42 runs, 7 schedules, bits=8 added)
  - Best: constant-2b (7.4287); progressive-down the worst
- [x] **Phase 2 — Inference eval** (3 nodes, early-exit to 9.47× speedup)
- [x] M2 Incremental forward (`y_next = y_prev + Δ`, 1.32× speedup)
- [x] M3 Early-exit generation (threshold ≤ 0.03 → 1 step, 5–9× speedup, 100% match)

### Documentation
- [x] `PROJECT_DOCUMENTATION.md` — 1042 lines, 13 sections
- [x] `README.md` — rewritten, public overview
- [x] Obsidian `_Project.md` — updated (Phase 1 + Phase 2 results)
- [x] `src/model.py` — docstring fixed (bits schemas)
- [x] `src/quantization.py` — docstring and EFFECTIVE_BITS updated

---

## 🔜 Direct next steps (recommended)

### 1. M0 — functional PD-LM baseline according to the new goal
- [x] Implement a deterministic M0/oracle evaluator for Q1/Q2/Q4/Q8/FP32
- [x] Verify evaluator smoke test at 10k `full_baseline` checkpoint
- [x] Verify a reproducible M0 inference run on `m1-256`, `m1-512` and `m4-air`
- [ ] Measure the quality of current mask-diffusion generation and save samples
- [ ] Separate metrics of diffusion steps, model depth, and degrees of accuracy

### 1A. Layer-wise grouped-precision prototype
- [x] Separate new 25-layer prototype from legacy `DiffusionLM`
- [x] Implement schedule 5× Q1/Q2/Q4/Q8/FP16 and real FP16 matmul path
- [x] Add shared LM head after each layer from layer 5
- [x] Add masked deep supervision for all intermediate outputs
- [x] Allow sequence-wide early exit inside the precision group
- [x] Verify the first three different smoke tests on three nodes
- [x] Connect prototype to real tokenizer and 69M-token Wikipedia cache
- [x] Run a 5000-step FP32 and progressive pilot and measure layers 5/10/15/20/25
- [ ] Remove collapse to most common tokens; deeper outputs do not improve quality yet
- [x] Measure constant-token baseline: new line = 3.743 % validation accuracy
- [x] Confirm with mask sweep that 5k FP32 checkpoint exactly copies this baseline
- [x] Validate learning ability on one sequence: 59.7% after 1000 exposures
- [x] Finish one-sequence overfit: 100% masked accuracy, loss 0.000718
- [ ] Repeat the 95% gate on 100 sequences with sufficient exposures
- [ ] Add mask-rate curriculum 15% → 30% → 50% → 75% → 100%
- [ ] Run the remaining five checks only after a successful language quality gate
- [ ] Calibrate early-exit threshold on validation data without leakage

### 2. Oracle adaptive precision analysis
- [x] Prepare continuous evaluation of the same inputs in Q1/Q2/Q4/Q8/FP32 without storing full logits
- [x] Measure fixed and newly introduced errors between degrees of accuracy on the 3node pilot
- [x] Add provenance-safe aggregation of distributed M0 runs
- [x] Implement the actual FP16 grade in a separate layer-wise prototype; legacy internal `bits=16` remains FP32 identity path
- [x] Create offline Pareto curve quality versus proxy calculation with calibration and held-out distribution
- [x] Verify the predictive value of confidence, entropy, top-1/top-2 margin and stability top-1
- [x] Add direct precision baseline and paired cluster bootstrap without held-out policy selection

### 3. M1 — minimal adaptive inference
- [ ] Implement decision `commit / defer / escalate` after token positions
- [ ] Compare fixed schedule, rule adaptive schedule and oracle
- [ ] Log the accuracy achieved and the number of diffusion steps for each token
- [ ] Continue to multi-precision training only with a positive oracle analysis

### 4. PTQ study (historical branch, checkpoints exist)
- [ ] Compare Direct/Naive PTQ vs. native QAT (Q1, Q2, Q3, Q4)
- [ ] Run `python scripts/ptq_study.py` at checkpoints from Phase 1
- [ ] Optional: `--include-ternary` for ternary variant
- [ ] Attention: native const_4bit uses the old scheme (15 levels) → Q4 the comparison is approximate

### 5. More seeds for Phase 2 (historical branch)
- [ ] Phase 2 has only 2 seeds (s201, s203) — seed sensitivity is high
- [ ] Recommended: 5-10 seeds for statistically reliable conclusions
- [ ] s201 gives almost schedule-invariant results → possibly wrong seed

### 6. Longer Phase 2 training (historical branch)
- [ ] Phase 2 model trained only 2000 steps vs. 10,000 for ablation
- [ ] Scaling to 5000-10000 steps for fair comparison

---

## 🔭 Research hypotheses (mid-term)

- [ ] **Native Q3 training** — no ablation counterpart exists yet
- [ ] **Calibrated PTQ** (GPTQ-style, AWQ-style) as a separate study
- [ ] **Binary decomposition** — representation FP32 by a matrix of more Q1 matrices
- [ ] **Adaptive compute** — early-exit thresholds trained, not hardcoded
- [ ] **Scaling** — bigger d_model, more layers; check if the trends hold
- [ ] **Real low-bit kernels** — packed integer weights, real memory measurements

---

## ⚠️ Open questions

- Why does `constant-2b` defeat `baseline-fp16` even progressive schedules? Is it regularization or an artifact of short training?
- Why does seed s201 produce almost identical results for all schedules?
- Q4 comparison — retrain `const_4bit` ablation with new scheme (16 levels without zero)?
- Early-exit the model always terminates after 1 step at threshold ≤ 0.03 — does it depend on the training length or is it a feature of the architecture?

---

## 🧹 Tech debt

- [ ] SSH config not set (nodes accessible via ZeroTier, but without `~/.ssh/config`)
- [ ] Passwords in Obsidian plain text (`SSH-pristupy.md`) — move to Keychain/1Password
- [ ] `configs/ptq/ptq_baseline_s{42,123,7}.json` — check if they match the new quantization schemes
- [ ] `results/full_progressive_1_2_4/` — check if it is included in the aggregated results
