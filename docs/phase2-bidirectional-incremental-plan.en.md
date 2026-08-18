# Phase 2 Plan — Bidirectional Incremental Progressive Precision

This document captures the post-P1 implementation phase aligned with the canonical principle:

- Progressive Up: `1b → 2b → 4b → 8b`
- Progressive Down: `8b → 4b → 2b → 1b`
- Constant precision baselines
- Full-precision baseline (FP16 and FP32 tracked separately)

## Scope

Current code does **full recompute** at each diffusion step with 1/2/4 only. Phase 2 introduces:

1. New precision level: 8b (alongside existing 1b/2b/4b).
2. Incremental computation interface: `y_next = y_prev + Δ`.
3. Reuse of intermediate activations where valid.
4. Early-exit inference with confidence threshold.
5. Bidirectional schedules as first-class configs.

## Milestones

### M1 — Quantization extension

- Add 8b quantization mode in `src/quantization.py`.
- Keep 1b/2b/4b as existing standard low-bit variants.
- Add tests for levels, symmetry, and storage accounting.

### M2 — Incremental forward API

- Introduce residual/delta path in model forward.
- Keep legacy full-recompute path behind flag for A/B validation.
- Add parity tests against full recompute.

### M3 — Early-exit inference

- Confidence metric (entropy/top-1 margin).
- Exit decision and fallback to next precision.
- Add accuracy-vs-latency evaluation script.

### M4 — Campaigns and analysis

- Up/Down/Constant/Baseline campaign configs.
- Multi-seed runs on M1-256, M1-512, M4 Air.
- Summary + publication-quality tables.

## Run policy

- One long task per node.
- Node-local runtime/cache.
- Immutable result bundles only.
