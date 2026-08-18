# First 3node smoke: layer-wise grouped precision

[English](layerwise-first-3node-smoke-2026-08-04.en.md) | [Čeština](layerwise-first-3node-smoke-2026-08-04.md)

<!-- doc-status: historical; verified: 2026-08-18 -->
> **Document status:** Historical report. Numbers and conclusions apply to the named campaign and date, not to the project’s current operational state.

Date: 2026-08-04
Commit: `ee5737f`

## What was verified

A standalone 25-layer prototype uses a fixed schedule of 5× Q1, 5× Q2, 5× Q4,
5× Q8, and 5× FP16. The shared LM head provides an intermediate output after
every layer from layer 5. Early exit is still shared for the entire sequence.

## Results

| Node | Test | Result |
|---|---|---|
| m1-256 | schedule and intermediate outputs | layers 5–8 available; cost of layer 8 = 11; entire schedule = 155; FP32 reference = 800 |
| m1-512 | 3 steps of deep-supervision training | loss 4.652355 → 2.936595 → 2.508478; gradients through outputs of layers 5–25 |
| m4-air | physical sequence-wide early exit | stop after layer 5; cost 5; layers 6–25 were not needed |

All three SSH processes exited with return code 0. Locally, all 189 tests also
passed, including verification of actual FP16 matmul, precision assignment to
specific blocks, and gradients in both the first and last layer.

## Interpretation and limitation

The test confirms the functionality of the architectural skeleton, not the
quality of the language model or real speedup. The early-exit threshold `-1`
was intentionally chosen in this smoke to demonstrate actual skipping of later
layers. A meaningful threshold must be calibrated after training on validation
data.

Source JSON artifacts are in `results/layerwise/first_smoke/`.
