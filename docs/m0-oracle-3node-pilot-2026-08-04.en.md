# M0 oracle pilot on three nodes — 2026-08-04

[English](m0-oracle-3node-pilot-2026-08-04.en.md) | [Čeština](m0-oracle-3node-pilot-2026-08-04.md)

<!-- doc-status: historical; verified: 2026-08-18 -->
> **Document status:** Historical report. Numbers and conclusions apply to the named campaign and date, not to the project’s current operational state.

## Verdict

The pilot confirmed that the distributed M0 pipeline is reproducible, but the
current FP32 checkpoint does not provide effective progressive escalation
Q1 → Q2 → Q4 → Q8 → FP32. The oracle manages to select slightly more correct
tokens than the FP32 pass alone, but at the cost of nearly the entire precision
ladder and significantly higher cumulative proxy cost than a single FP32 pass.

This result does not reject the project's new goal. The `full_baseline`
checkpoint was trained in FP32 and lower precisions are merely direct/naive PTQ
probes here. The pilot shows that a model trained only in FP32 is not a suitable
foundation for the expected adaptive behavior and that multi-precision training
will be necessary.

## Configuration

- commit: `b2c5802`
- checkpoint: `checkpoints/full_baseline/step_0010000.npz`
- checkpoint step: 10 000
- precision order: Q1, Q2, Q4, Q8, FP32
- nodes: `m1-256`, `m1-512`, `m4-air`
- fixture seeds: 20260804, 20260805, 20260806
- 10 fixture batches on each node
- 33 338 masked tokens total
- all nodes used the same SHA-256 checkpoint and validation data

## Aggregated Results

| Precision | Masked accuracy | Masked loss |
|---|---:|---:|
| Q1 | 4.496 % | 7.7584 |
| Q2 | 4.703 % | 7.4360 |
| Q4 | 4.787 % | 7.4313 |
| Q8 | 4.805 % | 7.4314 |
| FP32 | 4.805 % | 7.4314 |
| Oracle | 5.255 % | not defined |

The oracle increased masked accuracy over FP32 by approximately 0.45 percentage
points, because it could settle a token at lower precision in cases where the
coarser prediction was correct and the finer prediction was not.

## Precision Transitions

| Transition | Corrected tokens | Newly worsened tokens | Changed prediction |
|---|---:|---:|---:|
| Q1 → Q2 | 200 | 131 | 5 251 |
| Q2 → Q4 | 74 | 46 | 1 964 |
| Q4 → Q8 | 8 | 2 | 144 |
| Q8 → FP32 | 0 | 0 | 18 |

Q8 and FP32 have identical aggregate accuracy on this set. The last transition
changed 18 predictions, but no change converted a correct answer to incorrect
or an incorrect answer to correct.

## Proxy Computation

The original summary of this pilot incorrectly counted internal identifier `16`
as a 16-bit pass. In fact, it is an identity FP32 path. The runs were therefore
repeated on the corrected commit with the same inputs and seeds. All 33 338
per-token records have identical predictions and qualitative metrics; only the
proxy accounting changed.

- actually measured ladder: Q1 → Q2 → Q4 → Q8 → FP32
- correct full proxy cost of this ladder: 47 (`1 + 2 + 4 + 8 + 32`)
- one FP32 reference pass: 32
- stopping after Q4 would cost: 7 (`1 + 2 + 4`)
- corrected oracle average: 44.605 proxy units per token
- oracle savings against the full measured ladder: 5.097 %
- oracle cost against one FP32 pass: 1.394×, i.e. approximately 39.4 % more

The target future ladder is Q1 → Q2 → Q4 → Q8 → FP16 with cost 31 against
FP32=32. However, an actual FP16 stage does not yet exist in this pilot or in
the M0 evaluator.

The proxy metric assumes cost proportional to bit width. Today's MLX
implementation performs simulated full recasts and does not use packed low-bit
kernels or actual residual reuse. The numbers are therefore not a claim about
real hardware speedup.

## Decision for Next Step

1. Do not use this FP32 checkpoint as proof that adaptive inference will be
   efficient.
2. Use per-token pilot data to design the first Pareto analysis of thresholds.
3. Prepare a small model jointly trained in Q1/Q2/Q4/Q8/FP16 (with FP32
   master/reference).
4. Repeat the same M0 probe over the multi-precision checkpoint.
5. Proceed to a learned controller only when higher precisions substantially
   correct more errors than they introduce, and the oracle shows meaningful cost
   versus FP32.

## Artifacts

- `results/m0/pilot_aggregate_summary.json`
- `results/m0/pilot_m1-256_s20260804/`
- `results/m0/pilot_m1-512_s20260805/`
- `results/m0/pilot_m4-air_s20260806/`
