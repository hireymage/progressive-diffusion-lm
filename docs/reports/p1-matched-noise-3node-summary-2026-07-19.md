# P1 matched-noise campaign — 3-node summary (2026-07-19)

[English](p1-matched-noise-3node-summary-2026-07-19.md) | [Čeština](p1-matched-noise-3node-summary-2026-07-19.cs.md)

<!-- doc-status: historical; verified: 2026-08-18 -->
> **Document status:** Historical report. Numbers and conclusions apply to the named campaign and date, not to the project’s current operational state.

All three nodes finished successfully (24/24 training tasks).

| Node | Seeds | Tasks | Status |
|---|---|---:|---|
| m1-256 | 11, 29 | 8 | ✅ complete |
| m1-512 | 47, 73 | 8 | ✅ complete |
| m4-air | 101, 103 | 8 | ✅ complete |

## m1-256

| Variant | Mean best val loss | Mean train time | n |
|---|---:|---:|---:|
| clean-fp32 | 7.388631 | 1.401 h | 2 |
| constant-q1 | 7.400684 | 1.506 h | 2 |
| gaussian-matched-fp32 | 7.455089 | 1.584 h | 2 |
| uniform-matched-fp32 | 7.455248 | 1.574 h | 2 |

## m1-512

| Variant | Mean best val loss | Mean train time | n |
|---|---:|---:|---:|
| clean-fp32 | 7.423557 | 1.356 h | 2 |
| constant-q1 | 7.438940 | 1.462 h | 2 |
| gaussian-matched-fp32 | 7.460356 | 1.541 h | 2 |
| uniform-matched-fp32 | 7.461154 | 1.528 h | 2 |

## m4-air

| Variant | Mean best val loss | Mean train time | n |
|---|---:|---:|---:|
| clean-fp32 | 7.452910 | 1.040 h | 2 |
| constant-q1 | 7.441736 | 1.120 h | 2 |
| gaussian-matched-fp32 | 7.453775 | 1.170 h | 2 |
| uniform-matched-fp32 | 7.454047 | 1.200 h | 2 |

## Combined across all nodes (6 seeds per variant)

| Variant | Mean best val loss | Std dev | Mean train time | n |
|---|---:|---:|---:|---:|
| clean-fp32 | 7.421699 | 0.027529 | 1.266 h | 6 |
| constant-q1 | 7.427120 | 0.019807 | 1.363 h | 6 |
| gaussian-matched-fp32 | 7.456407 | 0.002908 | 1.432 h | 6 |
| uniform-matched-fp32 | 7.456817 | 0.003247 | 1.434 h | 6 |

## Ranking (lower loss is better)

1. **clean-fp32** — 7.421699
2. **constant-q1** — 7.427120
3. **gaussian-matched-fp32** — 7.456407
4. **uniform-matched-fp32** — 7.456817

- Δ(constant-q1 − clean-fp32): +0.005420
- Δ(gaussian-matched-fp32 − clean-fp32): +0.034708
- Δ(uniform-matched-fp32 − clean-fp32): +0.035117

## Interpretation

- This campaign is complete and internally consistent across 3 nodes and 6 total seeds per variant.
- Results remain specific to the current implementation (1/2/4 simulated quantization with full recompute), not yet the future incremental 1/2/4/8 design.
- Next phase should implement standard 1→2→4→8 / 8→4→2→1 with incremental computation and early-exit inference.
