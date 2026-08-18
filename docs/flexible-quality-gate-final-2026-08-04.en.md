# Flexible multi-route quality-gate, 2026-08-04
[English](flexible-quality-gate-final-2026-08-04.en.md) | [Čeština](flexible-quality-gate-final-2026-08-04.md)

<!-- doc-status: historical; verified: 2026-08-18 -->
> **Document status:** Historical report. Numbers and conclusions apply to the named campaign and date, not to the project’s current operational state.

## Quality-gate A/B/C

One shared model was deterministically cycled between routes `25× Q8`,
`Q8 → FP16`, and `Q2 → Q8 → FP16` during training. Each report evaluated all
routes on the same fixed 50% mask; gate accuracy was the worst accuracy and
gate loss the worst loss across all routes.

| Strategy | Steps | Worst accuracy | Worst loss | Gate |
|---|---:|---:|---:|---|
| A | 27 000 | **98.60 %** | **0.0520** | passed |
| B | 39 000 | 95.59 % | 0.1504 | passed |
| C | 40 000 | 89.77 % | 0.3577 | failed |

A is the winning strategy. It is faster and higher-quality than B; C failed the
hardest route within the given budget.

## Flexible A route×exit

| Route | L5 | L10 | L15 | L20 | L25 |
|---|---:|---:|---:|---:|---:|
| Q8-only | 49.57 % | 88.07 % | 99.55 % | 99.93 % | 99.95 % |
| Q8 → FP16 | 49.57 % | 88.07 % | 99.52 % | 99.92 % | 99.95 % |
| Q2 → Q8 → FP16 | 34.93 % | 59.34 % | 90.86 % | 97.60 % | 98.60 % |

Weighted multi-route training thus learned all three runtime routes and their
intermediate outputs. The hardest Q2 route exceeds the 95% threshold already at
layer 20.

## Simulated stable-confidence policy

At confidence threshold 0.9:

| Route | Accuracy | Average proxy cost | Proxy savings |
|---|---:|---:|---:|
| Q8-only | 99.95 % | 115.81 / 200 | 42.10 % |
| Q8 → FP16 | 99.95 % | 134.53 / 296 | 54.55 % |
| Q2 → Q8 → FP16 | 98.09 % | 104.77 / 210 | 50.11 % |

## Conclusion and limitations

The first stage is complete: one model with shared master weights learns 100
sequences from the start across multiple optional precision routes and provides
usable intermediate outputs. The policy results are an algorithmic simulation,
not measured tok/s; actual token savings require a sparse/gather kernel.

The next stage is a new Czech tokenizer, cswiki-only train/validation cache,
and the first small real flexible model validated on unseen Czech sequences.

Raw reports are in `results/layerwise/flexible_quality_gate_2026-08-04/` and
`results/layerwise/flexible_diagnostics_2026-08-04/`.
