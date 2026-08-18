# M0 Pareto Test of Adaptive Stopping — 2026-08-04

[English](m0-pareto-policy-test-2026-08-04.en.md) | [Čeština](m0-pareto-policy-test-2026-08-04.md)

<!-- doc-status: historical; verified: 2026-08-18 -->
> **Document status:** Historical report. Numbers and conclusions apply to the named campaign and date, not to the project’s current operational state.

## Verdict

The offline held-out test found a simple adaptive policy that maintained
masked-token accuracy statistically comparable to a direct FP32 pass at an
average proxy cost of 8.954 instead of 32. This corresponds to a theoretical
reduction in proxy computation of approximately 72%.

The result is the first positive signal for adaptive stopping, not proof of
actual speedup. The current implementation, when building M0 data, fully
recomputed all precisions in simulated FP32 and lacks low-bit kernels as well
as actual skipping of further computation.

## Methodology Without Held-Out Leakage

- calibration: `m1-256`, fixture seed 20260804, 11,167 tokens
- held-out test: `m1-512` and `m4-air`, seeds 20260805 and 20260806
- held-out tokens: 22,171
- candidate thresholds arose solely from feature quantiles of the calibration seed
- calibration ground truth selected the calibration Pareto frontier
- names of selected policies were frozen before held-out evaluation
- held-out ground truth served only for final scoring
- oracle remains separate and is explicitly an unimplementable upper bound

The policy, when making decisions, may read only prediction, confidence,
entropy, margin, and stability against the previously paid step. `target`,
`correct`, `loss`, and oracle fields are not part of the controller's input
data structure.

## Best Calibration-Selected Policy

Policy: `margin_ge_or_le_0.00243663603834`

The rule starts at Q1 and accepts the first step whose top-1/top-2 margin
reaches the calibrated threshold; otherwise it continues to the FP32 fallback.

| Metric | Adaptive policy | Direct Q4 | Direct Q8 | Direct FP32 |
|---|---:|---:|---:|---:|
| Held-out accuracy | 4.9163% | 4.8532% | 4.8802% | 4.8802% |
| Correct tokens | 1,090 | 1,076 | 1,082 | 1,082 |
| Average proxy cost | 8.9536 | 4 | 8 | 32 |

Termination distribution of the adaptive policy:

| Final step | Tokens | Share |
|---|---:|---:|
| Q1 | 10,705 | 48.28% |
| Q2 | 5,258 | 23.72% |
| Q4 | 2,948 | 13.30% |
| Q8 | 57 | 0.26% |
| FP32 fallback | 3,203 | 14.45% |

## Paired-Cluster Bootstrap

The bootstrap uses 2,000 deterministic resamples and preserves the entire pair
`(node/run, fixture_index)` as a cluster. There were 20 held-out clusters in
total. Tokens within a single fixture batch are therefore not treated as
independent samples.

| Adaptive policy comparison | Accuracy delta | 95% CI | Proxy cost delta | Accuracy CI excludes zero |
|---|---:|---:|---:|---|
| vs direct Q1 | +0.2526 pp | +0.0529 to +0.4680 | +7.9536 | yes |
| vs direct Q2 | +0.1443 pp | +0.0224 to +0.3012 | +6.9536 | yes |
| vs direct Q4 | +0.0631 pp | -0.0139 to +0.2045 | +4.9536 | no |
| vs direct Q8 | +0.0361 pp | -0.0139 to +0.1210 | +0.9536 | no |
| vs direct FP32 | +0.0361 pp | -0.0139 to +0.1210 | -23.0464 | no |

Against FP32, therefore, higher quality cannot be claimed. It can be claimed
that on this held-out data no significant accuracy loss was detected and the
policy has substantially lower proxy cost. This conclusion must be repeated on
a model trained in all precisions and with more independent seeds.

## Direct Baselines Versus Ladder Stop

The test separates two different costs:

- `direct_q4` runs only Q4 and costs 4,
- `ladder_stop_q4` goes through Q1 → Q2 → Q4 and costs 7.

Likewise, direct FP32 baseline costs 32, while a pass through the entire current
ladder Q1 → Q2 → Q4 → Q8 → FP32 costs 47. Without this distinction, the
adaptive policy would be compared against artificially expensive baselines.

## Decisions

1. The adaptive controller has sufficient offline signal to continue research.
2. The current margin threshold is a baseline, not the final controller.
3. The next model must be trained simultaneously in Q1/Q2/Q4/Q8/FP16 with an
   FP32 master/reference.
4. After multi-precision training, repeat the same calibration/held-out protocol.
5. Real tokens/s measurements only after implementation of actual early stop
   and low-bit computation.

## Artifacts

- `results/m0/pareto_policy_test/pareto_m0.json`
- `results/m0/pareto_policy_test/pareto_m0_policies.csv`
- `scripts/pareto_m0.py`
