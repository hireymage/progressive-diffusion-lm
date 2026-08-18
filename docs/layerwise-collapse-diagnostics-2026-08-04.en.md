# Layer-wise Model Collapse Diagnostics

[English](layerwise-collapse-diagnostics-2026-08-04.en.md) | [Čeština](layerwise-collapse-diagnostics-2026-08-04.md)

<!-- doc-status: historical; verified: 2026-08-18 -->
> **Document status:** Historical report. Numbers and conclusions apply to the named campaign and date, not to the project’s current operational state.

Date: 2026-08-04

Source commit: `a3d41ed`

## Verdict

The model is technically capable of learning, but all real pilots to date have been significantly undertrained. The best FP32 checkpoint effectively predicts a new line at every masked position. A larger dataset does not solve this problem now.

## Three Node Tests

### m1-256: frequency baseline

The most frequent token in the training data is a new line, ID 167. It constitutes 3.688% of training tokens, and constant prediction of this token achieves 3.743% on validation.

### m4-air: mask-rate sweep of the best FP32 checkpoint

| Mask rate | Model accuracy | Constant-newline baseline |
|---:|---:|---:|
| 15% | 3.762% | 3.762% |
| 30% | 4.663% | 4.663% |
| 50% | 4.015% | 4.015% |
| 75% | 3.845% | 3.845% |
| 100% | 4.016% | 4.016% |

Exact match at all points proves that this checkpoint uses constant prediction rather than context.

### m1-512: FP32 overfit

On 100 fixed sequences, each sequence received only approximately 40 exposures during 1,000 steps with batch size 4. Loss dropped from 9.887 to 7.020, but accuracy remained at 3.65%, so the 95% gate failed.

Testing on a single sequence achieved approximately 58% accuracy after 1,000 exposures, first exceeded the 95% gate at step 1,500, and reached 100% after 5,000 steps. Final loss was 0.000718 and full sequence reconstruction was exact. This is direct proof that gradients, optimizer, masking, and output head all function.

## Cause and Next Gate

The original 5,000-step pilot with batch size 1 processed only 1.28 million tokens and saw approximately 2% of 256,180 training sequences. It was therefore not a full pass through the 69M-token dataset.

We are not launching another long distributed training yet. First, the FP32 model must:

1. repeat the 95% gate on 100 sequences with a comparable number of exposures per sequence;
2. pass a masking curriculum from 15% toward 100%;
3. only then transition to the full dataset and progressive variant.

## 100-Sequence Quality Gate (Resumable)

`scripts/layerwise_diagnostics.py --mode overfit` now atomically saves a JSON report continuously, and checkpoints `latest` and `best` including optimizer and metadata. `--resume` picks up exactly at the next step; batch order and mask are derived from seed and step number. The gate is always evaluated on the same fixed 50% mask set and only terminates after three consecutive reports above the threshold.

Three limited, directly comparable runs (all: 100 sequences, batch 4, max. 40k steps) are:

| Strategy | Training mask curriculum | LR |
|---|---|---:|
| A | constant 50% for 40k steps | 1e-3 |
| B | 15% / 12k → 30% / 12k → 50% / 16k | 1e-3 |
| C | 15% / 8k → 30% / 8k → 50% / 12k → 75% / 12k | 7e-4 |

For example, strategy B:

```bash
.venv/bin/python scripts/layerwise_diagnostics.py --mode overfit --strategy B \
  --steps 40000 --output results/layerwise/quality-gate-B.json \
  --checkpoint-dir results/layerwise/quality-gate-B-checkpoints
```

The default objective is `final-only`; the experiment with auxiliary milestones is activated explicitly using `--auxiliary-loss weighted-milestones --milestone-weights 5:0.1,10:0.2,15:0.3,20:0.4,25:1.0`.

Machine-readable numbers are in `results/layerwise/diagnostics_2026-08-04/summary.json`.
