# Progressive A exit sweep, 2026-08-04
[English](layerwise-exit-sweep-A-2026-08-04.en.md) | [Čeština](layerwise-exit-sweep-A-2026-08-04.md)

<!-- doc-status: historical; verified: 2026-08-18 -->
> **Document status:** Historical report. Numbers and conclusions apply to the named campaign and date, not to the project’s current operational state.

## Question

Does weighted deep supervision actually learn usable layer-wise outputs at
layers 5, 10, 15, and 20, or does it merely alter the final output at layer 25?

## Comparison conditions

- same first 100 training sequences as in the overfit quality-gate,
- same fixed 50% mask with seed `21160804`,
- checkpoint `latest.npz` corresponding to the final report of each run,
- identical architecture 5× Q1 → 5× Q2 → 5× Q4 → 5× Q8 → 5× FP16,
- each output evaluated separately in batches of two sequences.

## Results

| Layer | Active grade | Proxy cost | Weighted accuracy | Final-only accuracy | Weighted loss | Final-only loss |
|---:|---|---:|---:|---:|---:|---:|
| 5 | Q1 | 5 | 31.81 % | 0.64 % | 3.1422 | 19.3337 |
| 10 | Q2 | 15 | 46.36 % | 1.01 % | 2.1511 | 17.3953 |
| 15 | Q4 | 35 | 79.30 % | 4.16 % | 0.7150 | 11.9844 |
| 20 | Q8 | 75 | 93.91 % | 14.45 % | 0.2220 | 6.7312 |
| 25 | FP16 | 155 | 97.26 % | 99.24 % | 0.1056 | 0.0465 |

## Conclusion

Weighted deep supervision demonstrably learns intermediate outputs. The control
final-only model concentrates almost all capability into the last five FP16
layers; its outputs at layers 5–20 are not usable. The weighted model, by
contrast, achieved 93.91% accuracy at layer 20 with proxy cost 75 versus full
cost 155, i.e., at 51.6% lower algorithmic budget.

The output at layer 20 does not yet meet the hard 95% quality-gate. The result
therefore demonstrates a functional intermediate output and the potential for
early-exit, not yet a safe automatic stopping policy or hardware acceleration.
The next step is the same sweep for weighted B and C followed by calibration
of the decision threshold over individual sequences or tokens.

Raw reports are in `results/layerwise/exit_sweep_2026-08-04/`.
