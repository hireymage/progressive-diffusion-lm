# Progressive A/B/C exit policy sweep, 2026-08-04
[English](layerwise-exit-policy-abc-2026-08-04.en.md) | [Čeština](layerwise-exit-policy-abc-2026-08-04.md)

## Conditions

- weighted checkpoints A, B and C after passing the 100-sequential quality-gate,
- same first 100 training sequences,
- same fixed 50% mask with seed `21160804`,
- exits at layers 5, 10, 15, 20 and 25,
- proxy costs 5, 15, 35, 75 and 155.

## Fixed exit at layer 20

| Strategy | Accuracy | Loss | Proxy cost | Proxy savings vs. layer 25 |
|---|---:|---:|---:|---:|
| A weighted | 93.91 % | 0.2220 | 75 | 51.61 % |
| B weighted | 94.03 % | 0.2341 | 75 | 51.61 % |
| C weighted | **96.64 %** | **0.1535** | 75 | 51.61 % |

Only C exceeded the fixed 95% quality-gate already at layer 20. Final-only
controls in the same layer reached only 14.45 %, 20.01 % and 22.39 %. The
difference confirms that weighted deep supervision actually teaches
intermediate outputs.

## Simulated token policy at confidence 0.8

The token exits at the first output where it has top-1 softmax confidence of at
least 0.8 and the prediction is stable against the previous milestone. Otherwise
it continues to layer 25.

| Strategy | Accuracy | Average proxy cost | Simulated proxy savings |
|---|---:|---:|---:|
| A weighted | 96.13 % | 77.32 | 50.12 % |
| B weighted | 96.20 % | 85.58 | 44.79 % |
| C weighted | **97.64 %** | **75.96** | **51.00 %** |

C is the best of the tested strategies. At threshold 0.8 it simultaneously
exceeds the 95% gate and uses approximately half of the full proxy budget. A
higher threshold of 0.9 for C achieves 98.46 % at proxy cost 85.98, thus
simulated savings of 44.53 %.

## Oracle upper bound

The ground-truth oracle selects the first layer that has the correct token. It
is not deployable, but shows the potential of the architecture:

| Strategy | Oracle accuracy | Average proxy cost | Proxy savings |
|---|---:|---:|---:|
| A | 98.56 % | 31.98 | 79.37 % |
| B | 98.57 % | 35.33 | 77.21 % |
| C | **99.29 %** | **29.77** | **80.79 %** |

## Conclusion and limitations

The experiment confirms the algorithmic potential of early exit and selects
strategy C as the candidate for the next phase. It does not prove actual
speedup in tok/s. Today's Transformer still performs dense computation for all
tokens and the current runtime controller is sequence-wide. Token proxy savings
will translate into hardware savings only with sparse/gather computation or a
corresponding kernel.

The next decision gate is precision-flexible overfit training via paths `Q8`,
`Q8 → FP16` and `Q2 → Q8 → FP16`, followed by validation on unseen Wiki-EN
sequences.

Raw reports are in `results/layerwise/exit_sweep_2026-08-04/` and
`results/layerwise/policy_sweep_2026-08-04/`.
