# Czech flexible real-data pilot — 2026-08-05

[English](cswiki-flexible-real-pilot-2026-08-05.en.md) | [Čeština](cswiki-flexible-real-pilot-2026-08-05.md)

## Scope

This is the first bounded real-data run of the shared-master flexible model. It is not a claim of a usable language model.

- data: 50,000 Czech Wikipedia articles only
- tokenizer: newly trained Czech byte-level BPE, vocabulary 16,000
- cache: 272,702 train and 15,214 held-out validation sequences, length 256
- model: 25 layers, `d_model=64`, `d_ff=256`, 4 heads
- routes: `q8_only`, `q8_fp16`, `q2_q8_fp16`
- training: strategy A (constant 50% mask), 40,000 updates, batch 4
- selection: minimum worst-route held-out loss

The dump SHA1 was verified as `7501b901ec7889db1460cca3c6d7cc9a1c01ae2c`. Cache and tokenizer metadata include their own SHA256 checksums. No English corpus or tokenizer was used.

## Final held-out validation

| Route | Loss | Perplexity | Masked-token accuracy |
|---|---:|---:|---:|
| `q8_only` | 5.1559 | 173.46 | 22.52% |
| `q8_fp16` | 5.1560 | 173.47 | 22.53% |
| `q2_q8_fp16` | 5.3567 | 212.03 | 20.31% |

The conservative worst route is `q2_q8_fp16`. All routes remained finite and stable. The best checkpoint was the final step 40,000 checkpoint.

## Route × exit diagnostic

The diagnostic used the same fixed 50% mask over the first 32 held-out validation sequences. The conservative worst route was `q2_q8_fp16` at every exit.

| Exit | Precision at exit | Proxy cost | Worst loss | Worst accuracy |
|---:|---|---:|---:|---:|
| 5 | Q2 | 10 | 6.5151 | 9.57% |
| 10 | Q8 | 26 | 6.3781 | 10.06% |
| 15 | Q8 | 66 | 6.3136 | 10.55% |
| 20 | FP16 | 130 | 6.2887 | 10.57% |
| 25 | FP16 | 210 | 6.2871 | 10.69% |

These proxy costs are algorithmic precision accounting, not measured throughput. The smaller diagnostic slice is intentionally different from the fixed random validation batches used during training, so its absolute loss is not directly comparable to the final training report.

## Conclusion

The experiment proves that one shared FP32-master checkpoint can be trained on real Czech data and executed through all three requested precision routes. Q8 and Q8→FP16 are effectively tied; adding the Q2 front section currently causes a measurable but bounded degradation.

The model learned Czech token statistics and reconstruction behavior, but four-pass all-mask generation is still incoherent. This is expected for the deliberately tiny architecture and bounded run. The result is therefore a successful architecture/data-pipeline pilot, not yet a usable Czech language model.

Raw immutable reports:

- `results/layerwise/cswiki_flexible_real_2026-08-05/training_report.json`
- `results/layerwise/cswiki_flexible_real_2026-08-05/diagnostics.json`
