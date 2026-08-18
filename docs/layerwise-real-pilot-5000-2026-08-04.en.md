# Real layer-wise pilot: FP32 versus progressive

[English](layerwise-real-pilot-5000-2026-08-04.en.md) | [Čeština](layerwise-real-pilot-5000-2026-08-04.md)

Date: 2026-08-04

Source commit: `223e71f`

## Verdict

The training pipeline works, but language quality has not yet passed. Both models
reduced loss from approximately 10 to below 7.5, however their unconditioned
generation collapsed to the most frequent tokens. The remaining five control
models therefore have not been launched yet.

## Setup

- English Wikipedia, 69 033 984 tokens;
- vocabulary 16 000 tokens, sequence length 256;
- 25 layers, `d_model=256`, `d_ff=1024`, 8 attention heads;
- 5 000 steps, batch size 1, seed 20260804;
- identical deep supervision on layers 5, 10, 15, 20 and 25;
- evaluation of the best checkpoint on 64 validation batches.

## Results

| Variant and output | Loss | Accuracy | Proxy cost |
|---|---:|---:|---:|
| FP32, layer 5 | 7.4274 | 4.46 % | 160 |
| FP32, layer 10 | 7.4267 | 4.66 % | 320 |
| FP32, layer 25 | 7.4349 | 4.32 % | 800 |
| Progressive, layer 5 (Q1) | **7.3986** | **5.19 %** | **5** |
| Progressive, layer 10 (Q2) | 7.4059 | 4.62 % | 15 |
| Progressive, layer 25 (FP16) | 7.4094 | 4.40 % | 155 |

The best checkpoint of both variants occurred at step 3 500. The FP32 run on m1-512
took 841 seconds, the progressive QAT simulation on m4-air 642 seconds. These times
cannot be used as a hardware comparison because they ran on different machines and
low-bit operations are still simulated.

## Interpretation

The progressive model did not lose against FP32 in the pilot, which justifies
continuing development. At the same time, however, deeper outputs do not add
quality and both models during generation prefer spaces, newlines and very
frequent words. This means that the current loss and generator do not yet
demonstrate a functional language model.

Before we launch Q1, Q2, Q4, Q8 and FP16 controls, we need to diagnose in
particular:

1. token distribution and accuracy against the baseline "most frequent token";
2. learning-rate schedule, number of effectively processed tokens and batch size;
3. weights of auxiliary losses so that early outputs do not impede learning of deeper layers;
4. reconstruction quality according to mask rate and actual progress of diffusion denoising.

The machine-readable summary is in `results/layerwise/real_pilot_5000/summary.json`.
