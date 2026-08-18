# Czech flexible model status — 2026-08-18

[English](cswiki-flexible-project-status-2026-08-18.en.md) | [Čeština](cswiki-flexible-project-status-2026-08-18.md)

## What has been verified

The current experiment has verified that one model with shared master weights
can be trained and evaluated for an extended period through three precision
routes:

- `q8_only`,
- `q8_fp16`,
- `q2_q8_fp16`.

This is a Czech diffusion-style masked-token model, not an instruction-tuned
chatbot. Training and validation use only the Czech Wikipedia and a new Czech
BPE tokenizer with a vocabulary of 16,000. No English corpus or English
tokenizer is part of this run.

## Reproducible configuration

| Item | Value |
|---|---:|
| Layers | 25 |
| `d_model` | 64 |
| `d_ff` | 256 |
| Attention heads | 4 |
| Sequence length | 256 |
| Batch | 4 sequences |
| Masking | constant 50% |
| Training chunks | 272,702 |
| Validation chunks | 15,214 |
| Total tokens | 73,706,496 |
| Checkpoint selection | minimum worst-route validation loss |

The cache is versioned and verified with SHA-256. Full checksums are in the
machine-readable snapshot
[`results/layerwise/cswiki_d64_status_2026-08-18/summary.json`](../results/layerwise/cswiki_d64_status_2026-08-18/summary.json).

## Worst-route validation metrics

All three routes are evaluated at every listed point. The table reports the
worst route. Lower loss and perplexity are better; higher accuracy is better.

| Step | Loss | Accuracy | Perplexity |
|---:|---:|---:|---:|
| 500,000 | 4.7227 | 26.97% | 112.47 |
| 1,000,000 | 4.6002 | 28.48% | 99.50 |
| 1,500,000 | 4.5208 | 29.49% | 91.91 |
| 2,000,000 | 4.5276 | 29.85% | 92.54 |
| 2,500,000 | 4.4384 | 30.90% | 84.64 |
| 3,000,000 | 4.4130 | 31.17% | 82.51 |

The best worst-route loss so far is **4.3451 at step 2,891,500**. The final
measurement at 3,000,000 was worse than this best checkpoint, so qualitative
tests must continue to distinguish `best` from `latest`.

## Current resumed run

Training resumed safely from step 3,000,000 and did not restart from zero.
Snapshot at step 3,189,500:

| Route | Loss | Accuracy | Perplexity |
|---|---:|---:|---:|
| `q8_only` | 4.0839 | 35.25% | 59.38 |
| `q8_fp16` | 4.0839 | 35.23% | 59.38 |
| `q2_q8_fp16` | 4.3757 | 31.60% | 79.49 |

The conservative worst route remains `q2_q8_fp16`. The target was explicitly
raised to 20,000,000 steps. `latest` and a possible new `best` are saved after
500-step validation blocks. Immutable long-term snapshots are stored every
100,000 steps after three million to avoid exhausting disk space.

## Preliminary conclusions

1. **The basic technical hypothesis works.** One checkpoint with shared master
   weights remains finite and trainable across all three routes after more than
   three million updates.
2. **The Q8 routes are effectively tied.** `q8_only` and `q8_fp16` have nearly
   identical validation metrics.
3. **The Q2 prefix has a measurable cost.** `q2_q8_fp16` is consistently the
   worst route, but its degradation is bounded and training remains stable.
4. **The model is still learning slowly.** Worst-route loss fell from 4.7227
   to 4.4130 and accuracy rose from 26.97% to 31.17% between 500,000 and three
   million steps.
5. **This is not yet a useful chat model.** Interactive completion now creates
   Czech words and occasionally related phrases, but longer output remains
   unreliable. Accuracy measures masked-token reconstruction, not factual or
   conversational quality.
6. **The data has been reused many times.** Three million updates represent
   roughly 44 passes over the training tokens. Additional steps optimize the
   same distribution rather than adding new knowledge.
7. **The wider model needs stabilization.** The `d_model=128` pilot diverged
   numerically near step 51,500. A new attempt needs a lower learning rate,
   warm-up, gradient clipping, and finite gradient and weight checks.

## Preliminary joint-route benchmark

Two short tests were run on M1-256 from the same step-2,270,000 checkpoint,
each with 300 route forward/backward passes:

- joint mode averaged **6.7% more route passes per second**,
- it showed a larger accuracy increase during this short test,
- alternating mode had a slightly better average reduction in worst-route loss,
- the test was too short to justify changing the primary training strategy.

Joint mode is therefore a promising experiment, not the new default trainer.
The primary run continues to use verified strategy A with alternating routes.

## Implemented backends and operational tools

The project is no longer limited to MLX. MLX on Apple Silicon remains the
verified primary path, while the repository also contains an optional
**PyTorch/CUDA backend**:

- conversion from MLX `.npz` to PyTorch `.pt`, including AdamW moments and step,
- continuation from the converted checkpoint without retraining from zero,
- the same shared FP32 master weights and all three routes,
- cache and checkpoint compatibility checks, finite loss and gradient guards,
  and `latest`, `best`, and immutable archive checkpoints,
- an automated tolerance test for matching Q8 logits between MLX and PyTorch.

The CUDA backend is a functional training backend, not a packed low-bit
runtime. Q2 and Q8 remain STE-based fake quantization over FP32 master weights.
CUDA support therefore does not by itself provide Q2/Q8 memory savings or
acceleration. See [`docs/cuda-training.en.md`](cuda-training.en.md).

The repository also includes tools for:

- remote multi-node monitoring and checkpoint tables every 10,000 steps,
- automatic Czech completion evaluation for new checkpoints,
- generation stop reasons and early-exit layer reporting,
- an HTML checkpoint dashboard,
- interactive completion and experimental whole-sentence refinement,
- short joint-route and distributed experiments.

Training routes independently and averaging their checkpoints remains
experimental and is not the default strategy. Independent branches do not
share identical optimizer state, so simple averaging is not equivalent to one
correct shared training update.

## Interpretation limits

- Low-bit computation is simulated in both MLX and CUDA; this is not evidence
  of real Q2/Q8 acceleration or memory savings.
- The model is intentionally tiny and capacity may be its primary limit.
- Individual validation points fluctuate; long-term trend and the best
  worst-route checkpoint matter more than one local accuracy peak.
- Milestones such as 50% or 75% accuracy and loss 1 cannot be promised from the
  current trend. Longer training measures the ceiling but may not remove it.
- Checkpoints and the full live report remain outside Git because of their
  size. This snapshot contains reproducible parameters and verified summaries.
