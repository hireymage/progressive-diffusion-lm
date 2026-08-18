# Progressive Precision — Canonical Experiment Principle

[English](PROGRESSIVE-PRECISION-PRINCIPLE.md) | [Čeština](PROGRESSIVE-PRECISION-PRINCIPLE.cs.md)

<!-- doc-status: living; verified: 2026-08-18 -->
> **Document status:** Living documentation, verified against the current code and published results on 2026-08-18.

> **Canonical source of truth for the experiment design.**
> This document separates the long-term principle from what the repository and
> the current Czech experiment have actually verified. Last reviewed against
> the code and published results on 2026-08-18.

## Core idea

Test changes in numerical precision while reusing as much already-computed
information as possible. The long-term goal is not sequentially retraining a
different model for every quantization level. It is one model with shared
master weights whose computation can begin cheaply and add precision or depth
only where confidence is insufficient.

The desired refinement identity is:

```text
y_next = y_previous + delta_missing_information
```

The delta should eventually be cheaper than recomputing the complete forward
pass. Merely subtracting two fully recomputed outputs is an API and correctness
milestone, not the final efficiency result.

## Experimental categories

### 1. Progressive Up

```text
1b -> 2b -> 4b -> 8b -> full precision
```

Gradually add information. Each stage should refine the previous result rather
than replace it with an unrelated computation.

### 2. Progressive Down

```text
full precision -> 8b -> 4b -> 2b -> 1b
```

Gradually remove information to study how much quality survives and whether
the lower-precision representation is a useful reduction of the higher one.

### 3. Constant Precision

```text
always 1b / always 2b / always 4b / always 8b
```

These runs are controls for deciding whether a progressive schedule provides
value beyond simply choosing one precision.

### 4. Full-precision Baseline

A standard non-progressive reference. FP16 and FP32 baselines must be named
separately because they are not interchangeable.

### 5. Flexible layerwise routes

The current long Czech run uses a newer, deployment-oriented experiment: one
checkpoint with shared FP32 master weights is trained across three layerwise
routes:

```text
q8_only
q8_fp16
q2_q8_fp16
```

The active route changes the precision assigned across the 25 layers. It does
not create three independent models. Training alternates routes, evaluates all
of them, and selects checkpoints conservatively by the **worst-route**
held-out validation loss.

This flexible-route experiment complements the four original categories; it
does not erase or retroactively redefine their historical results.

## Inference hypothesis

```text
cheap route / shallow exit
        |
        +-- sufficiently confident -> emit or keep token -> stop
        |
        +-- uncertain -> add layers and/or precision -> evaluate again
```

The final system should support both:

- **precision refinement**, adding higher-precision computation; and
- **depth refinement**, stopping tokens or sequences at an earlier layer when
  confidence is adequate.

The current project contains early-exit and route-by-exit diagnostic paths.
Those diagnostics demonstrate controllable execution and measure proxy cost;
they do not yet prove production latency savings or a calibrated learned
stopping policy.

## Current implementation status

| Capability | Verified status on 2026-08-18 |
|---|---|
| Q1, Q2, Q3, Q4 training/evaluation | Implemented through fake quantization over FP32 master weights |
| Q8 and FP16 layerwise execution | Implemented in the flexible model |
| Progressive Up and Down schedules | Implemented and evaluated for 1/2/4/8-bit Phase 2 experiments |
| Constant-precision controls | Implemented and evaluated |
| Shared-master flexible routes | Implemented: `q8_only`, `q8_fp16`, `q2_q8_fp16` |
| Conservative worst-route checkpointing | Implemented in the Czech flexible trainer |
| Layerwise early exit | Implemented and covered by tests; calibration remains experimental |
| Diffusion-step early exit | Implemented and evaluated as an experimental inference mode |
| Incremental API `y_next = y_previous + delta` | Implemented with parity tests |
| Genuine cheaper residual reuse | **Not implemented**: the current delta path still performs a full forward computation to obtain the delta |
| Packed low-bit kernels | **Not implemented**: low-bit arithmetic is simulated |
| Apple Silicon backend | Implemented with MLX; primary verified training path |
| NVIDIA backend | Implemented with PyTorch/CUDA, checkpoint conversion and resume |

## Current Czech shared-master experiment

The current real-data pilot is intentionally small and Czech-only:

| Item | Value |
|---|---:|
| Layers | 25 |
| `d_model` | 64 |
| `d_ff` | 256 |
| Attention heads | 4 |
| Sequence length | 256 |
| Tokenizer | Czech BPE, vocabulary 16,000 |
| Data | Czech Wikipedia only |
| Training routes | `q8_only`, `q8_fp16`, `q2_q8_fp16` |
| Master weights | Shared FP32 |

At the published three-million-step boundary all routes remained finite. The
worst-route held-out metrics were loss **4.4130**, accuracy **31.17%**, and
perplexity **82.51**. The run then resumed from the checkpoint toward the
explicit 20,000,000-step ceiling. These values measure masked-token
reconstruction, not chatbot quality.

The current evidence supports a limited but important conclusion: one set of
master weights can remain trainable across the three routes for millions of
updates. It does **not** yet establish useful conversational quality, real
low-bit speedups, memory compression, or optimal early-exit behavior.

The reproducible configuration, metric table, limitations, and CUDA status are
recorded in the [current project status](docs/cswiki-flexible-project-status-2026-08-18.en.md).

## Compared metrics

- held-out loss, token accuracy, and perplexity for every route;
- the worst route, not only the average or best route;
- numerical stability and finite gradients/weights;
- generation and masked-token reconstruction quality;
- route-by-exit quality and stopping behavior;
- measured wall-clock latency and memory, clearly separated from proxy cost;
- amount of computation and data actually reused;
- comparison against constant-precision and full-precision controls.

## Interpretation rules

1. Do not call fake quantization a packed low-bit speedup. MLX and CUDA
   currently retain and optimize FP32 master weights.
2. Do not call the present incremental API computational reuse. It preserves
   the refinement equation but still uses full recomputation internally.
3. Do not select a flexible checkpoint by its best route. Decisions are made
   conservatively from the worst route.
4. Do not treat masked-token accuracy as factual, conversational, or general
   language-model accuracy.
5. Keep historical Phase 1/Phase 2 schedule results distinct from the newer
   Czech flexible-route run.
6. Distinguish `best` from `latest`; validation metrics fluctuate locally.
7. Treat early-exit savings as experimental until quality-calibrated stopping
   and measured end-to-end acceleration are demonstrated.

## Success criterion for the full vision

The principle is fully demonstrated only when a shared model can add precision
or depth incrementally, reuse previous computation without a complete forward
recompute, stop safely from calibrated confidence, and show a measured
quality/latency/memory advantage against matched constant-precision and
full-precision baselines.
