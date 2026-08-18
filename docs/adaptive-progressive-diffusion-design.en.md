# Adaptive Progressive-Diffusion LM — proposal of a new direction

[English](adaptive-progressive-diffusion-design.en.md) | [Čeština](adaptive-progressive-diffusion-design.md)

Proposal date: 2026-08-04

> **Goal update:** The primary experimental branch no longer uses repeated
> full passes Q1 → Q2 → Q4 → Q8. A single 25-layer Transformer has fixed groups
> of 5× Q1, 5× Q2, 5× Q4, 5× Q8, and 5× FP16. From layer 5 onward, computation
> can stop after any layer, for example after layer 8 at the cost of
> `5×Q1 + 3×Q2 = 11` proxy units. The first version terminates the entire
> sequence; token-level early-exit comes only after resolving dependencies across
> self-attention. The older full-pass M0 analysis below remains a historical
> side experiment, not the target architecture implementation.

> **Target modular variant:** Fixed groups are a validation baseline, not the
> final distribution format. The target model has shared master weights and each
> of its layers must be trained for multiple precisions. If only Q8 is
> available, all 25 layers will run in Q8. If Q8 and FP16 are available, depth
> is split between them (first default policy evenly, later according to
> calibration and hardware benchmark). The model must learn different supported
> paths during training; switching quantization at inference alone does not
> guarantee this interchangeability. Precision packages, manifest, and hardware
> path selection are a further milestone after verifying the trainability of the
> fixed baseline.

## 1. Objective

Create a diffusion language model that generates a block of multiple tokens
simultaneously and refines each token position only as long as its prediction
remains uncertain. Computation starts with the cheapest coarse representation
and higher precision is added as a correction to an already computed result.

Basic inference should have this shape:

```text
MASK sequence
  → Q1 coarse prediction of all positions
  → lock sufficiently confident positions
  → Q2 correction of uncertain positions
  → re-evaluate confidence
  → Q4/Q8 correction only where still needed
  → terminate once quality criteria are met
```

The project analogy is a map: world → Europe → country → city. Refinement stops
at the lowest level sufficient for correct decision. It is not necessary to
reach street level.

## 2. Precise definition of terms

Three independent axes must be separated:

1. **Model depth** — number of Transformer layers in one pass.
2. **Diffusion step** — new processing of a partially revealed sequence.
3. **Precision level** — Q1, Q2, Q4, Q8, possibly FP16/FP32.

Higher precision level therefore does not automatically mean "another ten
layers". Layers process the sequence, diffusion steps gradually reveal it, and
the precision level determines the cost and fineness of a particular
computation. Only a later experiment may explore whether to assign different
depth to individual levels.

## 3. Main research hypothesis

A shared model can be trained so that:

- Q1 provides a useful coarse estimate of token distribution,
- each higher level adds residual information instead of an independent new
  prediction,
- simple token positions finish cheaper than ambiguous positions,
- adaptive inference achieves lower average computation at the same quality
  than a model using highest precision at every step.

The project claim will not be "Q1 is faster" until a real low-bit kernel exists.
Until then, we measure algorithmic work and simulated bit budget, not proven
hardware acceleration.

## 4. Proposed architecture

### 4.0 Two separate experimental levels

1. **Fixed baseline:** 5× Q1 → 5× Q2 → 5× Q4 → 5× Q8 → 5× FP16. Serves for
   cheap verification that progressive depth and intermediate outputs can be
   learned at all.
2. **Precision-flexible model:** each layer uses one of currently available
   precisions. A single level covers all layers; multiple levels are split across
   depth according to chosen runtime policy.

Examples for 100 layers:

```text
downloaded Q8:              100× Q8
downloaded Q8 + FP16:        50× Q8 → 50× FP16
downloaded Q2 + Q8 + FP16:   Q2 → Q8 → FP16 in roughly thirds
```

Runtime switching over today's FP32 master weights is only an implementation
baseline. Claims about the quality of these paths will be valid only after
joint training with sampling of various schedules and evaluation of each
supported combination.

### 4.1 Basic denoiser

Keep bidirectional Transformer and absorbing-mask diffusion. The model receives
a partially masked sequence and simultaneously predicts the original token for
all masked positions. The current implementation can already do this and it is
a suitable foundation for the first PD-LM.

### 4.2 Nested precision representation

Weights are to be conceptually decomposed into a base and additional bit
planes:

```text
W_Q1 = B1
W_Q2 = B1 + Δ2
W_Q4 = B1 + Δ2 + Δ4
W_Q8 = B1 + Δ2 + Δ4 + Δ8
```

The same principle applies to logits:

```text
z_Q2 = z_Q1 + δ2
z_Q4 = z_Q2 + δ4
z_Q8 = z_Q4 + δ8
```

This is a target property, not an assumption that today's `set_bits()` already
satisfies it. Current quantization switching uses the same master weights, but
by itself does not guarantee precise additive decomposition or computational
savings.

### 4.3 Adaptive controller

After each level, the controller makes a decision for each masked position:

- **commit** — token is sufficiently certain and can be revealed,
- **defer** — position remains masked for the next diffusion step,
- **escalate** — position needs higher precision level in this step.

The first version will use transparent rules based on:

- entropy of distribution,
- difference between top-1 and top-2 probabilities,
- stability of top-1 token between two precision levels.

A learned controller will come only after a reliable baseline and logs for its
training exist. High top-1 probability alone is insufficient, because the model
can be overconfident and yet wrong.

### 4.4 Granularity of computation

The first prototype will use a global pass at a given precision, but decisions
and revealing will be per individual token positions. Actual computation of
higher precision only for selected positions requires sparse/gather computation
or a custom kernel and belongs to a later milestone.

This separation will first allow verification of whether the decision
principle is correct, and only then invest in low-level optimization.

## 5. Training Strategy

### Phase A — Functional PD-LM Baseline

Train a small model in a single stable precision such that it actually generates
readable token sequences via masked diffusion. The purpose is to separate
generation problems from progressive-precision problems.

### Phase B — Joint Multi-Precision Training

Each batch samples both mask rate and precision level. The model optimizes
reconstruction loss across all supported precisions so that Q1 is not merely a
post-training degradation of an FP model.

Proposed loss:

```text
L = L_reconstruction
  + λ_distill · L_coarse_to_fine
  + λ_stability · L_prediction_stability
  + λ_cost · L_expected_compute
```

- `L_reconstruction`: correct token at masked positions for each level.
- `L_coarse_to_fine`: coarse distributions learn from finer level but need not
  copy its full sharpness.
- `L_prediction_stability`: penalizes unnecessary changes to already-correct
  predictions when precision is increased.
- `L_expected_compute`: added only once the learned controller is in place and
  penalizes unnecessary escalations.

### Phase C — Residual Precision

Replace mere repeated quantization with explicit residual/bit-plane blocks and
verify that `base + delta` matches full computation within defined tolerance.

### Phase D — Learned Stopping

The controller learns the probability that the next precision level will change
a wrong or unstable prediction. The goal is to minimize quality and cost jointly,
not to maximize confidence per se.

## 6. First-Version Inference

For each diffusion step:

1. Run Q1 over the current sequence.
2. Evaluate confidence of each masked position.
3. Reveal positions meeting the safe-commit threshold.
4. For remaining positions, escalate to Q2 and monitor distribution change.
5. Repeat for Q4 and Q8, up to the specified budget at most.
6. Positions that remain unsafe are kept masked for the next diffusion step;
   in the final step, use the defined fallback.

Decisions must be driven by both absolute confidence and stability across levels.
For example, a position commits if it has low entropy, sufficient top-1/top-2
margin, and its top-1 token did not change after adding one level.

## 7. What to Measure Exactly

### Quality

- masked-token accuracy and cross-entropy by mask rate,
- quality of completed text against FP16/FP32 baseline,
- proportion of tokens that were locked incorrectly,
- number of top-1 token changes between Q1 → Q2 → Q4 → Q8,
- confidence calibration, e.g., ECE or reliability bins.

### Adaptive Compute

- average achieved precision level per token,
- proportion of positions committed at Q1, Q2, Q4, and Q8,
- number of diffusion steps per token position,
- normalized bit-operation proxy budget,
- number of actually performed full and residual operations.

### System Metrics

- wall-clock latency and throughput,
- peak memory and loaded weight size,
- results separately for simulated quantization and later for actual low-bit
  kernels.

The project's main graph will be the Pareto curve of quality versus compute budget.

## 8. Baselines and Ablations

Every adaptive experiment must be compared against at least:

1. FP16/FP32 at all diffusion steps.
2. Constant Q1, Q2, Q4, and Q8.
3. Fixed progressive schedule without early exit.
4. Adaptive schedule with the same maximum number of steps.
5. Oracle controller that determines from ground truth whether additional precision helped.

The oracle does not belong in real inference. It establishes the upper bound on
whether adaptive refinement has sufficient potential at all. If the oracle does
not yield a useful Pareto benefit, there is no point yet in building a complex
learned controller or custom kernels.

## 9. Milestones and Decision Gates

### M0 — specification and reproducible baseline

- unify definitions of layers, diffusion steps, and precision levels,
- measure quality of current generation,
- save samples, checkpoint, configuration, and metrics.

**Gate:** the current model must demonstrably generate and the run must be
reproducible on all three nodes.

### M1 — minimal adaptive inference

- token-wise confidence logits for all precisions,
- rule-based controller,
- fixed versus adaptive comparison,
- oracle analysis.

**Gate:** both oracle and rule-based controller must show measurable quality
advantage per unit of proxy compute. If only oracle shows it, the controller is
improved. If neither shows it, training or precision representation is revised.

### M2 — model trained in all precisions

- multi-precision sampling in training,
- distillation and stability loss,
- comparison with today's fixed-schedule checkpoints.

**Gate:** Q1/Q2 must be useful coarse predictors and higher precision must fix
more errors than it introduces.

### M3 — true additive refinement

- nested/bit-plane weights,
- verified `base + delta` computation,
- accounting for actually saved operations.

**Gate:** the incremental path result matches the reference and requires less
work than full recomputation.

### M4 — system optimization

- selective computation for uncertain positions,
- packed low-bit weights and appropriate Apple Silicon kernels,
- measurement of real latency and memory.

**Gate:** speedup must exist in real wall-clock measurement, not just in proxy
metric.

## 10. Single user versus multiple users

Adaptive precision does not make the model a fundamentally single-user system.
Individual requests or tokens do not "fight" over precision; the controller merely
chooses different further work for them. The practical problem is efficient
batching: sequences ending at different precision levels break regular batching.

The first implementation should be optimized for a single user, because this most
purely measures latency and adaptive behavior. The multi-user version may later
group waiting positions or requests by current precision level.

## 11. Nearest experiment

The shortest path to a decision is not to immediately train a new large model.
First, for the same set of masked inputs, save logits Q1, Q2, Q4, Q8, and
FP16/FP32 over the existing checkpoint and create an oracle analysis:

- how many positions are already correct in Q1,
- how many erroneous positions higher precision fixes,
- how many correct positions it instead corrupts,
- whether these cases can be predicted from entropy, margin, and stability,
- what is the best achievable quality versus proxy compute curve.

This experiment cheaply verifies the very assumption of adaptive refinement.
Only then does it make sense to implement M1 and subsequently train the first
model under the new strategy.

## 12. Success criteria for the first stage

The first stage is successful if:

- a reproducible small PD-LM exists that generates multiple tokens at once,
- oracle demonstrates room for adaptive decision-making,
- rule-based controller achieves comparable quality as highest precision at lower
  average proxy budget,
- results hold across multiple seeds and are not derived from a single example,
- report clearly separates algorithmic savings from actual hardware speedup.

## 13. M0 implementation

The first evaluator is available as `scripts/oracle_m0.py`. Over identical
deterministic masked inputs it compares Q1, Q2, Q4, Q8, and FP32, measures fixed
and newly introduced errors, and saves `summary.json` and compact
`per_token.csv`. Full vocabulary logits are not saved and only one fixture batch
is processed at runtime so that analysis does not have high memory requirements.

Example over an existing 10k FP32 checkpoint:

```bash
.venv/bin/python scripts/oracle_m0.py \
  --config configs/full_baseline.json \
  --checkpoint checkpoints/full_baseline/step_0010000.npz \
  --validation-data data/cache/val_seq256_art50000_bytes500000000.npy \
  --output-dir results/m0/<run-name> \
  --eval-steps 20 \
  --fixture-seed 20260804
```

The output distinguishes final chosen precision from cumulative cost of all
visited levels. Wall-clock data describe today's simulated full recomputations,
not speed of future low-bit kernels. Because the mentioned checkpoint was trained
in FP32, its Q1/Q2/Q4/Q8 results are only a preliminary M0/PTQ probe, not a test
of a future model trained simultaneously in all precisions.

In the current evaluator, the internal identifier `16` is a compatible label for
the identity FP32 path: it uses FP32 master weights and FP32 computation and
has proxy cost 32, not 16. The measured ladder Q1 → Q2 → Q4 → Q8 → FP32
therefore costs 47 units (`1 + 2 + 4 + 8 + 32`); stopping at Q4 costs 7. The
target, not-yet-implemented ladder Q1 → Q2 → Q4 → Q8 → FP16 would cost 31 versus
32 for a single FP32 pass. M0 does not yet contain an actual FP16 level.

Results from multiple nodes are joined only on matching provenance:

```bash
.venv/bin/python scripts/aggregate_oracle_m0.py \
  results/m0/<run-a> results/m0/<run-b> results/m0/<run-c> \
  --output results/m0/aggregate_summary.json
```

The aggregator refuses to mix different commit, checkpoint, validation data,
configuration, or precision ordering.
