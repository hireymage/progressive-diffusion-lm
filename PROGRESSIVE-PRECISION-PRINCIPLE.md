# Progressive Precision — Canonical Experiment Principle

> **Canonical source of truth for the experiment design.**
> This document defines the full scope of the Progressive Precision LM
> experiment. The current codebase implements only a subset of this vision.

## Core idea

Test **both directions** of bit-precision change with **maximum reuse of
already-computed results**. The goal is not sequential retraining between
quantization levels, but a truly progressive representation of weights and
computation where precision can be dynamically added or removed.

## Four experimental categories

### 1. Progressive Up

```
1b → 2b → 3b → 4b → 5b
```

Gradually **adding information**:

```
y_1b = base coarse computation
y_2b = y_1b + Δ_2b
y_3b = y_2b + Δ_3b
y_4b = y_3b + Δ_4b
y_5b = y_4b + Δ_5b
```

Each step should ideally **only add the missing precision** and build on
the previous result, not recompute the full forward pass from scratch.

### 2. Progressive Down

```
5b → 4b → 3b → 2b → 1b
```

Gradually **removing precision and information**. Investigates whether
lower precisions can be understood as progressively reduced or simplified
representations of higher precision.

### 3. Constant Precision

```
always 1b / always 2b / always 3b / always 4b / always 5b
```

### 4. Baseline

Standard non-progressive model / full precision per specific experiment.

## Inference — main hypothesis

```
1b → sufficiently confident? → PREDICTION → STOP
1b → not enough? → add information → 2b
2b → not enough? → 3b
3b → not enough? → 4b
4b → not enough? → 5b
```

Higher precision should **add to the already-computed result**, not
recompute the full forward pass.

## Compared metrics

- model quality
- validation loss
- training stability
- information retention
- inference speed
- amount of data loaded
- amount of computation actually performed
- opportunity for intermediate result reuse

## Key concepts

```
PROGRESSIVE PRECISION
INCREMENTAL COMPUTATION
BIDIRECTIONAL PRECISION EXPERIMENTS
EARLY EXIT AT INFERENCE
MAXIMUM REUSE OF ALREADY-COMPUTED RESULTS
```

## Gap between current implementation and this principle

The current codebase (as of 2026-07-19) implements a **subset** of this
vision:

| Feature | Status |
|---|---|
| Precision levels | 1b, 2b, 4b (3 levels) — **missing 3b and 5b** |
| Progressive Up schedule | ✅ 1→2→4 (partial) |
| Progressive Down schedule | ✅ 4→2→1 (partial) |
| Constant precision | ✅ |
| Incremental computation (yₙ₊₁ = yₙ + Δ) | ❌ each step does full forward pass |
| Early exit at inference | ❌ |
| Intermediate result reuse | ❌ |
| 5-level schedule (1→2→3→4→5) | ❌ |

Current results (constant and progressive schedules with full recompute)
remain valid for what they actually test, but represent only the first
step toward the full experiment, not the complete vision.

## Important constraints for interpretation

- Do not confuse Progressive Up with sequential retraining between
  quantization levels.
- The long-term goal is to determine whether a truly progressive weight
  representation can be created where precision is dynamically added or
  removed and intermediate results are maximally reused.
- Always distinguish the four experimental categories when analyzing
  results.