# Progressive-Precision Diffusion Language Model

[English](README.md) | [Čeština](README.cs.md)

> **Private source-only staging snapshot.** Experimental results and public-release documentation are still under validation.

An Apple-MLX research prototype for testing timestep-dependent weight precision in a masked-diffusion language model. The central question is whether coarse, high-noise denoising steps can use lower precision than late, fine-grained refinement steps without materially reducing language-model quality.

## Current scope

This snapshot contains source code, configurations, tests, campaign tooling, and licensing only. It intentionally excludes datasets, tokenizer artifacts, caches, checkpoints, runtime logs, and numerical result bundles.

Previously generated results are being re-audited after reproducibility issues were found in the original research code. Until corrected result manifests are published, this repository makes **no claim that progressive precision, Q1, or any schedule outperforms FP32**.

## Method

- Bidirectional Transformer conditioned on the token mask rate.
- Absorbing/masked discrete diffusion.
- FP32 master weights with simulated low-bit forward passes via a Straight-Through Estimator (STE).
- Runtime precision schedule: schedule index `0` is the highest-noise/coarsest step; the last index is the lowest-noise/finest step.
- Optional matched-noise FP32 controls for separating quantization structure from noise regularization.

Quantization modes:

| Config value | Mode | Forward levels | Stored master weights |
|---:|---|---|---|
| `0` | Optional ternary | `{-1, 0, +1} × scale` | FP32 |
| `1` | Binary Q1 | `{-1, +1} × mean(abs(w))` per row | FP32 |
| `2` | True Q2 | 4 signed levels | FP32 |
| `3` | True Q3 | 8 signed levels | FP32 |
| `4` | True Q4 | 16 signed levels | FP32 |
| `16` | Identity / FP32 baseline | Unquantized | FP32 (32 storage bits) |

All current low-bit operations are simulated using FP32 MLX operations. The implementation has no packed integer weights or low-bit kernels, so it currently provides **no real model-size reduction or low-bit speedup**. A schedule-average bit width is a temporal compute descriptor, not a deployed model size.

## Requirements

- macOS with Apple Silicon and a supported MLX release.
- 8 GB unified memory for the tested small configurations; 16 GB recommended.
- Python 3.11 or compatible.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

python -m pytest -q
./run_smoke_test.sh
```

The smoke workflow trains a small tokenizer when one is not present, prepares a tiny Wikipedia cache, runs the complete test suite, trains tiny baseline/progressive models, performs paired evaluation, and attempts generation.

## Data preparation

```bash
python scripts/train_tokenizer.py \
  --dataset-name wikimedia/wikipedia \
  --dataset-config 20231101.en \
  --dataset-revision b04c8d1ceb2f5cd4588862100d08de323dccfbaa \
  --vocab-size 16000 \
  --max-articles 500 \
  --max-bytes 5000000

python scripts/prepare_data.py \
  --dataset-name wikimedia/wikipedia \
  --dataset-config 20231101.en \
  --dataset-revision b04c8d1ceb2f5cd4588862100d08de323dccfbaa \
  --max-articles 50000 \
  --max-bytes 500000000 \
  --seq-len 256
```

The checked-in workflow pins the current Wikipedia dataset revision shown above and records it in both tokenizer and dataset-cache provenance. Existing historical arrays predate revision pinning and are identified by preserved checksums rather than a retroactively asserted revision.

Dataset-cache identity includes dataset name/config/revision, tokenizer SHA-256, split ratio, seed, sequence length, and collection limits. Cache metadata also records train/validation SHA-256 checksums and is validated before reuse.

## Training

```bash
python -m src.train --config configs/full_baseline.json
python -m src.train --config configs/full_progressive_1_2_4.json
```

The checked-in experiment configurations use `dropout=0.0` and `weight_decay=0.0`, matching the effective protocol of the historical comparison runs. The implementation now supports attention-probability dropout and AdamW weight decay when explicitly enabled in a new configuration; such runs are a different experimental protocol and must not be mixed silently with the legacy controls.

## Evaluation

```bash
python -m src.evaluate \
  --baseline checkpoints/full_baseline/step_0010000.npz \
  --progressive checkpoints/full_progressive_1_2_4/step_0010000.npz \
  --config configs/full_progressive_1_2_4.json \
  --eval-steps 100
```

Comparative evaluation recreates the validation iterator and resets the MLX random seed for each model so both receive matched batches and corruption masks.

## Checkpoints and restart semantics

Checkpoints contain model and optimizer state plus checkpoint-specific step metadata. Training restart restores those states and rejects checkpoints without optimizer state. Iterator position, all RNG state, prior CSV rows, and running best-metric state are not yet restored, so this is a **warm restart**, not a bit-exact continuation.

## Tests

```bash
python -m py_compile src/*.py scripts/*.py tests/*.py
python -m pytest -q
```

The rebuilt source-only tree was verified locally with 132 passing tests before the final staged audit.

## Known limitations

- Small research architecture and limited dataset scale; conclusions cannot be generalized to large LLMs.
- Fake quantization/STE only; no packed storage or low-bit hardware benchmark.
- Dynamic progressive schedules require an explicit deployment representation before any storage-compression claim is valid.
- Historical non-constant schedule results require corrected direction labels because an earlier mapper inverted the documented coarse-to-fine convention.
- Historical runs configured dropout and weight decay but the old implementation did not apply either; those artifacts must be described as unregularized Adam runs.
- Historical dataset arrays have checksums but no provable immutable upstream dataset revision.
- Apple Silicon execution is not guaranteed bitwise deterministic across sessions.

## License

Source code is licensed under the [Apache License 2.0](LICENSE). External datasets, dependencies, tokenizers, generated artifacts, and third-party models retain their own licenses and are not relicensed by this repository.

---

*Experimental research software. No warranty expressed or implied.*
