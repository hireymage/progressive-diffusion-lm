# PyTorch/CUDA continuation of the Czech flexible model

[English](cuda-training.en.md) | [Čeština](cuda-training.md)

The CUDA backend preserves the 25-layer architecture, Czech tokenizer and
cache, shared FP32 master weights, and the `q8_only`, `q8_fp16`, and
`q2_q8_fp16` routes. It also converts the MLX AdamW `m` and `v` moments and
the global step. It uses the same compatible AdamW update without bias
correction as the current MLX trainer.

## Installation

Create an isolated environment on the CUDA machine. Install PyTorch using the
official selector for the CUDA version supported by the driver, then run:

```bash
python -m pip install numpy tokenizers
```

Verify the environment:

```bash
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

## Safe conversion

First stop the MLX trainer or choose an immutable `step_NNNNNNN.npz` together
with its JSON sidecar. Then run:

```bash
python scripts/convert_mlx_checkpoint_to_torch.py \
  --input /data/checkpoints/step_2110000.npz \
  --output /data/cuda-run/checkpoints/initial.pt
```

The converter rejects missing weights, missing optimizer moments, mismatched
names, and attempts to overwrite an existing destination.

## Short verified resume

Start with only ten additional steps and evaluate every route:

```bash
python scripts/train_cswiki_flexible_cuda.py \
  --resume /data/cuda-run/checkpoints/initial.pt \
  --cache-dir /data/cswiki/cache \
  --output-dir /data/cuda-run/checkpoints \
  --steps 2110010 \
  --eval-every 10 \
  --eval-steps 32 \
  --device cuda
```

`--steps` is the absolute target step, not the number of new steps. After
checking loss, accuracy, finite values, and throughput, resume from `latest.pt`
with a higher target. The source `.npz` history is never overwritten.

## GTX 1080 Ti note

Pascal supports CUDA and FP16 operations but has no Tensor Cores, so any speed
benefit must be established by a short benchmark. Q2/Q8 remain QAT fake-quant
operations over FP32 master weights, not packed integer CUDA kernels.
