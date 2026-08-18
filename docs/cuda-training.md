# PyTorch/CUDA pokračování českého flexibilního modelu

[English](cuda-training.en.md) | [Čeština](cuda-training.md)

CUDA backend zachovává architekturu 25 vrstev, český tokenizer/cache, sdílené
FP32 master váhy a routes `q8_only`, `q8_fp16`, `q2_q8_fp16`. Převádí také
MLX AdamW momenty `m`, `v` a číslo kroku. Používá kompatibilní AdamW bez bias
correction, stejně jako současný MLX trenér.

## Instalace

Na CUDA stroji vytvořte samostatné prostředí. PyTorch nainstalujte příkazem z
oficiálního selectoru pro verzi CUDA podporovanou ovladačem, potom:

```bash
python -m pip install numpy tokenizers
```

Ověření prostředí:

```bash
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

## Bezpečný převod

Nejdříve zastavte MLX trenér nebo vyberte neměnný `step_NNNNNNN.npz` včetně
stejnojmenného JSON souboru. Potom:

```bash
python scripts/convert_mlx_checkpoint_to_torch.py \
  --input /data/checkpoints/step_2110000.npz \
  --output /data/cuda-run/checkpoints/initial.pt
```

Převod odmítne chybějící váhy, optimizer momenty, rozdílné názvy nebo přepsání
existujícího cíle.

## Krátký ověřovací resume

Nejdříve spusťte jen 10 kroků a vyhodnoťte všechny routes:

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

`--steps` je celkový cílový krok, nikoliv počet nových kroků. Po kontrole loss,
accuracy, konečnosti hodnot a rychlosti lze pokračovat z `latest.pt` s vyšším
cílem. Zdrojová `.npz` historie se nikdy nepřepisuje.

## Poznámka k GTX 1080 Ti

Pascal umí CUDA a FP16 operace, ale nemá Tensor Cores. Přínos proto musí potvrdit
krátké měření. Q2/Q8 jsou QAT fake-quant výpočty nad FP32 master vahami, nikoliv
packed integer CUDA kernels.
