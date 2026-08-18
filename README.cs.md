# Diffuzní jazykový model s progresivní přesností

[English](README.md) | [Čeština](README.cs.md)

[Dokumentace](docs/) · [Přispívání](CONTRIBUTING.md) · [Bezpečnost](SECURITY.md) · [Citace](CITATION.cff)

> **Experimentální proof-of-concept** pro Apple Silicon a volitelně CUDA. Nejde o produkční chatbot.

**Stav vydání:** výzkumný náhled zdrojového kódu. Součástí tohoto vydání nejsou
váhy modelu; dlouhý český trénink stále pokračuje.

## Cíl projektu

Projekt zkoumá diffusion jazykový model, který začíná hrubou predikcí a postupně
ji zpřesňuje pouze tam, kde je to potřeba. Přesnost se nemá jen mechanicky
měnit podle pevného plánu; model má přidávat výpočet a vyšší přesnost po
krocích, dokud není aktuální rozhodnutí o tokenu dostatečně jisté.

Cílem je model, který:

- generuje text diffusion postupem nad více tokeny současně,
- začíná levným hrubým průchodem,
- přidává další zpřesnění pouze tam, kde zůstává nejistota,
- a přestane zvyšovat přesnost, jakmile je výsledek dostatečný.

Navrženou architekturu, fáze tréninku, metriky a rozhodovací brány popisuje
[`docs/adaptive-progressive-diffusion-design.md`](docs/adaptive-progressive-diffusion-design.md).

---

## Výzkumná hypotéza

Pracovní hypotéza říká, že diffusion LM lze trénovat přes více přesností a
používat je jako fáze zpřesnění při generování. Časné hrubé průchody mohou
používat low-bit výpočet, zatímco pozdější průchody přidají přesnost pouze tam,
kde je sekvence stále nejednoznačná.

**Dosavadní klíčový výsledek** z původních 18 ablačních běhů
(6 variant × 3 seedy × 10 000 kroků):

- binární `const_1bit` skončil první s průměrným `best_val_loss` 7,4336 oproti
  7,4434 u baseline,
- progresivní schedule `[1,1,1,1,2,2,4,4]` byl statisticky vyrovnaný s FP32,
- rozdíly jsou malé a rozptyl mezi seedy velký, takže tři seedy neurčují
  definitivní pořadí,
- všechny low-bit operace jsou simulované v FP32 pomocí STE, takže zatím
  nepřinášejí skutečnou úsporu paměti ani rychlosti.

---

## Architektura

Původní bidirectional Transformer má 28,3 milionu parametrů:

- `d_model=512`, `n_layers=6`, `n_heads=8`, `d_ff=2048`, `max_seq_len=256`,
- každá lineární projekce používá `QuantizedLinear` s přepínáním přesnosti,
- embeddingy a LayerNorm zůstávají FP32,
- LM head sdílí token embedding matrix,
- sinusoidální embedding míry maskování podmiňuje úroveň šumu.

Kvantizační schémata jsou simulovaná v FP32 přes STE:

| Bity | Schéma | Úrovně | Efektivní bity |
|---|---|---|---:|
| 1 | binární | 2: `{-1,+1}` × průměr `|w|` | 1,0 |
| 2 | skutečné 2-bit | 4: `{-3,-1,+1,+3}` × krok | 2,0 |
| 3 | skutečné 3-bit | 8: `{-7,…,+7}` × krok | 3,0 |
| 4 | skutečné 4-bit | 16: `{-15,…,+15}` × krok | 4,0 |
| 16 (interní ID) | FP32 | identita bez kvantizace | 32,0 proxy |
| 0 | ternární, volitelné | 3: `{-1,0,+1}` × maximum `|w|` | přibližně 1,585 |

---

## Rychlý start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

./run_smoke_test.sh

python -m src.train --config configs/full_baseline.json
python -m src.train --config configs/full_progressive_1_2_4.json
```

---

## Příkazy krok za krokem

### Příprava dat

```bash
python scripts/train_tokenizer.py --vocab-size 16000 --max-articles 500 --max-bytes 5000000
python scripts/prepare_data.py --max-articles 50000 --max-bytes 500000000 --seq-len 256
```

### Trénování modelů

```bash
python -m src.train --config configs/full_baseline.json
python -m src.train --config configs/full_progressive_1_2_4.json
python -m src.train --config configs/<vlastni-config>.json
```

### Celá ablační studie

```bash
python scripts/ablation_study.py --phase screen
python scripts/ablation_study.py --phase full --resume
python scripts/ablation_study.py --analyze-only --phase full
```

### Reprodukce PTQ studie

```bash
python scripts/ptq_study.py
python scripts/ptq_study.py --dry-run
python scripts/ptq_study.py --skip-training
```

### Evaluace a generování

```bash
python -m src.evaluate \
    --baseline checkpoints/full_baseline/step_0010000.npz \
    --progressive checkpoints/full_progressive_1_2_4/step_0010000.npz \
    --config configs/full_progressive_1_2_4.json --eval-steps 100

python -m src.generate \
    --checkpoint checkpoints/full_progressive_1_2_4/step_0010000.npz \
    --config configs/full_progressive_1_2_4.json \
    --n-sequences 4 --seq-len 128
```

---

## Stav projektu

> **Aktuální snapshot českého flexibilního modelu (2026-08-18):** model se
> sdílenými master vahami a `d_model=64` dokončil 3 000 000 aktualizací s
> konečnými held-out metrikami na všech třech precision routes a pokračuje
> směrem k 20 000 000 aktualizací. Ověřené metriky, první závěry a omezení jsou
> v [`docs/cswiki-flexible-project-status-2026-08-18.md`](docs/cswiki-flexible-project-status-2026-08-18.md).
> Stejný český checkpoint lze převést a dál trénovat přes volitelný
> PyTorch/CUDA backend; viz [`docs/cuda-training.md`](docs/cuda-training.md).

| Experiment | Stav | Běhy |
|---|---|---:|
| Smoke testy | HOTOVO | 2 varianty, 50 kroků |
| Krátké experimenty | HOTOVO | 3 varianty, seed 42 |
| Úvodní porovnání 10k | HOTOVO | 2 varianty, seed 42 |
| Ablační screening 3k | HOTOVO | 6 variant × 3 seedy = 18 |
| Plná ablace 10k | HOTOVO | 6 variant × 3 seedy = 18 |
| PTQ studie | HOTOVO | 18/18 evaluací Q1/Q2/Q3/Q4/FP32/ternární |

Úplnou technickou dokumentaci původních studií obsahuje český dokument
[`PROJECT_DOCUMENTATION.cs.md`](PROJECT_DOCUMENTATION.cs.md); k dispozici je
také [anglická verze](PROJECT_DOCUMENTATION.md).

---

## Požadavky a hardware

- hlavní backend: macOS 13.5+ na Apple Silicon a MLX,
- volitelný backend: Linux nebo Windows s NVIDIA GPU podporující CUDA a PyTorch,
- CUDA podporuje převod checkpointu a pokračování tréninku, ale pro hlavní MLX
  implementaci není povinná,
- 16 GB unified memory stačí pro malé MLX experimenty, ověřeno na M4 16 GB,
- MLX závislosti: `mlx>=0.21.0`, `tokenizers`, `datasets`, `numpy`, `tqdm`,
- CUDA závislosti a postup: [`requirements-cuda.txt`](requirements-cuda.txt) a
  [`docs/cuda-training.md`](docs/cuda-training.md).

---

## Známá omezení

Nejdůležitější omezení: všechny 1bitové až 4bitové operace jsou simulované v
FP32 přes Straight-Through Estimation. Nejsou implementované packed integer
váhy ani kernely. Odhadovaná komprese proto není v současném wall-clock čase
ani paměti skutečně realizovaná.

Stejné omezení platí pro MLX i CUDA. CUDA backend provádí stejné fake-quant
učení v PyTorch; zatím neposkytuje packed Q2/Q8 kernely ani zaručené low-bit
zrychlení.

Další omezení:

- malý model a dataset, takže závěry nelze automaticky zobecnit na produkční měřítko,
- původní ablace má jen tři seedy,
- Apple Silicon nemusí být mezi samostatnými běhy bitově deterministický,
- starší `const_4bit` ablace používala 15 úrovní, zatímco PTQ používá 16.

---

*Experimentální výzkumný software bez záruky.*
