# Diagnostika kolapsu layer-wise modelu

Datum: 2026-08-04  
Zdrojový commit: `a3d41ed`

## Verdikt

Model se technicky učit dokáže, ale dosavadní reálné piloty byly výrazně
podtrénované. Nejlepší FP32 checkpoint fakticky předpovídá nový řádek na každé
maskované pozici. Větší dataset tento problém nyní neřeší.

## Tři node testy

### m1-256: frekvenční baseline

Nejčastější token v train datech je nový řádek, ID 167. Tvoří 3,688 % train
tokenů a konstantní předpověď tohoto tokenu dosahuje na validation 3,743 %.

### m4-air: mask-rate sweep nejlepšího FP32 checkpointu

| Mask rate | Model accuracy | Constant-newline baseline |
|---:|---:|---:|
| 15 % | 3,762 % | 3,762 % |
| 30 % | 4,663 % | 4,663 % |
| 50 % | 4,015 % | 4,015 % |
| 75 % | 3,845 % | 3,845 % |
| 100 % | 4,016 % | 4,016 % |

Přesná shoda ve všech bodech dokazuje, že tento checkpoint používá konstantní
předpověď a ne kontext.

### m1-512: FP32 overfit

Na 100 fixních sekvencích dostala každá sekvence během 1 000 kroků a batch size
4 přibližně jen 40 expozic. Loss klesla z 9,887 na 7,020, ale accuracy zůstala
3,65 %, takže 95% gate neprošel.

Kontrola na jediné sekvenci dosáhla po 1 000 expozicích přibližně 58 % accuracy,
na kroku 1 500 poprvé překročila 95% gate a po 5 000 krocích dosáhla 100 %.
Finální loss byla 0,000718 a rekonstrukce celé sekvence byla přesná. To je přímý
důkaz, že gradienty, optimizer, maskování i výstupní head fungují.

## Příčina a další gate

Původní 5 000krokový pilot s batch size 1 zpracoval jen 1,28 milionu tokenů a
viděl přibližně 2 % z 256 180 trénovacích sekvencí. Nešlo tedy o plný průchod
69M-tokenovým datasetem.

Další dlouhý distribuovaný trénink ještě nespouštíme. Nejprve musí FP32 model:

1. zopakovat 95% gate na 100 sekvencích při srovnatelném počtu expozic na sekvenci;
2. projít curriculum maskování od 15 % směrem k 100 %;
3. teprve potom přejít na celý dataset a progresivní variantu.

## 100-sekvenční quality gate (resumovatelná)

`scripts/layerwise_diagnostics.py --mode overfit` nyní ukládá atomický JSON
report průběžně, a checkpointy `latest` a `best` včetně optimizeru a metadat.
`--resume` naváže přesně dalším krokem; batch pořadí i maska jsou odvozené ze
seed a čísla kroku. Gate se vyhodnocuje vždy na stejném fixním 50% mask setu a
skončí až po třech po sobě jdoucích reportech nad prahem.

Tři omezené, přímo porovnatelné běhy (všechny: 100 sekvencí, batch 4, max. 40k
kroků) jsou:

| Strategie | Trénovací mask curriculum | LR |
|---|---|---:|
| A | konstantní 50 % po 40k krocích | 1e-3 |
| B | 15 % / 12k → 30 % / 12k → 50 % / 16k | 1e-3 |
| C | 15 % / 8k → 30 % / 8k → 50 % / 12k → 75 % / 12k | 7e-4 |

Například strategie B:

```bash
.venv/bin/python scripts/layerwise_diagnostics.py --mode overfit --strategy B \
  --steps 40000 --output results/layerwise/quality-gate-B.json \
  --checkpoint-dir results/layerwise/quality-gate-B-checkpoints
```

Výchozí objective je `final-only`; experiment s pomocnými milníky se aktivuje
explicitně pomocí `--auxiliary-loss weighted-milestones --milestone-weights
5:0.1,10:0.2,15:0.3,20:0.4,25:1.0`.

Strojově čitelná čísla jsou v
`results/layerwise/diagnostics_2026-08-04/summary.json`.
