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

Strojově čitelná čísla jsou v
`results/layerwise/diagnostics_2026-08-04/summary.json`.
