# Progressive A exit sweep, 2026-08-04

[English](layerwise-exit-sweep-A-2026-08-04.en.md) | [Čeština](layerwise-exit-sweep-A-2026-08-04.md)

<!-- doc-status: historical; verified: 2026-08-18 -->
> **Stav dokumentu:** Historický report. Čísla a závěry platí pro uvedenou kampaň a datum, nikoli jako současný provozní stav projektu.

## Otázka

Naučí weighted deep supervision skutečně použitelné výstupy po vrstvách
5, 10, 15 a 20, nebo pouze změní finální výstup ve vrstvě 25?

## Srovnávací podmínky

- stejných prvních 100 trénovacích sekvencí jako v overfit quality-gate,
- stejná fixní 50% maska se seedem `21160804`,
- checkpoint `latest.npz` odpovídající finálnímu reportu každého běhu,
- shodná architektura 5× Q1 → 5× Q2 → 5× Q4 → 5× Q8 → 5× FP16,
- každý výstup vyhodnocen samostatně v dávkách po dvou sekvencích.

## Výsledky

| Vrstva | Aktivní stupeň | Proxy cena | Weighted accuracy | Final-only accuracy | Weighted loss | Final-only loss |
|---:|---|---:|---:|---:|---:|---:|
| 5 | Q1 | 5 | 31,81 % | 0,64 % | 3,1422 | 19,3337 |
| 10 | Q2 | 15 | 46,36 % | 1,01 % | 2,1511 | 17,3953 |
| 15 | Q4 | 35 | 79,30 % | 4,16 % | 0,7150 | 11,9844 |
| 20 | Q8 | 75 | 93,91 % | 14,45 % | 0,2220 | 6,7312 |
| 25 | FP16 | 155 | 97,26 % | 99,24 % | 0,1056 | 0,0465 |

## Závěr

Weighted deep supervision prokazatelně učí mezivýstupy. Kontrolní final-only
model soustředí téměř všechnu schopnost do posledních pěti FP16 vrstev; jeho
výstupy ve vrstvách 5–20 nejsou použitelné. Weighted model naproti tomu dosáhl
ve vrstvě 20 přesnosti 93,91 % při proxy ceně 75 oproti plné ceně 155, tedy při
o 51,6 % nižším algoritmickém rozpočtu.

Výstup ve vrstvě 20 ještě nesplňuje pevnou 95% quality-gate. Výsledek proto
prokazuje funkční mezivýstup a potenciál early-exitu, nikoli zatím bezpečnou
automatickou stop politiku ani hardwarové zrychlení. Další krok je stejný sweep
pro weighted B a C a následně kalibrace rozhodovacího prahu nad jednotlivými
sekvencemi nebo tokeny.

Surové reporty jsou v `results/layerwise/exit_sweep_2026-08-04/`.
