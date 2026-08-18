# Progressive A/B/C exit a policy sweep, 2026-08-04

[English](layerwise-exit-policy-abc-2026-08-04.en.md) | [Čeština](layerwise-exit-policy-abc-2026-08-04.md)

<!-- doc-status: historical; verified: 2026-08-18 -->
> **Stav dokumentu:** Historický report. Čísla a závěry platí pro uvedenou kampaň a datum, nikoli jako současný provozní stav projektu.

## Podmínky

- weighted checkpointy A, B a C po splnění 100sekvenční quality-gate,
- stejných prvních 100 trénovacích sekvencí,
- stejná fixní 50% maska se seedem `21160804`,
- výstupy ve vrstvách 5, 10, 15, 20 a 25,
- proxy ceny 5, 15, 35, 75 a 155.

## Pevný výstup ve vrstvě 20

| Strategie | Accuracy | Loss | Proxy cena | Proxy úspora proti vrstvě 25 |
|---|---:|---:|---:|---:|
| A weighted | 93,91 % | 0,2220 | 75 | 51,61 % |
| B weighted | 94,03 % | 0,2341 | 75 | 51,61 % |
| C weighted | **96,64 %** | **0,1535** | 75 | 51,61 % |

Pouze C překonala pevnou 95% quality-gate už ve vrstvě 20. Final-only kontroly
ve stejné vrstvě dosáhly jen 14,45 %, 20,01 % a 22,39 %. Rozdíl potvrzuje, že
weighted deep supervision skutečně učí mezivýstupy.

## Simulovaná tokenová policy při confidence 0,8

Token skončí na prvním výstupu, kde má top-1 softmax confidence alespoň 0,8 a
predikce je stabilní proti předchozímu milníku. Jinak pokračuje do vrstvy 25.

| Strategie | Accuracy | Průměrná proxy cena | Simulovaná proxy úspora |
|---|---:|---:|---:|
| A weighted | 96,13 % | 77,32 | 50,12 % |
| B weighted | 96,20 % | 85,58 | 44,79 % |
| C weighted | **97,64 %** | **75,96** | **51,00 %** |

C je nejlepší z testovaných strategií. Při prahu 0,8 zároveň překonává 95%
bránu a používá přibližně polovinu plného proxy rozpočtu. Vyšší práh 0,9 u C
dosahuje 98,46 % při proxy ceně 85,98, tedy simulované úspoře 44,53 %.

## Oracle horní mez

Ground-truth oracle volí první vrstvu, která má správný token. Není nasaditelný,
ale ukazuje potenciál architektury:

| Strategie | Oracle accuracy | Průměrná proxy cena | Proxy úspora |
|---|---:|---:|---:|
| A | 98,56 % | 31,98 | 79,37 % |
| B | 98,57 % | 35,33 | 77,21 % |
| C | **99,29 %** | **29,77** | **80,79 %** |

## Závěr a omezení

Experiment potvrzuje algoritmický potenciál early-exitu a vybírá strategii C
jako kandidáta pro další fázi. Neprokazuje skutečné zrychlení v tok/s. Dnešní
Transformer stále provádí dense výpočet pro všechny tokeny a aktuální runtime
controller je sequence-wide. Tokenové proxy úspory se změní na hardwarovou
úsporu až se sparse/gather výpočtem nebo odpovídajícím kernelem.

Další rozhodovací brána je precision-flexible overfit trénink cest `Q8`,
`Q8 → FP16` a `Q2 → Q8 → FP16`, následovaný validací na neviděných Wiki-EN
sekvencích.

Surové reporty jsou v `results/layerwise/exit_sweep_2026-08-04/` a
`results/layerwise/policy_sweep_2026-08-04/`.
