# Kampaň P1 matched-noise – Úplný 3-uzlový souhrn (2026-07-20)

[English](p1-matched-noise-3node-summary-2026-07-20.md) | [Čeština](p1-matched-noise-3node-summary-2026-07-20.cs.md)

<!-- doc-status: historical; verified: 2026-08-18 -->
> **Stav dokumentu:** Historický report. Čísla a závěry platí pro uvedenou kampaň a datum, nikoli jako současný provozní stav projektu.

Všechny tři uzly dokončily dvě fáze kampaně (48/48 tréninkové úkoly celkem):

| Uzel | Hardware | P1 Seeds | P1-next Seeds | Úkoly | Stav |
|---|---|---|---|---:|---|
| m1-256 | M1 8 GB | 11, 29 | 131, 137 | 16 | ✅ kompletní |
| m1-512 | M1 8 GB | 47, 73 | 149, 151 | 16 | ✅ kompletní |
| m4-air | M4 16 GB | 101, 103 | 157, 163 | 16 | ✅ kompletní |

## ID běhů

| Uzel | Fáze | ID běhu |
|---|---|---|
| m1-256 | P1 | `20260719-060513_matched-noise_s11-29_3afd9c76` |
| m1-512 | P1 | `20260719-060530_matched-noise_s47-73_28e8a7fd` |
| m4-air | P1 | `20260719-120028_matched-noise_s101-103_14070a9e` |
| m1-256 | P1-next | `20260719-182527_matched-noise-next_s131-137_ef0786b9` |
| m1-512 | P1-next | `20260719-182528_matched-noise-next_s149-151_4917cde7` |
| m4-air | P1-next | `20260719-210751_matched-noise-next_s157-163_31b3261c` |

## P1 (originál, 6 seeds za variantu)

| Varianta | Průměrná nejlepší hodnota ztráty | Std dev | Průměrný čas tréninku | n |
|---|---:|---:|---:|---:|
| clean-fp32 | 7,421699 | 0,030157 | 1,275 h | 6 |
| constant-q1 | 7,427120 | 0,021697 | 1,363 h | 6 |
| gaussian-matched-fp32 | 7,456407 | 0,003186 | 1,432 h | 6 |
| uniform-matched-fp32 | 7,456817 | 0,003557 | 1,434 h | 6 |

## P1-next (nové, 6 seeds na variantu)

| Varianta | Průměrná nejlepší hodnota ztráty | Std dev | Průměrný čas tréninku | n |
|---|---:|---:|---:|---:|
| clean-fp32 | 7,440174 | 0,028590 | 1,323 h | 6 |
| constant-q1 | 7,442006 | 0,020045 | 1,380 h | 6 |
| gaussian-matched-fp32 | 7,458250 | 0,008061 | 1,451 h | 6 |
| uniform-matched-fp32 | 7,458563 | 0,008336 | 1,436 h | 6 |

## Kombinované P1 + P1-next (12 seeds na variantu)

| Varianta | Průměrná nejlepší hodnota ztráty | Std dev | Průměrný čas tréninku | n |
|---|---:|---:|---:|---:|
| clean-fp32 | 7,430937 | 0,029631 | 1,299 h | 12 |
| constant-q1 | 7,434563 | 0,021379 | 1,372 h | 12 |
| gaussian-matched-fp32 | 7,457329 | 0,005923 | 1,441 h | 12 |
| uniform-matched-fp32 | 7,457690 | 0,006178 | 1,435 h | 12 |

## Pořadí (nižší ztráta je lepší)

1. **clean-fp32** — 7,430937
2. **constant-q1** — 7,434563
3. **gaussian-matched-fp32** — 7,457329
4. **uniform-matched-fp32** — 7,457690

### Rozdíly proti clean-fp32 (kombinované, 12 seeds)

- Δ(constant-q1 − clean-fp32): +0,003626
- Δ(gaussian-matched-fp32 − clean-fp32): +0,026392
- Δ(uniform-matched-fp32 − clean-fp32): +0,026753

## Rozpad P1-next podle nodů

| Uzel | clean-fp32 | constant-q1 | gaussian-matched-fp32 | uniform-matched-fp32 |
|---|---:|---:|---:|---:|
| m1-256 | 7,458103 | 7,443998 | 7,459747 | 7,459800 |
| m1-512 | 7,423768 | 7,425200 | 7,451015 | 7,451283 |
| m4-air | 7,438650 | 7,456819 | 7,463988 | 7,464605 |

## Výklad

- Tato kampaň je kompletní a vnitřně konzistentní napříč 3 uzly a celkem 12 seeds na variantu.
- **clean-fp32** je v průměru nejlepší varianta, těsně následovaná **constant-q1** (Δ = +0,0036).
- Obě matched-noise FP32 varianty (Gaussova a Uniformní) jsou horší než čisté FP32 o ~0,026–0,027, s velmi nízkým rozptylem (std < 0,009).
- Nativní kvantizace Q1 neuškodí zobecnění v tomto modelovém měřítku — deficit Q1 je v rámci jedné standardní odchylky FP32 spreadu.
- Matched-weight-noise FP32 trénink degraduje ztrátu validace více než nativní Q1 kvantizace, což naznačuje, že přínos Q1 není čistě jen regularizační efekt.
- Výsledky zůstávají specifické pro současnou implementaci (1/2/4 simulovaná kvantizace s plným přepočtem), zatím ne pro budoucí inkrementální 1/2/4/8 design.
- Další fáze by měla implementovat standard 1→2→4→8 / 8→4→2→1 s inkrementálním výpočtem a early-exit inferencí.

## Poznámka k výkonu M4 Air

M4 Air (16GB sjednocená paměť) dokončil 8 úkolů za ~9,7 h doby zdi, v průměru ~1,2 h na běh po 10 000 krocích – přibližně o 15 % rychlejší na úlohu než M1 uzly (~1,3–1,4 h). Na 16GB uzlu nebyly pozorovány žádné problémy s tlakem paměti ani s odkládáním.
