# P1 matched-noise kampaň — 3-uzlový souhrn (2026-07-19)

[English](p1-matched-noise-3node-summary-2026-07-19.md) | [Čeština](p1-matched-noise-3node-summary-2026-07-19.cs.md)

Všechny tři uzly úspěšně skončily (24/24 tréninkové úkoly).

| Uzel | Seeds | Úkoly | Stav |
|---|---|---:|---|
| m1-256 | 11, 29 | 8 | ✅ kompletní |
| m1-512 | 47, 73 | 8 | ✅ kompletní |
| m4-air | 101, 103 | 8 | ✅ kompletní |

## m1-256

| Varianta | Průměrná nejlepší hodnota ztráty | Průměrný čas tréninku | n |
|---|---:|---:|---:|
| clean-fp32 | 7,388631 | 1,401 h | 2 |
| constant-q1 | 7,400684 | 1,506 h | 2 |
| gaussian-matched-fp32 | 7,455089 | 1,584 h | 2 |
| uniform-matched-fp32 | 7,455248 | 1,574 h | 2 |

## m1-512

| Varianta | Průměrná nejlepší hodnota ztráty | Průměrný čas tréninku | n |
|---|---:|---:|---:|
| clean-fp32 | 7,423557 | 1,356 h | 2 |
| constant-q1 | 7,438940 | 1,462 h | 2 |
| gaussian-matched-fp32 | 7,460356 | 1,541 h | 2 |
| uniform-matched-fp32 | 7,461154 | 1,528 h | 2 |

## m4-air

| Varianta | Průměrná nejlepší hodnota ztráty | Průměrný čas tréninku | n |
|---|---:|---:|---:|
| clean-fp32 | 7,452910 | 1,040 h | 2 |
| constant-q1 | 7,441736 | 1.120 h | 2 |
| gaussian-matched-fp32 | 7,453775 | 1,170 h | 2 |
| uniform-matched-fp32 | 7,454047 | 1.200 hod | 2 |

## Kombinováno napříč všemi uzly (6 seeds na variantu)

| Varianta | Průměrná nejlepší hodnota ztráty | Std dev | Průměrný čas tréninku | n |
|---|---:|---:|---:|---:|
| clean-fp32 | 7,421699 | 0,027529 | 1,266 h | 6 |
| constant-q1 | 7,427120 | 0,019807 | 1,363 h | 6 |
| gaussian-matched-fp32 | 7,456407 | 0,002908 | 1,432 h | 6 |
| uniform-matched-fp32 | 7,456817 | 0,003247 | 1,434 h | 6 |

## Pořadí (nižší ztráta je lepší)

1. **clean-fp32** — 7,421699
2. **constant-q1** — 7,427120
3. **gaussian-matched-fp32** — 7,456407
4. **uniform-matched-fp32** — 7,456817

- Δ(constant-q1 − clean-fp32): +0,005420
- Δ(gaussian-matched-fp32 − clean-fp32): +0,034708
- Δ(uniform-matched-fp32 − clean-fp32): +0,035117

## Výklad

- Tato kampaň je kompletní a vnitřně konzistentní napříč 3 uzly a celkem 6 seeds na variantu.
- Výsledky zůstávají specifické pro současnou implementaci (1/2/4 simulovaná kvantizace s plným přepočtem), zatím ne pro budoucí inkrementální 1/2/4/8 design.
- Další fáze by měla implementovat standard 1→2→4→8 / 8→4→2→1 s inkrementálním výpočtem a early-exit inferencí.
