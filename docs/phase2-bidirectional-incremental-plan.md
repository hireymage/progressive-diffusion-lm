# Plán fáze 2 — obousměrná inkrementální progresivní přesnost

[English](phase2-bidirectional-incremental-plan.en.md) | [Čeština](phase2-bidirectional-incremental-plan.md)

<!-- doc-status: historical; verified: 2026-08-18 -->
> **Stav dokumentu:** Historický report. Čísla a závěry platí pro uvedenou kampaň a datum, nikoli jako současný provozní stav projektu.

Tento dokument zachycuje implementační fázi po P1 v souladu s hlavním principem:

- Progressive Up: `1b → 2b → 4b → 8b`,
- Progressive Down: `8b → 4b → 2b → 1b`,
- baseline s konstantní přesností,
- baseline v plné přesnosti, kde jsou FP16 a FP32 sledovány odděleně.

## Rozsah

Aktuální kód provádí v každém diffusion kroku **úplný přepočet** pouze s 1/2/4 bity. Fáze 2 zavádí:

1. novou 8bitovou úroveň přesnosti vedle stávajících 1b/2b/4b,
2. rozhraní inkrementálního výpočtu `y_next = y_prev + Δ`,
3. opětovné použití mezivýsledků aktivací tam, kde je platné,
4. inference s předčasným ukončením podle prahu jistoty,
5. obousměrné schedules jako plnohodnotné konfigurace.

## Milníky

### M1 — rozšíření kvantizace

- Přidat 8bitový režim kvantizace do `src/quantization.py`.
- Zachovat 1b/2b/4b jako stávající standardní low-bit varianty.
- Přidat testy úrovní, symetrie a výpočtu úložiště.

### M2 — inkrementální forward API

- Zavést residual/delta cestu ve forward průchodu modelu.
- Zachovat původní úplný přepočet za přepínačem pro A/B ověření.
- Přidat parity testy proti úplnému přepočtu.

### M3 — inference s předčasným ukončením

- Metrika jistoty (entropie / rozdíl top-1).
- Rozhodnutí o ukončení a fallback na další přesnost.
- Přidat skript pro vyhodnocení přesnosti proti latenci.

### M4 — kampaně a analýza

- Konfigurace kampaní Up/Down/Constant/Baseline.
- Běhy s více seedy na M1-256, M1-512 a M4 Air.
- Souhrn a tabulky v publikační kvalitě.

## Pravidla běhů

- Jeden dlouhý úkol na node.
- Runtime a cache lokální pro každý node.
- Pouze neměnné balíčky výsledků.
