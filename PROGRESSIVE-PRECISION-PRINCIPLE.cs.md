# Progresivní přesnost — princip kanonického experimentu

[English](PROGRESSIVE-PRECISION-PRINCIPLE.md) | [Čeština](PROGRESSIVE-PRECISION-PRINCIPLE.cs.md)

> **Kanonický zdroj pravdy pro návrh experimentu.**
> Tento dokument definuje plný rozsah Progressive Precision LM
> experimentu. Současná kódová základna implementuje pouze podmnožinu této vize.

## Základní myšlenka

Testujte **oba směry** změny bitové přesnosti s **maximálním opětovným použitím již vypočítaných výsledků**. Cílem není sekvenční přetrénování mezi úrovněmi kvantizace, ale skutečně progresivní reprezentace vah a výpočtu, kde lze přesnost dynamicky přidávat nebo odebírat.

## Čtyři experimentální kategorie

### 1. Progresivní nahoru

```
1b → 2b → 4b → 8b
```

Postupně **doplňování informací**:

```
y_1b = base coarse computation
y_2b = y_1b + Δ_2b
y_4b = y_2b + Δ_4b
y_8b = y_4b + Δ_8b
```

Každý krok by měl v ideálním případě **pouze přidat chybějící přesnost** a navazovat na předchozí výsledek, nikoli přepočítávat celou přihrávku vpřed od začátku.

### 2. Progresivní dolů

```
8b → 4b → 2b → 1b
```

Postupné **odstranění přesnosti a informací**. Zkoumá, zda lze nižší přesnosti chápat jako postupně redukované nebo zjednodušené reprezentace vyšší přesnosti.

### 3. Konstantní přesnost

```
always 1b / always 2b / always 4b / always 8b
```

### 4. Baseline

Standardní non-progressive model / plná přesnost na konkrétní experiment. Baseline rodiny jsou sledovány samostatně pro FP16 a FP32 tam, kde je to relevantní.

## Inference — hlavní hypotéza

```
1b → sufficiently confident? → PREDICTION → STOP
1b → not enough? → add information → 2b
2b → not enough? → 4b
4b → not enough? → 8b
```

Vyšší přesnost by měla **přidat k already-computed výsledku**, nikoli přepočítávat celou přihrávku vpřed.

## Porovnané metriky

- kvalita modelu
- ztráta validace
- stabilita tréninku
- uchovávání informací
- inferenční rychlost
- množství načtených dat
- množství skutečně provedených výpočtů
- příležitost pro opětovné použití mezivýsledků

## Klíčové pojmy

```
PROGRESSIVE PRECISION
INCREMENTAL COMPUTATION
BIDIRECTIONAL PRECISION EXPERIMENTS
EARLY EXIT AT INFERENCE
MAXIMUM REUSE OF ALREADY-COMPUTED RESULTS
```

## Rozdíl mezi současnou implementací a tímto principem

Současná kódová základna (k 2026-07-19) implementuje **podmnožinu** této vize:

| Funkce | Stav |
|---|---|
| Úrovně přesnosti | 1b, 2b, 4b (3 úrovně) — **chybí 8b** |
| Progresivní nahoru schedule | ✅ 1→2→4 (částečné) |
| Progresivní dolů schedule | ✅ 4→2→1 (částečné) |
| Neustálá přesnost | ✅ |
| Inkrementální výpočet (yₙ₊₁ = yₙ + Δ) | ❌ každý krok projde úplně vpřed |
| Předčasný odchod při závěru | ❌ |
| Opakované použití mezivýsledku | ❌ |
| 4úrovňové schedule (1→2→4→8) | ❌ |

Současné výsledky (konstantní a progresivní schedules s plným přepočtem) zůstávají platné pro to, co skutečně testují, ale představují pouze první krok k úplnému experimentu, nikoli kompletní vizi.

## Důležitá omezení pro interpretaci

- Nezaměňujte Progressive Up se sekvenčním přetrénováním mezi
  kvantizační úrovně.
- Cílem long-term je zjistit, zda jde o skutečně progresivní váhu
  reprezentace lze vytvořit tam, kde se dynamicky přidává přesnost nebo
  odstraněny a mezivýsledky jsou maximálně znovu použity.
- Při analýze vždy rozlišujte čtyři experimentální kategorie
  výsledky.
