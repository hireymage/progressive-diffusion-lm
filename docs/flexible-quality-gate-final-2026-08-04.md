# Flexible multi-route quality-gate, 2026-08-04

[English](flexible-quality-gate-final-2026-08-04.en.md) | [Čeština](flexible-quality-gate-final-2026-08-04.md)

<!-- doc-status: historical; verified: 2026-08-18 -->
> **Stav dokumentu:** Historický report. Čísla a závěry platí pro uvedenou kampaň a datum, nikoli jako současný provozní stav projektu.

## Quality-gate A/B/C

Jeden sdílený model se při tréninku deterministicky střídal mezi cestami
`25× Q8`, `Q8 → FP16` a `Q2 → Q8 → FP16`. Každý report hodnotil všechny cesty
na stejné fixní 50% masce; gate accuracy byla nejhorší accuracy a gate loss
nejhorší loss ze všech cest.

| Strategie | Kroky | Nejhorší accuracy | Nejhorší loss | Gate |
|---|---:|---:|---:|---|
| A | 27 000 | **98,60 %** | **0,0520** | prošla |
| B | 39 000 | 95,59 % | 0,1504 | prošla |
| C | 40 000 | 89,77 % | 0,3577 | neprošla |

A je vítězná strategie. Je rychlejší i kvalitnější než B; C při daném rozpočtu
nezvládla nejtěžší cestu.

## Flexible A route×exit

| Route | L5 | L10 | L15 | L20 | L25 |
|---|---:|---:|---:|---:|---:|
| Q8-only | 49,57 % | 88,07 % | 99,55 % | 99,93 % | 99,95 % |
| Q8 → FP16 | 49,57 % | 88,07 % | 99,52 % | 99,92 % | 99,95 % |
| Q2 → Q8 → FP16 | 34,93 % | 59,34 % | 90,86 % | 97,60 % | 98,60 % |

Weighted multi-route trénink tedy naučil všechny tři runtime cesty i jejich
mezivýstupy. Nejtěžší Q2 cesta překonává 95% hranici už ve vrstvě 20.

## Simulovaná stable-confidence policy

Při confidence prahu 0,9:

| Route | Accuracy | Průměrná proxy cena | Proxy úspora |
|---|---:|---:|---:|
| Q8-only | 99,95 % | 115,81 / 200 | 42,10 % |
| Q8 → FP16 | 99,95 % | 134,53 / 296 | 54,55 % |
| Q2 → Q8 → FP16 | 98,09 % | 104,77 / 210 | 50,11 % |

## Závěr a omezení

První etapa je splněna: jeden model se sdílenými master vahami se dokáže od
začátku naučit 100 sekvencí přes více volitelných precision cest a poskytuje
použitelné mezivýstupy. Výsledky policy jsou algoritmická simulace, nikoli
naměřené tok/s; skutečná tokenová úspora vyžaduje sparse/gather kernel.

Další etapa je nový český tokenizer, cswiki-only train/validation cache a první
malý reálný flexibilní model validovaný na neviděných českých sekvencích.

Surové reporty jsou v `results/layerwise/flexible_quality_gate_2026-08-04/` a
`results/layerwise/flexible_diagnostics_2026-08-04/`.
