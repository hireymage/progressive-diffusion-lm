# První 3node smoke: layer-wise grouped precision

Datum: 2026-08-04  
Commit: `ee5737f`

## Co se ověřovalo

Samostatný 25vrstvý prototyp používá pevný rozvrh 5× Q1, 5× Q2, 5× Q4,
5× Q8 a 5× FP16. Sdílený LM head poskytuje mezivýstup po každé vrstvě od
vrstvy 5. Early exit je zatím společný pro celou sekvenci.

## Výsledky

| Node | Test | Výsledek |
|---|---|---|
| m1-256 | schedule a mezivýstupy | vrstvy 5–8 dostupné; cena vrstvy 8 = 11; celý schedule = 155; FP32 reference = 800 |
| m1-512 | 3 kroky deep-supervision tréninku | loss 4,652355 → 2,936595 → 2,508478; gradienty přes výstupy vrstev 5–25 |
| m4-air | fyzický sequence-wide early exit | stop po vrstvě 5; cena 5; vrstvy 6–25 nebyly potřeba |

Všechny tři SSH procesy skončily návratovým kódem 0. Lokálně navíc prošlo
všech 189 testů včetně kontroly skutečné FP16 matmul, přiřazení přesnosti
konkrétním blokům a gradientů v první i poslední vrstvě.

## Interpretace a limit

Test potvrzuje funkčnost architektonické kostry, nikoli kvalitu jazykového
modelu ani reálné zrychlení. Early-exit práh `-1` byl v tomto smoke zvolen
záměrně tak, aby prokázal skutečné přeskočení pozdějších vrstev. Smysluplný
práh se musí kalibrovat až po tréninku na validačních datech.

Zdrojové JSON artefakty jsou v `results/layerwise/first_smoke/`.
