# Český flexibilní pilot na reálných datech — 2026-08-05

[English](cswiki-flexible-real-pilot-2026-08-05.en.md) | [Čeština](cswiki-flexible-real-pilot-2026-08-05.md)

## Rozsah

Jde o první omezený běh flexibilního modelu se sdílenými master vahami na reálných datech. Výsledek není tvrzením, že jde o použitelný jazykový model.

- data: pouze 50 000 článků české Wikipedie,
- tokenizer: nově natrénovaný český byte-level BPE se slovníkem 16 000,
- cache: 272 702 trénovacích a 15 214 oddělených validačních sekvencí délky 256,
- model: 25 vrstev, `d_model=64`, `d_ff=256`, 4 hlavy,
- routes: `q8_only`, `q8_fp16`, `q2_q8_fp16`,
- trénink: strategie A (konstantní maskování 50 %), 40 000 aktualizací, batch 4,
- výběr: minimální held-out loss nejhorší route.

SHA1 dumpu byl ověřen jako `7501b901ec7889db1460cca3c6d7cc9a1c01ae2c`. Metadata cache i tokenizeru obsahují vlastní SHA256 kontrolní součty. Nebyl použit anglický korpus ani anglický tokenizer.

## Konečná held-out validace

| Route | Loss | Perplexita | Přesnost maskovaných tokenů |
|---|---:|---:|---:|
| `q8_only` | 5,1559 | 173,46 | 22,52 % |
| `q8_fp16` | 5,1560 | 173,47 | 22,53 % |
| `q2_q8_fp16` | 5,3567 | 212,03 | 20,31 % |

Konzervativně nejhorší route je `q2_q8_fp16`. Všechny routes zůstaly konečné a stabilní. Nejlepší checkpoint byl konečný checkpoint z kroku 40 000.

## Diagnostika route × exit

Diagnostika použila stejnou pevnou masku 50 % nad prvními 32 oddělenými validačními sekvencemi. Ve všech exitech byla konzervativně nejhorší route `q2_q8_fp16`.

| Exit | Přesnost na exitu | Proxy náklad | Nejhorší loss | Nejhorší přesnost |
|---:|---|---:|---:|---:|
| 5 | Q2 | 10 | 6,5151 | 9,57 % |
| 10 | Q8 | 26 | 6,3781 | 10,06 % |
| 15 | Q8 | 66 | 6,3136 | 10,55 % |
| 20 | FP16 | 130 | 6,2887 | 10,57 % |
| 25 | FP16 | 210 | 6,2871 | 10,69 % |

Proxy náklady vyjadřují algoritmické účtování přesnosti, nikoli změřenou propustnost. Menší diagnostický výřez se záměrně liší od pevných náhodných validačních batchů použitých během tréninku, takže jeho absolutní loss nelze přímo porovnávat s konečným trénovacím reportem.

## Závěr

Experiment prokazuje, že jediný checkpoint se sdílenými FP32 master vahami lze trénovat na reálných českých datech a spouštět přes všechny tři požadované precision routes. Q8 a Q8→FP16 jsou prakticky vyrovnané; přidání úvodní Q2 části zatím způsobuje měřitelné, ale omezené zhoršení.

Model se naučil české tokenové statistiky a rekonstrukční chování, ale čtyřprůchodová generace z úplného zamaskování je stále nesouvislá. U záměrně malé architektury a omezeného běhu je to očekávané. Výsledek je tedy úspěšným pilotem architektury a datové pipeline, nikoli použitelným českým jazykovým modelem.

Neměnné surové reporty:

- `results/layerwise/cswiki_flexible_real_2026-08-05/training_report.json`
- `results/layerwise/cswiki_flexible_real_2026-08-05/diagnostics.json`
