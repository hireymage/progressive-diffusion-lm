# M0 oracle pilot na třech nodech — 2026-08-04

[English](m0-oracle-3node-pilot-2026-08-04.en.md) | [Čeština](m0-oracle-3node-pilot-2026-08-04.md)

<!-- doc-status: historical; verified: 2026-08-18 -->
> **Stav dokumentu:** Historický report. Čísla a závěry platí pro uvedenou kampaň a datum, nikoli jako současný provozní stav projektu.

## Verdikt

Pilot potvrdil, že distribuovaná M0 pipeline je reprodukovatelná, ale současný
FP32 checkpoint neposkytuje efektivní progresivní eskalaci Q1 → Q2 → Q4 → Q8 →
FP32. Oracle dokáže vybrat o něco více správných tokenů než samotný FP32
průchod, avšak za cenu téměř celé precision ladder a výrazně vyšší kumulativní
proxy ceny než jeden FP32 průchod.

Tento výsledek nezamítá nový cíl projektu. Checkpoint `full_baseline` byl učen
ve FP32 a nižší přesnosti jsou zde pouze direct/naive PTQ sondy. Pilot ukazuje,
že model učený jen ve FP32 není vhodným základem pro očekávané adaptivní chování
a že multi-precision trénink bude nezbytný.

## Nastavení

- commit: `b2c5802`
- checkpoint: `checkpoints/full_baseline/step_0010000.npz`
- checkpoint step: 10 000
- precision order: Q1, Q2, Q4, Q8, FP32
- nody: `m1-256`, `m1-512`, `m4-air`
- fixture seedy: 20260804, 20260805, 20260806
- 10 fixture dávek na každý node
- celkem 33 338 maskovaných tokenů
- všechny nody použily shodný SHA-256 checkpointu i validačních dat

## Agregované výsledky

| Přesnost | Masked accuracy | Masked loss |
|---|---:|---:|
| Q1 | 4,496 % | 7,7584 |
| Q2 | 4,703 % | 7,4360 |
| Q4 | 4,787 % | 7,4313 |
| Q8 | 4,805 % | 7,4314 |
| FP32 | 4,805 % | 7,4314 |
| Oracle | 5,255 % | není definováno |

Oracle zvýšil masked accuracy proti FP32 přibližně o 0,45 procentního bodu,
protože mohl ukončit token na nižší přesnosti v případech, kdy byla nižší
predikce správná a jemnější predikce nikoli.

## Přechody mezi přesnostmi

| Přechod | Opravené tokeny | Nově zhoršené tokeny | Změněná predikce |
|---|---:|---:|---:|
| Q1 → Q2 | 200 | 131 | 5 251 |
| Q2 → Q4 | 74 | 46 | 1 964 |
| Q4 → Q8 | 8 | 2 | 144 |
| Q8 → FP32 | 0 | 0 | 18 |

Q8 a FP32 mají na této sadě shodnou agregovanou accuracy. Poslední přechod
změnil 18 predikcí, ale žádná změna nepřevedla správnou odpověď na chybnou ani
chybnou na správnou.

## Proxy výpočet

Původní souhrn tohoto pilotu nesprávně započítal interní identifikátor `16`
jako 16bitový průchod. Ve skutečnosti jde o identity FP32 cestu. Běhy byly proto
zopakovány na opraveném commitu se stejnými vstupy a seedy. Všech 33 338
per-token záznamů má shodné predikce a kvalitativní metriky; změnilo se pouze
proxy účetnictví.

- skutečně měřená ladder: Q1 → Q2 → Q4 → Q8 → FP32
- správná plná proxy cena této ladder: 47 (`1 + 2 + 4 + 8 + 32`)
- jeden FP32 referenční průchod: 32
- zastavení po Q4 by stálo 7 (`1 + 2 + 4`)
- opravený oracle průměr: 44,605 proxy jednotek na token
- oracle úspora proti celé měřené ladder: 5,097 %
- oracle cena proti jednomu FP32 průchodu: 1,394×, tedy přibližně o 39,4 % více

Cílová budoucí ladder je Q1 → Q2 → Q4 → Q8 → FP16 s cenou 31 proti FP32=32.
Skutečný FP16 stupeň však v tomto pilotu ani v M0 evaluatoru dosud neexistuje.

Proxy metrika předpokládá cenu úměrnou bitové šířce. Dnešní MLX implementace
provádí simulované plné přepočty a nepoužívá packed low-bit kernely ani skutečné
reziduální reuse. Čísla proto nejsou tvrzením o reálném hardwarovém zrychlení.

## Rozhodnutí pro další krok

1. Nepoužívat tento FP32 checkpoint jako důkaz, že adaptivní inference bude
   efektivní.
2. Použít per-token pilotní data k návrhu první Pareto analýzy prahů.
3. Připravit malý model učený společně v Q1/Q2/Q4/Q8/FP16 (s FP32 master/reference).
4. Stejnou M0 sondu zopakovat nad multi-precision checkpointem.
5. Pokračovat k naučenému řadiči až tehdy, když vyšší přesnosti opraví podstatně
   více chyb, než kolik jich zavedou, a oracle ukáže smysluplnou cenu vůči FP32.

## Artefakty

- `results/m0/pilot_aggregate_summary.json`
- `results/m0/pilot_m1-256_s20260804/`
- `results/m0/pilot_m1-512_s20260805/`
- `results/m0/pilot_m4-air_s20260806/`
