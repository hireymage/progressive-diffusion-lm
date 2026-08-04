# M0 oracle pilot na třech nodech — 2026-08-04

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

- plná ladder: 31 proxy bitů na token (`1 + 2 + 4 + 8 + 16`)
- oracle průměr: 29,445 proxy bitů na token
- oracle úspora proti plné ladder: 5,015 %
- oracle cena proti jednomu FP32 průchodu: 1,840×

Proxy metrika předpokládá cenu úměrnou bitové šířce. Dnešní MLX implementace
provádí simulované plné přepočty a nepoužívá packed low-bit kernely ani skutečné
reziduální reuse. Čísla proto nejsou tvrzením o reálném hardwarovém zrychlení.

## Rozhodnutí pro další krok

1. Nepoužívat tento FP32 checkpoint jako důkaz, že adaptivní inference bude
   efektivní.
2. Použít per-token pilotní data k návrhu první Pareto analýzy prahů.
3. Připravit malý model učený společně v Q1/Q2/Q4/Q8/FP32.
4. Stejnou M0 sondu zopakovat nad multi-precision checkpointem.
5. Pokračovat k naučenému řadiči až tehdy, když vyšší přesnosti opraví podstatně
   více chyb, než kolik jich zavedou, a oracle ukáže smysluplnou cenu vůči FP32.

## Artefakty

- `results/m0/pilot_aggregate_summary.json`
- `results/m0/pilot_m1-256_s20260804/`
- `results/m0/pilot_m1-512_s20260805/`
- `results/m0/pilot_m4-air_s20260806/`
