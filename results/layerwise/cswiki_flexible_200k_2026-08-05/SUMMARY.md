# CSWiki flexible model — 200 000 kroků

Datum: 2026-08-05

## Tréninkový quality gate

- Finální checkpoint, krok 200 000: worst route `q2_q8_fp16`, accuracy 24,82 %, loss 4,8830.
- Nejlepší checkpoint podle worst-route loss, krok 198 000: accuracy 25,19 %, loss 4,8655.
- Finální `q8_only`: accuracy 28,71 %, loss 4,5413.
- Finální `q8_fp16`: accuracy 28,71 %, loss 4,5412.
- Všechny tři routes zůstaly konečné; Q8 a Q8+FP16 jsou prakticky shodné.

## Route × exit diagnostika nejlepšího checkpointu

Krátká diagnostika používá 32 held-out sekvencí a proto její absolutní accuracy není přímo
srovnatelná s plným training quality gate.

| Route | vrstva 5 | vrstva 10 | vrstva 15 | vrstva 20 | vrstva 25 |
|---|---:|---:|---:|---:|---:|
| `q8_only` | 15,00 % | 16,59 % | 17,32 % | 17,68 % | 17,75 % |
| `q8_fp16` | 15,00 % | 16,59 % | 17,32 % | 17,68 % | 17,75 % |
| `q2_q8_fp16` | 11,69 % | 12,69 % | 14,17 % | 14,56 % | 14,64 % |

Hloubka modelu pomáhá všem routes. Nízkobitový začátek zůstává hlavním omezením.

## Předběžné generování, 5 nových tokenů

- Česká republika, Q8: `Hlavním městem České republiky je v Praze. Praha.`
- Francie, Q8: `Hlavním městem Francie je největší město (2002)`
- Kočka, Q8: `Kočka leze dírou,, Světová,`
- Q8 a Q8+FP16 v těchto testech generovaly totožný text.
- Oprava diakritiky stále nefunguje spolehlivě.

## Konzervativní závěr

Model se naučil krátké české konstrukce a některé časté vztahy, ale ještě negeneruje
spolehlivý souvislý text ani neopravuje diakritiku. Běh na 200 000 kroků potvrzuje funkční
sdílený flexibilní model a kompatibilitu routes; nepotvrzuje prakticky použitelný jazykový
model. Další delší trénink se bez nového souhlasu nespouští.

