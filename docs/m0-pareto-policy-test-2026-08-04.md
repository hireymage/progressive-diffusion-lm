# M0 Pareto test adaptivního zastavení — 2026-08-04

## Verdikt

Offline held-out test našel jednoduchou adaptivní politiku, která zachovala
masked-token accuracy statisticky srovnatelnou s přímým FP32 průchodem při
průměrné proxy ceně 8,954 místo 32. To odpovídá teoretickému snížení proxy
výpočtu přibližně o 72 %.

Výsledek je první pozitivní signál pro adaptivní zastavení, nikoli důkaz
reálného zrychlení. Současná implementace při tvorbě M0 dat všechny přesnosti
plně přepočítala v simulovaném FP32 a nemá low-bit kernely ani skutečné
přeskakování dalšího výpočtu.

## Metodika bez held-out leakage

- kalibrace: `m1-256`, fixture seed 20260804, 11 167 tokenů
- held-out test: `m1-512` a `m4-air`, seedy 20260805 a 20260806
- held-out tokeny: 22 171
- kandidátní prahy vznikly pouze z feature kvantilů kalibračního seedu
- kalibrační ground truth vybral kalibrační Pareto frontier
- názvy vybraných politik byly před held-out vyhodnocením zmrazené
- held-out ground truth sloužil pouze k závěrečnému scoringu
- oracle zůstává oddělený a je výslovně neimplementovatelná horní mez

Politika při rozhodování smí číst pouze predikci, confidence, entropy, margin a
stabilitu vůči předchozímu již zaplacenému stupni. `target`, `correct`, `loss` a
oracle pole nejsou součástí vstupní datové struktury řadiče.

## Nejlepší kalibračně vybraná politika

Politika: `margin_ge_or_le_0.00243663603834`

Pravidlo začne na Q1 a přijme první stupeň, jehož top-1/top-2 margin dosáhne
kalibrovaného prahu; jinak pokračuje až k FP32 fallbacku.

| Metrika | Adaptivní politika | Direct Q4 | Direct Q8 | Direct FP32 |
|---|---:|---:|---:|---:|
| Held-out accuracy | 4,9163 % | 4,8532 % | 4,8802 % | 4,8802 % |
| Správné tokeny | 1 090 | 1 076 | 1 082 | 1 082 |
| Průměrná proxy cena | 8,9536 | 4 | 8 | 32 |

Rozložení ukončení adaptivní politiky:

| Konečný stupeň | Tokeny | Podíl |
|---|---:|---:|
| Q1 | 10 705 | 48,28 % |
| Q2 | 5 258 | 23,72 % |
| Q4 | 2 948 | 13,30 % |
| Q8 | 57 | 0,26 % |
| FP32 fallback | 3 203 | 14,45 % |

## Párový cluster bootstrap

Bootstrap používá 2 000 deterministických resamplů a jako cluster zachovává
celou dvojici `(node/run, fixture_index)`. Celkem bylo 20 held-out clusterů.
Tokeny v jedné fixture dávce se tedy nepovažují za nezávislé vzorky.

| Srovnání adaptivní politiky | Accuracy delta | 95% CI | Proxy cost delta | Accuracy CI mimo nulu |
|---|---:|---:|---:|---|
| vs direct Q1 | +0,2526 p. b. | +0,0529 až +0,4680 | +7,9536 | ano |
| vs direct Q2 | +0,1443 p. b. | +0,0224 až +0,3012 | +6,9536 | ano |
| vs direct Q4 | +0,0631 p. b. | −0,0139 až +0,2045 | +4,9536 | ne |
| vs direct Q8 | +0,0361 p. b. | −0,0139 až +0,1210 | +0,9536 | ne |
| vs direct FP32 | +0,0361 p. b. | −0,0139 až +0,1210 | −23,0464 | ne |

Proti FP32 tedy nelze tvrdit vyšší kvalitu. Lze tvrdit, že na těchto held-out
datech nebyla zjištěna průkazná ztráta accuracy a politika má výrazně nižší
proxy cenu. Tento závěr se musí zopakovat nad modelem učeným ve všech
přesnostech a s více nezávislými seedy.

## Přímé baseline versus ladder stop

Test odděluje dvě různé ceny:

- `direct_q4` spustí pouze Q4 a stojí 4,
- `ladder_stop_q4` projde Q1 → Q2 → Q4 a stojí 7.

Stejně tak přímý FP32 baseline stojí 32, zatímco průchod celé současné ladder
Q1 → Q2 → Q4 → Q8 → FP32 stojí 47. Bez tohoto rozlišení by adaptivní politika
byla porovnávána s uměle drahými baseline.

## Rozhodnutí

1. Adaptivní řadič má dostatečný offline signál pro pokračování výzkumu.
2. Současný margin práh je baseline, nikoli finální řadič.
3. Další model musí být učen současně v Q1/Q2/Q4/Q8/FP16 s FP32 master/reference.
4. Po multi-precision tréninku zopakovat stejný kalibrační/held-out protokol.
5. Reálné tokens/s měřit až po implementaci skutečného early stop a low-bit
   výpočtu.

## Artefakty

- `results/m0/pareto_policy_test/pareto_m0.json`
- `results/m0/pareto_policy_test/pareto_m0_policies.csv`
- `scripts/pareto_m0.py`
