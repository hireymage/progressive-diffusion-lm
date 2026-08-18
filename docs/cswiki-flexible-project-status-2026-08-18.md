# Stav českého flexibilního modelu — 2026-08-18

## Co bylo ověřeno

Současný experiment ověřil, že jeden model se sdílenými master vahami lze
dlouhodobě trénovat a vyhodnocovat přes tři přesnostní routes:

- `q8_only`,
- `q8_fp16`,
- `q2_q8_fp16`.

Jde o český diffusion-style masked-token model, nikoli instrukční chatbot.
Trénink i validace používají výhradně českou Wikipedii a nový český BPE
tokenizer s velikostí slovníku 16 000. Anglický corpus ani anglický tokenizer
nejsou součástí tohoto běhu.

## Reprodukovatelná konfigurace

| Položka | Hodnota |
|---|---:|
| Vrstvy | 25 |
| `d_model` | 64 |
| `d_ff` | 256 |
| Hlavy | 4 |
| Délka sekvence | 256 |
| Batch | 4 sekvence |
| Maskování | konstantních 50 % |
| Trénovací bloky | 272 702 |
| Validační bloky | 15 214 |
| Tokeny celkem | 73 706 496 |
| Výběr checkpointu | minimum worst-route validačního loss |

Cache je verzovaná a kontrolovaná SHA-256. Úplné kontrolní součty jsou ve
strojově čitelném snapshotu
[`results/layerwise/cswiki_d64_status_2026-08-18/summary.json`](../results/layerwise/cswiki_d64_status_2026-08-18/summary.json).

## Vývoj worst-route validačních metrik

V každém bodě se vyhodnocují všechny tři routes a tabulka uvádí nejhorší z
nich. Nižší loss a perplexita jsou lepší, vyšší accuracy je lepší.

| Krok | Loss | Accuracy | Perplexita |
|---:|---:|---:|---:|
| 500 000 | 4,7227 | 26,97 % | 112,47 |
| 1 000 000 | 4,6002 | 28,48 % | 99,50 |
| 1 500 000 | 4,5208 | 29,49 % | 91,91 |
| 2 000 000 | 4,5276 | 29,85 % | 92,54 |
| 2 500 000 | 4,4384 | 30,90 % | 84,64 |
| 3 000 000 | 4,4130 | 31,17 % | 82,51 |

Nejlepší dosavadní worst-route loss je **4,3451 na kroku 2 891 500**.
Finální měření na kroku 3 000 000 bylo horší než tento nejlepší checkpoint,
proto musí být pro kvalitativní testy zachováno rozlišení `best` a `latest`.

## Průběžný stav po pokračování

Trénink byl bezpečně obnoven z kroku 3 000 000, nezačal od nuly. Snapshot na
kroku 3 189 500:

| Route | Loss | Accuracy | Perplexita |
|---|---:|---:|---:|
| `q8_only` | 4,0839 | 35,25 % | 59,38 |
| `q8_fp16` | 4,0839 | 35,23 % | 59,38 |
| `q2_q8_fp16` | 4,3757 | 31,60 % | 79,49 |

Konzervativní worst route zůstává `q2_q8_fp16`. Cíl běhu byl na výslovný
pokyn zvýšen na 20 000 000 kroků. `latest` a případný nový `best` se ukládají
po validačních blocích po 500 krocích; neměnné dlouhodobé snapshoty se po
3 milionech ukládají po 100 000 krocích, aby archiv nevyčerpal disk.

## První závěry

1. **Základní hypotéza technicky funguje.** Jeden checkpoint se sdílenými
   master vahami zůstává konečný a trénovatelný přes všechny tři routes i po
   více než třech milionech aktualizací.
2. **Q8 routes jsou prakticky shodné.** `q8_only` a `q8_fp16` mají ve
   validačním snapshotu téměř totožné metriky.
3. **Q2 prefix má měřitelnou cenu.** `q2_q8_fp16` je soustavně nejhorší route,
   ale její degradace je omezená a trénink zůstává stabilní.
4. **Model se stále pomalu učí.** Od 500 tisíc do 3 milionů klesl worst-route
   loss z 4,7227 na 4,4130 a accuracy vzrostla z 26,97 % na 31,17 %.
5. **Nejde zatím o použitelný chatovací model.** Interaktivní doplňování už
   vytváří česká slova a místy tematicky související pokračování, ale dlouhé
   výstupy zůstávají nespolehlivé. Accuracy měří rekonstrukci maskovaných
   tokenů, nikoli faktickou správnost nebo kvalitu konverzace.
6. **Data už byla opakovaně použita.** Při 3 milionech kroků odpovídá objem
   zpracování přibližně 44 průchodům trénovacími tokeny; další kroky nepřidávají
   nové znalosti, pouze dále optimalizují stejné rozdělení.
7. **Větší model vyžaduje stabilizaci.** Pilot `d_model=128` numericky
   divergoval kolem kroku 51 500. To není důkaz, že větší šířka nefunguje, ale
   před dalším během vyžaduje nižší learning rate, warm-up, gradient clipping
   a kontrolu konečnosti gradientů a vah.

## Předběžný joint-route benchmark

Na M1-256 byly ze stejného checkpointu 2 270 000 provedeny dva krátké testy,
každý s 300 route forward/backward průchody:

- joint režim byl průměrně o **6,7 % rychlejší** v route-průchodech za sekundu,
- v krátkém testu měl větší růst accuracy,
- střídavý režim měl v průměru mírně lepší pokles worst-route loss,
- test byl příliš krátký pro změnu hlavní trénovací strategie.

Joint režim je proto označen jako slibný experiment, nikoli jako nový výchozí
trenér. Hlavní běh dál používá ověřenou strategii A se střídáním routes.

## Omezení interpretace

- Low-bit výpočty jsou stále simulované; nejde o důkaz reálného Q2/Q8
  zrychlení ani úspory paměti.
- Model je záměrně velmi malý a jeho kapacita může být hlavním limitem.
- Jednotlivé validační body kolísají; rozhoduje dlouhodobý trend a nejlepší
  worst-route checkpoint, ne jediný lokální vrchol accuracy.
- Milníky 50 % nebo 75 % accuracy a loss 1 nelze ze současného trendu poctivě
  slíbit. Delší trénink ověří strop, ale nemusí jej odstranit.
- Checkpointy a plný průběžný report jsou kvůli velikosti uloženy mimo Git;
  tento snapshot obsahuje reprodukovatelné parametry a ověřené souhrnné metriky.
