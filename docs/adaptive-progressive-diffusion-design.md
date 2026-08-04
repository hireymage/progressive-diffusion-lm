# Adaptive Progressive-Diffusion LM — návrh nového směru

Datum návrhu: 2026-08-04

> **Aktualizace cíle:** Primární experimentální větev už nepoužívá opakované
> celé průchody Q1 → Q2 → Q4 → Q8. Jeden 25vrstvý Transformer má pevné skupiny
> 5× Q1, 5× Q2, 5× Q4, 5× Q8 a 5× FP16. Od vrstvy 5 lze výpočet ukončit po
> libovolné vrstvě, například po vrstvě 8 za cenu `5×Q1 + 3×Q2 = 11` proxy
> jednotek. První verze ukončuje celou sekvenci; tokenové early-exit přijde až
> po vyřešení vazeb přes self-attention. Starší full-pass M0 analýza níže
> zůstává historickým vedlejším experimentem, nikoli implementací cílové
> architektury.

## 1. Cíl

Vytvořit difuzní jazykový model, který generuje blok více tokenů současně a
každou tokenovou pozici zpřesňuje pouze tak dlouho, dokud je její predikce
nejistá. Výpočet začíná nejlevnější hrubou reprezentací a vyšší přesnost se
přidává jako oprava již vypočteného výsledku.

Základní inference má mít tento tvar:

```text
MASK sekvence
  → Q1 hrubá predikce všech pozic
  → uzamknout dostatečně jisté pozice
  → Q2 oprava nejistých pozic
  → znovu vyhodnotit jistotu
  → Q4/Q8 oprava pouze tam, kde je stále potřeba
  → ukončit, jakmile jsou splněna kritéria kvality
```

Analogie projektu je mapa: svět → Evropa → stát → město. Zpřesňování končí na
nejnižší úrovni, která stačí pro správné rozhodnutí. Není nutné dojít až k ulici.

## 2. Přesné vymezení pojmů

Je nutné oddělit tři nezávislé osy:

1. **Hloubka modelu** — počet Transformerových vrstev v jednom průchodu.
2. **Difuzní krok** — nové zpracování částečně odhalené sekvence.
3. **Stupeň přesnosti** — Q1, Q2, Q4, Q8, případně FP16/FP32.

Vyšší stupeň přesnosti proto automaticky neznamená „dalších deset vrstev“.
Vrstvy zpracovávají sekvenci, difuzní kroky ji postupně odhalují a stupeň
přesnosti určuje cenu a jemnost konkrétního výpočtu. Teprve pozdější experiment
může zkoumat, zda jednotlivým stupňům přiřadit také odlišnou hloubku.

## 3. Hlavní výzkumná hypotéza

Sdílený model lze natrénovat tak, aby:

- Q1 poskytla užitečný hrubý odhad distribuce tokenů,
- každý vyšší stupeň přidal reziduální informaci místo nezávislé nové predikce,
- jednoduché tokenové pozice skončily levněji než nejednoznačné pozice,
- adaptivní inference dosáhla při stejné kvalitě menšího průměrného výpočtu než
  model používající nejvyšší přesnost při každém kroku.

Tvrzení projektu nebude „Q1 je rychlejší“, dokud neexistuje skutečný low-bit
kernel. Do té doby měříme algoritmickou práci a simulovaný bitový rozpočet,
nikoli prokázané hardwarové zrychlení.

## 4. Navržená architektura

### 4.1 Základní denoiser

Zachovat bidirekční Transformer a absorbing-mask difuzi. Model dostane částečně
maskovanou sekvenci a současně předpoví původní token pro všechny maskované
pozice. To už současná implementace umí a je to vhodný základ prvního PD-LM.

### 4.2 Vnořená reprezentace přesnosti

Váhy mají být koncepčně rozloženy na základ a přídavné bitové roviny:

```text
W_Q1 = B1
W_Q2 = B1 + Δ2
W_Q4 = B1 + Δ2 + Δ4
W_Q8 = B1 + Δ2 + Δ4 + Δ8
```

Stejný princip platí pro logity:

```text
z_Q2 = z_Q1 + δ2
z_Q4 = z_Q2 + δ4
z_Q8 = z_Q4 + δ8
```

Toto je cílová vlastnost, ne předpoklad, že ji dnešní `set_bits()` už splňuje.
Současné přepnutí kvantizace používá stejné master váhy, ale samo o sobě
nezaručuje přesnou aditivní dekompozici ani úsporu výpočtu.

### 4.3 Adaptivní řadič

Po každém stupni vytvoří řadič pro každou maskovanou pozici rozhodnutí:

- **commit** — token je dostatečně jistý a může se odhalit,
- **defer** — pozice zůstává maskovaná do dalšího difuzního kroku,
- **escalate** — pozice potřebuje vyšší stupeň přesnosti v tomto kroku.

První verze použije průhledná pravidla založená na:

- entropii distribuce,
- rozdílu pravděpodobností top-1 a top-2,
- stabilitě top-1 tokenu mezi dvěma stupni přesnosti.

Naučený řadič přijde až poté, co bude existovat spolehlivý baseline a logy pro
jeho trénování. Samotná vysoká top-1 pravděpodobnost nestačí, protože model může
být sebejistý a přitom chybný.

### 4.4 Granularita výpočtu

První prototyp použije globální průchod danou přesností, ale rozhodnutí a
odhalování budou po jednotlivých tokenových pozicích. Skutečný výpočet vyšší
přesností pouze pro vybrané pozice vyžaduje sparse/gather výpočet nebo vlastní
kernel a patří do pozdějšího milníku.

Toto rozdělení umožní nejprve ověřit, zda je rozhodovací princip správný, a až
potom investovat do nízkoúrovňové optimalizace.

## 5. Tréninková strategie

### Fáze A — funkční PD-LM baseline

Natrénovat malý model v jedné stabilní přesnosti tak, aby skutečně generoval
čitelné tokenové sekvence pomocí maskované difuze. Účelem je oddělit problémy
samotného generování od problémů progresivní přesnosti.

### Fáze B — společný multi-precision trénink

Každý batch vzorkuje mask rate i stupeň přesnosti. Model optimalizuje
rekonstrukční loss ve všech podporovaných přesnostech, aby Q1 nebyla pouze
post-training degradací FP modelu.

Navržený loss:

```text
L = L_reconstruction
  + λ_distill · L_coarse_to_fine
  + λ_stability · L_prediction_stability
  + λ_cost · L_expected_compute
```

- `L_reconstruction`: správný token na maskovaných pozicích pro každý stupeň.
- `L_coarse_to_fine`: hrubé distribuce se učí od jemnějšího stupně, ale nemusí
  kopírovat jeho úplnou ostrost.
- `L_prediction_stability`: penalizuje zbytečné změny již správných predikcí při
  zvýšení přesnosti.
- `L_expected_compute`: přidá se až s naučeným řadičem a trestá zbytečné
  eskalace.

### Fáze C — reziduální přesnost

Nahradit pouhé opakované kvantování explicitními reziduálními/bit-plane bloky a
ověřit, že `base + delta` odpovídá plnému výpočtu v definované toleranci.

### Fáze D — naučené zastavení

Řadič se učí pravděpodobnost, že další přesnost změní chybnou nebo nestabilní
predikci. Cílem je minimalizovat kvalitu a cenu společně, nikoli maximalizovat
samotnou sebejistotu.

## 6. Inference první verze

Pro každý difuzní krok:

1. Spustit Q1 nad aktuální sekvencí.
2. Vyhodnotit jistotu každé maskované pozice.
3. Odhalit pozice splňující bezpečný commit práh.
4. Pro zbylé pozice přejít na Q2 a sledovat změnu distribuce.
5. Opakovat pro Q4 a Q8, nejvýše do stanoveného rozpočtu.
6. Pozice, které stále nejsou bezpečné, ponechat maskované do dalšího difuzního
   kroku; v posledním kroku použít definovaný fallback.

Rozhodnutí musí být vedeno jak absolutní jistotou, tak stabilitou mezi stupni.
Například pozice skončí, pokud má nízkou entropii, dostatečný top-1/top-2 margin
a její top-1 token se po přidání jedné úrovně nezměnil.

## 7. Co přesně měřit

### Kvalita

- masked-token accuracy a cross-entropy podle mask rate,
- kvalita dokončeného textu proti FP16/FP32 baseline,
- podíl tokenů, které byly uzamčeny chybně,
- počet změn top-1 tokenu mezi Q1 → Q2 → Q4 → Q8,
- kalibrace confidence, například ECE nebo reliability bins.

### Adaptivní výpočet

- průměrný dosažený stupeň přesnosti na token,
- podíl pozic ukončených na Q1, Q2, Q4 a Q8,
- počet difuzních kroků na tokenovou pozici,
- normalizovaný bit-operation proxy rozpočet,
- počet skutečně provedených plných a reziduálních operací.

### Systémové metriky

- wall-clock latence a propustnost,
- peak memory a velikost načtených vah,
- výsledky zvlášť pro simulovanou kvantizaci a později pro skutečné low-bit
  kernely.

Hlavní graf projektu bude Pareto křivka kvalita versus výpočetní rozpočet.

## 8. Baselines a ablace

Každý adaptivní experiment musí být porovnán alespoň s:

1. FP16/FP32 při všech difuzních krocích.
2. Konstantní Q1, Q2, Q4 a Q8.
3. Fixní progresivní schedule bez early exit.
4. Adaptivní schedule se stejným maximálním počtem kroků.
5. Oracle řadičem, který z ground-truth určí, zda další přesnost pomohla.

Oracle nepatří do reálné inference. Určuje horní mez toho, zda má adaptivní
zpřesňování vůbec dostatečný potenciál. Pokud oracle nepřinese užitečnou Pareto
výhodu, nemá smysl zatím stavět složitý naučený řadič ani vlastní kernely.

## 9. Milníky a rozhodovací brány

### M0 — specifikace a reprodukovatelný baseline

- sjednotit definice vrstev, difuzních kroků a stupňů přesnosti,
- změřit kvalitu současného generování,
- uložit ukázky, checkpoint, konfiguraci a metriky.

**Brána:** současný model musí prokazatelně generovat a běh musí být
reprodukovatelný na všech třech nodech.

### M1 — minimální adaptivní inference

- token-wise confidence logy pro všechny přesnosti,
- pravidlový řadič,
- fixed versus adaptive porovnání,
- oracle analýza.

**Brána:** oracle i pravidlový řadič musí ukázat měřitelnou výhodu v kvalitě na
jednotku proxy výpočtu. Pokud ji ukáže pouze oracle, zlepšuje se řadič. Pokud ji
neukáže ani oracle, reviduje se trénink nebo reprezentace přesností.

### M2 — model učený ve všech přesnostech

- multi-precision sampling v tréninku,
- distillation a stability loss,
- srovnání s dnešními fixed-schedule checkpointy.

**Brána:** Q1/Q2 musí být užitečné hrubé prediktory a vyšší přesnost musí
opravovat více chyb, než kolik nových chyb zavede.

### M3 — skutečné aditivní zpřesňování

- nested/bit-plane váhy,
- ověřený `base + delta` výpočet,
- účtování skutečně ušetřených operací.

**Brána:** výsledek inkrementální cesty odpovídá referenci a vyžaduje méně
práce než plný přepočet.

### M4 — systémová optimalizace

- výběrový výpočet pro nejisté pozice,
- packed low-bit váhy a vhodné Apple Silicon kernely,
- měření reálné latence a paměti.

**Brána:** zrychlení musí existovat v reálném wall-clock měření, ne jen v proxy
metrice.

## 10. Jeden uživatel versus více uživatelů

Adaptivní přesnost nedělá z modelu principiálně jednouživatelský systém.
Jednotlivé požadavky nebo tokeny se mezi sebou o přesnost „nehádají“; řadič pro
ně pouze zvolí různou další práci. Praktický problém je efektivní batching:
sekvence končící na různých stupních rozbíjejí pravidelný batch.

První implementace má být optimalizována pro jednoho uživatele, protože tak
nejčistěji změří latenci a adaptivní chování. Víceuživatelská verze může později
seskupovat čekající pozice nebo požadavky podle aktuálního stupně přesnosti.

## 11. Nejbližší experiment

Nejkratší cesta k rozhodnutí není hned trénovat nový velký model. Nejdříve nad
existujícím checkpointem pro stejnou sadu maskovaných vstupů uložit logity Q1,
Q2, Q4, Q8 a FP16/FP32 a vytvořit oracle analýzu:

- kolik pozic je správných už v Q1,
- kolik chybných pozic vyšší přesnost opraví,
- kolik správných pozic naopak pokazí,
- zda lze tyto případy předpovědět z entropy, marginu a stability,
- jaká je nejlepší dosažitelná křivka kvalita versus proxy výpočet.

Tento experiment levně ověří samotný předpoklad adaptivního zpřesňování. Potom
má smysl implementovat M1 a následně trénovat první model podle nové strategie.

## 12. Kritéria úspěchu první etapy

První etapa je úspěšná, pokud:

- existuje reprodukovatelný malý PD-LM, který generuje více tokenů současně,
- oracle prokáže prostor pro adaptivní rozhodování,
- pravidlový řadič dosáhne srovnatelné kvality jako nejvyšší přesnost při nižším
  průměrném proxy rozpočtu,
- výsledky drží napříč více seedy a nejsou odvozeny pouze z jednoho příkladu,
- report jasně odděluje algoritmickou úsporu od skutečného hardwarového
  zrychlení.

## 13. Implementace M0

První evaluator je dostupný jako `scripts/oracle_m0.py`. Nad identickými
deterministickými maskovanými vstupy porovnává Q1, Q2, Q4, Q8 a FP32, měří
opravené i nově zavedené chyby a ukládá `summary.json` a kompaktní
`per_token.csv`. Plné vocabulary logity se neukládají a při běhu se zpracovává
vždy jen jedna fixture dávka, aby analýza neměla vysoké nároky na paměť.

Příklad nad existujícím 10k FP32 checkpointem:

```bash
.venv/bin/python scripts/oracle_m0.py \
  --config configs/full_baseline.json \
  --checkpoint checkpoints/full_baseline/step_0010000.npz \
  --validation-data data/cache/val_seq256_art50000_bytes500000000.npy \
  --output-dir results/m0/<run-name> \
  --eval-steps 20 \
  --fixture-seed 20260804
```

Výstup rozlišuje konečnou zvolenou přesnost od kumulativní ceny všech
navštívených stupňů. Wall-clock údaje popisují dnešní simulované plné přepočty,
nikoli rychlost budoucích low-bit kernelů. Protože uvedený checkpoint byl učen
ve FP32, jeho Q1/Q2/Q4/Q8 výsledky jsou pouze předběžná M0/PTQ sonda, nikoli test
budoucího modelu učeného současně ve všech přesnostech.

V současném evaluatoru je interní identifikátor `16` kompatibilní označení pro
identity FP32 cestu: používá FP32 master weights i FP32 výpočet a má proxy cenu
32, nikoli 16. Měřená ladder Q1 → Q2 → Q4 → Q8 → FP32 proto stojí 47 jednotek
(`1 + 2 + 4 + 8 + 32`); zastavení na Q4 stojí 7. Cílová, dosud
neimplementovaná ladder Q1 → Q2 → Q4 → Q8 → FP16 by stála 31 proti 32 pro jeden
FP32 průchod. M0 zatím skutečný FP16 stupeň neobsahuje.

Výsledky z více nodů se spojí pouze při shodné provenance:

```bash
.venv/bin/python scripts/aggregate_oracle_m0.py \
  results/m0/<run-a> results/m0/<run-b> results/m0/<run-c> \
  --output results/m0/aggregate_summary.json
```

Agregátor odmítne smíchat rozdílný commit, checkpoint, validační data,
konfiguraci nebo pořadí přesností.
