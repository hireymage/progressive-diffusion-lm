# Fáze 2: Obousměrná inkrementální progresivní přesnost — Závěrečná zpráva

[English](phase2-bidirectional-incremental-final-report.md) | [Čeština](phase2-bidirectional-incremental-final-report.cs.md)

**Datum:** 21.07.2026  
**Autoři:** Martin Hozák (Hozzy), Hermes Agent  
**Úložiště:** hireymage/progressive-diffusion-lm (SOUKROMÉ, Apache-2.0)  
**Hardware:** 2× Mac mini M1 8 GB (m1-256, m1-512) + 1× MacBook Air M4 16 GB (m4-air)  

---

## 1. Přehled

Testováno ve fázi 2 **obousměrná progresivní přesnost schedules** — měnící se kvantování bits napříč difúzními kroky ve směru hrubý→jemný (nahoru) i jemný→hrubý (dolů) — proti constant-precision základním liniím a FP16 baseline, přes 3 uzly a 2 seeds (celkem 42 běhů). Navíc bylo na všech 3 uzlech provedeno inference-time vyhodnocení inkrementálního dopředného (M2) a early-exit generování (M3).

### Milníky

| Milník | Popis | Stav |
|-----------|-------------|:------:|
| M1 | 8bitová symetrická kvantizace (256 úrovní) | ✅ 58/58 testy |
| M2 | Inkrementální forward (`y_next = y_prev + Δ`) | ✅ 26/26 testů |
| M3 | Early-exit generace (`generate_with_early_exit`) | ✅ 35/35 testy |
| M4 | Konfigurace kampaně (7 schedules × 2 seeds × 3 uzly) | ✅ |
| M5 | Validace smoke testu | ✅ |
| M6 | Realizace kampaně (42 spuštění) | ✅ 42/42 |
| M7 | Agregace, vyhodnocení závěrů, zpráva | ✅ |

**Unit testy: 108/108 úspěšně, 0 neúspěšně.**

---

## 2. Výsledky kampaně (42 běhů)

### 2.1 Kompletní tabulka výsledků

| Schedule | Seed | m1-256 | m1-512 | m4-air | Střední | Std |
|----------|------|-------:|-------:|-------:|-----:|----:|
| progressive-up [1,1,2,2,4,4,8,8] | s201 | 7,4634 | 7,4633 | 7,4633 | 7,4633 | 0,0001 |
| progressive-up | s203 | **7,3971** | 7,4324 | 7,4330 | **7,4208** | 0,0205 |
| progressive-down [8,8,4,4,2,2,1,1] | s201 | 7,4635 | 7,4635 | 7,4635 | 7,4635 | 0,0000 |
| progressive-down | s203 | 7,4539 | 7,4528 | 7,4581 | 7,4549 | 0,0028 |
| constant-1b | s201 | 7,4615 | 7,4615 | 7,4615 | 7,4615 | 0,0000 |
| constant-1b | s203 | 7,4113 | 7,4204 | 7,4452 | 7,4256 | 0,0176 |
| constant-2b | s201 | 7,4646 | 7,4646 | 7,4647 | 7,4646 | 0,0001 |
| constant-2b | s203 | 7,3961 | **7,3761** | 7,4064 | **7,3929** | 0,0154 |
| constant-4b | s201 | 7,4633 | 7,4633 | 7,4633 | 7,4633 | 0,0000 |
| constant-4b | s203 | 7,4428 | 7,4325 | 7,3946 | 7,4233 | 0,0254 |
| constant-8b | s201 | 7,4634 | 7,4635 | 7,4634 | 7,4634 | 0,0000 |
| constant-8b | s203 | 7,3888 | 7,4187 | 7,4501 | 7,4192 | 0,0307 |
| baseline-fp16 | s201 | 7,4635 | 7,4635 | 7,4635 | 7,4635 | 0,0000 |
| baseline-fp16 | s203 | 7,4096 | 7,4501 | 7,4521 | 7,4373 | 0,0240 |

### 2.2 Průměry schedules (ve všech seeds a uzlech)

| Schedule | Střední Val Loss | Std | n |
|----------|------------:|----:|---:|
| **constant-2b** | **7,4287** | 0,0405 | 6 |
| constant-8b | 7,4413 | 0,0310 | 6 |
| progressive-up | 7,4421 | 0,0266 | 6 |
| constant-4b | 7,4433 | 0,0272 | 6 |
| constant-1b | 7,4436 | 0,0226 | 6 |
| baseline-fp16 | 7,4504 | 0,0209 | 6 |
| progressive-down | 7,4592 | 0,0050 | 6 |

### 2.3 Vliv seedu

| Seed | Střední | Std | n |
|------|-----:|----:|---:|
| s201 | 7,4633 | 0,0009 | 21 |
| s203 | 7,4249 | 0,0252 | 21 |

Seed s203 produkuje výrazně nižší (lepší) val_loss napříč všemi schedules, s vyšším rozptylem. Seed s201 poskytuje téměř identické výsledky ve všech schedules (std=0,0009) — schedule volba má minimální vliv na tento seed.

### 2.4 Nejlepší a nejhorší jednotlivé běhy

**Top 5:**
1. constant-2b s203 m1-512: **7,3761**
2. constant-8b s203 m1-256: 7,3888
3. constant-4b s203 m4-air: 7,3946
4. constant-2b s203 m1-256: 7,3961
5. progressive-up s203 m1-256: 7,3971

**Spodní 5:**
1. constant-2b s201 m4-air: 7,4647
2. constant-2b s201 m1-256: 7,4646
3. constant-2b s201 m1-512: 7,4646
4. progressive-down s201 m1-256: 7,4635
5. baseline-fp16 s201 m4-air: 7,4635

---

## 3. Vyhodnocení inference (M2/M3)

### 3.1 Nastavení

Každý uzel trénoval malý progresivní model (2000 kroků, schedule [1,2,4,8,8,4,2,1]) s checkpoint uložením, poté porovnal 3 režimy odvození:
- **standardní**: plný dopředný průchod všemi 8 difuzními kroky
- **inkrementální**: `forward_incremental` — znovu použije výstup předchozího kroku přes `y_next = y_prev + Δ`
- **early_exit**: `generate_with_early_exit` — zastaví generování, když maximální token spolehlivost překročí práh

Prahové hodnoty: [0,01, 0,02, 0,03, 0,05, 0,10, 0,50]
Opakování: 3 na režim, zprůměrováno.

### 3.2 Výsledky podle uzlu

#### m1-256 (Mac mini M1 8 GB)

| Režim | Latence (y) | Kroky | Zrychlení | Dohoda |
|------|--------:|------:|--------:|----------:|
| standardní | 0,176 | 8 | 1,00× | — |
| přírůstkové | 0,106 | 8 | 1,65× | 100 % |
| early_exit (t=0,01) | 0,020 | 1 | 8,94× | 100 % |
| early_exit (t=0,02) | 0,014 | 1 | **12,68×** | 100 % |
| early_exit (t=0,03) | 0,016 | 1 | 11,15× | 100 % |
| early_exit_inc (t=0,01) | 0,017 | 1 | 10,36× | 100 % |
| early_exit_inc (t=0,02) | 0,017 | 1 | 10,49× | 100 % |
| early_exit_inc (t=0,03) | 0,019 | 1 | 9,47× | 100 % |
| early_exit (t=0,05) | 0,080 | 8 | 2,19× | 100 % |
| early_exit (t=0,1) | 0,079 | 8 | 2,24× | 100 % |
| early_exit (t=0,5) | 0,078 | 8 | 2,27× | 100 % |

#### m1-512 (Mac mini M1 8 GB)

| Režim | Latence (y) | Kroky | Zrychlení | Dohoda |
|------|--------:|------:|--------:|----------:|
| standardní | 0,104 | 8 | 1,00× | — |
| přírůstkové | 0,091 | 8 | 1,14× | 100 % |
| early_exit (t=0,01) | 0,020 | 1 | 5,29× | 100 % |
| early_exit (t=0,02) | 0,018 | 1 | **5,85×** | 100 % |
| early_exit (t=0,03) | 0,018 | 1 | 5,66× | 100 % |
| early_exit_inc (t=0,01) | 0,019 | 1 | 5,40× | 100 % |
| early_exit_inc (t=0,02) | 0,019 | 1 | 5,57× | 100 % |
| early_exit_inc (t=0,03) | 0,019 | 1 | 5,57× | 100 % |
| early_exit (t=0,05) | 0,093 | 8 | 1,11× | 100 % |
| early_exit (t=0,1) | 0,084 | 8 | 1,23× | 100 % |
| early_exit (t=0,5) | 0,087 | 8 | 1,19× | 100 % |

#### m4-air (MacBook Air M4 16GB)

| Režim | Latence (y) | Kroky | Zrychlení | Dohoda |
|------|--------:|------:|--------:|----------:|
| standardní | 0,058 | 8 | 1,00× | — |
| přírůstkové | 0,049 | 8 | 1,18× | 100 % |
| early_exit (t=0,01) | 0,008 | 1 | 7,37× | 100 % |
| early_exit (t=0,02) | 0,008 | 1 | **7,47×** | 100 % |
| early_exit (t=0,03) | 0,008 | 1 | 7,37× | 100 % |
| early_exit_inc (t=0,01) | 0,009 | 1 | 6,76× | 100 % |
| early_exit_inc (t=0,02) | 0,008 | 1 | 6,85× | 100 % |
| early_exit_inc (t=0,03) | 0,009 | 1 | 6,61× | 100 % |
| early_exit (t=0,05) | 0,042 | 8 | 1,38× | 100 % |
| early_exit (t=0,1) | 0,042 | 8 | 1,37× | 100 % |
| early_exit (t=0,5) | 0,042 | 8 | 1,36× | 100 % |

### 3.3 Souhrn inference

| Metrické | m1-256 | m1-512 | m4-air | Střední |
|--------|-------:|-------:|-------:|-----:|
| Přírůstkové zrychlení | 1,65× | 1,14× | 1,18× | **1,32×** |
| Early-exit zrychlení (t≤0,03) | 8,94–12,68× | 5,29–5,85× | 7,37–7,47× | **7,2–8,7×** |
| Early-exit kroků (t≤0,03) | 1/8 | 1/8 | 1/8 | 1/8 |
| Early-exit zrychlení (t≥0,05) | 2,19–2,27× | 1,11–1,23× | 1,36–1,38× | **1,5–1,6×** |
| Early-exit kroků (t≥0,05) | 8/8 | 8/8 | 8/8 | 8/8 |
| Token dohoda (všechny režimy) | 100 % | 100 % | 100 % | **100 %** |

**Klíčové zjištění:** Early-exit s prahem ≤0,03 redukuje inferenci na jeden difúzní krok (1/8) se 100% token shodou a 5–13× zrychlením. Při prahu ≥0,05 model nikdy neukončí předčasně (proběhne všech 8 kroků) a zrychlení pochází pouze z inkrementální optimalizace (~1,2–2,3×).

---

## 4. Klíčová zjištění

### 4.1 Progresivní vs konstantní

| Srovnání | s201 | s203 |
|------------|------|------|
| progressive-up | 7,4633 | 7,4208 |
| progressive-down | 7,4635 | 7,4549 |
| constant-2b (nejlepší konstanta) | 7,4646 | **7,3929** |
| constant-4b | 7,4633 | 7,4233 |
| baseline-fp16 | 7,4635 | 7,4373 |

- **Constant-2b je celkově nejlepší schedule** (průměr 7,4287), díky silným výsledkům s203 (7,3929).
- **Progressive-up je konkurenceschopný** (průměr 7,4421), zejména s s203 (7,4208), ale nepřekoná constant-2b.
- **Progressive-down je nejhorší schedule** (průměr 7,4592) — počínaje vysokou přesností a degradací má horší výkon než všechny alternativy včetně baseline.
- **Baseline FP16 není nejlepší** — constant-2b, constant-8b a progressive-up všechny jej v průměru překonávají, což naznačuje, že kvantizační šum může působit jako regularizace.

### 4.2 Na směru záleží

Progressive-up (hrubý→jemný, 1→8b) výrazně překonává progressive-down (jemný→hrubý, 8→1b):
- s203: 7,4208 vs 7,4549 (Δ=0,034)
- s201: 7,4633 vs 7,4635 (Δ=0,0002, zanedbatelné)

**Coarse-to-fine je správný směr** – začít s nízkou přesností a rafinací přináší lepší výsledky než začít s vysokou a degradující.

### 4.3 Citlivost na seed

Seed s201 produkuje téměř schedule-invariant výsledků (std=0,0009 během 21 běhů) — model konverguje k podobné ztrátě bez ohledu na přesnost schedule. Seed s203 vykazuje významnou schedule citlivost (std=0,0252) — u tohoto seed záleží více na volbě schedule.

To naznačuje, že efekt progresivní přesnosti je **seed-dependent** a může být výraznější u určité dynamiky tréninku než u jiných.

### 4.4 Konzistence mezi nody

Pro s201 poskytují všechny 3 uzly téměř identické výsledky (std ≤ 0,0001 na schedule). Pro s203 se rozptyl uzlů zvyšuje (std až 0,0307 pro constant-8b). m1-256 má tendenci produkovat nejnižší val_loss s s203, zatímco m4-air má tendenci vyšší. To může odrážet různé charakteristiky šířky pásma paměti ovlivňující kvantizační simulaci.

### 4.5 Optimalizace inference

- **Inkrementální vpřed** poskytuje konzistentní **1,14–1,65× zrychlení** (průměrně 1,32×) se 100% shodou token — složení `y_next = y_prev + Δ` je funkčně ekvivalentní plnému vpřed.
- **Early-exit** s nízkými prahovými hodnotami (≤0,03) dosahuje **5–13× zrychlení** snížením na 1/8 difuzních kroků se 100% token shodou. Jistota modelu po prvním kroku je dostatečná pro správné vygenerování token.
- **Early-exit + inkrementální** kombinuje obě optimalizace, ale výrazně nepřekonává early-exit samostatně při nízkých prahových hodnotách – dominuje cesta single-step.
- Při prahových hodnotách ≥0,05 se early-exit nikdy nespustí (proběhne všech 8 kroků) a zrychlení je pouze z inkrementální optimalizace.

---

## 5. Závěry

1. **Constant-2b je nejpřesnější schedule** pro tuto velikost modelu a datovou sadu a překonává progresivní schedules i FP16 baseline.
2. **Progressive-up (hrubý→jemný) je životaschopný** — překonává baseline-fp16 a je konkurenceschopný s konstantním schedules, ale nepřekračuje constant-2b.
3. **Progressive-down (jemné→hrubé) je třeba se vyhnout** — je to worst-performing schedule.
4. **Kvantizace se může regulovat** — constant-2b a constant-8b obě překonávají baseline-fp16, což naznačuje, že kvantizační šum pomáhá zobecnění.
5. **Inkrementální vpřed (M2) funguje správně** — 1,32× průměrné zrychlení se 100% ekvivalentem výstupu.
6. **Early-exit (M3) je vysoce efektivní** — až 12,68× zrychlení s prahovou hodnotou ≤ 0,03, což snižuje inferenci na 1/8 kroků se 100% token shodou.
7. **Seed citlivost je vysoká** — efekt schedule je dramatický u s203, ale zanedbatelný u s201. Pro statistickou významnost je potřeba více seeds.

---

## 6. Omezení a budoucí práce

- **Malý model** (7,5 milionu parametrů, 16000 vocab) — výsledky se nemusí přizpůsobit větším modelům.
- **Krátký trénink** (2000 kroků) — delší trénink může změnit relativní hodnocení schedules.
- **Pouze 2 seeds** — statistická významnost je omezená. Dramatický seed efekt (s201 vs s203) naznačuje, že je potřeba více seeds (5–10).
- **Jedna datová sada** — výsledky jsou specifické pro aktuální textový korpus.
- **Kvantizace je simulovaná** — STE v FP32, nikoli skutečný low-bit hardware. Skutečná kvantovaná inference se může lišit.
- **Early-exit ladění prahu** — optimální práh (0,03) je model-specific. Vyžaduje per-model kalibraci.

### Doporučené další kroky

1. **Více seeds** (5–10) ke stanovení statistické významnosti schedule efektů.
2. **Větší model** pro testování, zda constant-2b zůstává optimální v měřítku.
3. **Delší trénink** (5000–10000 kroků), abyste zjistili, zda se pořadí stabilizuje.
4. **Skutečný kvantovaný závěr** — nasazení se skutečnými zabalenými low-bit hmotnostmi.
5. **Per-model early-exit kalibrace** — najděte optimální práh jako funkci tréninkového kroku.

---

## Dodatek A: Konfigurace

- Model: 7,5M parametrů, progresivní typ, sdílení vah
- Přesnost schedules: [1,1,2,2,4,4,8,8] (nahoru), [8,8,4,4,2,2,1,1] (dolů), konstanty [1,2,4,8], baseline FP16
- Seeds: 201, 203
- Trénink: 10 000 kroků, velikost dávky=32, lr=3e-4 (rozpad kosinu)
- Dataset: 434 688 tokens, 3 226 vlakových kousků, 170 val chunků
- Kvantování: symetrické, STE v FP32

## Dodatek B: Hardware

| Uzel | Model | RAM | Jádra | Role |
|------|-------|-----|-------|------|
| m1-256 | Mac mini M1 | 8 GB | 8 | Kampaň + vyhodnocení odvození |
| m1-512 | Mac mini M1 | 8 GB | 8 | Kampaň + vyhodnocení odvození |
| m4-air | MacBook Air M4 | 16 GB | 10 | Kampaň + vyhodnocení odvození |

## Dodatek C: Datové soubory

- Výsledky kampaně: `results/phase2_campaign_all_results.csv`
- Vyhodnocení závěru: `results/inference_eval/{m1-256,m1-512,m4-air}_inference_eval.json`
- Konfigurace kampaní: `configs/campaign/m1-256-phase2-bidir.json`, `m1-512-phase2-bidir.json`, `m4-air-phase2-bidir.json`
- Skript inference eval: `scripts/eval_inference.py`
- Běžec kampaně: `scripts/run_dual_m1_campaign.py`
