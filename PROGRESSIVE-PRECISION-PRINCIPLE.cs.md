# Progresivní přesnost — kanonický princip experimentu

[English](PROGRESSIVE-PRECISION-PRINCIPLE.md) | [Čeština](PROGRESSIVE-PRECISION-PRINCIPLE.cs.md)

> **Kanonický zdroj pravdy pro návrh experimentu.**
> Tento dokument odděluje dlouhodobý princip od toho, co skutečně ověřil
> současný kód a český experiment. Naposledy porovnáno s kódem a publikovanými
> výsledky 18. 8. 2026.

## Základní myšlenka

Testovat změny numerické přesnosti s maximálním využitím již vypočtených
informací. Dlouhodobým cílem není postupně přetrénovat jiný model pro každou
úroveň kvantizace. Cílem je jeden model se sdílenými master vahami, který může
začít levným výpočtem a přidávat přesnost nebo hloubku pouze tam, kde dosavadní
jistota nestačí.

Požadovaný vztah zpřesnění je:

```text
y_dalsi = y_predchozi + delta_chybejici_informace
```

Výpočet delty musí být nakonec levnější než nový kompletní forward pass. Pouhé
odečtení dvou plně přepočtených výstupů je milníkem správnosti API, nikoli
konečným výsledkem efektivity.

## Kategorie experimentů

### 1. Progresivně nahoru

```text
1b -> 2b -> 4b -> 8b -> plná přesnost
```

Postupně přidává informace. Každá fáze má zpřesnit předchozí výsledek, nikoli
jej nahradit nesouvisejícím výpočtem.

### 2. Progresivně dolů

```text
plná přesnost -> 8b -> 4b -> 2b -> 1b
```

Postupně odebírá informace a zkoumá, kolik kvality zůstane a zda je nižší
přesnost užitečnou redukcí vyšší přesnosti.

### 3. Konstantní přesnost

```text
vždy 1b / vždy 2b / vždy 4b / vždy 8b
```

Tyto běhy slouží jako kontrola, zda progresivní schedule přináší něco navíc
oproti prostému výběru jedné přesnosti.

### 4. Baseline s plnou přesností

Standardní neprogresivní reference. FP16 a FP32 baseline musí být označeny
samostatně, protože nejsou zaměnitelné.

### 5. Flexibilní routes napříč vrstvami

Současný dlouhý český běh používá novější experiment zaměřený na nasazení:
jeden checkpoint se sdílenými FP32 master vahami je trénován přes tři routes
napříč vrstvami:

```text
q8_only
q8_fp16
q2_q8_fp16
```

Aktivní route mění přesnost přiřazenou napříč 25 vrstvami. Nevytváří tři
nezávislé modely. Trénink routes střídá, všechny je vyhodnocuje a checkpoint
vybírá konzervativně podle validačního loss **nejhorší route** na neviděných
datech.

Flexibilní experiment doplňuje čtyři původní kategorie. Neruší ani zpětně
nemění význam jejich historických výsledků.

## Hypotéza pro inference

```text
levná route / časný exit
        |
        +-- dostatečná jistota -> vypsat nebo ponechat token -> konec
        |
        +-- nejistota -> přidat vrstvy a/nebo přesnost -> znovu vyhodnotit
```

Konečný systém má podporovat obě formy zpřesnění:

- **zpřesnění přesnosti**, tedy přidání výpočtu ve vyšší přesnosti;
- **zpřesnění hloubky**, tedy ukončení tokenu nebo sekvence v dřívější vrstvě,
  pokud je jistota dostatečná.

Projekt už obsahuje diagnostické cesty pro early exit a route-by-exit. Tyto
diagnostiky dokazují řiditelné provádění a měří proxy cenu. Zatím nedokazují
produkční úsporu latence ani kalibrovanou naučenou ukončovací policy.

## Aktuální stav implementace

| Schopnost | Ověřený stav k 18. 8. 2026 |
|---|---|
| Trénink a evaluace Q1, Q2, Q3 a Q4 | Implementováno pomocí fake quantization nad FP32 master vahami |
| Výpočet Q8 a FP16 po vrstvách | Implementováno ve flexibilním modelu |
| Progressive Up a Down schedules | Implementováno a vyhodnoceno pro 1/2/4/8bitové experimenty Phase 2 |
| Kontroly s konstantní přesností | Implementovány a vyhodnoceny |
| Flexibilní routes se sdílenými vahami | Implementováno: `q8_only`, `q8_fp16`, `q2_q8_fp16` |
| Konzervativní výběr podle nejhorší route | Implementováno v českém flexibilním trenéru |
| Early exit podle vrstev | Implementováno a pokryto testy; kalibrace zůstává experimentální |
| Early exit podle diffusion kroků | Implementováno a vyhodnoceno jako experimentální inference režim |
| Inkrementální API `y_dalsi = y_predchozi + delta` | Implementováno s testy parity |
| Skutečně levnější opakované využití rezidua | **Neimplementováno**: současná cesta stále provede plný forward výpočet pro získání delty |
| Packed low-bit kernely | **Neimplementováno**: low-bit aritmetika je simulovaná |
| Backend pro Apple Silicon | Implementován v MLX; hlavní ověřená cesta tréninku |
| Backend pro NVIDIA | Implementován v PyTorch/CUDA včetně převodu checkpointu a resume |

## Současný český experiment se sdílenými vahami

Aktuální reálný pilot je záměrně malý a používá pouze česká data:

| Položka | Hodnota |
|---|---:|
| Vrstvy | 25 |
| `d_model` | 64 |
| `d_ff` | 256 |
| Attention heads | 4 |
| Délka sekvence | 256 |
| Tokenizer | český BPE, slovník 16 000 |
| Data | pouze česká Wikipedie |
| Trénované routes | `q8_only`, `q8_fp16`, `q2_q8_fp16` |
| Master váhy | sdílené FP32 |

Na publikované hranici tří milionů kroků zůstaly všechny routes numericky
konečné. Nejhorší route měla na neviděných datech loss **4,4130**, accuracy
**31,17 %** a perplexitu **82,51**. Běh poté pokračoval z checkpointu k výslovně
nastavenému limitu 20 000 000 kroků. Tyto hodnoty měří rekonstrukci maskovaných
tokenů, nikoli kvalitu chatbota.

Současná evidence podporuje omezený, ale důležitý závěr: jedna sada master vah
může zůstat trénovatelná přes všechny tři routes po miliony aktualizací. Zatím
nedokazuje použitelnou konverzační kvalitu, skutečné low-bit zrychlení, úsporu
paměti ani optimální chování early exit.

Reprodukovatelná konfigurace, tabulka metrik, omezení a stav CUDA jsou uvedeny
v [aktuálním stavu projektu](docs/cswiki-flexible-project-status-2026-08-18.md).

## Porovnávané metriky

- loss, přesnost tokenů a perplexita na neviděných datech pro každou route;
- nejhorší route, nikoli pouze průměrná nebo nejlepší route;
- numerická stabilita a konečné gradienty i váhy;
- kvalita generování a rekonstrukce maskovaných tokenů;
- kvalita route-by-exit a chování ukončování;
- změřená latence a paměť, jasně oddělené od proxy ceny;
- množství skutečně provedených a znovu využitých výpočtů a dat;
- porovnání s konstantní přesností a baseline s plnou přesností.

## Pravidla interpretace

1. Fake quantization se nesmí označovat jako packed low-bit zrychlení. MLX i
   CUDA nyní uchovávají a optimalizují FP32 master váhy.
2. Současné inkrementální API se nesmí vydávat za úsporu výpočtu. Zachovává
   vztah zpřesnění, ale uvnitř stále používá plný přepočet.
3. Flexibilní checkpoint se nesmí vybírat podle nejlepší route. Rozhoduje se
   konzervativně podle nejhorší route.
4. Přesnost maskovaných tokenů není faktická, konverzační ani obecná přesnost
   jazykového modelu.
5. Historické výsledky schedules z Phase 1 a Phase 2 musí zůstat oddělené od
   novějšího českého flexibilního běhu.
6. Je nutné rozlišovat `best` a `latest`; validační metriky lokálně kolísají.
7. Úspory early exit zůstávají experimentální, dokud nebude prokázáno
   kalibrované ukončování se zachováním kvality a změřené end-to-end zrychlení.

## Kritérium úspěchu úplné vize

Princip bude plně prokázán teprve tehdy, až sdílený model dokáže inkrementálně
přidávat přesnost nebo hloubku, znovu využívat předchozí výpočet bez kompletního
forward přepočtu, bezpečně končit podle kalibrované jistoty a prokáže změřenou
výhodu kvality, latence nebo paměti proti srovnatelným kontrolám s konstantní a
plnou přesností.
