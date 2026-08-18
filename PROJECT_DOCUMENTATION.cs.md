# Progressive-Precision Diffusion Language Model — Technická výzkumná dokumentace

[English](PROJECT_DOCUMENTATION.md) | [Čeština](PROJECT_DOCUMENTATION.cs.md)

<!-- doc-status: living; verified: 2026-08-18 -->
> **Stav dokumentu:** Živá dokumentace, obsah ověřen proti aktuálnímu kódu a publikovaným výsledkům 18. 8. 2026.

*Původní reference experimentů vytvořena 18. 7. 2026; živý stav a roadmapa
ověřeny 18. 8. 2026. Historická čísla zůstávají svázána se svými kampaněmi.*

> Sekce popisující 28M model nad anglickou Wikipedií jsou zachovanou referencí,
> nikoli aktivní trénovací konfigurací. Aktivním experimentem je český
> 25vrstvý model `d_model=64` se sdílenými master vahami, popsaný v aktuálním
> dodatku níže. Kanonické hranice výzkumu uvádí
> [PROGRESSIVE-PRECISION-PRINCIPLE.cs.md](PROGRESSIVE-PRECISION-PRINCIPLE.cs.md).

---

## 1. PŘEHLED PROJEKTU

### Co to je

Progressive-Precision Diffusion LM je výzkumný projekt zkoumající, zda lze maskovaný difúzní jazykový model efektivně trénovat s extrémně low-bit hmotnostními reprezentacemi a zda přiřazování různých přesností různým krokům odšumování („progresivní přesnost schedule“) přináší nějakou výhodu oproti constant-precision alternativám.

Model je obousměrný Transformer trénovaný pomocí masked-diffusion cíle (absorbing-state difúze nad diskrétními tokeny). Od autoregresních modelů se liší tím, že v každém kroku odšumování predikuje všechny pozice tokenů paralelně. Progresivní přesnost znamená, že hrubé počáteční kroky odšumování používají low-bit váhy (například binární 1bitové), zatímco pozdější jemné kroky používají vyšší přesnost (například 4bitovou), protože podle hypotézy hrubé kroky nepotřebují vysokou aritmetickou přesnost.

### Základní motivace a hypotéza

Ústřední hypotézou je, že přesnost potřebná pro odšumování koreluje s úrovní šumu: kroky s vysokým šumem potřebují pouze vytvořit hrubé rozložení tokenů a mohou tolerovat silně komprimované váhy, zatímco jemné kroky s nízkým šumem musí rozlišovat mezi podobnými tokeny a využijí vyšší přesnost. Jeden model lze napříč kroky vyhodnocovat s různou přesností, protože všechny vrstvy `QuantizedLinear` jsou za běhu řízeny schedule přesnosti pro jednotlivé kroky.

Sekundární hypotézou je, že nativní low-bit trénink (trénink od nuly s low-bit vahami přes STE) může produkovat lepší modely než aplikace stejné kvantizace post-hoc na high-precision checkpoint. direct/naive PTQ kampaň je dokončena: 18/18 hodnocení napříč seeds 42/123/7 a Q1/Q2/Q3/Q4/FP32/optional ternární byla obnovena a ověřena.

### Kritický rozdíl: simulovaný vs. skutečný low-bit

**Toto rozlišení je zásadní pro interpretaci všech výsledků.**

- **FP32 master váhy**: Všechny vrstvy `QuantizedLinear` ukládají své váhy vždy jako float32. Právě tyto parametry aktualizuje optimalizátor Adam.
- **Simulovaná kvantizace (STE dopředný průchod)**: Během dopředného průchodu jsou váhy předávány přes `quantize_weights(w, bits)`, což vrací aproximaci float32 kvantizovaných hodnot. Není použita žádná celočíselná aritmetika. To je plně simulováno v float32 na Apple Silicon.
- **Fake/simulated kvantizace**: STE trik (`w_ste = w + stop_gradient(quantize(w) - w)`) znamená, že zpětný průchod obdrží gradienty identity, jako by se kvantizace nestala. Toto je standardní quantization-aware školení (QAT).
- **Žádná skutečná zabalená jádra**: Neexistují žádné vlastní metalové shadery, žádné celočíselné MAC adresy, žádné úložiště hmotnosti. Všechny operace se provádějí v float32. Simulace 1 bitu v float32 je ve skutečnosti pomalejší než nativní float32 na současném hardwaru.
- **Teoretická komprese je skutečná**: Pokud by byly hmotnosti skutečně uloženy jako zabalená celá čísla (ne tento případ), uvedené kompresní poměry by se vztahovaly na inference-time ukládání hmotnosti.

### Dlouhodobý cíl

long-term výzkumná trajektorie má za cíl:
1. Potvrďte, že nativní low-bit školení funguje (dokončeno ablační studií)
2. Kvantifikujte rozdíl v kvalitě mezi nativní QAT a post-training kvantizací (direct/naive PTQ kampaň dokončena; širší replikované nativní Q3/Q4/ternary srovnání zůstává otevřené)
3. Potenciálně implementujte skutečná zabalená jádra v MLX pro měření skutečných výhod paměti a rychlosti
4. Testujte hypotézy binárního rozkladu ve větším měřítku

Všechny dosavadní výsledky jsou na malém modelu (~28 milionů parametrů) s omezeným souborem dat (~69 milionů tokens). Zobecnění na větší modely je hypotéza, nikoli demonstrovaný výsledek.

---

## 2. VÝZKUMNÉ OTÁZKY

Následující otázky byly výslovně uvedeny ve skriptech ablačního studia (`scripts/ablation_study.py`) aPTQ studijních skript (`scripts/ptq_study.py`). Otázky s výsledky jsou označeny [MÁ DATA]; otázky, které jsou stále otevřené, jsou označeny jako [OTEVŘENÉ] nebo [HYPOTETICKÉ].

**Q1: Lze difúzní LM úspěšně trénovat s extrémně low-bit váhami?** [MÁ DATA]
Ano. Varianta const_1bit (binární váhy, bits=1 v každém kroku) konverguje a dosahuje střední hodnoty best_val_loss 7,4336 napříč 3 seeds, což je nižší (lepší) než baseline průměr 7,4434. Trénink se s 1bitovou přesností nerozchází.

**Q2: Dokáže nativní low-bit trénink vyrovnat nebo překonat high-precision baseline?** [MÁ DATA]
Varianta const_1bit překonává baseline na 2 ze 3 seeds (seeds 7 a 123 překonává baseline per-seed průměr; seed 42 nikoli). Průměrný rozdíl je 0,0098 nats ve prospěch const_1bit, ale seed rozptyl je velký (std ~0,024). Výsledek je sugestivní, ale ne průkazný při 3 seeds.

**Q3: Poskytuje progresivní přesné plánování výhodu?** [MÁ ÚDAJE — NEPŘESVĚDNÉ]
prog_1_2_4 (1→2→4) dosahuje střední hodnoty best_val_loss 7,4428, což je v rozmezí 0,0006 od baseline (7,4434). Při 3 seeds s touto úrovní rozptylu je výsledek nerozeznatelný od šumu.

**Q4: Záleží na přesnosti směru (1→2→4 vs 4→2→1)?** [MÁ ÚDAJE – NEPŘESVĚDNÉ]
prog_1_2_4 průměr = 7,4428, prog_4_2_1 průměr = 7,4571. Směr coarse-to-fine se zdá o něco lepší (o 0,014 nats průměr), ale při 3 seeds to není statisticky prokázáno.

**Q5: Je zlepšení z progresivní struktury nebo low-bit regularizace?** [MÁ ÚDAJE — NEZÁVĚRNÉ]
prog_1_2_4 (průměr 7,4428) vs const_2bit (průměr 7,4586, stejný průměr 2,0 efektivní bits). Progresivní struktura se zdá o něco lepší o 0,016 nats, ale skript ablační studie klasifikuje rozdíly <0,002 jako neprůkazné a ty <0,02 jako "nerozlišitelné v této škále." Výsledek je na hranici.

**Q6: Nativní low-bit trénink vs post-training kvantizace (PTQ)?** [MÁ DATA — DIRECT/NAIVE PTQ DOKONČENO]
Ozdravná kampaň vytvořila a ověřila všech 18 požadovaných `(seed, bits)` hodnocení. Interpretace zůstává omezena legacy-Q4/current-Q4 nesouladem schématu a pouze jedním nativním seed pro aktuální pravdivé Q3 a ternární.

**Q7: Zvyšuje se native-vs-PTQ rozdíl v kvalitě při nižších bits?** [MÁ DATA — INTERPRETOVAT S VÝSTRAHY]
Hotový agregát direct/naive PTQ řeší tuto otázku pro Q1/Q2 a přibližně pro Q4. Current-Q4 postrádá scheme-matched nativní baseline, zatímco nativní Q3 a ternární mají každý pouze jeden později seed.

**Q8: Existuje práh přesnosti, kdy PTQ kolabuje, ale nativní low-bit zůstává stabilní?** [MÁ DATA — PŘEDBĚŽNÉ]
Hotový PTQ agregát podává collapse-threshold analýzu. Silná inference je stále omezená, protože současný Q3/ternary nativní důkaz je single-seed a současný Q4 postrádá scheme-matched nativní baseline.

**Q9 (HYPOTETICKÉ): Může více 1bitových komponent nahradit aritmetiku s širší přesností?** [HYPOTETICKÉ]
Toto je budoucí výzkumná hypotéza (binární rozklad). Neimplementováno ani testováno.

**Q10 (HYPOTETICKÉ): Může progresivní přesnost umožnit adaptivní compute/memory při vyvozování?** [HYPOTETICKÉ]
Architektura to podporuje (model.set_bits() lze volat v každém kroku), ale nebyl vytvořen ani vyhodnocen žádný adaptivní inferenční systém.

---

## 3. ARCHITEKTURA MODELU

### Hyperparametry (plný/ablační model)

| Parametr | Hodnota |
|---|---|
| vocab_size | 16 000 |
| d_model | 512 |
| n_layers | 6 |
| n_heads | 8 |
| head_dim | 64 (= d_model / n_heads) |
| d_ff | 2,048 |
| max_seq_len | 256 |
| odpadnutí | 0,1 |
| n_diffusion_steps | 8 |
| tie_word_embeddings | Pravda |

Zdroj: `configs/full_baseline.json`, `configs/ablation/ablation_baseline_s42_full.json`, potvrzeno `results/full_baseline/final_summary.json`.

Poznámka: Konfigurace smoke-test a short-exp používají menší architektury (viz část 6).

### Počet parametrů

Celkové parametry: **28 295 808** (28,3 milionů).

Rozdělení od `results/full_baseline/final_summary.json` (úložná část):
- Celkové parametry: 28 295 808
- QuantizedLinear hmotnostní parametry (kvantované během training/inference): 18 874 368
- Non-quantized parametry (embedding, LayerNorm, biases, lm_head_bias): 9,421,440

S váhovým vázáním (tie_word_embeddings=True) sdílí hlava LM tabulku token vkládání (vocab_size × d_model = 16 000 × 512 = 8 192 000 parametrů), čímž ušetří ~8M parametrů ve srovnání se samostatnou matricí hlavy LM. Stále je přidělen malý naučený vektor zkreslení (vocab_size = 16 000 hodnot).

FP32 úložiště: 113,18 MB (28,3 M × 4 bajty).

Odhad tréninkové paměti (mistrovské váhy + gradienty + stavy Adam mav, vše FP32): ~452,7 MB.

### Komponenty specifické pro diffusion model

**SinusoidalEmbedding (step_embed)**: Mapuje míru maskování (skalární v [0,1], kde 1,0 = zcela zašuměný, 0,0 = čistý) na d_model-rozměrný vektor úpravy. Realizace: sinusové kódování → 2vrstvé MLP (d_model → d_model*2 → d_model) s aktivací SiLU. Toto vložení se vysílá do všech token pozic a přidá se k vložení token+ pozice před bloky Transformer.

**Dopředná difúze (maskování)**: Každé token je nezávisle nahrazeno MASK_TOKEN (= vocab_size = 16 000) s pravděpodobností rovnou maskovací frekvenci vzorkované z Uniform (0,1, 1,0). Jedná se o standardní absorbing-state maskovanou difuzi.

**Cíl školení**: Cross-entropy ztráta počítaná pouze na maskovaných pozicích. Model předpovídá původní token vzhledem k částečně maskované sekvenci a rychlosti maskování jako podmiňování.

**Přesnost schedule během tréninku**: Na dávku se vypočítá průměrná míra maskování v celé dávce. To je mapováno na index kroku (krok = minimum (střední_rychlost * n_kroků), upnutý na [0, T-1]). Přesnost tohoto indexu kroku se vyhledá z precision_schedule a aplikuje se na všechny vrstvy QuantizedLinear pomocí model.set_bits(bits).

**Odvozování / generování**: Počínaje plně maskovanou sekvencí jsou provedeny T kroky pro odstranění šumu. V kroku i model používá precision_schedule[i] bits, předpovídá všechny pozice a odmaskuje maskované pozice top-k highest-confidence (max. softmax pravděpodobnost). Toto pokračuje, dokud nejsou obsazeny všechny pozice.

### Mechanismus pozornosti

Obousměrný (non-causal) multi-head self-attention. Při tréninku na non-padded vstupech není aplikována žádná maska ​​pozornosti. Skóre pozornosti softmax se vypočítá v float32 (vyvoláno ze vstupu dtype), aby se předešlo numerické nestabilitě, a poté se přenese zpět. Důvodem pro obousměrnou pozornost je to, že maskovaná difúze je fill-in-the-blank úkolem: model vidí všechny nezamaskované tokens a musí předvídat maskované tokens, což vyžaduje pozornost v obou směrech.

### Kvantizovaná lineární vrstva

Každá lineární projekce v pozornosti (Q, K, V, out_proj) a každá feed-forward vrstva (ff1, ff2) je QuantizedLinear. Vrstvy (token_embed, pos_embed, lm_head_bias) a LayerNorm zůstávají vždy ve float32.

Vrstva QuantizedLinear:
1. Ukládá full-precision float32 závaží (hlavní závaží aktualizováno Adamem)
2. V dopředném čase zavolá `ste_quantize(self.weight, self.bits)` pro získání kvantovaných vah
3. Vypočítá násobek matice: `x @ w_quantized.T`
4. Přidá zkreslení float32 (pokud zkreslení=True)

STE provedení: `w_ste = w + stop_gradient(quantize(w) - w)`. Dopředný průchod vidí `quantize(w)` (přiblížení low-bit). Zpětný průchod vidí identitu (gradient teče do `w` nezměněn). MLX to realizuje pomocí `mx.stop_gradient`.

`bits` je měnitelný atribut runtime na každém QuantizedLinear. Volání `model.set_bits(bits)` aktualizuje všechny vrstvy QuantizedLinear současně. To umožňuje vyhodnotit jeden model checkpoint s jakoukoli úrovní přesnosti bez překládání závaží.

### Rozdíl v chování `model_type`

- `model_type="baseline"`: Model vždy používá bits=16 (identita pass-through, žádná kvantizace). Precision_schedule uložený v konfiguraci je přepsán pomocí [16]*n_diffusion_steps.
- `model_type="progressive"`: Model používá precision_schedule[step_idx] pro aktuální krok, který může zahrnovat bits=1, 2, 3, 4 nebo 16.

Poznámka: const_1bit až const_4bit varianty používají model_type="progressive" s konstantou schedule (všechny položky jsou stejné). "baseline" model_type je strukturálně identický, ale obchází veškerou kvantizaci.

### Sdílení vah

Když tie_word_embeddings=True (výchozí a použité ve všech úplných experimentech), výstupní LM hlava znovu použije token matici vkládání. Projekce v dopředném čase je: `logits = x @ token_embed.weight[:vocab_size].T + lm_head_bias`. To ušetří 8 192 000 parametrů ve srovnání se samostatnou lineární vrstvou.

---

## 4. SOUBOR DAT

### Zdroj

Dataset: `wikimedia/wikipedia`, konfigurace `20231101.en` (snímek anglické Wikipedie, listopad 2023). Načteno prostřednictvím knihovny Hugging Face `datasets` s `streaming=True`, aby nedošlo k načtení celých ~22 GB do paměti.

### Limity používané v hlavních experimentech (ablace a úplné běhy)

- `max_articles`: 50 000
- `max_text_bytes`: 500 000 000 (500 MB nezpracovaného textu)
- `seq_len`: 256

### Tokenizer

- Typ: Byte-Pair Kódování (BPE), natrénováno pomocí knihovny HuggingFace `tokenizers`
- Velikost slovní zásoby: 16 000 podslov tokens
- Místo: `tokenizer/wiki_bpe/tokenizer.json`
- Speciální tokens: `[PAD]=0, [UNK]=1, [MASK]=2, [BOS]=3, [EOS]=4`
- Poznámka: Difúzní model používá samostatnou MASK_TOKEN na ID 16 000 (= vocab_size), odlišnou od tokenizer `[MASK]` na ID 2. Tím se zabrání nejednoznačnosti mezi „tato pozice byla maskována difúzí“ a běžným textem.

### Pipeline předzpracování

1. Streamujte články z wikimedia/wikipedia jeden po druhém (streaming=True)
2. Tokenizujte pole `text` každého článku a připojte `[EOS]` token
3. Spojte všechna token ID do jedné dlouhé vyrovnávací paměti
4. Rozdělit na non-overlapping kusy po `seq_len` tokens (neúplný poslední kus vyřazen)
5. Zamíchejte všechny kousky s pevným náhodným seed
6. Rozdělit na vlak (95 %) a ověření (5 %)
7. Uložit jako numpy int32 pole do `data/cache/`

### Statistiky datové sady (hlavní experimenty)

Od `data/cache/meta_seq256_art50000_bytes500000000.json`:
- Kusy vlaku: **256 180**
- Val chunks: **13 484**
- Celkem tokens: **69 033 984** (přibližně 69 mil. tokens)
- Tvar kusu: (256,) na kus (seq_len=256)

### Rozdělení train/validation

- Vlak: 95 % (`train_split=0.95`)
- Ověření: 5 %

### Umístění souborů cache

Hlavní mezipaměť experimentu:
- `data/cache/train_seq256_art50000_bytes500000000.npy` (~250 MB)
- `data/cache/val_seq256_art50000_bytes500000000.npy` (~13 MB)
- `data/cache/meta_seq256_art50000_bytes500000000.json`

Smoke test / short exp cache:
- `data/cache/train_seq64_art100_bytes1000000.npy` (~800 kB)
- `data/cache/val_seq64_art100_bytes1000000.npy` (~89 kB)
- `data/cache/train_seq128_art100_bytes1000000.npy` (~800 kB)
- `data/cache/val_seq128_art100_bytes1000000.npy` (~89 kB)

---

## 5. Kvantizační schémata

Veškerá kvantizace je implementována v `src/quantization.py`. Klíčovou volbou návrhu je „jednotné no-zero symetrické“ schéma pro Q1–Q4: úrovně jsou vždy liché násobky kroku, takže nula nikdy není reprezentovatelná hodnota. Tím se vyhnete problému zero-collapse v binary-style schématech.

### Kompletní tabulka schémat

| bits param | Název schématu | úrovně | Hodnoty úrovní | Scale/step vzorec | Nula reprezentativní | Eff. bits |
|---|---|---|---|---|---|---|
| 1 | Q1 / Binární | 2 | {-1, +1} × měřítko | měřítko = průměr(|w|) za output-row | Ne (0 map až +1) | 1,0 |
| 2 | Q2 / True 2-bit | 4 | {-3, -1, +1, +3} × krok | krok = max(|w|) / 3 za output-row | Ne | 2,0 |
| 3 | Q3 / True 3-bit | 8 | {-7, -5, -3, -1, +1, +3, +5, +7} × krok | krok = max(|w|) / 7 za output-row | Ne | 3.0 |
| 4 | Q4 / True 4-bit | 16 | {-15, -13, …, -1, +1, …, +15} × krok | krok = max(|w|) / 15 za output-row | Ne | 4,0 |
| 16 | FP32 pass-through | kontinuální | identita (w nezměněna) | — | Ano | 16.0 (float32) |
| 0 | Ternární (volitelné) | 3 | {-1, 0, +1} × měřítko | měřítko = max(|w|) za output-row | Ano | ~1,585 (log2(3)) |

### Obecný vzorec pro Q2–Q4

Pro bits=n (n ∈ {2,3,4}): úrovně jsou {±1, ±3, ±5, …, ±(2^n−1)} × krok. Velikost kvantované hmotnosti se vypočítá jako `mag = 2·podlaha(|w_norm|/2) + 1`, capped at `2^n−1`. Hranice mezi po sobě jdoucími úrovněmi jsou ±2, ±4, … × krok.

### Sémantika `bits=16`

`bits=16` (a jakákoli hodnota ≥16) je případ pass-through: `ste_quantize` vrací `w` přímo a model baseline efektivně trénuje s plnou přesností float32. To se používá pro model_type="baseline" a pro FP32 referenční bod ve studii PTQ.

### `bits=0` (ternární) — VOLITELNÉ / EXPERIMENTÁLNÍ

Ternary používá 3 úrovně {-1, 0, +1} × max(|w|) s hranicí ±0,5 × měřítko. Je záměrně oddělena od hlavní Q1–Q4 matice (k níž se přistupuje prostřednictvím sentinelové hodnoty bits=0 spíše než bits=3, která je nyní True 3-bit). Ternární je vyhodnocen pouze tehdy, když je `--include-ternary` předáno `ptq_study.py`. Původní ablační matrice nemá žádnou ternární variantu; jeden pozdější nativní ternární běh existuje na seed 31415 a musí zůstat označen single-seed evidence.

Komentáře `src/model.py` a `src/config.py` nyní odrážejí tuto sémantiku. Autoritativní doba běhu zůstává `src/quantization.py`.

### KRITICKÉ: historie změn schématu Q4

Funkce `_quantize_4bit` byla aktualizována z 15úrovňového with-zero schématu na současné 16úrovňové no-zero schéma. Komentář v `src/quantization.py` řádku 27–29 to explicitně dokumentuje:

> POZNÁMKA: Před tímto schématem používal 4bitový režim 15 úrovní {-7,…,+7}×scale/7 (s nulou). Varianta ablace const_4bit byla trénována podle starého schématu — upozornění viz ptq_study.py.

**Důsledek**: Varianta `const_4bit` ablace (jak screening, tak celé běhy po 10 000 krocích) byla trénována podle STARÉHO 15úrovňového with-zero Q4 schématu. Hodnocení PTQ studie Q4 využívá NOVÉ schéma 16 úrovní. Jakékoli srovnání mezi PTQ Q4 výsledky a nativními const_4bit výsledky porovnává dvě různé kvantizační funkce. Skript studie PTQ (`scripts/ptq_study.py`) označí všechna Q4 srovnání upozorněním `*` a ve výstupních datech obsahuje příznak `q4_scheme_caveat: true`.

Pro studii PTQ bylo přidáno schéma Q3 (True 3-bit). Původní ablační matrice nemá Q3 variantu; jeden pozdější nativní Q3 běh existuje na seed 31415, takže cross-seed nativní důkaz stále chybí.

### Implementace STE

```python
def ste_quantize(w: mx.array, bits: int) -> mx.array:
    if bits >= 16:
        return w
    w_q = quantize_weights(w, bits)
    return w + mx.stop_gradient(w_q - w)
```

Forward: výraz se vyhodnotí na `w_q` (kvantované váhy). Zpětně: `mx.stop_gradient(w_q - w)` má nulový gradient, takže gradient `w + stop_gradient(...)` vzhledem k `w` je 1 (identita). full-precision hlavní závaží obdrží plný gradient.

### Registr `EFFECTIVE_BITS`

```python
EFFECTIVE_BITS = {0: math.log2(3), 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 16: 16.0}
```

To se používá pro teoretické odhady komprese. Pro prog_1_2_4 schedule [1,1,1,1,2,2,4,4] je efektivní průměr bits = (1+1+1+1+2+2+4+4)/8 = 2,0.

---

## 6. HISTORIE EXPERIMENTU (CHRONOLOGICKÉ)

### Fáze 0: Smoke testy

**Stav: DOKONČENO**

**Účel**: Než se pustíte do delších běhů, ověřte, zda funguje celý kanál. Příčetnost zkontroluje, že trénování nepadne, že kvantizace funguje správně a že datový kanál funguje.

**Config/setup**:
- Model: d_model=128, n_layers=2, n_heads=4, d_ff=512 (malé, ~0,7 milionů parametrů na základě zprávy o uložení pro short_exp)
- Údaje: 100 článků na Wikipedii, max. 1 MB textu, seq_len=64
- Trénink: 50 kroků, dávka=4, LR=1e-3, rozcvička=5 kroků
- Konfigurace: `configs/smoke_test_baseline.json`, `configs/smoke_test.json`

**Výsledky**: Checkpoints uloženy v `checkpoints/smoke_test_baseline/` a `checkpoints/smoke_test_progressive/` (každý ~2 soubory v kroku_25 a kroku_50). Soubory konečných výsledků nejsou podrobně zkontrolovány (výsledky nejsou v hlavním stromu výsledků pro smoke_test).

**Závěr**: Potrubí funguje. Žádné smysluplné kvalitní závěry z 50 kroků.

---

### Fáze 1: Krátké experimenty (500 kroků, malý model)

**Stav: DOKONČENO**

**Účel**: První srovnání baseline a progresivních schedules v mírně větším měřítku než smoke test, stále dostatečně rychlé pro iterace.

**Config/setup**:
- Model: d_model=256, n_layers=4, n_heads=8, d_ff=1024 (~7,6 milionů parametrů ze zprávy úložiště)
- Údaje: 100 článků na Wikipedii, max. 1 MB textu, seq_len=128
- Trénink: 500 kroků, dávka=8, LR=3e-4, rozcvička=50 kroků, vyhodnocení každých 100 kroků
- Tři varianty: short_exp_baseline, short_exp_progressive_2bit, short_exp_progressive_ternary
- Konfigurace: `configs/short_exp_baseline.json`, `configs/short_exp_progressive_2bit.json`, `configs/short_exp_progressive_ternary.json`
- Všechny jednotlivé seed=42

**Výsledky** (z `results/short_exp_*/final_summary.json`):

| Experimentujte | Přesnost schedule | best_val_loss | celkem_sekund |
|---|---|---|---|
| short_exp_baseline | [16]*8 | 7,529157 | 39,0 |
| short_exp_progressive_2bit | [1,1,1,1,2,2,4,4] | 7,530751 | 41,5 |
| short_exp_progressive_ternary | [1,1,1,1,3,3,4,4] | 7,533732 | 40.8 |

Poznámka: Ternární schedule v short_exp_progressive_ternary používá OLD bits=3 ternární interpretaci (před refaktorizací, kde bits=3 se stalo True 3-bit a bits=0 se stalo ternární). Toto je historický artefakt.

**Závěr**: Při 500 krocích s malým modelem a malou sadou dat všechny tři varianty produkují téměř identickou ztrátu ověření (~7,53). Žádný signál v tomto měřítku. Progresivní jednoznačně nevyhrává ani neprohrává. Ternary je o něco horší, ale rozdíl je v rámci šumu. Experiment odůvodnil zvětšení.

---

### Fáze 2: Úvodní srovnání v plném měřítku (10 000 kroků)

**Stav: DOKONČENO**

**Účel**: Úplný trénink 10 000 kroků v měřítku hlavního modelu s 50 000 články na Wikipedii, porovnání baseline vs. progressive_1_2_4. Jeden seed (seed=42) pro každého.

**Config/setup**:
- Model: d_model=512, n_layers=6, n_heads=8, d_ff=2048, tie_word_embeddings=True (28,3 milionů parametrů)
- Data: 50 000 článků, max. 500 MB textu, seq_len=256 (vlak: 256 180 kusů; hodnota: 13 484 kusů)
- Trénink: 10 000 kroků, dávka=8, LR=3e-4, zahřívání=500 kroků, vyhodnocení každých 500 kroků (100 stejných dávek)
- Dvě varianty: full_baseline (seed=42), full_progressive_1_2_4 (seed=42)
- Konfigurace: `configs/full_baseline.json`, `configs/full_progressive_1_2_4.json`

**Výsledky** (z `results/full_baseline/final_summary.json` a `results/full_progressive_1_2_4/final_summary.json`):

| Experimentujte | Typ modelu | Přesnost schedule | best_val_loss | celkem_sekund |
|---|---|---|---|---|
| úplný_základ | baseline | [16]*8 | 7,432665 | 4 990,3 (83,2 min) |
| full_progressive_1_2_4 | progresivní | [1,1,1,1,2,2,4,4] | 7,414686 | 5 663,1 (94,4 min) |

**Checkpoints**: Oba checkpoints jsou plně zachovány na `checkpoints/full_baseline/` a `checkpoints/full_progressive_1_2_4/`, každý obsahuje 17 checkpoint souborů po 324 MB (krok_500 až krok_10000). Studie PTQ byla navržena tak, aby používala nové tréninkové běhy s `save_checkpoints` konfigurovaným odlišně (pouze jeden checkpoint v posledním kroku), takže tyto nejsou zamýšleným PTQ zdrojem checkpoints.

**Teoretická komprese pro progresivní**:
- Efektivní průměr bits: 2,0
- Teoretické úložiště Q: 42,4 MB (vs. 113,2 MB FP32, vs 56,6 MB BF16)
- Komprese vs FP32: 2,67×; oproti BF16: 1,33×

**Pozor**: Progresivní model porazil baseline o 0,018 nats při seed=42. Jedná se však o jediný seed — Apple Silicon non-determinism napříč relacemi znamená, že ani totéž seed nezaručuje reprodukovatelnost. Ablační studie byla navržena tak, aby to řešila s 3 seeds.

---

### Fáze 3: Ablační screening (3 000 kroků, 6 variant × 3 seeds)

**Stav: DOKONČENO (18/18 běží)**

**Účel**: Systematický screening, aby se zjistilo, které varianty stojí za úplný trénink 10 000 kroků. Kontrola pro seed rozptyl spuštěním 3 seeds. Varianty určené k rozkladu: (a) funguje nějaké low-bit školení? (b) pomáhá progresivní struktura za konstantní low-bit? c) záleží na směru?

**Varianty**:
- baseline: [16]*8 (FP32, bez kvantizace)
- const_1bit: [1]*8 (v celém rozsahu binárně)
- const_2bit: [2]*8 (skutečně 2bitové)
- const_4bit: [4]*8 (4bitové celé)
- prog_1_2_4: [1,1,1,1,2,2,4,4] (coarse-to-fine)
- prog_4_2_1: [4,4,2,2,1,1,1,1] (fine-to-coarse, obráceně)

**Seeds**: 42, 123, 7

**Config/setup** (od `configs/ablation/ablation_baseline_s42_screen.json`):
- Model: stejný jako plný (d_model=512, n_layers=6, n_heads=8, d_ff=2048, 28,3 milionů parametrů)
- Data: stejná (50 tisíc článků, 500 MB, seq_len=256)
- Trénink: 3000 kroků, dávka=8, LR=3e-4, rozcvička=300 kroků, vyhodnocení každých 500 kroků
- Checkpoints zakázáno (save_checkpoints=False); uloženy pouze metriky
- Výsledky uloženy do `results/ablation/`

**Výsledky screeningu** (z `results/ablation/aggregate_screen.json`):

| Varianta | Prům. efekt bits | Průměr best_val_loss | Std | Min | Max |
|---|---|---|---|---|---|
| baseline | 16.0 | 7,489006 | 0,004098 | 7,485992 | 7,493672 |
| const_1bit | 1,0 | 7,487793 | 0,003420 | 7,485748 | 7,491741 |
| const_2bit | 2,0 | 7,489598 | 0,004093 | 7,485646 | 7,493819 |
| const_4bit | 4,0 | 7,488849 | 0,003643 | 7,485985 | 7,492949 |
| prog_1_2_4 | 2,0 | 7,491556 | 0,004255 | 7,488473 | 7,496411 |
| prog_4_2_1 | 2,0 | 7,489618 | 0,001666 | 7,487710 | 7,490788 |

Při 3 000 krocích se všechny varianty shlukují extrémně těsně (rozsah: 7,4857–7,4964). Žádná varianta nevykazuje jasnou výhodu. Všechny varianty se stále sbližují. Obrazovka identifikovala všech 6 variant, které stojí za to spustit na plných 10 000 kroků (protože všechny rozdíly jsou v rámci šumu).

Rychlost konvergence (krok, při kterém val_loss nejprve klesne pod 7,50, z aggregate_screen.json):
- baseline: průměr 2833 kroků
- const_1bit: průměrně 2500 kroků (nejrychlejší)
- const_2bit: průměr 2500 kroků
- const_4bit: průměr 2833 kroků
- prog_1_2_4: průměrně 2833 kroků
- prog_4_2_1: průměr 2500 kroků

Tréninkové časy při 3000 krocích se výrazně lišily napříč seeds (1314s až 3175s), což odráželo Apple Silicon non-determinism a kolísání zátěže pozadí během běhů.

---

### Fáze 4: Úplná ablace (10 000 kroků, všech 6 variant × 3 seeds)

**Stav: DOKONČENO (18/18 běží)**

Toto je primární dokončený experiment. Všech 18 běhů skončilo; žádné běhy se nezdařily.

**Config/setup** (od `configs/ablation/ablation_baseline_s42_full.json`):
- Model: d_model=512, n_layers=6, n_heads=8, d_ff=2048, tie_word_embeddings=True (28,3 milionů parametrů)
- Data: 50 tisíc článků, 500 MB, seq_len=256
- Trénink: 10 000 kroků, dávka=8, LR=3e-4, zahřívání=500 kroků, vyhodnocení každých 500 kroků (100 stejných dávek)
- Checkpoints zakázáno (save_checkpoints=False); uloženy pouze metriky
- Výsledky uloženy do `results/ablation_full/`

#### Tabulka výsledků jednotlivých běhů

Všechna čísla se přečtou z `results/ablation_full/*/eval_history.json` a `results/ablation_full/*/final_summary.json`.

| Varianta | Seed | best_val_loss | nejlepší_krok | final_val_loss | best_val_acc | tréninkové_sekundy |
|---|---|---|---|---|---|---|
| baseline | 42 | 7,419442 | 9500 | 7,435617 | 0,047905 | 13724,7 |
| baseline | 123 | 7,462691 | 8500 | 7,472883 | 0,039417 | 8529,9 |
| baseline | 7 | 7,448140 | 10 000 | 7,448140 | 0,044462 | 9159,9 |
| const_1bit | 42 | 7,458013 | 5000 | 7,477719 | 0,039497 | 9312,6 |
| const_1bit | 123 | 7,409514 | 9500 | 7,410912 | 0,047712 | 9207,4 |
| const_1bit | 7 | 7,433141 | 10 000 | 7,433141 | 0,048047 | 9378,9 |
| const_2bit | 42 | 7,445650 | 9500 | 7,458969 | 0,042673 | 9961,8 |
| const_2bit | 123 | 7,462503 | 8500 | 7,473033 | 0,039417 | 8646,6 |
| const_2bit | 7 | 7,467745 | 6500 | 7,468079 | 0,039473 | 8202,7 |
| const_4bit | 42 | 7,426016 | 9500 | 7,441531 | 0,046293 | 7471,4 |
| const_4bit | 123 | 7,463445 | 8500 | 7,473605 | 0,039417 | 10032,6 |
| const_4bit | 7 | 7,445710 | 10 000 | 7,445710 | 0,043417 | 12879,6 |
| prog_1_2_4 | 42 | 7,412376 | 9500 | 7,428867 | 0,047496 | 12675,6 |
| prog_1_2_4 | 123 | 7,454207 | 9500 | 7,490064 | 0,043799 | 9972,0 |
| prog_1_2_4 | 7 | 7,461684 | 10 000 | 7,461684 | 0,042976 | 7809,0 |
| prog_4_2_1 | 42 | 7,445405 | 7500 | 7,461251 | 0,043121 | 7932,5 |
| prog_4_2_1 | 123 | 7,459054 | 9500 | 7,466103 | 0,043628 | 8443,2 |
| prog_4_2_1 | 7 | 7,466833 | 10 000 | 7,466833 | 0,040836 | 8060,2 |

#### Agregace podle variant (od `results/ablation/aggregate_full.json`)

| Varianta | Prům. efekt bits | Průměr best_val_loss | Std | Min | Max |
|---|---|---|---|---|---|
| **const_1bit** | 1,0 | **7,433556** | 0,024252 | 7,409514 | 7,458013 |
| **prog_1_2_4** | 2,0 | **7,442756** | 0,026574 | 7,412376 | 7,461684 |
| **baseline** | 16.0 | **7,443424** | 0,022007 | 7,419442 | 7,462691 |
| **const_4bit** | 4,0 | **7,445057** | 0,018723 | 7,426016 | 7,463445 |
| **prog_4_2_1** | 2,0 | **7,457097** | 0,010847 | 7,445405 | 7,466833 |
| **const_2bit** | 2,0 | **7,458633** | 0,011545 | 7,445650 | 7,467745 |

Seřazeno podle střední hodnoty best_val_loss vzestupně (nižší = lepší).

#### Pořadí

1. const_1bit — průměr 7,4336, tepy baseline za 2/3 seeds (seeds 7 a 123)
2. prog_1_2_4 — průměr 7,4428, tepy baseline za 1/3 seeds (seed 42)
3. baseline — průměr 7,4434 (referenční)
4. const_4bit — průměr 7,4451, tepy baseline za 1/3 seeds (seed 42)
5. prog_4_2_1 — průměr 7,4571, tepy baseline za 0/3 seeds
6. const_2bit — průměr 7,4586, tepy baseline za 0/3 seeds

Delta prog_1_2_4 vs baseline: −0,0007 (lepší, ale v rámci šumu). Delta const_1bit vs baseline: −0,0099 (lepší, větší signál, stále v rámci 1σ vzhledem k std ~0,022).

#### Odpovědi na výzkumné otázky (z úplných ablačních dat)

**Q1 — Má progresivní přesnost konzistentně outperform/match baseline?**
prog_1_2_4 průměr (7,4428) vs baseline průměr (7,4434): delta = −0,0007 ve prospěch progresivního. Práh ablačního skriptu pro "PROG WINS" je delta < −0,001. Při delta = -0,0007 nedosahuje prahové hodnoty. Klasifikace: TIED (v rámci hluku). prog_1_2_4 překonává baseline per-seed průměr na 1 ze 3 seeds.

**Q2 — Předčí progresivní výkon constant-bit alternativ při stejném průměru bits (const_2bit)?**
prog_1_2_4 (7,4428) vs const_2bit (7,4586): delta = −0,016. Ablační skript klasifikuje |delta| < 0,002 jako neprůkazné a 0,002–0,02 jako „nerozlišitelné v této škále“. Při delta = −0,016 je to hraniční. Na výstupu skriptu by bylo uvedeno „NEZÁVĚRNÉ: schedule struktura vs regularizace nerozlišitelná v tomto měřítku.“

**Q3 — Záleží na směru (1→2→4 vs 4→2→1)?**
prog_1_2_4 (7,4428) vs prog_4_2_1 (7,4571): delta = -0,014. Coarse-to-fine se jeví lépe, ale při 3 seeds se std ~0,011–0,027 to není statisticky prokázáno.

**Q4 — const_1bit vs baseline:**
Největší signál v datové sadě. const_1bit na prvním místě. Zdá se, že binární váhy fungují jako regularizace, která může pomoci zobecnění. seed rozptyl je však velký (const_1bit std = 0,024 vs baseline std = 0,022) a rozdíly per-seed nejsou konzistentní napříč seeds.

#### Pozorování o rozptylu

Seed rozptyl je značný napříč všemi variantami. Směrodatné odchylky 0,011–0,027 nats v best_val_loss ve srovnání s rozdíly mezi variantami 0,001–0,025 nats. To znamená, že překrývající se intervaly spolehlivosti jsou pravděpodobně u většiny párových srovnání. Tréninková doba se také podstatně liší napříč seeds (např. const_4bit: 7471s až 12879s), což pravděpodobně odráží tepelné škrcení Apple Silicon a procesy na pozadí během dlouhých běhů.

Mnoho běhů se stále zlepšuje při kroku 10 000 (best_step = 10 000 pro 8 z 18 běhů), což naznačuje, že delší trénink by mohl odhalit jasnější rozdíly.

---

## 7. METODICKÁ OMEZENÍ

### 7.1 Malý model

S parametry 28,3M se jedná o model toy-scale. Chování pozorované v tomto měřítku se nemusí přenést do modelů, kde je kvantizace prakticky relevantní (např. parametry 1B+, kde je skutečným omezením paměť). Výzkumné otázky jsou zde zkoumány jako důkazy konceptu.

### 7.2 Omezený soubor dat

~69 milionů tokens z 50 000 článků na Wikipedii. Moderní malé jazykové modely jsou trénovány na bilionech tokens. Při 10 000 krocích s batch_size=8 a seq_len=256, celkové zpracované školení tokens = 10 000 × 8 × 256 × 0,5 (maskovaný zlomek) ≈ 10 milionů unikátních maskovaných token predikcí. Mnoho tréninkových sekvencí se může opakovat. To omezuje strop kvality.

### 7.3 Statistická síla: pouze 3 seedy

Při n=3 je standardní chyba průměru std/sqrt(3) ≈ 0,013 pro const_1bit a ≈ 0,013 pro baseline. Vzhledem k pozorovaným rozdílům mezi variantami 0,001–0,016 nats většina párových srovnání nedosahuje statistické významnosti při konvenčních prahových hodnotách. Ablační studie poskytuje směrový důkaz, ale nemůže definitivně seřadit varianty.

### 7.4 Rozdíly validačního loss jsou malé

Všechny rozdíly mezi variantami spadají do rozmezí 0,001–0,025 nats. Není známo, zda jsou tyto údaje prakticky smysluplné (z hlediska kvality textu, výkonu následné úlohy); rozdíly zmatenosti této velikosti v tomto měřítku nemusí být v kvalitě generovaného textu zjistitelné.

### 7.5 Pouze simulovaná kvantizace

Nejkritičtější omezení: všechny kvantované operace jsou simulovány v float32 přes STE. Hodnoty hmotnosti použité v matmulu jsou aproximace typu float32 (např. znaménko (w) × střední (|w|) pro binární), nikoli sbalené celočíselné kódy. Důsledky:
- Žádná skutečná redukce paměti v době tréninku (všechny váhy jsou float32 master váhy)
- Žádné skutečné zrychlení inference (simulace float32 je pomalejší než nativní float32)
- Odhady komprese úložiště (např. 2,67× pro prog_1_2_4) jsou teoretické a platily by pouze v případě, že by váhy byly skutečně zabaleny do 1/2/4-bit celých čísel
- Experiment testuje, zda STE gradientový signál během trénování ovlivňuje kvalitu modelu, ne zda je low-bit inference rychlá

### 7.6 Nedeterminismus na Apple Silicon

Navzdory nastavení `mx.random.seed(seed)` i `np.random.seed(seed)`, MLX na Apple Silicon nezaručuje bit-for-bit reprodukovatelnost napříč samostatnými běhy procesu. Stejná konfigurace se stejným seed spuštěním dvakrát v různých relacích může produkovat mírně odlišné val_loss křivky. Všech 18 úplných ablačních běhů byly samostatné procesy. To je výslovně uvedeno v kódu a je to přirozené omezení kombinace hardware/framework.

### 7.7 Odhady paměti jsou teoretické

`training_memory_estimate_mb` v sestavách úložiště (≈452 MB pro celý model) se vypočítá jako `total_params × 4 × 4 / 1e6` (4 bajty na parametr × 4 kopie: hlavní váhy, přechody, Adam m, Adam v). Toto nezohledňuje aktivační paměť během forward/backward průchodu, která se mění s velikostí dávky × seq_len × d_model. Skutečná špičková paměť na zařízení se může lišit.

### 7.8 Staré Q4 schéma v const_4bit

const_4bit ablační varianta byla trénována podle staršího 15-úrovňového with-zero Q4 schématu, zatímco PTQ studie používá nové 16-úrovňové no-zero Q4 schéma. Jakékoli srovnání mezi těmito dvěma je porovnáváním různých kvantizačních funkcí. To je zdokumentováno v `src/quantization.py` a označeno v `scripts/ptq_study.py`.

---

## 8. PTQ STUDIE — historický stav v době dokumentace

### Vědecká otázka

Vzhledem k naprosto stejné kvantizační funkci (`quantize_weights(w, bits)`), je lepší trénovat pod STE gradientním tlakem na cíli bit-width během tréninku (nativní QAT), nebo aplikovat stejnou kvantizaci post-hoc na high-precision checkpoint (Direct/Naive PTQ)?

Toto NENÍ srovnání s state-of-the-art PTQ metodami (GPTQ, AWQ, calibration-based). Jde o řízené srovnání, kde jedinou proměnnou je, když je aplikována kvantizace (během tréninku vs. v době hodnocení).

### Proč Direct/Naive PTQ

V době vyhodnocování probíhá jak nativní QAT, tak přímý PTQ hovor `model.set_bits(bits)` před každým dopředným průchodem, který spouští `ste_quantize()` → `quantize_weights()`. Protože během vyhodnocování neexistují žádné gradienty, rozlišení STE vs. přímé kvantování se zhroutí: obě cesty platí `quantize_weights(w, bits)` pro uložené váhy. Rozdíl je zcela v tom, co bylo provedeno během tréninku: nativní QAT optimalizovalo hlavní váhy pod tlakem gradientu STE při bits, zatímco PTQ optimalizovalo při bits=16 (plná přesnost).

### Experimentální matice

- Seeds: 42, 123, 7 (odpovídající ablační studie)
- PTQ bits (hlavní matice): 1, 2, 3, 4, 16
- PTQ bits (volitelné): 0 (ternární, přístupné přes `--include-ternary`)
- Celkem hlavních hodnocení: 3 seeds × 5 bits = 15 hodnocení
- S ternárním: 3 seeds × 6 bits = 18 hodnocení

### Fáze

**Fáze 1 — Trénink 3 baseline checkpoints** (konfigurace existují, checkpoints ne):
Každý baseline je plný tréninkový běh o 10 000 krocích s použitím `configs/ptq/ptq_baseline_s{42,123,7}.json`. Tyto konfigurace jsou totožné s konfiguracemi ablace full-phase baseline, kromě `save_checkpoints=True` a `checkpoint_every=999999` (uložená pouze jedna checkpoint: poslední krok). Cílová hodnota checkpoint pro každou z nich je `checkpoints/ptq_baselines/ptq_baseline_s{seed}/step_0010000.npz`.

Poznámka: Stávající `checkpoints/full_baseline/` a `checkpoints/full_progressive_1_2_4/` checkpoints (pouze z fáze 2, seed=42) NEPOUŽÍVÁ studie PTQ. Studie PTQ vyžaduje samostatné baseline běhy ve všech 3 seeds, do jiného checkpoint adresáře.

**Fáze 2 — Aplikujte Direct/Naive PTQ hodnocení**:
Načtěte FP32 checkpoint, zavolejte `model.set_bits(bits)` jednotně pro všechny kroky a vyhodnoťte jej na validační sadě (100 batchů na evaluaci). Bez přetrénování a bez kalibrace.

**Fáze 3 – Analýza**:
Porovnejte PTQ výsledky s výsledky nativní ablace (načtené z `results/ablation_full/*/final_summary.json` — není potřeba žádný přepočet). Sestavte srovnávací tabulku: Δ(PTQ vs FP32), Δ(nativní vs PTQ) na každé bitové úrovni. Otestujte, zda nativní QAT poskytuje výhodu kvality oproti Direct PTQ a zda se tato výhoda zvyšuje při nižších bitových úrovních.

### Q4 upozornění na schéma

Hodnocení PTQ Q4 využívá nové schéma 16 úrovní. Nativní const_4bit baseline bylo trénováno podle starého 15úrovňového with-zero schématu. Skript PTQ tuto neshodu výslovně označí.

### Q3 poznámka

Q3 (skutečný 3bitový, 8 úrovní) v původní sadě nativní ablace chyběl. Pozdější kampaň vyškolila jeden nativní Q3 protějšek na seed 31415 (nejlepší hodnota ztráty 7,402252), což je užitečný single-seed důkaz, ale ne replikované srovnání. Zbývají další spárované seeds.

### Konfigurace, které existují

- `configs/ptq/ptq_baseline_s42.json`
- `configs/ptq/ptq_baseline_s123.json`
- `configs/ptq/ptq_baseline_s7.json`

### Existující checkpointy

Žádné PTQ baseline checkpoints zatím neexistují. Adresář `checkpoints/ptq_baselines/` neexistuje.

### Skript

`scripts/ptq_study.py` — plně implementováno, připraveno k provozu.

### Příkazy

```bash
# Full study (Phase 1 + Phase 2 + Phase 3, ~6-7h estimated)
python scripts/ptq_study.py

# Skip training if checkpoints already exist
python scripts/ptq_study.py --skip-training

# Skip training + PTQ eval; load saved ptq_eval_results.json and analyze
python scripts/ptq_study.py --eval-only

# Dry run: print plan, train nothing
python scripts/ptq_study.py --dry-run

# Include optional ternary evaluation
python scripts/ptq_study.py --include-ternary

# Reduce eval batches for faster (less accurate) evaluation
python scripts/ptq_study.py --eval-steps 50
```

### Odhadovaná doba běhu

3 × ~1,5h trénink + 15 × ~5min vyhodnocení = ~4,5h + 1,25h ≈ 6–7h celkem.

---

## 9. MAPA SOUBORU VÝSLEDKŮ

Ověřeno podle skutečného obsahu úložiště.

```
results/
  ablation/                           # 3k-step screening phase (COMPLETED)
    per_run_screen.csv                # 18 rows: one per (variant, seed)
    per_run_full.csv                  # 18 rows: full-phase summary (written by ablation_study.py --analyze-only --phase full)
    aggregate_screen.json             # mean/std/min/max per variant across 3 seeds (screen phase)
    aggregate_full.json               # mean/std/min/max per variant across 3 seeds (full phase)
    abl_baseline_s7_scr/
      final_summary.json
      eval_history.json
      train_metrics.csv
    abl_baseline_s42_scr/             # (similar structure for all 18 screening runs)
    abl_baseline_s123_scr/
    abl_const_1bit_s7_scr/
    abl_const_1bit_s42_scr/
    abl_const_1bit_s123_scr/
    abl_const_2bit_s7_scr/
    abl_const_2bit_s42_scr/
    abl_const_2bit_s123_scr/
    abl_const_4bit_s7_scr/
    abl_const_4bit_s42_scr/
    abl_const_4bit_s123_scr/
    abl_prog_1_2_4_s7_scr/
    abl_prog_1_2_4_s42_scr/
    abl_prog_1_2_4_s123_scr/
    abl_prog_4_2_1_s7_scr/
    abl_prog_4_2_1_s42_scr/
    abl_prog_4_2_1_s123_scr/

  ablation_full/                      # 10k-step full ablation (COMPLETED)
    abl_baseline_s7_full/
      final_summary.json
      eval_history.json
      train_metrics.csv
    abl_baseline_s42_full/            # (same structure for all 18 full runs)
    abl_baseline_s123_full/
    abl_const_1bit_s7_full/
    abl_const_1bit_s42_full/
    abl_const_1bit_s123_full/
    abl_const_2bit_s7_full/
    abl_const_2bit_s42_full/
    abl_const_2bit_s123_full/
    abl_const_4bit_s7_full/
    abl_const_4bit_s42_full/
    abl_const_4bit_s123_full/
    abl_prog_1_2_4_s7_full/
    abl_prog_1_2_4_s42_full/
    abl_prog_1_2_4_s123_full/
    abl_prog_4_2_1_s7_full/
    abl_prog_4_2_1_s42_full/
    abl_prog_4_2_1_s123_full/

  full_baseline/                      # Phase 2 single-seed baseline (seed=42)
    final_summary.json                # best_val_loss: 7.432665
    eval_history.json
    train_metrics.csv

  full_progressive_1_2_4/             # Phase 2 single-seed progressive (seed=42)
    final_summary.json                # best_val_loss: 7.414686
    eval_history.json
    train_metrics.csv

  short_exp_baseline/                 # Phase 1 short experiment
    final_summary.json                # best_val_loss: 7.529157 (500 steps)
    eval_history.json
    train_metrics.csv

  short_exp_progressive_2bit/         # Phase 1 short experiment
    final_summary.json                # best_val_loss: 7.530751 (500 steps)
    eval_history.json
    train_metrics.csv

  short_exp_progressive_ternary/      # Phase 1 short experiment (old ternary scheme)
    final_summary.json                # best_val_loss: 7.533732 (500 steps)
    eval_history.json
    train_metrics.csv

checkpoints/
  full_baseline/                      # Phase 2 baseline: 17 checkpoints at 324 MB each
    latest_meta.json
    step_0000500.npz  ... step_0010000.npz

  full_progressive_1_2_4/             # Phase 2 progressive: 16 checkpoints at 324 MB each
    latest_meta.json
    step_0000500.npz  ... step_0010000.npz

  short_exp_baseline/                 # Phase 1 baseline: 4 checkpoints at ~87 MB each
    latest_meta.json
    step_0000100.npz, step_0000200.npz, step_0000400.npz, step_0000500.npz

  short_exp_progressive_2bit/
  short_exp_progressive_ternary/

  smoke_test_baseline/                # Smoke test: 2 checkpoints
    step_0000025.npz, step_0000050.npz

  smoke_test_progressive/
    step_0000025.npz, step_0000050.npz

  # NOTE: checkpoints/ptq_baselines/ does NOT exist yet (PTQ study not run)

configs/
  baseline.json
  full_baseline.json
  full_progressive_1_2_4.json
  progressive_1_2_4.json
  short_exp_baseline.json
  short_exp_progressive_2bit.json
  short_exp_progressive_ternary.json
  smoke_test.json                     # smoke test progressive
  smoke_test_baseline.json

  ablation/                           # 36 auto-generated configs (18 screen + 18 full)
    ablation_baseline_s42_screen.json
    ablation_baseline_s42_full.json
    ablation_baseline_s123_screen.json
    ablation_baseline_s123_full.json
    ablation_baseline_s7_screen.json
    ablation_baseline_s7_full.json
    ablation_const_1bit_s42_screen.json
    ... (similar pattern for all 6 variants × 3 seeds × 2 phases)

  ptq/                                # PTQ baseline configs (exist, not yet used)
    ptq_baseline_s42.json
    ptq_baseline_s123.json
    ptq_baseline_s7.json

scripts/
  ablation_study.py
  ptq_study.py
  prepare_data.py
  train_tokenizer.py

src/
  __init__.py
  config.py
  quantization.py
  model.py
  diffusion.py
  data.py
  train.py
  evaluate.py
  generate.py

tests/
  test_quantization.py
  test_model.py
  test_diffusion.py
  test_training.py

tokenizer/
  wiki_bpe/
    tokenizer.json                    # BPE tokenizer (1.1 MB)
    vocab_info.json

data/
  cache/
    train_seq256_art50000_bytes500000000.npy  (~250 MB)
    val_seq256_art50000_bytes500000000.npy    (~13 MB)
    meta_seq256_art50000_bytes500000000.json
    (plus smaller caches for smoke test / short exp)
```

---

## 10. REPRODUKOVATELNOST

### Prostředí

- Platforma: Apple Silicon macOS (testováno na M4, 16 GB unifikované paměti)
- Rámec: MLX >= 0,21,0
- Závislosti Pythonu: viz `requirements.txt`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Příprava dat

```bash
# Step 1: Train BPE tokenizer (needed once)
python scripts/train_tokenizer.py \
    --vocab-size 16000 \
    --max-articles 500 \
    --max-bytes 5000000 \
    --output tokenizer/wiki_bpe

# Step 2: Prepare main dataset (will cache to data/cache/)
python scripts/prepare_data.py \
    --max-articles 50000 \
    --max-bytes 500000000 \
    --seq-len 256 \
    --tokenizer-path tokenizer/wiki_bpe \
    --cache-dir data/cache
```

### Smoke test (end-to-end kontrola pipeline)

```bash
./run_smoke_test.sh
```

### Trénink jednoho modelu

```bash
# Baseline (FP32, no quantization)
python -m src.train --config configs/full_baseline.json

# Progressive [1,1,1,1,2,2,4,4]
python -m src.train --config configs/full_progressive_1_2_4.json

# Any custom config
python -m src.train --config configs/<your_config>.json
```

### Ablační studie

```bash
# Screening phase only (3k steps × 18 runs, ~9h)
python scripts/ablation_study.py --phase screen

# Full phase only (10k steps × 18 runs, ~45h)
python scripts/ablation_study.py --phase full

# Full study (screen then full)
python scripts/ablation_study.py --phase both

# Resume skipping completed runs
python scripts/ablation_study.py --phase full --resume

# Dry run: print plan only
python scripts/ablation_study.py --dry-run

# Analysis only (requires existing results)
python scripts/ablation_study.py --analyze-only --phase full

# Analysis of both phases
python scripts/ablation_study.py --analyze-only --phase both
```

### PTQ studie

```bash
# Full study (~6-7h: 3 training runs + 15 PTQ evals)
python scripts/ptq_study.py

# Skip training, run PTQ evals only (requires checkpoints/ptq_baselines/)
python scripts/ptq_study.py --skip-training

# Skip training + evals, analyze saved ptq_eval_results.json only
python scripts/ptq_study.py --eval-only

# Dry run: print plan without executing
python scripts/ptq_study.py --dry-run

# Include optional ternary (3-state) evaluation
python scripts/ptq_study.py --include-ternary

# Use fewer eval batches (faster, less accurate)
python scripts/ptq_study.py --eval-steps 50
```

### Evaluace a generování

```bash
# Compare two checkpoints
python -m src.evaluate \
    --baseline checkpoints/full_baseline/step_0010000.npz \
    --progressive checkpoints/full_progressive_1_2_4/step_0010000.npz \
    --config configs/full_progressive_1_2_4.json \
    --eval-steps 100

# Generate text
python -m src.generate \
    --checkpoint checkpoints/full_progressive_1_2_4/step_0010000.npz \
    --config configs/full_progressive_1_2_4.json \
    --n-sequences 4 \
    --seq-len 128
```

---

## 11. PLÁN BUDOUCÍCH EXPERIMENTŮ

### Stav: NEXT (okamžitá priorita)

**A. Dokončete Direct/Naive PTQ studii**

Spustit `scripts/ptq_study.py`. To vyžaduje:
1. Trénink 3 baseline checkpoints (seeds 42, 123, 7) do `checkpoints/ptq_baselines/`
2. Spuštění PTQ hodnocení při bits ∈ {1, 2, 3, 4, 16} pro každé checkpoint
3. Vytvoření srovnávací tabulky s výsledky nativní ablace

Tím odpovíte přímo na RQ6–RQ8.

**B. Interpretovat PTQ vs. nativní výsledky v kontextu Q4 nesouladu schémat**

Po PTQ studii vyžaduje Q4 srovnání opatrnost. K získání čistého Q4 srovnání by byla potřeba nativní ablace podle nového 16-úrovňového Q4 schématu (viz bod E níže).

---

### Stav: PLÁNOVANÝ (následující kroky)

**C. Přidejte další seeds, pokud výsledky zůstanou neprůkazné**

Při n=3 má většina párových srovnání překrývající se intervaly spolehlivosti. Přidání seeds 456, 789, 2024 by zvýšilo statistickou sílu. Doporučuje se, pokud PTQ výsledky jsou také neprůkazné.

**D. Trénujte nativní True-Q3 model (bits=3, 8 úrovní, žádná nula)**

Q3 v původní nativní ablaci chyběl, ale jeden nativní Q3 běh na seed 31415 je nyní dokončen. Nesmí se s nimi nakládat jako s replikovanými důkazy; další srovnání by mělo spárovat aktuální true Q3 s aktuálním true Q4, ternary a FP32 napříč více seeds.

**E. Trénujte nativní True-Q4 model (bits=4, 16 úrovní, žádná nula)**

Stávající const_4bit ablace využívala staré 15-ti úrovňové schéma. Nová ablace podle současného 16-úrovňového no-zero Q4 schématu by:
- Uveďte čisté baseline pro Q4 PTQ srovnání (bez upozornění scheme-mismatch)
- Umožněte Q4 porovnat s Q3 ve stejné no-zero symetrické rodině

**F. Kalibrované / pokročilé PTQ jako samostatná studie**

To by porovnalo současný přímý PTQ přístup s kalibrovanými metodami (GPTQ-style rekonstrukce hmotnosti, AWQ-style activation-aware škálování, fine-tuning po PTQ). Jedná se o podstatně větší rozsah a mělo by se s ním nakládat jako se samostatnou fází projektu.

---

### Stav: HYPOTETICKÝ / BUDOUCÍ VÝZKUM

Jde o myšlenky a hypotézy, které nebyly realizovány nebo systematicky prozkoumány. Jsou uvedeny jako směry výzkumu, nikoli jako ověřené výsledky.

**G. Škálování velikosti modelu**

Historické kampaně používaly přibližně 28M a 7,5M modely. Aktivní český pilot
je samostatný 25vrstvý model `d_model=64`; `d_model=32` slouží jako kontrola a
pokus `d_model=128` byl numericky nestabilní. Ověření na 100M, 500M nebo 1B+
parametrech zůstává budoucí prací.

**H. Větší datové sady a delší trénink**

Historická anglická datová sada měla přibližně 69M tokenů. Aktivní česká cache
má 73 706 496 tokenů a v dlouhém běhu se opakovaně používá; další kroky přidávají
optimalizační expozici, nikoli nové znalosti. Budoucí rozšíření má přidávat
ověřená česká data bez nahrazení původního korpusu.

**I. Skutečně zabalená low-bit jádra**

Implementace skutečné celočíselné aritmetiky pomocí vlastních MLX Metal shaderů nebo využití Apple Neural Engine by převedlo teoretické odhady komprese na skutečnou paměť a měření rychlosti. To by umožnilo měření skutečného tokens/second, skutečného špičkového využití paměti a skutečné spotřeby energie – žádné z nich v současnosti nelze měřit.

**J. Binární rozklad (HYPOTETICKÝ)**

Hypotéza, že více 1bitových maticových operací lze kombinovat k aproximaci higher-precision výpočtu. Například dva 1bitové matmuly s různými měřítkovými faktory by teoreticky mohly aproximovat 2bitovou operaci. Toto je neověřený architektonický nápad, neprozkoumaný v žádném současném experimentu.

**K. Progresivní přesnost při inference pro adaptivní výpočet (ČÁSTEČNĚ)**

Repozitář už obsahuje pevné flexibilní routes, early exit podle vrstev i
diffusion kroků a route-by-exit diagnostiku. Hypotetická zůstává kalibrovaná
produkční policy se skutečným přeskakováním kernelů pro jednotlivé tokeny a
změřenou end-to-end úsporou.

**L. Přenos zjištění do podstatně větších modelů**

Všechna zjištění zůstávají omezená na malé výzkumné modely a širší český pilot
`d_model=128` numericky divergovala. Hypotéza, že výhody low-bit tréninku nebo
flexibilních routes přetrvají ve výrazně větším měřítku, není prokázána.

---

## 12. AKTUÁLNÍ STAV PROJEKTU

### AKTIVNÍ ČESKÝ BĚH SE SDÍLENÝMI VAHAMI (ověřeno 18. 8. 2026 13:56 CEST)

- M1-512 trénuje z obnoveného checkpointu český 25vrstvý model `d_model=64`,
  `d_ff=256`, se čtyřmi heads a délkou sekvence 256.
- Routes jsou `q8_only`, `q8_fp16` a `q2_q8_fp16`; checkpoint vybírá nejhorší
  route.
- Živý krok: 3 206 000 / 20 000 000. Nejhorší route `q2_q8_fp16`: loss 4,4015,
  accuracy 30,87 %, perplexita 81,57. Nejlepší worst-route loss: 4,3451 na
  kroku 2 891 500.
- MLX je hlavní backend. Převod a resume přes PyTorch/CUDA jsou implementovány,
  ale oba backendy stále simulují low-bit aritmetiku nad FP32 master vahami.
- Jde o výzkumný model pro doplňování maskovaných tokenů, nikoli instruction
  chatbot. Podrobnosti obsahuje datovaný
  [aktuální stav](docs/cswiki-flexible-project-status-2026-08-18.md).

### DOKONČENO

- Plná softwarová implementace: model, kvantizace, difúzní proces, trénovací smyčka, vyhodnocení, generování, ablační rámec, PTQ framework
- Tokenizer školení (BPE, 16k vocab, uloženo na `tokenizer/wiki_bpe/`)
- Příprava datové sady a ukládání do mezipaměti (50 000 článků, ~69 milionů tokens, mezipaměť `data/cache/`)
- Smoke testy (50 kroků, ověření pipeline)
- Fáze 1 krátké experimenty (500 kroků, 3 varianty, jeden seed)
- Fáze 2 full-scale počáteční srovnání (10 000 kroků, baseline vs. prog_1_2_4, seed=42)
- **Ablační screening 3. fáze** (3 000 kroků, 6 variant × 3 seeds = 18 běhů, vše dokončeno)
- **Fáze 4 úplná ablace** (10 000 kroků, 6 variant × 3 seeds = 18 běhů, vše dokončeno)
- Q4 aktualizace schématu (z 15 úrovně with-zero na 16 úrovní no-zero pro konzistenci)
- Q3 (skutečně 3bitová) implementace
- PTQ studie návrh a implementace skript
- **Direct/naive PTQ zotavení**: 18/18 hodnocení napříč 3 seeds a Q1/Q2/Q3/Q4/FP32/ternary, s celkovým JSON/CSV ověřeným
- Jedna nativní true-Q3 a jedna nativní ternární 10k běží na seed 31415
- Dvě další párové baseline/Q1/progressive replikace (seeds 31415 a 27182)

### DALŠÍ KROKY HISTORICKÉ RODINY ANGLICKÝCH EXPERIMENTŮ

1. Spusťte scheme-matched multi-seed nativní srovnání FP32, aktuální pravdivé Q3, aktuální skutečné Q4 a trojčlenné.
2. Rozdělte šest spárovaných seeds rovnoměrně napříč m1-256 a m1-512, se čtyřmi sériovými 10k variantami na seed a neměnnými node-local výstupy.
3. Pre-register párové delty a intervaly spolehlivosti; ponechat legacy-Q4 a old-bits=3 výsledky historické a mimo primární matici.

### BUDOUCNOST

- Přidejte seeds (lepší statistická spolehlivost)
- Replikovaný nativní Q3 trénink se shodným schématem nad rámec dokončeného single-seed běhu
- Nativní Q4 podle aktualizovaného schématu
- Kalibrovaná PTQ studie
- Scale-up experimenty
- Reálná low-bit implementace jádra (aktuální memory/speed měření)

---

## 13. VÝZKUMNÝ DENNÍK

Chronologický příběh rekonstruovaný z časových razítek souborů, obsahu konfigurace a dat výsledků.

**[2026-07-15] — Počáteční nastavení a smoke testy**

Úložiště vytvořeno. Zapsané základní zdrojové soubory: `src/model.py`, `src/quantization.py`, `src/diffusion.py`, `src/train.py`, `src/data.py`, `src/config.py`, `src/evaluate.py`, `src/generate.py`. Písemné jednotkové testy. BPE tokenizer vyškoleni. Vytvořeny a spuštěny počáteční konfigurace testu kouře (50 kroků, malý model d_model=128). Checkpoints uloženo do `checkpoints/smoke_test_baseline/` a `checkpoints/smoke_test_progressive/`. V této fázi quantization.py použilo jiné schéma pro bits=3 (ternární) a bits=4 (15-úrovňové s nulou) ve srovnání se současnou implementací.

Byly vytvořeny konfigurace krátkého experimentu (d_model=256, 4 vrstvy, 500 kroků) a spuštěny: `short_exp_baseline`, `short_exp_progressive_2bit`, `short_exp_progressive_ternary`. Výsledky: všechny varianty ~7,53 val_loss, žádný signál při 500 krocích se 100 články. Rozhodnutí: zvětšit na úplný model.

Full-scale zapsané konfigurace (`configs/baseline.json`, `configs/progressive_1_2_4.json`, `configs/full_baseline.json`, `configs/full_progressive_1_2_4.json`).

**[2026-07-15 / 07-16] — Full-scale počáteční srovnání**

Celý tréninkový běh o 10 000 krocích pro `full_baseline` (seed=42) a `full_progressive_1_2_4` (seed=42). Oba používali aktualizované konfigurace s `tie_word_embeddings=True` a `results_dir="results"`. Výsledky: baseline best_val_loss = 7,432665 (4990s), progresivní best_val_loss = 7,414686 (5663s). Progresivní takt baseline o 0,018 nats při seed=42. To motivovalo ablační studii k testování s více seeds.

**[2026-07-16] — Návrh ablační studie a fáze screeningu**

`scripts/ablation_study.py` napsáno. Definováno 6 variant: baseline, const_1bit, const_2bit, const_4bit, prog_1_2_4, prog_4_2_1. Všech 18 konfigurací promítání auto-generated do `configs/ablation/`. 18 cyklů screeningu (každý 3 000 kroků) provedených postupně. Vše dokončeno. Souhrnné výsledky ukazují všechny varianty v rozmezí 0,007 nats od sebe ve 3k krocích; rozhodnuto spustit všech 6 variant na plných 10 000 kroků.

**[2026-07-16 / 07-18] — Fáze úplné ablace**

Všech 18 úplných 10k-krokových ablačních běhů bylo provedeno. Spuštění trvala přibližně od 16. července 18:00 do 18. července 15:46 (odvozeno z časových razítek konfiguračního souboru). Všech 18 dokončilo bez chyb. `aggregate_full.json` a `per_run_full.csv` napsáno. Klíčové zjištění: const_1bit dosahuje nejlepšího průměru best_val_loss (7,4336), přičemž překonává baseline (7,4434) o 0,0099 nat, i když s vysokým seed rozptylem. prog_1_2_4 (7,4428) je v podstatě svázán s baseline (7,4434).

**[2026-07-18] — PTQ návrh studie a Q4/Q3 aktualizace schématu**

Byl vytvořen `scripts/ptq_study.py`. Q4 schéma bylo změněno z 15 úrovní s
nulou na 16 úrovní bez nuly, Q3 bylo přidáno jako skutečná 3bitová úroveň a
ternární varianta byla přesunuta z `bits=3` na sentinel `bits=0`. Konfigurace
baseline byly uloženy do `configs/ptq/`. Tento odstavec zaznamenává stav z
18. 7. 2026; pozdější obnova direct/naive PTQ dokončila všech 18 požadovaných
evaluací.

---

*Konec technické dokumentace. Všechny číselné výsledky pocházejí přímo z JSON/CSV souborů v úložišti.*
