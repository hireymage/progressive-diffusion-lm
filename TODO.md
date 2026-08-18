# TODO — Progressive Diffusion LM

Poslední aktualizace: 2026-08-18

---

## 🎯 Nový směr projektu

- Cílem už není jen porovnávat progresivní kvantizaci jako fixed schedule.
- Nový směr je vyvíjet difuzní jazykový model, který začne hrubým odhadem a postupně se zpřesňuje jen tam, kde je to ještě potřeba.
- Model má umět generovat více tokenů najednou a při nejistotě přidávat další výpočetní krok nebo vyšší přesnost.
- Prakticky to znamená navrhnout novou strategii generování, vyjasnit potřebné komponenty a pak znovu poskládat experimenty podle tohoto cíle.
- Jako užitečný mezikrok dává smysl vytvořit jednoduchý funkční PD model, který umí text skutečně generovat, i kdyby ještě nebyl finálně optimalizovaný.
- Původní experimenty a výsledky zůstávají zachované jako historie projektu, ale nový vývoj se bude řídit tímto zpřesněným cílem.
- Cílový první model mění přesnost **mezi skupinami Transformerových vrstev**, nikoli po celém průchodu modelem: 5× Q1, 5× Q2, 5× Q4, 5× Q8 a 5× FP16.
- Od vrstvy 5 lze sekvenčně ukončit výpočet po libovolné další vrstvě, tedy i uvnitř skupiny (např. po vrstvě 8).

## ✅ Hotovo

### Infrastruktura a pipeline
- [x] Základní pipeline: tokenizer, dataset, model, training loop, eval
- [x] BPE tokenizer trénování (`vocab_size=16 000`, wikimedia/wikipedia)
- [x] `QuantizedLinear` + Straight-Through Estimator
- [x] `model.set_bits()` — runtime přepínání precize
- [x] Campaign harness `scripts/run_dual_m1_campaign.py` (lock, artifact publish)
- [x] Multi-node SSH spouštění (m1-256, m1-512, m4-air přes ZeroTier)
- [x] **PyTorch/CUDA backend pro Windows a Linux**
  - Implementován převod MLX checkpointů včetně vah, AdamW momentů a globálního kroku.
  - Implementováno pokračování českého flexibilního tréninku přes CUDA se stejnou architekturou, tokenizerem, cache a routes.
  - Převod i CUDA trenér jsou pokryté testy; postup je v `docs/cuda-training.md` a `docs/cuda-training.en.md`.
  - Q2/Q8 zatím zůstávají QAT fake-quant výpočty nad FP32 master vahami, nikoli packed integer CUDA kernely.
- [x] Unit testy — 280/280 passed

### Kvantizační schémata (opravené)
- [x] Q1 binary (2 úrovně, bits=1)
- [x] Q2 true 2-bit (4 úrovně, bits=2)
- [x] Q3 true 3-bit (8 úrovní, bits=3) — **implementováno v této session**
- [x] Q4 true 4-bit (16 úrovní bez nuly, bits=4) — **opraveno z 15 úrovní**
- [x] Q8 8-bit symmetric (bits=8, Phase 2)
- [x] Ternary/3-state přesunut na bits=0 (odděleno od Q3)
- [x] PTQ přejmenováno na "Direct/Naive PTQ" (bez kalibrace/GPTQ/AWQ)

### Experimenty
- [x] Smoke testy (50 kroků, tiny model)
- [x] Krátké experimenty (500 kroků, 3 varianty)
- [x] Iniciální full comparison (10 000 kroků, seed=42) — progressive o 0,018 lepší
- [x] Ablation screening (3 000 kroků, 18 běhů = 6 variant × 3 seedy)
- [x] **Plný ablation (10 000 kroků, 18 běhů)** — const_1bit nejlepší průměr (7,4336)
- [x] **Phase 1 — PTQ + native gaps** (m1-256, seeds 42/123/7/31415)
- [x] **Phase 1 — Paired replication** (m1-512, seeds 31415/27182)
- [x] **Phase 1 — Matched-noise campaign** (3 nody, 12 seedů/varianta, 48 běhů)
  - Závěr: Q1 výhoda NENÍ noise regularizace; matched-noise FP32 horší o ~0,026
- [x] **Phase 2 — Bidirectional incremental** (42 běhů, 7 schedules, bits=8 přidán)
  - Nejlepší: constant-2b (7,4287); progressive-down nejhorší
- [x] **Phase 2 — Inference eval** (3 nody, early-exit až 9,47× speedup)
- [x] M2 Incremental forward (`y_next = y_prev + Δ`, 1,32× speedup)
- [x] M3 Early-exit generation (threshold ≤ 0,03 → 1 krok, 5–9× speedup, 100% shoda)

### Dokumentace
- [x] `PROJECT_DOCUMENTATION.md` — 1 042 řádků, 13 sekcí
- [x] `README.md` — přepsán, veřejný přehled
- [x] Obsidian `_Project.md` — aktualizován (Phase 1 + Phase 2 výsledky)
- [x] `src/model.py` — docstring opraven (bits schémata)
- [x] `src/quantization.py` — docstring a EFFECTIVE_BITS aktualizovány

---

## 🔜 Přímé další kroky (doporučené)

### 1. M0 — funkční PD-LM baseline podle nového cíle
- [x] Implementovat deterministický M0/oracle evaluator pro Q1/Q2/Q4/Q8/FP32
- [x] Ověřit evaluator smoke testem na 10k `full_baseline` checkpointu
- [x] Ověřit reprodukovatelný M0 inference běh na `m1-256`, `m1-512` a `m4-air`
- [ ] Změřit kvalitu současného mask-diffusion generování a uložit ukázky
- [ ] Oddělit metriky difuzních kroků, hloubky modelu a stupňů přesnosti

### 1A. Layer-wise grouped-precision prototyp
- [x] Oddělit nový 25vrstvý prototyp od legacy `DiffusionLM`
- [x] Implementovat schedule 5× Q1/Q2/Q4/Q8/FP16 a skutečnou FP16 matmul cestu
- [x] Přidat sdílený LM head po každé vrstvě od vrstvy 5
- [x] Přidat masked deep supervision pro všechny mezivýstupy
- [x] Umožnit sequence-wide early exit uvnitř precision skupiny
- [x] Ověřit první tři rozdílné smoke testy na třech nodech
- [x] Napojit prototyp na reálný tokenizer a 69M-tokenový Wikipedia cache
- [x] Spustit 5 000krokový FP32 a progressive pilot a změřit vrstvy 5/10/15/20/25
- [ ] Odstranit kolaps k nejčastějším tokenům; hlubší výstupy zatím nezlepšují kvalitu
- [x] Změřit constant-token baseline: nový řádek = 3,743 % validation accuracy
- [x] Potvrdit mask sweepem, že 5k FP32 checkpoint přesně kopíruje tuto baseline
- [x] Ověřit schopnost učení na jedné sekvenci: 59,7 % po 1 000 expozicích
- [x] Dokončit one-sequence overfit: 100 % masked accuracy, loss 0,000718
- [ ] Zopakovat 95% gate na 100 sekvencích s dostatečným počtem expozic
- [ ] Přidat mask-rate curriculum 15 % → 30 % → 50 % → 75 % → 100 %
- [ ] Spustit zbývajících pět kontrol až po úspěšném jazykovém quality gate
- [ ] Kalibrovat early-exit práh na validačních datech bez leakage

### 2. Oracle analýza adaptivní přesnosti
- [x] Připravit průběžné vyhodnocení stejných vstupů v Q1/Q2/Q4/Q8/FP32 bez ukládání plných logitů
- [x] Změřit opravené i nově zavedené chyby mezi stupni přesnosti na 3node pilotu
- [x] Přidat provenance-safe agregaci distribuovaných M0 běhů
- [x] Implementovat skutečný FP16 stupeň v odděleném layer-wise prototypu; legacy interní `bits=16` zůstává FP32 identity cesta
- [x] Vytvořit offline Pareto křivku kvalita versus proxy výpočet s kalibračním a held-out rozdělením
- [x] Ověřit prediktivní hodnotu confidence, entropy, top-1/top-2 marginu a stability top-1
- [x] Doplnit přímé precision baseline a párový cluster bootstrap bez held-out policy selection

### 3. M1 — minimální adaptivní inference
- [ ] Implementovat rozhodnutí `commit / defer / escalate` po tokenových pozicích
- [ ] Porovnat fixed schedule, pravidlový adaptive schedule a oracle
- [ ] Logovat dosaženou přesnost a počet difuzních kroků pro každý token
- [ ] Pokračovat k multi-precision tréninku pouze při pozitivní oracle analýze

### 4. PTQ studie (historická větev, checkpointy existují)
- [ ] Porovnat Direct/Naive PTQ vs. native QAT (Q1, Q2, Q3, Q4)
- [ ] Spustit `python scripts/ptq_study.py` na checkpointech z Phase 1
- [ ] Volitelně: `--include-ternary` pro ternary variantu
- [ ] Pozor: native const_4bit používá staré schéma (15 úrovní) → Q4 srovnání je přibližné

### 5. Více seedů pro Phase 2 (historická větev)
- [ ] Phase 2 má pouze 2 seedy (s201, s203) — seed sensitivity je vysoká
- [ ] Doporučeno: 5–10 seedů pro statisticky spolehlivé závěry
- [ ] s201 dává téměř schedule-invariantní výsledky → možná špatný seed

### 6. Delší trénink Phase 2 (historická větev)
- [ ] Phase 2 model trénován jen 2 000 kroků vs. 10 000 u ablationu
- [ ] Rozšíření na 5 000–10 000 kroků pro spravedlivé srovnání

---

## 🔭 Výzkumné hypotézy (střednědobé)

- [ ] **Native Q3 trénink** — žádný ablation protějšek zatím neexistuje
- [ ] **Kalibrované PTQ** (GPTQ-style, AWQ-style) jako samostatná studie
- [ ] **Binární dekompozice** — reprezentace FP32 maticí více Q1 matic
- [ ] **Adaptivní compute** — early-exit prahy trénovány, ne hardcoded
- [ ] **Škálování** — větší d_model, více vrstev; ověřit, zda trendy drží
- [ ] **Skutečné low-bit kernely** — packed integer weights, měření reálné paměti

---

## ⚠️ Otevřené otázky

- Proč `constant-2b` poráží `baseline-fp16` i progresivní scheduly? Je to regularizace, nebo artefakt krátkého tréninku?
- Proč seed s201 produkuje téměř identické výsledky pro všechny scheduly?
- Q4 srovnání — přetrénovat `const_4bit` ablation s novým schématem (16 úrovní bez nuly)?
- Early-exit model vždy skončí po 1 kroku při threshold ≤ 0,03 — závisí to na délce tréninku nebo je to vlastnost architektury?

---

## 🧹 Tech debt

- [ ] SSH config není nastaven (nody přístupné přes ZeroTier, ale bez `~/.ssh/config`)
- [ ] Hesla v Obsidian plain textu (`SSH-pristupy.md`) — přesunout do Keychain/1Password
- [ ] `configs/ptq/ptq_baseline_s{42,123,7}.json` — zkontrolovat, zda odpovídají novým kvantizačním schématům
- [ ] `results/full_progressive_1_2_4/` — ověřit, zda je zahrnuto v agregovaných výsledcích
