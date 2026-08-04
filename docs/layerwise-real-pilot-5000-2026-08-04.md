# Reálný layer-wise pilot: FP32 versus progressive

Datum: 2026-08-04  
Zdrojový commit: `223e71f`

## Verdikt

Trénovací pipeline funguje, ale jazyková kvalita ještě neprošla. Oba modely
snížily loss z přibližně 10 pod 7,5, jejich volné generování se však zhroutilo
k nejčastějším tokenům. Zbývajících pět kontrolních modelů se proto zatím
nespustilo.

## Nastavení

- anglická Wikipedia, 69 033 984 tokenů;
- slovník 16 000 tokenů, délka sekvence 256;
- 25 vrstev, `d_model=256`, `d_ff=1024`, 8 attention heads;
- 5 000 kroků, batch size 1, seed 20260804;
- stejná deep supervision na vrstvách 5, 10, 15, 20 a 25;
- vyhodnocení nejlepšího checkpointu na 64 validačních batchech.

## Výsledky

| Varianta a výstup | Loss | Accuracy | Proxy cost |
|---|---:|---:|---:|
| FP32, vrstva 5 | 7,4274 | 4,46 % | 160 |
| FP32, vrstva 10 | 7,4267 | 4,66 % | 320 |
| FP32, vrstva 25 | 7,4349 | 4,32 % | 800 |
| Progressive, vrstva 5 (Q1) | **7,3986** | **5,19 %** | **5** |
| Progressive, vrstva 10 (Q2) | 7,4059 | 4,62 % | 15 |
| Progressive, vrstva 25 (FP16) | 7,4094 | 4,40 % | 155 |

Nejlepší checkpoint obou variant vznikl na kroku 3 500. FP32 běh na m1-512
trval 841 sekund, progresivní QAT simulace na m4-air 642 sekund. Tyto časy
nelze použít jako hardwarové srovnání, protože běžely na různých strojích a
low-bit operace jsou zatím simulované.

## Interpretace

Progresivní model v pilotu neztratil proti FP32, což opravňuje pokračovat ve
vývoji. Současně ale hlubší výstupy nepřidávají kvalitu a oba modely při
generování preferují mezery, nové řádky a velmi častá slova. To znamená, že
aktuální loss a generátor ještě neprokazují funkční jazykový model.

Než spustíme Q1, Q2, Q4, Q8 a FP16 kontroly, je potřeba diagnostikovat zejména:

1. distribuci tokenů a přesnost vůči baseline „nejčastější token“;
2. learning-rate schedule, počet efektivně zpracovaných tokenů a batch size;
3. váhy pomocných lossů, aby rané výstupy nebránily učení hlubších vrstev;
4. kvalitu rekonstrukce podle mask rate a skutečný postup difuzního odhalování.

Strojově čitelný souhrn je v `results/layerwise/real_pilot_5000/summary.json`.
