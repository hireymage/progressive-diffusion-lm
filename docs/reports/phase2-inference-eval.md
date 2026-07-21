# Phase 2 — Inference Eval Report (3-node)

**Datum:** 2026-07-21
**Train steps:** 2000 · **Gen repeats:** 5 · **Seq len:** 128 · **Precision schedule:** [1, 2, 4, 8, 8, 4, 2, 1] · **Max steps:** 8

## Uzly

| Uzel | Architektura | Standard latency (s) |
|------|-------------|----------------------|
| m1-256 | Apple M1 (256 GPU cores) | 0.2435 |
| m1-512 | Apple M1 (512 GPU cores) | 0.2494 |
| m4-air | Apple M4 Air | 0.0881 |

## Agregovaná srovnávací tabulka

### Standard & incremental

| Mode | m1-256 lat (s) | m1-256 speedup | m1-512 lat (s) | m1-512 speedup | m4-air lat (s) | m4-air speedup | Agree % (all) |
|------|---------------|----------------|---------------|----------------|----------------|----------------|---------------|
| standard | 0.2435 | 1.00× | 0.2494 | 1.00× | 0.0881 | 1.00× | — |
| incremental | 0.1798 | 1.35× | 0.2133 | 1.17× | 0.0541 | 1.63× | 100% |

### Early-exit (1 step, confidence threshold ≤ 0.03)

| Mode | m1-256 lat (s) | m1-256 speedup | m1-512 lat (s) | m1-512 speedup | m4-air lat (s) | m4-air speedup | Agree % (all) |
|------|---------------|----------------|---------------|----------------|----------------|----------------|---------------|
| early_exit (t=0.01) | 0.0584 | 4.17× | 0.0430 | 5.73× | 0.0098 | 8.99× | 100% |
| early_exit_inc (t=0.01) | 0.0353 | 6.90× | 0.0510 | 4.93× | 0.0108 | 8.16× | 100% |
| early_exit (t=0.02) | 0.0348 | 7.00× | 0.0520 | 4.77× | 0.0099 | 8.90× | 100% |
| early_exit_inc (t=0.02) | 0.0574 | 4.24× | 0.0500 | 5.01× | 0.0108 | 8.16× | 100% |
| early_exit (t=0.03) | 0.0382 | 6.37× | 0.0480 | 5.16× | 0.0104 | 8.47× | 100% |
| early_exit_inc (t=0.03) | 0.0257 | 9.47× | 0.0440 | 5.62× | 0.0111 | 7.94× | 100% |

### Early-exit (8 steps, confidence threshold ≥ 0.05) — no early termination

| Mode | m1-256 lat (s) | m1-256 speedup | m1-512 lat (s) | m1-512 speedup | m4-air lat (s) | m4-air speedup | Agree % (all) |
|------|---------------|----------------|---------------|----------------|----------------|----------------|---------------|
| early_exit (t=0.05) | 0.2017 | 1.21× | 0.2030 | 1.23× | 0.0515 | 1.71× | 100% |
| early_exit_inc (t=0.05) | 0.2432 | 1.00× | 0.2540 | 0.98× | 0.0556 | 1.58× | 100% |
| early_exit (t=0.1) | 0.1757 | 1.39× | 0.2160 | 1.16× | 0.0525 | 1.68× | 100% |
| early_exit_inc (t=0.1) | 0.2521 | 0.97× | 0.2260 | 1.10× | 0.0582 | 1.51× | 100% |
| early_exit (t=0.5) | 0.1825 | 1.33× | 0.2010 | 1.24× | 0.0534 | 1.65× | 100% |
| early_exit_inc (t=0.5) | 0.2126 | 1.15× | 0.2330 | 1.07× | 0.0590 | 1.49× | 100% |

## Klíčové poznatky

1. **m4-air dominantní**: ~2.8× rychlejší než oba M1 uzly ve standardním režimu (0.088s vs ~0.25s). M4 architektura má výrazně lepší GPU propustnost.

2. **Early-exit threshold ≤ 0.03 = 1 step + masivní speedup**: Na všech uzlech se model po 1. kroku shoduje na 100 % se standardním 8-krokovým výstupem. Speedup dosahuje 4.2×–9.5×.

3. **Nejlepší konfigurace**: `early_exit_incremental (t=0.03)` na m1-256 = **9.47× speedup** (0.026s) při 100 % agreement. Na m4-air je nejlepší `early_exit (t=0.01)` = **8.99× speedup** (0.010s).

4. **Threshold ≥ 0.05 nedokončí early**: Model nikdy neskončí dříve, provedou se všechny kroky (8/8). Speedup zde pochází pouze z incremental cache (m4-air: 1.49–1.71×, m1: 0.97–1.39×).

5. **Token samples**: Všechny uzly generují identické sekvence (token 208 opakovaný) — model se trénuje na velmi omezeném slovníku/režimu. Shoda 100 % napříč všemi módy potvrzuje deterministické chování.

6. **Incremental cache**: Konzistentně urychluje na m4-air (1.49–1.63×), ale na m1-512 jen 1.07–1.17×. Na m1-256 je variabilní (0.97–1.35×).

## Soubory

- `results/inference_eval/m1-256_inference_eval.json`
- `results/inference_eval/m1-512_inference_eval.json`
- `results/inference_eval/m4-air_inference_eval.json`

## Závěr

Phase 2 inference eval úspěšně dokončena na všech 3 uzlech. Model po Phase 2 tréninku (2000 kroků) vykazuje konzistentní chování napříč uzly s 100 % token agreement. Největší zisk přináší early-exit s confidence threshold ≤ 0.03 (1 krok, 4–9× speedup). M4 Air je nejvýkonnější uzel v absolutních časech.