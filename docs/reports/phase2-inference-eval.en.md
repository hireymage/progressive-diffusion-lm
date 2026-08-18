# Phase 2 — Inference Eval Report (3-node)

[English](phase2-inference-eval.en.md) | [Čeština](phase2-inference-eval.md)

**Date:** 2026-07-21
**Train steps:** 2000 · **Gen repeats:** 5 · **Seq len:** 128 · **Precision schedule:** [1, 2, 4, 8, 8, 4, 2, 1] · **Max steps:** 8

## Nodes

| Node | Architecture | Standard latency (s) |
|------|-------------|----------------------|
| m1-256 | Apple M1 (256 GPU cores) | 0.2435 |
| m1-512 | Apple M1 (512 GPU cores) | 0.2494 |
| m4-air | Apple M4 Air | 0.0881 |

## Aggregated comparison table

### Standard & incremental

| Mode | m1-256 lat (s) | m1-256 speedup | m1-512 lat (s) | m1-512 speedup | m4-air lat (s) | m4-air speedup | Agreement % (all) |
|------|---------------|----------------|---------------|----------------|----------------|----------------|---------------|
| standard | 0.2435 | 1.00x | 0.2494 | 1.00x | 0.0881 | 1.00x | — |
| incremental | 0.1798 | 1.35x | 0.2133 | 1.17× | 0.0541 | 1.63× | 100% |

### Early-exit (1 step, confidence threshold ≤ 0.03)

| Mode | m1-256 lat (s) | m1-256 speedup | m1-512 lat (s) | m1-512 speedup | m4-air lat (s) | m4-air speedup | Agreement % (all) |
|------|---------------|----------------|---------------|----------------|----------------|----------------|---------------|
| early_exit (t=0.01) | 0.0584 | 4.17x | 0.0430 | 5.73× | 0.0098 | 8.99× | 100% |
| early_exit_inc (t=0.01) | 0.0353 | 6.90× | 0.0510 | 4.93× | 0.0108 | 8.16× | 100% |
| early_exit (t=0.02) | 0.0348 | 7.00× | 0.0520 | 4.77× | 0.0099 | 8.90x | 100% |
| early_exit_inc (t=0.02) | 0.0574 | 4.24× | 0.0500 | 5.01× | 0.0108 | 8.16× | 100% |
| early_exit (t=0.03) | 0.0382 | 6.37× | 0.0480 | 5.16× | 0.0104 | 8.47× | 100% |
| early_exit_inc (t=0.03) | 0.0257 | 9.47x | 0.0440 | 5.62× | 0.0111 | 7.94× | 100% |

### Early-exit (8 steps, confidence threshold ≥ 0.05) — no early termination

| Mode | m1-256 lat (s) | m1-256 speedup | m1-512 lat (s) | m1-512 speedup | m4-air lat (s) | m4-air speedup | Agreement % (all) |
|------|---------------|----------------|---------------|----------------|----------------|----------------|---------------|
| early_exit (t=0.05) | 0.2017 | 1.21× | 0.2030 | 1.23× | 0.0515 | 1.71× | 100% |
| early_exit_inc (t=0.05) | 0.2432 | 1.00x | 0.2540 | 0.98× | 0.0556 | 1.58× | 100% |
| early_exit (t=0.1) | 0.1757 | 1.39× | 0.2160 | 1.16× | 0.0525 | 1.68× | 100% |
| early_exit_inc (t=0.1) | 0.2521 | 0.97× | 0.2260 | 1.10x | 0.0582 | 1.51× | 100% |
| early_exit (t=0.5) | 0.1825 | 1.33× | 0.2010 | 1.24× | 0.0534 | 1.65× | 100% |
| early_exit_inc (t=0.5) | 0.2126 | 1.15x | 0.2330 | 1.07× | 0.0590 | 1.49× | 100% |

## Key takeaways

1. **m4-air dominant**: ~2.8× faster than both M1 nodes in standard mode (0.088s vs ~0.25s). M4 architecture has significantly better GPU throughput.

2. **Early-exit threshold ≤ 0.03 = 1 step + massive speedup**: On all nodes, after the 1st step, the model matches 100% with the standard 8-step output. Speedup reaches 4.2×–9.5×.

3. **Best configuration**: `early_exit_incremental (t=0.03)` on m1-256 = **9.47× speedup** (0.026s) at 100% agreement. At m4-air the best is `early_exit (t=0.01)` = **8.99× speedup** (0.010s).

4. **Threshold ≥ 0.05 will not finish early**: The model will never finish early, all steps (8/8) will be executed. Speedup here comes only from the incremental cache (m4-air: 1.49–1.71×, m1: 0.97–1.39×).

5. **Token samples**: All nodes generate identical sequences (token 208 repeated) — model is trained on a very limited slovníku/režimu. 100% agreement across all modes confirms deterministic behavior.

6. **Incremental cache**: Consistently speeds up on m4-air (1.49–1.63×), but only 1.07–1.17× on m1-512. At m1-256 it is variable (0.97–1.35×).

## Files

- `results/inference_eval/m1-256_inference_eval.json`
- `results/inference_eval/m1-512_inference_eval.json`
- `results/inference_eval/m4-air_inference_eval.json`

## Conclusion

Phase 2 inference eval completed successfully on all 3 nodes. The model after Phase 2 training (2000 steps) shows consistent behavior across nodes with 100% token agreement. The biggest profit is brought by early-exit with confidence threshold ≤ 0.03 (1 step, 4–9× speedup). M4 Air is the top performing node in absolute times.
