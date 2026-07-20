# Dual-M1 long campaign harness

Status: **historická dual-M1 campaign dokončena; P1 matched-noise campaign implementována, ověřena a spuštěna na obou nodech 2026-07-19**.

## Safety model

- Accepted node IDs are only `m1-256` and `m1-512`; each campaign config is bound to one node.
- The harness rejects a runtime root inside the synchronized project/iCloud tree and rejects any shared results root other than `<source>/results`.
- Every invocation creates a collision-resistant run ID of the form `YYYYMMDD-HHMMSS_<campaign>_s<seeds>_<nonce>`.
- Runtime source snapshot, dataset cache, MLX cache, logs, generated configs and checkpoints remain under the node-local root outside iCloud.
- `max_parallel_tasks` must be `1`. Workloads execute sequentially to fit the two 8GB unified-memory M1 nodes.
- Completion state and per-task exit codes are written atomically. `--resume --run-id ...` skips only tasks already marked completed with exit code 0; a failed/current task is retried. Metrics-only training tasks restart that task from step 0, while already completed tasks are preserved.
- Only a completed campaign is bundled. Publication to `results/<node>/<run-id>` is serialized with the shared `.dual-m1-publish.lock`, staged, atomically renamed, and made read-only. Existing destinations are never overwritten; resuming an already verified published run is a successful no-op.
- `--dry-run` is read-only and does not snapshot source, create runtime directories, import MLX, train, evaluate, or publish.

Each immutable result bundle contains the campaign manifest, environment/Git HEAD/RAM record, state with per-task exit codes, generated configs, full task logs, concise log tails, artifact inventory and selected metrics artifacts. Large local caches and checkpoints are not published.

## Workload split

### `m1-256`: PTQ plus missing native counterparts

Config: `configs/campaign/m1-256-ptq.json`

Sequential tasks:

1. Existing PTQ study: train missing FP32 checkpoints for seeds 42/123/7, then direct/naive PTQ evaluation at Q1/Q2/Q3/Q4/FP32 plus optional ternary.
2. Native true-Q3 10k run, seed 31415.
3. Native ternary 10k run, seed 31415.

Rationale: PTQ is the principal missing campaign, while the two added native runs fill the current Q3 and ternary comparison gaps. Based on three measured 10k FP32 runs (~1.4 h each), PTQ evaluation overhead and two additional 10k runs, this should occupy roughly 8+ hours; actual M1 time must be measured.

### `m1-512`: paired two-seed replication

Config: `configs/campaign/m1-512-paired-replication.json`

At each new seed 31415 and 27182, sequentially run 10k steps for:

1. FP32 baseline;
2. constant binary Q1;
3. progressive Q1→Q2→Q4.

Six historical-duration runs (~1.4–1.6 h each) imply roughly 8.4–9.6 h. Two paired seeds are more informative than padding a one-seed comparison with unrelated work.

All six replication runs are metrics-only to limit disk use on 8GB/512GB hardware. Architecture, batch size 8, data and optimizer settings stay aligned with existing full configs. Sequential execution and node-local caches are the 8GB safeguards; no claim is made until live RAM and runtime are measured.

## Exact commands

Run from the synchronized project directory, using a **node-local Python environment outside iCloud** with `requirements.txt` installed. The Python interpreter that launches the harness is also used for all tasks.

Dry-run on M1-256:

```bash
python3 scripts/run_dual_m1_campaign.py \
  --node m1-256 \
  --campaign configs/campaign/m1-256-ptq.json \
  --dry-run
```

Dry-run on M1-512:

```bash
python3 scripts/run_dual_m1_campaign.py \
  --node m1-512 \
  --campaign configs/campaign/m1-512-paired-replication.json \
  --dry-run
```

Actual launch commands (documented only; **not run during implementation**):

```bash
python3 scripts/run_dual_m1_campaign.py \
  --node m1-256 \
  --campaign configs/campaign/m1-256-ptq.json
```

```bash
python3 scripts/run_dual_m1_campaign.py \
  --node m1-512 \
  --campaign configs/campaign/m1-512-paired-replication.json
```

After a failure, copy the exact run ID printed in the plan/state and resume on the same node:

```bash
python3 scripts/run_dual_m1_campaign.py \
  --node m1-256 \
  --campaign configs/campaign/m1-256-ptq.json \
  --resume \
  --run-id 'YYYYMMDD-HHMMSS_ptq-plus-native-gaps_s42-123-7-31415_NNNNNNNN'
```

The nodes do not require SSH or coordination with each other. They may compute concurrently because runtime writes are local; final shared publication is automatically serialized.

## Verification (no training)

```bash
python3 -m py_compile scripts/run_dual_m1_campaign.py tests/test_campaign_harness.py
python3 -m unittest tests.test_campaign_harness -v
python3 scripts/run_dual_m1_campaign.py --node m1-256 --campaign configs/campaign/m1-256-ptq.json --dry-run
python3 scripts/run_dual_m1_campaign.py --node m1-512 --campaign configs/campaign/m1-512-paired-replication.json --dry-run
```

## P1 matched-noise regularization campaign

Configs:

- `configs/campaign/m1-256-matched-noise.json` — seeds `11`, `29`;
- `configs/campaign/m1-512-matched-noise.json` — seeds `47`, `73`;
- `configs/campaign/m1-256-matched-noise-smoke.json` — 20-step operational smoke only.

Na každý fresh seed běží paired pořadí: clean FP32, constant Q1, Gaussian matched-RMS FP32, Uniform matched-RMS FP32. Každý node provede osm sekvenčních 10k tasků. `expected_metrics_contract` navíc k exit code ověřuje CSV/JSON schema, konečné hodnoty a shodu experiment/model/noise/seed.

Ověření 2026-07-19:

```text
py_compile: exit 0
P1 matched-noise + metrics: 17 passed
campaign harness: 12 passed
full suite: 109 passed
m1-256 dry-run: exit 0, 8 tasks
m1-512 dry-run: exit 0, 8 tasks
wrong-node guard: exit 2
```

Node-local smoke `20260719-054925_matched-noise-smoke_s911_98480c03` dokončil 20 kroků s exit 0 a semantic contractem. Skutečný injected Gaussian RMS odpovídal Q1 residual RMS přibližně na `0.0341`; všechny CSV/JSON hodnoty byly konečné. Tiny smoke loss není vědecký výsledek.

Launch na M1-256:

```bash
"$HOME/Library/Application Support/ML-Experiments/progressive-diffusion-lm/m1-256/env/bin/python" \
  scripts/run_dual_m1_campaign.py \
  --node m1-256 \
  --campaign configs/campaign/m1-256-matched-noise.json
```

Na M1-512 se používá ověřený node-local source snapshot mimo iCloud, node-local Python a oddělený runtime root:

```bash
REMOTE_SOURCE="/Users/hozzy/Library/Application Support/ML-Experiments/progressive-diffusion-lm/m1-512/<p1-source>"
REMOTE_RUNTIME="/Users/hozzy/Library/Application Support/ML-Experiments/progressive-diffusion-lm/runtime"
"/Users/hozzy/Library/Application Support/ML-Experiments/progressive-diffusion-lm/m1-512/env/bin/python" \
  "$REMOTE_SOURCE/scripts/run_dual_m1_campaign.py" \
  --node m1-512 \
  --campaign "$REMOTE_SOURCE/configs/campaign/m1-512-matched-noise.json" \
  --source "$REMOTE_SOURCE" \
  --local-root "$REMOTE_RUNTIME" \
  --shared-results-root "$REMOTE_SOURCE/results"
```
