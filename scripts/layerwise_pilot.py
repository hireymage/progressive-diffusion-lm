#!/usr/bin/env python3
"""Offline-only pilot harness for the 25-layer layer-wise models.

It deliberately selects an existing verified seq256 cache and never calls the
network/data builder.  Both variants use the same masked-diffusion objective
and the same safe default milestone exits, making supervision comparable.
"""
from __future__ import annotations

import argparse, csv, json, shutil, sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import LayerwiseModelConfig
from src.data import load_tokenizer
from src.layerwise_model import LayerwiseProgressiveLM, masked_deep_supervision_loss

EXIT_LAYERS = (5, 10, 15, 20, 25)

def variant_schedule(name: str) -> list[str]:
    if name == "fp32": return ["fp32"] * 25
    if name == "progressive": return ["q1"]*5 + ["q2"]*5 + ["q4"]*5 + ["q8"]*5 + ["fp16"]*5
    raise ValueError(f"unknown variant {name}")

def select_cache(cache_dir: Path, seq_len: int = 256) -> tuple[np.ndarray, np.ndarray, dict, Path]:
    """Select the largest complete local cache of exactly the requested length."""
    candidates = []
    for meta_path in cache_dir.glob(f"meta_seq{seq_len}_*.json"):
        meta = json.loads(meta_path.read_text())
        suffix = meta_path.name.removeprefix(f"meta_seq{seq_len}_").removesuffix(".json")
        train, val = cache_dir / f"train_seq{seq_len}_{suffix}.npy", cache_dir / f"val_seq{seq_len}_{suffix}.npy"
        # A legacy filename-only cache is not sufficient for a reproducible
        # comparison.  Require the provenance/checksums written by the current
        # cache builder, but do not rehash 263 MiB on every pilot invocation.
        verified = all(meta.get(k) for k in ("tokenizer_sha256", "train_sha256", "val_sha256"))
        if train.exists() and val.exists() and verified and meta.get("n_train_chunks", 0) and meta.get("n_val_chunks", 0):
            candidates.append((meta.get("total_tokens", 0), train, val, meta, meta_path))
    if not candidates:
        raise FileNotFoundError(f"No complete local seq{seq_len} cache in {cache_dir}; network is intentionally disabled.")
    _, train, val, meta, meta_path = max(candidates, key=lambda item: item[0])
    return np.load(train, mmap_mode="r"), np.load(val, mmap_mode="r"), meta, meta_path

def make_batch(data: np.ndarray, rng: np.random.RandomState, batch_size: int) -> np.ndarray:
    return np.asarray(data[rng.randint(0, len(data), size=batch_size)], dtype=np.int32)

def corrupt(batch: np.ndarray, mask_id: int, rng: np.random.RandomState) -> tuple[mx.array, mx.array, mx.array]:
    rates = rng.uniform(.1, 1., size=(len(batch), 1))
    mask = rng.random_sample(batch.shape) < rates
    # Never allow an empty objective in a minibatch.
    mask[:, 0] = True
    return mx.array(np.where(mask, mask_id, batch), dtype=mx.int32), mx.array(batch, dtype=mx.int32), mx.array(mask)

def evaluate(model, val, rng, batch_size, steps, mask_id, exits):
    was_training = model.training; model.eval()
    totals = {e: [0., 0., 0.] for e in exits} # loss sum, correct, masked
    try:
        for _ in range(steps):
            x, targets, mask = corrupt(make_batch(val, rng, batch_size), mask_id, rng)
            outputs = model.forward_intermediates(x, exit_layer=max(exits), requested_layers=exits)
            flat_mask = mask.reshape(-1).astype(mx.float32); n = mx.maximum(mx.sum(flat_mask), mx.array(1.))
            for e in exits:
                logits = outputs[e].reshape(-1, model.cfg.vocab_size).astype(mx.float32)
                truth = targets.reshape(-1)
                loss = mx.sum(-nn.log_softmax(logits, axis=-1)[mx.arange(logits.shape[0]), truth] * flat_mask) / n
                correct = mx.sum((mx.argmax(logits, axis=-1) == truth) * flat_mask)
                mx.eval(loss, correct, n)
                totals[e][0] += float(loss); totals[e][1] += float(correct); totals[e][2] += float(n)
    finally: model.train(was_training)
    return {str(e): {"loss": v[0]/steps, "accuracy": v[1]/max(v[2], 1), "perplexity": float(np.exp(v[0]/steps)), "proxy_cost": model.proxy_cost(e)} for e,v in totals.items()}

def disk_guard(output: Path, required_free_gb: float, estimated_checkpoint_bytes: int):
    free = shutil.disk_usage(output.parent).free
    required = int(required_free_gb * 1024**3) + 2 * estimated_checkpoint_bytes
    if free < required:
        raise RuntimeError(f"disk budget guard: {free/1024**3:.1f} GiB free, need {required/1024**3:.1f} GiB (reserve + latest/best)")

def save_checkpoint(model, optimizer, directory: Path, kind: str, step: int, val_loss: float):
    directory.mkdir(parents=True, exist_ok=True); path = directory / f"{kind}.npz"
    payload = dict(mlx.utils.tree_flatten(model.parameters()))
    payload.update({"opt_" + k: v for k,v in mlx.utils.tree_flatten(optimizer.state)})
    mx.savez(str(path), **payload)
    (directory / f"{kind}.json").write_text(json.dumps({"step": step, "val_loss": val_loss}, indent=2))
    return path

def load_checkpoint(model, optimizer, path: Path) -> int:
    data = mx.load(str(path)); model.load_weights([(k,v) for k,v in data.items() if not k.startswith("opt_")])
    opt = [(k.removeprefix("opt_"),v) for k,v in data.items() if k.startswith("opt_")]
    if not opt: raise ValueError("checkpoint has no optimizer state")
    optimizer.state = mlx.utils.tree_unflatten(opt); mx.eval(model.parameters(), optimizer.state)
    return int(json.loads(path.with_suffix(".json").read_text())["step"])

def examples(model, tokenizer, val, mask_id, seed):
    rng=np.random.RandomState(seed); targets=make_batch(val,rng,1); mask=rng.random_sample(targets.shape)<.25; mask[:,0]=True
    x=mx.array(np.where(mask,mask_id,targets), dtype=mx.int32); logits=model(x, exit_layer=25); pred=np.array(mx.argmax(logits,axis=-1))
    # Four deterministic confidence-based refinement steps: every iteration
    # replaces one quarter of the still-masked positions with its argmax.
    generated = np.full(targets.shape, mask_id, dtype=np.int32)
    for _ in range(4):
        logits = model(mx.array(generated, dtype=mx.int32), exit_layer=25)
        probs = np.array(mx.max(nn.softmax(logits.astype(mx.float32), axis=-1), axis=-1))
        preds = np.array(mx.argmax(logits, axis=-1))
        for row in range(len(generated)):
            remaining = np.where(generated[row] == mask_id)[0]
            if len(remaining):
                take = remaining[np.argsort(probs[row, remaining])[-max(1, int(np.ceil(len(remaining)/4))):]]
                generated[row, take] = preds[row, take]
    def dec(ids): return tokenizer.decode([int(i) for i in ids if int(i) < model.cfg.vocab_size])
    return {"target":dec(targets[0]), "masked_reconstruction":dec(np.where(mask[0],pred[0],targets[0])), "all_mask_4step_refinement":dec(generated[0])}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--mode", choices=("benchmark","train","eval"), required=True); p.add_argument("--variant", choices=("fp32","progressive"), required=True)
    p.add_argument("--cache-dir", type=Path, default=ROOT/"data/cache"); p.add_argument("--tokenizer", type=Path, default=ROOT/"tokenizer/wiki_bpe"); p.add_argument("--output", type=Path, default=ROOT/"results/layerwise_pilot")
    p.add_argument("--steps", type=int, default=100); p.add_argument("--batch-size", type=int, default=1); p.add_argument("--eval-steps", type=int, default=4); p.add_argument("--eval-every", type=int, default=25); p.add_argument("--seed", type=int, default=20260804); p.add_argument("--d-model", type=int, default=256); p.add_argument("--d-ff", type=int, default=1024); p.add_argument("--n-heads", type=int, default=8); p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--min-free-gb", type=float, default=10.0); p.add_argument("--resume", action="store_true"); p.add_argument("--checkpoint", choices=("best","latest"), default="best"); p.add_argument("--supervision", choices=("all","milestones","final"), default="milestones")
    a=p.parse_args(); train,val,meta,meta_path=select_cache(a.cache_dir); tokenizer=load_tokenizer(str(a.tokenizer)); vocab=tokenizer.get_vocab_size()
    cfg=LayerwiseModelConfig(vocab_size=vocab,d_model=a.d_model,d_ff=a.d_ff,n_heads=a.n_heads,max_seq_len=256,layer_precisions=variant_schedule(a.variant))
    mx.random.seed(a.seed); np.random.seed(a.seed); model=LayerwiseProgressiveLM(cfg); mx.eval(model.parameters()); optimizer=optim.AdamW(learning_rate=a.lr); out=a.output/a.variant; ckpt=out/"checkpoints"
    # Create only the requested result root; checkpoint files are still
    # guarded before training starts.
    out.mkdir(parents=True, exist_ok=True)
    estimated=sum(np.asarray(x).nbytes for _,x in mlx.utils.tree_flatten(model.parameters())) * 4
    disk_guard(out, a.min_free_gb, estimated)
    exits=tuple(range(5,26)) if a.supervision=="all" else (EXIT_LAYERS if a.supervision=="milestones" else (25,)); rng=np.random.RandomState(a.seed); start=0
    if a.resume and (ckpt/"latest.npz").exists(): start=load_checkpoint(model,optimizer,ckpt/"latest.npz")
    if a.mode == "eval":
        selected = ckpt / f"{a.checkpoint}.npz"
        if not selected.exists(): raise FileNotFoundError(f"eval requires {selected}; train first or pass --output for an existing run")
        load_checkpoint(model, optimizer, selected)
        metrics=evaluate(model,val,rng,a.batch_size,a.eval_steps,cfg.mask_token_id(),EXIT_LAYERS); out.mkdir(parents=True,exist_ok=True); (out/"eval.json").write_text(json.dumps({"cache":str(meta_path),"metrics":metrics,"examples":examples(model,tokenizer,val,cfg.mask_token_id(),a.seed)},indent=2)); print(json.dumps(metrics,indent=2)); return
    if a.mode == "benchmark": a.steps=min(a.steps,3)
    # Preserve the true best across resumed invocations; otherwise a worse
    # first resumed evaluation could overwrite the best checkpoint.
    best_meta = ckpt / "best.json"
    best = float(json.loads(best_meta.read_text())["val_loss"]) if best_meta.exists() else float("inf")
    csv_path=out/"train_metrics.csv"; new=not csv_path.exists() or start==0
    with csv_path.open("a",newline="") as f:
      writer=csv.DictWriter(f,fieldnames=["step","loss","elapsed_s"]); 
      if new: writer.writeheader()
      grad_fn=nn.value_and_grad(model, lambda m,x,t,mask: masked_deep_supervision_loss(m,x,t,mask,supervised_layers=exits))
      began=time.time()
      for step in range(start+1,a.steps+1):
        x,t,m=corrupt(make_batch(train,rng,a.batch_size),cfg.mask_token_id(),rng); loss,grads=grad_fn(model,x,t,m); optimizer.update(model,grads); mx.eval(loss,model.parameters()); writer.writerow({"step":step,"loss":float(loss),"elapsed_s":time.time()-began}); f.flush()
        if step % a.eval_every == 0 or step == a.steps:
          metrics=evaluate(model,val,rng,a.batch_size,a.eval_steps,cfg.mask_token_id(),EXIT_LAYERS); value=metrics["25"]["loss"]; save_checkpoint(model,optimizer,ckpt,"latest",step,value)
          if value < best: best=value; save_checkpoint(model,optimizer,ckpt,"best",step,value)
    summary={"variant":a.variant,"mode":a.mode,"seed":a.seed,"supervised_exits":list(exits),"cache_metadata":meta,"cache_meta_path":str(meta_path),"metrics":metrics,"examples":examples(model,tokenizer,val,cfg.mask_token_id(),a.seed),"checkpoint_policy":"latest+best"}; (out/"summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__ == "__main__": main()
