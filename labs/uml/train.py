"""Training loops shared by the labs.  Small enough to read, seeded enough to trust."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import Config, TinyGPT


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


@torch.no_grad()
def accuracy(model, x, y, ignore_index=-100):
    model.eval()
    logits, _ = model(x)
    mask = y != ignore_index
    pred = logits.argmax(-1)
    return (pred[mask] == y[mask]).float().mean().item()


def train_lm(model, batches, steps=1500, lr=3e-3, warmup=50, weight_decay=0.01,
             eval_every=250, eval_batches=None, seed=0, log=True):
    """Train on an iterator of (x, y) batches, returning a loss history."""
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                            betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))
    )
    history, t0 = [], time.time()
    model.train()
    for step in range(steps):
        x, y = next(batches)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % eval_every == 0 or step == steps - 1:
            row = {"step": step, "loss": round(loss.item(), 4)}
            if eval_batches is not None:
                model.eval()
                with torch.no_grad():
                    vb = [next(eval_batches) for _ in range(8)]
                    vl = sum(F.cross_entropy(model(x)[0].flatten(0, 1), y.flatten(),
                                             ignore_index=-100).item() for x, y in vb) / len(vb)
                row["val_loss"] = round(vl, 4)
                model.train()
            history.append(row)
            if log:
                print(f"step {step:4d}  loss {row['loss']:.3f}"
                      + (f"  val {row['val_loss']:.3f}" if "val_loss" in row else ""))
    return {"history": history, "seconds": round(time.time() - t0, 1),
            "params": model.n_params()}


def save(model, path, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.cfg.to_dict(), "model": model.state_dict(),
                "extra": extra or {}}, path)


def fresh(cfg: Config, seed=0):
    set_seed(seed)
    return TinyGPT(cfg)
