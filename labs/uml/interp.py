"""The interpretability toolkit, implemented on the tiny model itself.

Nothing here is a black box: each function returns the tensor it measured, so a
claim in the book can be traced to a number produced by shipped code.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .model import TinyGPT
from .data import induction_prompt


@torch.no_grad()
def attention(model, idx):
    """Return attention patterns shaped [layer][head][query][key]."""
    store = {}
    model(idx, store=store)
    return [store[(l, "attn")][0] for l in range(model.cfg.n_layers)]


@torch.no_grad()
def head_table(model, prompt, target_pos=None):
    """Five attention statistics per head, measured on a single prompt.

    prev   attention to the immediately preceding position
    self   attention to the query's own position
    bos    attention to the first token
    same   attention to an earlier occurrence of the query's token
    succ   attention to the token *after* that earlier occurrence: the copy target
    """
    att = attention(model, prompt)
    T = prompt.shape[1]
    toks = prompt[0].tolist()
    rows = []
    for l, A in enumerate(att):
        for h in range(A.shape[0]):
            M = A[h]
            same_vals, succ_vals = [], []
            for i in range(1, T):
                prev = [j for j in range(i) if toks[j] == toks[i]]
                if prev:
                    j = prev[-1]
                    same_vals.append(M[i, j].item())
                    if j + 1 < T:
                        succ_vals.append(M[i, j + 1].item())
            row = {
                "layer": l, "head": h,
                "prev": M[1:, :-1].diagonal(offset=-1).mean().item(),
                "self": M.diagonal().mean().item(),
                "bos": M[:, 0].mean().item(),
                "same": sum(same_vals) / max(1, len(same_vals)),
                "succ": sum(succ_vals) / max(1, len(succ_vals)),
            }
            if target_pos is not None:
                row["final"] = M[-1, target_pos].item()
            rows.append(row)
    return rows


@torch.no_grad()
def head_stats(model, prompt, target_pos):
    """Prev-token and induction scores for every head on one prompt.

    prev   mean attention to the immediately preceding position
    ind    attention from the final query to the token after the first
           occurrence of the final token (the position induction must copy)
    """
    att = attention(model, prompt)
    rows = []
    T = prompt.shape[1]
    for l, A in enumerate(att):
        for h in range(A.shape[0]):
            M = A[h]
            prev = M[1:, :-1].diagonal(offset=-1).mean().item()
            ind = M[-1, target_pos].item()
            rows.append({"layer": l, "head": h, "prev": prev, "induction": ind})
    return rows


@torch.no_grad()
def induction_scores(model, tokens, n_prompts=32, seed=0):
    """Average the final-query induction attention over random repetitions."""
    g = torch.Generator().manual_seed(seed)
    totals = None
    for _ in range(n_prompts):
        t = torch.randint(1, model.cfg.vocab_size, (tokens,), generator=g)
        prompt = induction_prompt(t.tolist())
        # final token is t[-2]; its first occurrence is at position tokens-1,
        # so the copied next token sits at position tokens
        target = tokens
        rows = head_stats(model, prompt, target)
        vals = torch.tensor([[r["induction"] for r in rows if r["layer"] == l]
                             for l in range(model.cfg.n_layers)])
        totals = vals if totals is None else totals + vals
    return totals / n_prompts


@contextmanager
def ablate_head(model, layer, head):
    """Remove one head's write to the residual stream, then restore it."""
    W = model.blocks[layer].attn.Wo.weight
    dh = model.cfg.d_head
    saved = W[:, head * dh : (head + 1) * dh].clone()
    with torch.no_grad():
        W[:, head * dh : (head + 1) * dh] = 0
    try:
        yield
    finally:
        with torch.no_grad():
            W[:, head * dh : (head + 1) * dh] = saved


@torch.no_grad()
def loss_on(model, x, y):
    logits, _ = model(x)
    return F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100).item()


@torch.no_grad()
def logit_lens(model, idx, topk=5):
    """Decode the residual stream after every block with the final unembedding."""
    B, T = idx.shape
    pos = torch.arange(T, device=idx.device)
    x = model.tok(idx) + model.pos(pos)
    rows = []
    for i, block in enumerate(model.blocks):
        x = block(x)
        logits = model.unembed(model.ln_f(x))
        probs = logits[0, -1].softmax(-1)
        top = probs.topk(topk)
        rows.append({"layer": i + 1,
                     "tokens": top.indices.tolist(),
                     "probs": [round(v, 4) for v in top.values.tolist()]})
    return rows


@torch.no_grad()
def residual_cache(model, idx):
    """Residual stream after embedding and after each block (pre-final-LN)."""
    B, T = idx.shape
    x = model.tok(idx) + model.pos(torch.arange(T, device=idx.device))
    out = [x[0].clone()]
    for block in model.blocks:
        x = block(x)
        out.append(x[0].clone())
    return out


@torch.no_grad()
def patch_residual(model, clean_idx, corrupt_idx, layer, positions, answer):
    """Replace corrupt residuals at `positions` with clean ones.

    Returns the logit difference (answer minus the other candidate) before and
    after the patch, plus the fraction of the gap that was restored.
    """
    clean = residual_cache(model, clean_idx)
    corrupt = residual_cache(model, corrupt_idx)

    def run(x, replacement=None):
        B, T = x.shape
        h = model.tok(x) + model.pos(torch.arange(T, device=x.device))
        for i, block in enumerate(model.blocks):
            if replacement is not None and i == layer:
                h = h.clone()
                for p in positions:
                    h[:, p] = replacement[p]
            h = block(h)
        return model.unembed(model.ln_f(h))

    def diff(logits):
        return (logits[0, -1, answer[0]] - logits[0, -1, answer[1]]).item()

    before = diff(run(corrupt_idx))
    after = diff(run(corrupt_idx, clean[layer]))
    full = diff(run(clean_idx))
    restored = (after - before) / (full - before) if abs(full - before) > 1e-9 else 0.0
    return {"before": round(before, 3), "after": round(after, 3),
            "clean": round(full, 3), "restored": round(restored, 3)}


def head_table_many(model, prompts, target_pos=None):
    """Average head_table over several prompts of the same length."""
    rows = None
    for p in prompts:
        r = head_table(model, p, target_pos)
        rows = r if rows is None else [{k: a[k] + b[k] for k in a} for a, b in zip(rows, r)]
    return [{k: round(v / len(prompts), 4) for k, v in row.items()} for row in rows]


def train_probe(X, y, steps=400, lr=0.05, weight_decay=1e-3):
    """A linear probe trained with full-batch gradient descent."""
    X = X.float()
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    n_classes = int(y.max().item()) + 1
    W = torch.zeros(X.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        logits = X @ W + b
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = ((X @ W + b).argmax(-1) == y).float().mean().item()
    return round(acc, 4), W.detach(), b.detach()


@torch.no_grad()
def collect_resid(model, prompts, layer):
    """Mean residual over positions for each prompt, at one layer."""
    out = []
    for idx in prompts:
        out.append(residual_cache(model, idx)[layer].mean(0))
    return torch.stack(out)
