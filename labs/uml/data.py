"""Synthetic tasks with known structure.

Every task here is chosen because the mechanism that solves it is small enough
to inspect.  No downloads, no tokenizer, no pretrained weights.
"""
from __future__ import annotations

import random

import torch


BOS = 0
PAD = 0


def induction_batch(batch=64, seq_len=12, vocab=24, generator=None):
    """A random string, then the same string: the model must continue it.

    Input   [BOS] t1 ... tL t1 ... tL
    Target  -100 for the first pass, then t2 ... tL, then BOS
    """
    g = generator or torch.Generator()
    t = torch.randint(1, vocab, (batch, seq_len), generator=g)
    x = torch.cat([torch.full((batch, 1), BOS, dtype=torch.long), t, t], dim=1)
    y = torch.full_like(x, -100)
    # positions L+1 ... 2L-1 predict the next token of the second copy
    y[:, seq_len + 1 : 2 * seq_len] = t[:, 1:]
    return x, y


def induction_gap_batch(batch=128, L=10, gap=5, vocab=16, generator=None):
    """A string, a random filler, then the string again.

    The gap makes a fixed positional offset useless: the model has to find the
    earlier occurrence of the current token and copy what followed it.
    """
    g = generator or torch.Generator()
    t = torch.randint(1, vocab, (batch, L), generator=g)
    f = torch.randint(1, vocab, (batch, gap), generator=g)
    x = torch.cat([torch.full((batch, 1), BOS, dtype=torch.long), t, f, t], dim=1)
    y = torch.full_like(x, -100)
    start = 1 + L + gap
    y[:, start : start + L - 1] = t[:, 1:]
    return x, y


def induction_gap_prompt(seq, gap_tokens, device="cpu"):
    """Prompt ending on the second-to-last repeated token; next is the last."""
    t = torch.tensor(seq, dtype=torch.long, device=device)
    f = torch.tensor(gap_tokens, dtype=torch.long, device=device)
    x = torch.cat([torch.tensor([BOS], device=device), t, f, t[:-1]])
    return x.unsqueeze(0)


def induction_prompt(seq, device="cpu"):
    """One prompt whose final logits should be the next repeated token."""
    t = torch.tensor(seq, dtype=torch.long, device=device)
    x = torch.cat([torch.tensor([BOS], device=device), t, t[:-1]])
    return x.unsqueeze(0)


def parity_batch(batch=64, length=12, cot=False, generator=None):
    """Parity of a bit string, with or without a running-scratchpad target.

    Direct:   BOS b1 ... bL =  ->  p
    CoT:      BOS b1 ... bL =  ->  p1 p2 ... pL
    The vocabulary is {BOS=0, '='=1, 0=2, 1=3, 0/1 same symbols}, so the
    scratchpad writes the running parity after every input bit.
    """
    g = generator or torch.Generator()
    bits = torch.randint(0, 2, (batch, length), generator=g)
    eq = torch.full((batch, 1), 1, dtype=torch.long)
    prompt = torch.cat([torch.full((batch, 1), BOS, dtype=torch.long), bits + 2, eq], dim=1)
    if cot:
        running = torch.cumsum(bits, dim=1) % 2
        x = torch.cat([prompt, running[:, :-1] + 2], dim=1)
        y = torch.cat([torch.full((batch, prompt.shape[1] - 1), -100, dtype=torch.long),
                       running + 2], dim=1)
    else:
        x = prompt
        y = torch.cat([torch.full((batch, prompt.shape[1] - 1), -100, dtype=torch.long),
                       (bits.sum(dim=1) % 2).unsqueeze(1) + 2], dim=1)
    return x, y


def parity_prompt(bits, cot=False, device="cpu"):
    x = torch.tensor([BOS] + [b + 2 for b in bits] + [1], dtype=torch.long, device=device)
    return x.unsqueeze(0)


REL_VOCAB = 12          # nodes 2..11; 0 BOS, 1 '='


def relation_batch(batch=64, hops=6, nodes=10, cot=False, generator=None):
    """Follow a chain of directed edges a>b from a start node.

    Input   BOS s | a>b b>c ... | =
    Direct  -> endpoint
    CoT     -> every intermediate node, then the endpoint twice
    """
    g = generator or torch.Generator()
    xs, ys = [], []
    for _ in range(batch):
        path = torch.randperm(nodes, generator=g)[: hops + 1].tolist()
        start = path[0]
        seq = [BOS, start + 2, 1]                      # BOS start '|'
        for a, b in zip(path[:-1], path[1:]):
            seq += [a + 2, b + 2]
        seq += [1]                                     # '='
        if cot:
            target = [p + 2 for p in path[1:]] + [path[-1] + 2]
        else:
            target = [path[-1] + 2]
        xs.append(seq)
        ys.append([-100] * len(seq) + target)
    T = max(len(s) for s in xs) + max(len(t) for t in ys)
    x = torch.zeros(batch, T, dtype=torch.long)
    y = torch.full((batch, T), -100, dtype=torch.long)
    for i, (s, t) in enumerate(zip(xs, ys)):
        x[i, : len(s)] = torch.tensor(s)
        y[i, len(s) : len(s) + len(t)] = torch.tensor(t)
    return x, y


# ---------------------------------------------------------------- IOI, words
NAMES = ["mary", "john", "alice", "bob", "emma", "james", "sarah", "david"]
PLACES = ["store", "park", "office", "market"]
OBJECTS = ["drink", "book", "flower", "apple"]
WORDS = ["[pad]", "when", "and", "went", "to", "the", "gave", "a", "then"] + NAMES + PLACES + OBJECTS
W2I = {w: i for i, w in enumerate(WORDS)}


def ioi_batch(batch=64, generator=None):
    """'When A and B went to the P, B gave a O to' -> A."""
    g = random.Random(generator)
    xs, ys = [], []
    for _ in range(batch):
        a, b = g.sample(NAMES, 2)
        p = g.choice(PLACES)
        o = g.choice(OBJECTS)
        words = ["when", a, "and", b, "went", "to", "the", p, b, "gave", "a", o, "to"]
        xs.append([W2I[w] for w in words])
        ys.append(W2I[a])
    T = max(len(s) for s in xs) + 1
    x = torch.zeros(batch, T, dtype=torch.long)
    y = torch.full((batch, T), -100, dtype=torch.long)
    for i, (s, t) in enumerate(zip(xs, ys)):
        x[i, : len(s)] = torch.tensor(s)
        y[i, len(s)] = t
    return x, y


def ioi_prompt(a, b, p, o):
    words = ["when", a, "and", b, "went", "to", "the", p, b, "gave", "a", o, "to"]
    return torch.tensor([[W2I[w] for w in words]]), W2I[a]
