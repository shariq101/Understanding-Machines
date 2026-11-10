"""Meaning from co-occurrence, on a corpus small enough to hold in your hand."""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

import torch


ANIMALS = ["cat", "dog", "horse", "cow", "fox"]
COLORS = ["red", "blue", "green", "black", "white"]
COUNTRIES = ["france", "japan", "brazil", "egypt", "canada"]
CAPITALS = ["paris", "tokyo", "brasilia", "cairo", "ottawa"]
FOODS = ["bread", "rice", "apple", "cheese", "soup"]


def corpus(n=2400, seed=0):
    """A tiny world with enough regularity for distributional structure."""
    r = random.Random(seed)
    cap = dict(zip(COUNTRIES, CAPITALS))
    sents = []
    for _ in range(n):
        a, b = r.sample(ANIMALS, 2)
        c1, c2 = r.sample(COLORS, 2)
        co = r.choice(COUNTRIES)
        f1, f2 = r.sample(FOODS, 2)
        template = r.randrange(8)
        if template == 0:
            s = f"the {a} is {c1} and the {b} is {c2}"
        elif template == 1:
            s = f"i saw a {a} in {co} near a {b}"
        elif template == 2:
            s = f"the capital of {co} is {cap[co]}"
        elif template == 3:
            s = f"{cap[co]} is a city in {co}"
        elif template == 4:
            s = f"we ate {f1} and {f2} with a {c1} cup"
        elif template == 5:
            s = f"the {f1} is {c1} and tastes like {f2}"
        elif template == 6:
            s = f"a {a} in {co} likes {f1}"
        else:
            s = f"the flag of {co} is {c1} and {c2}"
        sents.append(s.split())
    return sents


def cooccurrence(sents, window=2):
    vocab = sorted({w for s in sents for w in s})
    idx = {w: i for i, w in enumerate(vocab)}
    C = torch.zeros(len(vocab), len(vocab))
    for s in sents:
        ids = [idx[w] for w in s]
        for i, w in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if i != j:
                    C[w, ids[j]] += 1.0
    return vocab, idx, C


def ppmi(C, eps=1e-9):
    total = C.sum()
    pw = C.sum(1, keepdim=True) / total
    pc = C.sum(0, keepdim=True) / total
    pwc = C / total
    pmi = torch.log((pwc + eps) / (pw @ pc + eps))
    return pmi.clamp(min=0)


def svd_embed(M, dim=30):
    U, S, Vh = torch.linalg.svd(M)
    return U[:, :dim] * S[:dim].sqrt()


def nearest(vocab, E, word, k=5):
    i = vocab.index(word)
    sims = torch.nn.functional.cosine_similarity(E[i][None], E, dim=1)
    order = sims.argsort(descending=True).tolist()
    out = []
    for j in order:
        if j != i:
            out.append((vocab[j], round(sims[j].item(), 3)))
        if len(out) == k:
            break
    return out


def analogy(vocab, E, a, b, c, k=3):
    """a is to b as c is to ?  (b - a + c)"""
    E = torch.nn.functional.normalize(E, dim=1)
    v = E[vocab.index(b)] - E[vocab.index(a)] + E[vocab.index(c)]
    sims = torch.nn.functional.cosine_similarity(v[None], E, dim=1)
    out = []
    for j in sims.argsort(descending=True).tolist():
        w = vocab[j]
        if w not in (a, b, c):
            out.append((w, round(sims[j].item(), 3)))
        if len(out) == k:
            break
    return out
