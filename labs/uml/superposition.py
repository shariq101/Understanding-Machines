"""A two-dimensional bottleneck asked to carry more than two sparse features.

This is the smallest model in which superposition appears: five feature
directions, two hidden dimensions, a sparsity knob, and a reconstruction loss.
The input is a sparse vector of feature activations, not a point in 2D; the
point in 2D is what the five features are squeezed into.
"""
from __future__ import annotations

import math

import torch


def directions(n_features=5, dim=2):
    ang = torch.arange(n_features) * (2 * math.pi / n_features)
    return torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)   # [F,2]


def sample(n, p, n_features=5, generator=None):
    """Sparse feature activations: each feature is on with probability p."""
    active = (torch.rand(n, n_features, generator=generator) < p).float()
    values = active * (0.7 + 0.6 * torch.rand(n, n_features, generator=generator))
    return values


def train_model(p, steps=3000, n=4096, n_features=5, seed=0, lr=0.03):
    torch.manual_seed(seed)
    D = directions(n_features)
    W = torch.nn.Parameter(torch.randn(n_features, 2) * 0.1)   # encoder
    b = torch.nn.Parameter(torch.zeros(n_features))
    a = sample(n, p, n_features)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        h = a @ W                                              # [n,2]
        recon = torch.relu(h @ W.T + b)                         # [n,F]
        loss = ((recon - a) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        cos = torch.nn.functional.cosine_similarity(
            W / W.norm(dim=1, keepdim=True), D, dim=1)
        return {"p": p, "loss": round(loss.item(), 4),
                "cosines": [round(c, 3) for c in cos.tolist()],
                "represented": int((cos.abs() > 0.7).sum().item()),
                "W": W.detach().clone(), "b": b.detach().clone(), "D": D}


def sweep(ps=(1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05), seed=0):
    return [train_model(p, seed=seed) for p in ps]
