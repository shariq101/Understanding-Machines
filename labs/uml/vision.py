"""Pixels to patches: a tiny vision lab with no downloaded data.

The images are 16x16 grayscale shapes.  A CNN sees them through local filters;
a ViT cuts them into 4x4 patches and lets attention mix the patches.  Both are
small enough to train on a CPU in seconds, and the ViT's attention is a tensor
we can draw.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Block, Config


def shapes(n=1600, size=16, seed=0):
    """Return images [n,1,size,size], labels [n] (0 circle, 1 square)."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    imgs, labels = [], []
    for _ in range(n):
        cx = int(torch.randint(4, size - 4, (1,), generator=g))
        cy = int(torch.randint(4, size - 4, (1,), generator=g))
        r = float(torch.randint(3, 5, (1,), generator=g))
        lab = int(torch.randint(0, 2, (1,), generator=g))
        dx, dy = (xx - cx).float(), (yy - cy).float()
        if lab == 0:
            mask = (dx ** 2 + dy ** 2) <= r ** 2
        else:
            mask = (dx.abs() <= r) & (dy.abs() <= r)
        img = mask.float() * (0.75 + 0.25 * torch.rand(1, generator=g))
        img = img + 0.08 * torch.randn(size, size, generator=g)
        imgs.append(img.clamp(0, 1)[None])
        labels.append(lab)
    return torch.stack(imgs), torch.tensor(labels)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16 * 4 * 4, 2),
        )

    def forward(self, x):
        return self.net(x), None


class ViT(nn.Module):
    """16 patches, a class token, two pre-LN blocks, mean-free attention."""

    def __init__(self, d_model=32, n_heads=2, n_layers=2, patch=4, size=16, in_chans=1):
        super().__init__()
        self.patch, self.size, self.in_chans = patch, size, in_chans
        n = (size // patch) ** 2
        self.cfg = Config(vocab_size=2, n_layers=n_layers, d_model=d_model,
                          n_heads=n_heads, d_mlp=2 * d_model, max_len=n + 1,
                          causal=False)
        self.embed = nn.Linear(in_chans * patch * patch, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, d_model))
        self.blocks = nn.ModuleList([Block(self.cfg) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.pos, std=0.02)

    def patches(self, x):
        p = self.patch
        # [B,1,H,W] -> [B, n, p*p]
        u = x.unfold(2, p, p).unfold(3, p, p)          # B,1,H/p,W/p,p,p
        u = u.permute(0, 2, 3, 1, 4, 5).reshape(x.shape[0], -1, self.in_chans * p * p)
        return u

    def forward(self, x, store=None):
        z = self.embed(self.patches(x))
        z = torch.cat([self.cls.expand(z.shape[0], -1, -1), z], dim=1) + self.pos
        for i, b in enumerate(self.blocks):
            z = b(z, store, i)
        z = self.ln(z)
        return self.head(z[:, 0]), z

    def attention_maps(self, x):
        store = {}
        with torch.no_grad():
            self(x, store)
        return [store[(l, "attn")] for l in range(len(self.blocks))]


def train(model, x, y, epochs=30, lr=2e-3, seed=0, log_every=10):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    hist = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), 64):
            idx = perm[i : i + 64]
            logits, _ = model(x[idx])
            loss = F.cross_entropy(logits, y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        if ep % log_every == 0 or ep == epochs - 1:
            hist.append((ep, round(total / len(x), 4)))
    return hist


def occlude(x, n=5, seed=5):
    """Erase one n×n square per image, chosen at random."""
    g = torch.Generator().manual_seed(seed)
    y = x.clone()
    for i in range(len(y)):
        cx = int(torch.randint(0, x.shape[-1] - n, (1,), generator=g))
        cy = int(torch.randint(0, x.shape[-1] - n, (1,), generator=g))
        y[i, :, cy:cy + n, cx:cx + n] = 0
    return y


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    logits, _ = model(x)
    return (logits.argmax(-1) == y).float().mean().item()
