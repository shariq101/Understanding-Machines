"""A small, readable decoder-only transformer.

The point of this module is not speed.  It is that every tensor that matters
has a name, every block is reachable, and the attention pattern can be read
without a debugger.  The labs hook these modules directly.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int
    n_layers: int = 2
    d_model: int = 64
    n_heads: int = 4
    d_mlp: int = 256
    max_len: int = 64
    dropout: float = 0.0
    causal: bool = True

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    def to_dict(self):
        return asdict(self)


class Attention(nn.Module):
    """Multi-head causal self-attention, written out so heads stay separable."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.Wqkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.Wo = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.knockout = None      # optional (head, query_slice, key_slice)

    def forward(self, x, store=None, layer=None):
        B, T, C = x.shape
        H, dh = self.cfg.n_heads, self.cfg.d_head
        qkv = self.Wqkv(x).view(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]              # B,H,T,dh
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        if self.cfg.causal:
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        if self.knockout is not None:
            head, qslice, kslice = self.knockout
            attn[:, head, qslice, kslice] = 0
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        if store is not None:
            store[(layer, "attn")] = attn.detach()
        return self.Wo(out), attn


class MLP(nn.Module):
    """The block's key-value memory: expand, threshold, project back."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.fc2 = nn.Linear(cfg.d_mlp, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, store=None, layer=None):
        h = F.gelu(self.fc1(x))
        if store is not None:
            store[(layer, "mlp")] = h.detach()
        return self.dropout(self.fc2(h))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, store=None, layer=None):
        a, _ = self.attn(self.ln1(x), store, layer)
        x = x + a
        m = self.mlp(self.ln2(x), store, layer)
        return x + m


class TinyGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.unembed.weight = self.tok.weight          # tied embeddings
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("Wo.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, store=None, layers=None):
        B, T = idx.shape
        if T > self.cfg.max_len:
            raise ValueError(f"sequence of {T} exceeds max_len {self.cfg.max_len}")
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        for i, block in enumerate(self.blocks):
            if layers is None or i in layers:
                x = block(x, store, i)
            else:
                x = block(x, None, i)
        h = self.ln_f(x)
        return self.unembed(h), h

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def cache(self, idx):
        """Run once and return every attention pattern and MLP activation."""
        store = {}
        with torch.no_grad():
            self(idx, store=store)
        return store


def load(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = Config(**ckpt["config"])
    model = TinyGPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
