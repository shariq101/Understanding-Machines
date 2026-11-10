"""A sparse autoencoder: an overcomplete dictionary learned from activations.

The encoder is linear plus ReLU; the decoder is linear.  The only pressure
toward interpretability is the L1 penalty on the code, which asks the model to
explain each activation with as few dictionary elements as it can.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAE(nn.Module):
    def __init__(self, d_in, d_hidden=512):
        super().__init__()
        self.enc = nn.Linear(d_in, d_hidden)
        self.dec = nn.Linear(d_hidden, d_in, bias=True)
        self.d_in, self.d_hidden = d_in, d_hidden

    def forward(self, x):
        a = F.relu(self.enc(x))
        return self.dec(a), a


def train_sae(x, d_hidden=512, l1=3e-3, steps=1500, batch=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    mu, sd = x.mean(0, keepdim=True), x.std(0, keepdim=True) + 1e-6
    z = (x - mu) / sd
    sae = SAE(x.shape[1], d_hidden)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for step in range(steps):
        idx = torch.randint(0, len(z), (batch,))
        xb = z[idx]
        recon, a = sae(xb)
        loss = F.mse_loss(recon, xb) + l1 * a.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        recon, a = sae(z)
        mse = F.mse_loss(recon, z).item()
        var = z.var().item()
        l0 = (a > 1e-6).float().sum(1).mean().item()
    return sae, {"mse": round(mse, 4), "fvu": round(mse / var, 4),
                 "l0": round(l0, 1), "d_hidden": d_hidden,
                 "l1": l1, "mu": mu, "sd": sd}
