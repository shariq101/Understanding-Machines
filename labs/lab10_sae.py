# %% [markdown]
# # Lab 10 — Sparse autoencoders
#
# A dictionary learns to reconstruct activations with a sparse code. We first
# test it on synthetic sparse features, where the truth is known, and then on
# the transformer's MLP activations, where it is not.
#
# **Milestone:** compare FVU, L0, and feature purity in the clean and messy cases.

# %%
import torch
from uml.sae import train_sae

torch.set_num_threads(4)
# Synthetic: 10 sparse features in 64 dimensions, about one active on average.
g = torch.Generator().manual_seed(0)
n, d, F = 4096, 64, 10
active = (torch.rand(n, F, generator=g) < 0.1).float()
values = active * (0.5 + torch.rand(n, F, generator=g))
x = torch.zeros(n, d); x[:, :F] = values
_, s1 = train_sae(x, d_hidden=64, l1=0.5, steps=800)
print("synthetic   FVU:", s1["fvu"], " L0:", s1["l0"], " of", s1["d_hidden"])

# %%
from uml.model import load
from uml.data import induction_gap_prompt
model, _ = load("checkpoints/induction.pt")
L, gap = 10, 5
gg = torch.Generator().manual_seed(5)
store = {}
acts, toks = [], []
with torch.no_grad():
    for i in range(24):
        t = torch.randint(1, 16, (L,), generator=gg)
        f = torch.randint(1, 16, (gap,), generator=gg)
        model(induction_gap_prompt(t.tolist(), f.tolist()), store=store)
        acts.append(store[(1, "mlp")][0])
        toks.append(torch.cat([torch.zeros(1, dtype=torch.long), t, f, t[:-1]]))
acts = torch.cat(acts, 0).reshape(-1, 256)
tok = torch.cat(toks, 0)
sae, s2 = train_sae(acts, d_hidden=256, l1=1.0, steps=700)
print("model MLP   FVU:", s2["fvu"], " L0:", s2["l0"], " of", s2["d_hidden"])

# %%
z = (acts - s2["mu"]) / s2["sd"]
with torch.no_grad():
    a = sae.enc(z).relu()
purity = []
for f in range(20):
    top = a[:, f].topk(20).indices
    tt = tok.view(-1)[top].tolist()
    counts = {}
    for t in tt:
        counts[t] = counts.get(t, 0) + 1
    purity.append(max(counts.values()) / 20)
print("top-20 features, mean purity by top token:", round(sum(purity) / len(purity), 3))

# %% [markdown]
# On the synthetic data the dictionary is clean and sparse. On the model's
# activations the reconstruction is good but the features mix token identities:
# an FVU near zero does not imply an interpretable ontology. The honest test is
# causal: clamp a feature on and see whether the behaviour changes.
