# %% [markdown]
# # Lab 6 — Attention from scratch
#
# Attention is a soft dictionary lookup. We implement it in NumPy, compare it
# with PyTorch's reference, and inspect the effect of the causal mask. If the
# two implementations disagree, every downstream interpretation is suspect.
#
# **Milestone:** the maximum absolute difference is below 1e-6.

# %%
import numpy as np
import torch
import torch.nn.functional as F

rng = np.random.default_rng(0)
T, d = 6, 4
Q = rng.normal(size=(T, d))
K = rng.normal(size=(T, d))
V = rng.normal(size=(T, d))

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def attention(Q, K, V, causal=False):
    scores = Q @ K.T / np.sqrt(d)
    if causal:
        scores = np.where(np.triu(np.ones((T, T)), 1).astype(bool), -np.inf, scores)
    A = softmax(scores, axis=-1)
    return A @ V, A

out, A = attention(Q, K, V)
ref = F.scaled_dot_product_attention(
    torch.tensor(Q, dtype=torch.float32)[None],
    torch.tensor(K, dtype=torch.float32)[None],
    torch.tensor(V, dtype=torch.float32)[None],
)[0].numpy()
print("max |numpy - torch|:", f"{np.abs(out - ref).max():.2e}")

# %%
print("attention rows sum to 1:", np.allclose(A.sum(1), 1.0))
print("first row:", np.round(A[0], 3))

# %%
_, Ac = attention(Q, K, V, causal=True)
print("causal mask, upper triangle is exactly zero:", bool(np.all(Ac[np.triu_indices(T, 1)] == 0)))
print("causal row 3:", np.round(Ac[3], 3))

# %% [markdown]
# The causal mask sets the masked logits to −∞ **before** the softmax. Masking
# afterwards would leave the denominator including the masked terms, so the
# remaining weights would not sum to one. This is the most common implementation
# bug in a first transformer.
