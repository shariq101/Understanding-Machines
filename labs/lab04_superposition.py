# %% [markdown]
# # Lab 4 — Five features, two dimensions
#
# A two-dimensional bottleneck is asked to carry five sparse features. As the
# features become sparser, the model can afford to give each one a direction.
# The price is interference: activating one feature also excites its neighbours.
#
# **Milestone:** show that average loss can improve while the one-hot response matrix becomes less diagonal.

# %%
import torch
from uml.superposition import train_model

torch.set_num_threads(4)
results = [train_model(p, steps=1800, seed=1, n=2048) for p in (0.9, 0.3, 0.05)]
for r in results:
    print(f"p={r['p']:.2f}  loss={r['loss']:.4f}  |cos|>0.7: {r['represented']}  cos={r['cosines']}")

# %%
# The one-hot response matrix: what comes out when exactly one feature is on?
import torch.nn.functional as F
for r in results:
    W, b, D = r["W"], r["b"], r["D"]
    rec = F.relu(D @ W.T + b)
    off = rec - torch.diag(rec.diagonal())
    print(f"p={r['p']:.2f}  diag mean={rec.diagonal().mean():.2f}  off-diagonal mass={off.abs().mean():.2f}")
    print("  ", [[round(float(x), 2) for x in row] for row in rec])

# %% [markdown]
# At high sparsity the model represents more of the five features, but each
# one-hot input also writes a second feature. The scalar loss is low because
# most features are off most of the time; the matrix is where the packing shows.
# This is the trade-off the chapter calls superposition: capacity bought with
# interference, not a free lunch.

# %%
r = results[-1]
print("feature directions D:")
print(r["D"].round(decimals=2))
print("learned encoder W:")
print(r["W"].round(decimals=2))
