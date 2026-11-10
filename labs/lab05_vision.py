# %% [markdown]
# # Lab 5 — Pixels to patches
#
# A CNN and a ViT see the same 16×16 shapes through different priors. We train
# both, test them on clean and occluded images, and inspect the class token's
# attention. The comparison is not a leaderboard; it is a lesson in inductive bias.
#
# **Milestone:** explain one result that the attention map alone would have misled you about.

# %%
import torch
from uml.vision import shapes, CNN, ViT, train, evaluate, occlude

torch.set_num_threads(4)
xtr, ytr = shapes(900, seed=0)
xte, yte = shapes(300, seed=1)

cnn = CNN(); vit = ViT()
train(cnn, xtr, ytr, epochs=15, seed=0)
train(vit, xtr, ytr, epochs=15, seed=0)
print("parameters  CNN:", sum(p.numel() for p in cnn.parameters()),
      " ViT:", sum(p.numel() for p in vit.parameters()))
print("clean accuracy  CNN:", evaluate(cnn, xte, yte), " ViT:", evaluate(vit, xte, yte))

# %%
for n in (5, 7):
    xo = occlude(xte, n=n, seed=5)
    print(f"occluder {n}x{n}  CNN: {evaluate(cnn, xo, yte):.3f}  ViT: {evaluate(vit, xo, yte):.3f}")

# %%
att = vit.attention_maps(xte[:1])[-1][0, 0, 0, 1:]   # class token -> patches
p = att / att.sum()
entropy = float(-(p * (p + 1e-9).log()).sum())
print("class-token attention, last layer:", [round(float(v), 3) for v in att])
print("normalized entropy:", round(entropy, 3), "of", round(float(torch.tensor(16.0).log()), 3), "maximum")

# %% [markdown]
# The attention is nearly uniform. That does **not** mean the model ignores the
# image: a head can attend broadly and still write a selective value. The
# occlusion result is the causal evidence. The CNN degrades more gracefully at
# this scale because its local filters and pooling spread the evidence; the ViT
# loses a larger fraction of its input when a patch is erased.
