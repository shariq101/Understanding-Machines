# %% [markdown]
# # Lab 17 — A dictionary of visual features, and what steering costs
#
# An SAE on CLIP's class token asks whether the embedding is a sparse mixture of
# reusable concepts. The second half of the lab is the harder question: if a
# feature is real, does its decoder direction steer the model?
#
# **Milestone:** report FVU, L0, feature purity, and the selectivity of two
# steering interventions.

# %%
import torch
import torch.nn.functional as F
from uml.multimodal import (load_clip, scene_batch, one_per_caption, caption,
                            encode_caption, clip_retrieval, COLORS, SHAPES, POSITIONS)
from uml.sae import train_sae

torch.set_num_threads(4)
clip = load_clip("checkpoints/clip.pt")
g = torch.Generator().manual_seed(9)
imgs, shape, color, pos, _ = scene_batch(3000, g)
with torch.no_grad():
    X = clip.vision(imgs)[1][:, 0]
print("activation shape:", tuple(X.shape))

sae, info = train_sae(X, d_hidden=192, l1=0.8, steps=1200, batch=256, lr=1e-3, seed=0)
print("SAE:", {k: info[k] for k in ("fvu", "l0", "d_hidden", "l1")})
with torch.no_grad():
    Z = (X - info["mu"]) / info["sd"]
    recon, A = sae(Z)
print("active features:", int((A > 1e-6).any(0).sum()), "of", A.shape[1])

# %%
# Purity: what fraction of a feature's top activations share a label?
def purity(feature_acts, labels, k=20):
    vals = []
    for f in range(feature_acts.shape[1]):
        top = feature_acts[:, f].topk(k).indices
        counts = torch.bincount(labels[top], minlength=3).float()
        vals.append((counts.max() / k).item())
    return vals

for name, labels in (("colour", color), ("shape", shape), ("position", pos)):
    vals = torch.tensor(purity(A, labels))
    print(f"{name:8s} mean purity={vals.mean():.3f}  best={vals.max():.3f}")

# %%
# Pick the most colour-pure feature and ask which colour it prefers.
best = None
for f in range(A.shape[1]):
    top = A[:, f].topk(30).indices
    counts = torch.bincount(color[top], minlength=3).float()
    score = counts.max().item() / 30
    if best is None or score > best[0]:
        best = (score, f, int(counts.argmax()))
score, feat, pref = best
print("best colour feature:", feat, "purity", round(score, 3), "preferred", COLORS[pref])

# %%
# Evaluate attribute accuracy from the nearest of the 27 captions.
all_imgs, labels = one_per_caption()
caps = torch.stack([encode_caption(caption(COLORS[c], SHAPES[s], POSITIONS[p]))
                    for s, c, p in labels.tolist()])
with torch.no_grad():
    te = clip.text_embed(caps)

def attr_accuracy(z):
    vi = F.normalize(clip.vproj(z), dim=-1)
    pred = (vi @ te.T).argmax(1)
    pl = labels[pred]
    return {
        "colour": round((pl[:, 1] == labels[:, 1]).float().mean().item(), 3),
        "shape": round((pl[:, 0] == labels[:, 0]).float().mean().item(), 3),
        "position": round((pl[:, 2] == labels[:, 2]).float().mean().item(), 3),
        "red_frac": round((pl[:, 1] == 0).float().mean().item(), 3),
    }

with torch.no_grad():
    base_z = clip.vision(all_imgs)[1][:, 0]
print("base:", attr_accuracy(base_z))

# %%
# Steering with the SAE decoder direction: the feature is decodable, but is its
# direction a control knob?  Increase the strength and watch selectivity.
d_sae = sae.dec.weight[:, feat].detach()
d_sae = d_sae / d_sae.norm()
for alpha in (2.0, 5.0, 10.0):
    with torch.no_grad():
        z = base_z + alpha * d_sae * X.std(0).mean()
    print("SAE direction alpha", alpha, attr_accuracy(z))

# %%
# A difference-in-means direction is not a dictionary feature; it is the
# simplest supervised probe direction.  Compare its selectivity.
with torch.no_grad():
    d_mean = X[color == 0].mean(0) - X[color == 2].mean(0)
d_mean = d_mean / d_mean.norm()
for alpha in (2.0, 5.0, 10.0):
    with torch.no_grad():
        z = base_z + alpha * d_mean * X.std(0).mean()
    print("mean-shift alpha", alpha, attr_accuracy(z))
