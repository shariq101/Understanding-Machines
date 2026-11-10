# %% [markdown]
# # Lab 13 — Align two modalities
#
# A CLIP-style model is two towers trained so that a picture and its caption
# land in the same place. We train one on a synthetic world of three shapes,
# three colours, and three positions, then measure what alignment buys and
# what it does not.
#
# **Milestone:** report retrieval, the modality gap, and the binding accuracy
# on colour- and shape-swapped captions.

# %%
from pathlib import Path
import torch
import torch.nn.functional as F

from uml.multimodal import (TinyCLIP, train_clip, save_clip, load_clip,
                            scene_batch, one_per_caption, caption, encode_caption,
                            clip_retrieval, modality_gap, COLORS, SHAPES, POSITIONS)

torch.set_num_threads(4)
CKPT = Path("checkpoints/clip.pt")

if CKPT.exists():
    model = load_clip(CKPT)
    print("loaded", CKPT)
else:
    model = TinyCLIP(seed=3)
    print("parameters:", sum(p.numel() for p in model.parameters()))
    train_clip(model, steps=900, batch=128, lr=2e-3, seed=3)
    save_clip(model, CKPT)

print("parameters:", sum(p.numel() for p in model.parameters()))

# %%
# The 27 distinct combinations, one image and one caption each.
imgs, labels = one_per_caption()
txt = torch.stack([encode_caption(caption(COLORS[c], SHAPES[s], POSITIONS[p]))
                   for s, c, p in labels.tolist()])
print("retrieval:", clip_retrieval(model, imgs, txt))
print("modality gap:", modality_gap(model, imgs, txt))

# %%
# A linear map from image space to text space: how much of the alignment is
# captured by a single affine transform?
with torch.no_grad():
    X = model.image_embed(imgs)
    Y = model.text_embed(txt)
A = X.T @ X + 1e-2 * torch.eye(X.shape[1])
W = torch.linalg.solve(A, X.T @ Y)
mapped = X @ W
cos_before = F.cosine_similarity(X, Y).mean().item()
cos_after = F.cosine_similarity(mapped, Y).mean().item()
print(f"linear map cosine: before={cos_before:.3f} after={cos_after:.3f}")

# %%
# Compositionality: keep the words, change one attribute.  A bag-of-words
# model cannot tell the correct caption from the swap.
def score(image_vec, text_ids):
    te = model.text_embed(text_ids)
    return (image_vec @ te.T)

correct = bad_color = bad_shape = 0
for i, (s, c, p) in enumerate(labels.tolist()):
    good = caption(COLORS[c], SHAPES[s], POSITIONS[p])
    c2 = (c + 1) % 3
    s2 = (s + 1) % 3
    swap_c = caption(COLORS[c2], SHAPES[s], POSITIONS[p])
    swap_s = caption(COLORS[c], SHAPES[s2], POSITIONS[p])
    ids = torch.stack([encode_caption(good), encode_caption(swap_c), encode_caption(swap_s)])
    sc = score(model.image_embed(imgs[i:i+1])[0], ids)
    correct += int(sc.argmax() == 0)
    bad_color += int(sc.argmax() == 1)
    bad_shape += int(sc.argmax() == 2)
n = len(labels)
print(f"binding: correct={correct/n:.3f} colour-swap={bad_color/n:.3f} shape-swap={bad_shape/n:.3f}")

# %%
# The gap is not a bug in the loss; it is a geometric fact about two towers
# trained with a symmetric objective.  Images and captions occupy separate
# cones that face each other.
with torch.no_grad():
    vi = model.image_embed(imgs)
    te = model.text_embed(txt)
    print("mean image norm:", round(vi.norm(dim=1).mean().item(), 3))
    print("mean text norm: ", round(te.norm(dim=1).mean().item(), 3))
    print("within-pair cosine:", round((vi * te).sum(-1).mean().item(), 3))
    print("cross-pair cosine: ", round((vi @ te.T).mean().item(), 3))
