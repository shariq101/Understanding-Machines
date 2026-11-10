# %% [markdown]
# # Lab 15 — Trace the image into the answer
#
# Causal tracing asks a counterfactual: if this activation had been clean, would
# the corrupted run have answered correctly? We patch one site at a time and
# measure how much of the clean logit difference returns.
#
# **Milestone:** report the layer curve for the final token and the image-token
# map at the layer where the visual information enters.

# %%
import torch
from uml.multimodal import (load_clip, load_vlm, scene_pair, patch_vlm,
                            QUESTIONS, WORD_ID, BOS)

torch.set_num_threads(4)
clip = load_clip("checkpoints/clip.pt")
model = load_vlm(clip.vision, "checkpoints/vlm.pt")

q = torch.tensor([[BOS] + [WORD_ID[w] for w in QUESTIONS["color"]] + [WORD_ID["="]]])
clean, corrupt = (x[None] for x in scene_pair("circle", "left", "red", "green", seed=21))
T = 17 + q.shape[1]
answer = (WORD_ID["red"], WORD_ID["green"])
print("sequence length:", T, " image tokens:", 17, " text starts at 17")

# %%
# Patch the final text position after each block.  The curve says when the
# answer becomes readable from that site.
print("layer  restored  before    after    clean")
for layer in range(4):
    r = patch_vlm(model, clean, corrupt, q, layer=layer, positions=[T - 1], answer=answer)
    print(f"{layer:5d}  {r['restored']:8.3f}  {r['before']:7.3f}  {r['after']:7.3f}  {r['clean']:7.3f}")

# %%
# Patch one image token at a time, at the layer where the visual signal first
# enters.  The 4x4 grid is the patch layout; the class token is position 0.
grid = []
for tok in range(1, 17):
    r = patch_vlm(model, clean, corrupt, q, layer=0, positions=[tok], answer=answer)
    grid.append(r["restored"])
print("image-token restoration at layer 0:")
for row in range(4):
    print(" ".join(f"{grid[row * 4 + col]:+.2f}" for col in range(4)))
ranked = sorted(enumerate(grid, start=1), key=lambda t: -abs(t[1]))
print("top tokens:", [(i, round(v, 3)) for i, v in ranked[:5]])

# %%
# The class token is a summary, but it is not the only path.  Patch it alone
# and compare with the best patch token.
r = patch_vlm(model, clean, corrupt, q, layer=0, positions=[0], answer=answer)
print("class token alone:", r["restored"])

# %%
# A negative control: patch a token that does not contain the object.  The
# restoration should be near zero, and the difference is the evidence that the
# map is spatial rather than a generic reset.
for tok in (1, 16):
    r = patch_vlm(model, clean, corrupt, q, layer=0, positions=[tok], answer=answer)
    print("control token", tok, "restored:", r["restored"])
