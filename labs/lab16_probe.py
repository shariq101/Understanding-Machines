# %% [markdown]
# # Lab 16 — Probe the visual stream
#
# A probe asks whether a property is linearly decodable from an activation. It
# does not ask whether the model uses the property. The distinction is the
# whole lab: a feature can be present and still be ignored downstream.
#
# **Milestone:** report the layerwise probe table and the shuffled-label control.

# %%
import torch
from uml.multimodal import (load_clip, load_vlm, scene_batch, vit_residuals,
                            probe_eval, question_batch, vlm_residual_cache,
                            COLORS, SHAPES, POSITIONS)

torch.set_num_threads(4)
clip = load_clip("checkpoints/clip.pt")
vlm = load_vlm(clip.vision, "checkpoints/vlm.pt")

g = torch.Generator().manual_seed(12)
imgs, shape, color, pos, _ = scene_batch(1800, g)
resid = vit_residuals(clip.vision, imgs)

# %%
print("CLIP ViT: class token")
print("layer  colour  shape  position")
for layer, z in enumerate(resid):
    accs = [probe_eval(z[:, 0], y) for y in (color, shape, pos)]
    print(f"{layer:5d}  {accs[0]:6.3f}  {accs[1]:5.3f}  {accs[2]:8.3f}")

print("CLIP ViT: mean patch token")
print("layer  colour  shape  position")
for layer, z in enumerate(resid):
    accs = [probe_eval(z[:, 1:].mean(1), y) for y in (color, shape, pos)]
    print(f"{layer:5d}  {accs[0]:6.3f}  {accs[1]:5.3f}  {accs[2]:8.3f}")

# %%
# The control: same features, shuffled labels.  A probe that still succeeds is
# reading a confound, not the label.
print("shuffled-label controls at the best layer:")
for name, y in (("colour", color), ("shape", shape), ("position", pos)):
    print(" ", name, probe_eval(resid[-1][:, 0], y, shuffle=True))

# %%
# The same question inside the language model.  The final token is where the
# answer is read; the probes say when each property becomes linearly available.
g2 = torch.Generator().manual_seed(31)
imgs2, q2, ans2, meta2 = question_batch(900, g2, types=("color", "shape", "position"))
labels = torch.tensor([[m[1], m[0], m[2]] for m in meta2])   # colour, shape, position
n_layers = len(vlm.blocks)
rows = [[] for _ in range(n_layers + 1)]
with torch.no_grad():
    for start in range(0, len(imgs2), 150):
        cache = vlm_residual_cache(vlm, imgs2[start:start+150], q2[start:start+150])
        for layer, x in enumerate(cache):
            rows[layer].append(x[:, -1])
rows = [torch.cat(r) for r in rows]
print("VLM: final text token")
print("layer  colour  shape  position")
for layer, x in enumerate(rows):
    accs = [probe_eval(x, labels[:, i]) for i in range(3)]
    print(f"{layer:5d}  {accs[0]:6.3f}  {accs[1]:5.3f}  {accs[2]:8.3f}")
