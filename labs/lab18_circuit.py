# %% [markdown]
# # Lab 18 — Heads, neurons, and the visual pathway
#
# A neuron-level story is tempting because neurons are easy to count. This lab
# puts a head-level intervention and a neuron-level intervention on the same
# task and lets the effect sizes argue.
#
# **Milestone:** name the head whose knockout costs the most accuracy, and
# compare that cost with ablating the most modality-selective neurons.

# %%
import torch
from uml.multimodal import (load_clip, load_vlm, question_batch,
                            attention_knockout, attention_maps,
                            modality_selectivity, ablate_mlp_neurons, scene,
                            QUESTIONS, WORD_ID, BOS)

torch.set_num_threads(4)
clip = load_clip("checkpoints/clip.pt")
model = load_vlm(clip.vision, "checkpoints/vlm.pt")
g = torch.Generator().manual_seed(4)
imgs, q, ans, meta = question_batch(512, g)

def accuracy():
    return round((model(imgs, q)[0][:, -1].argmax(-1) == ans).float().mean().item(), 3)

base = accuracy()
print("base accuracy:", base)

# %%
# Knock out the final query's attention to every image token, one head at a
# time.  A head that matters should produce a large drop.
print("layer head  accuracy  drop")
rows = []
for layer in range(len(model.blocks)):
    for head in range(model.cfg.n_heads):
        with attention_knockout(model, layer, head, slice(-1, None), slice(0, 17)):
            acc = accuracy()
        rows.append((layer, head, acc, base - acc))
        print(f"{layer:5d} {head:4d}  {acc:8.3f}  {base - acc:+.3f}")

best = max(rows, key=lambda r: r[3])
print("largest knockout:", best)

# %%
# Modality-selective MLP neurons: mean activation on image tokens minus mean on
# text tokens, normalized.  Count them, then ablate the most visual ones.
for layer in range(len(model.blocks)):
    sel, img, txt = modality_selectivity(model, imgs, q, layer)
    strong = int((sel > 0.8).sum().item())
    top = sel.topk(32).indices
    with ablate_mlp_neurons(model, layer, top):
        acc = accuracy()
    print(f"layer {layer}: strong visual neurons={strong}/{len(sel)}  "
          f"ablate top32 accuracy={acc:.3f}  drop={base - acc:+.3f}")

# %%
# Attention mass is not necessity.  The final query reads a large fraction of
# its attention from the image, but the knockout table shows that only some of
# that mass is load-bearing.
qv = torch.tensor([[BOS] + [WORD_ID[w] for w in QUESTIONS["color"]] + [WORD_ID["="]]])
for pos in ("left", "center", "right"):
    img = scene("circle", "red", pos)[None]
    att = attention_maps(model, img, qv)
    mass = [round(att[l][:, -1, :17].sum(-1).mean().item(), 3) for l in range(len(att))]
    print(pos, "image attention mass by layer:", mass)

# %%
# A head-level ablation, for comparison with knockout.  Zeroing a head's write
# is a different intervention: it removes the head everywhere, not just on one
# edge.
with torch.no_grad():
    W = model.blocks[best[0]].attn.Wo.weight
    dh = model.cfg.d_head
    saved = W[:, best[1] * dh:(best[1] + 1) * dh].clone()
    W[:, best[1] * dh:(best[1] + 1) * dh] = 0
    ablated = accuracy()
    W[:, best[1] * dh:(best[1] + 1) * dh] = saved
print("write ablation of", best[:2], "accuracy:", ablated, "drop:", round(base - ablated, 3))
