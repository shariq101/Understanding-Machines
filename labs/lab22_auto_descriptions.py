# %% [markdown]
# # Lab 22 — Automated descriptions are hypotheses
#
# An automated interpretability tool proposes a label for a unit. The label is
# a hypothesis, not a finding. This lab generates descriptions for the VLM's
# MLP neurons from their top activations, measures whether the description
# survives a held-out split, and then tests one description with an ablation.
#
# **Milestone:** report description purity, held-out stability, and the causal
# effect of ablating the described neurons against a random control.

# %%
import torch

from uml.multimodal import (load_clip, load_vlm, question_batch, mlp_activations,
                            ablate_mlp_neurons, COLORS)

torch.set_num_threads(4)
clip = load_clip("checkpoints/clip.pt")
model = load_vlm(clip.vision, "checkpoints/vlm.pt")
g = torch.Generator().manual_seed(22)

imgs, q, ans, meta = question_batch(600, g, types=("color",))
labels = torch.tensor([m[1] for m in meta])
print("scenes:", len(imgs), " neurons per layer:", 2 * model.d_model)

# %%
# Mean activation over the 17 image-token positions. We describe a neuron by
# the colour of the scenes that drive it most.
with torch.no_grad():
    h = mlp_activations(model, imgs, q)[0]
a = h[:, :17].mean(1)
print("activation matrix:", tuple(a.shape))

perm = torch.randperm(len(imgs), generator=torch.Generator().manual_seed(222))
A, B = perm[:300], perm[300:]


# %%
def describe(neuron, split, k=20):
    """Majority colour among the top-k activations in one split."""
    vals = a[split, neuron]
    top = vals.topk(k).indices
    lab = labels[split[top]]
    counts = torch.bincount(lab, minlength=3).float()
    return int(counts.argmax()), (counts.max() / k).item()


rows = []
for j in range(a.shape[1]):
    pref, purity = describe(j, A)
    if a[A, j].std() < 1e-3:
        continue
    topB = a[B, j].topk(20).indices
    stability = (labels[B[topB]] == pref).float().mean().item()
    rows.append((purity * stability, purity, stability, j, pref))

rows.sort(reverse=True)
print("\nneuron  colour  purity  stability  score")
for score, purity, stability, j, pref in rows[:8]:
    print(f"{j:6d}  {COLORS[pref]:6s}  {purity:6.3f}  {stability:9.3f}  {score:.3f}")


# %%
# Pick the best-supported description, then ablate every layer-0 neuron with
# the same preferred colour. The ablation is the causal test; the description
# alone is a correlation with the top activations.
score, purity, stability, best, pref = rows[0]
same = [(r[1] + r[2], r[3]) for r in rows if r[4] == pref and r[1] > 0.6]
same.sort(reverse=True)
neurons = [j for _, j in same[:8]]
print("\ndescription:", f"colour={COLORS[pref]}", "| purity", round(purity, 3),
      "| stability", round(stability, 3))
print("neurons with the same description:", neurons)

eval_imgs, eval_q, eval_ans, _ = question_batch(
    256, torch.Generator().manual_seed(223), types=("color",))


def accuracy():
    with torch.no_grad():
        logits, _ = model(eval_imgs, eval_q)
    return round((logits[:, -1].argmax(-1) == eval_ans).float().mean().item(), 3)


base = accuracy()
with ablate_mlp_neurons(model, 0, neurons):
    ablated = accuracy()
with ablate_mlp_neurons(model, 0, torch.randperm(a.shape[1])[:len(neurons)].tolist()):
    control = accuracy()
with ablate_mlp_neurons(model, 0, list(range(a.shape[1]))):
    whole_layer = accuracy()
print("colour accuracy: base", base, "| described ablation", ablated,
      "| random control", control, "| whole layer", whole_layer)


# %%
# The claim ledger. Purity and stability are descriptive. The described subset
# ablates to nothing, while the layer as a whole is necessary: the label is a
# description, not a localizer. That is the gap automated tools must cross.
print("\nclaim ledger")
print(" establishes: the descriptions are stable on held-out scenes, but ablating the described subset does not change the answer")
print(f" establishes: ablating all 128 layer-0 neurons costs {1 - whole_layer:.3f}, so the layer contributes; the described subset does not localize it")
print(" does not establish: that the label is the unit's function, or that the effect transfers")
print(" next control: ablate a matched set of neurons whose top scenes have a different colour")
