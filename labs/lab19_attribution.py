# %% [markdown]
# # Lab 19 — Attribution, and what it costs
#
# Three maps for the same answer: occlude one image token and measure the drop,
# take the gradient of the correct logit with respect to the projected token,
# and read the final query's attention over the image. Then ask the only
# question that matters for an attribution method: which map, when used to
# delete tokens, destroys the answer fastest?
#
# **Milestone:** report the deletion AUC of the three maps and a random
# control, the rank correlation between them, and the non-additivity of token
# deletion: the joint effect of the top tokens versus the sum of their single
# effects.

# %%
import torch
import torch.nn.functional as F

from uml.multimodal import (load_clip, load_vlm, question_batch, attention_maps,
                            QUESTIONS, WORD_ID, BOS)

torch.set_num_threads(4)
clip = load_clip("checkpoints/clip.pt")
model = load_vlm(clip.vision, "checkpoints/vlm.pt")
g = torch.Generator().manual_seed(19)
B = 16
imgs, q, ans, meta = question_batch(B, g, types=("color",))

print("scenes:", B, " image tokens:", 17, " first answer:", meta[0][1])


# %%
def logits_from_tokens(model, img, qids):
    """Run the decoder on an explicit image-token tensor."""
    txt = model.tok(qids)
    x = torch.cat([img, txt], dim=1)
    x = x + model.pos(torch.arange(x.shape[1], device=x.device))
    for block in model.blocks:
        x = block(x)
    return model.unembed(model.ln_f(x))[:, -1]


@torch.no_grad()
def probs_keep(keep):
    """Probability of the correct colour when image tokens are masked by keep."""
    img = model.image_tokens(imgs) * keep[..., None]
    logits = logits_from_tokens(model, img, q)
    return F.softmax(logits, -1).gather(1, ans[:, None]).squeeze(1)


ones = torch.ones(B, 17)
base = probs_keep(ones)
print("base probability: mean", round(base.mean().item(), 3),
      " min", round(base.min().item(), 3))


# %%
# Occlusion: zero one token, measure the fall in the correct probability.
with torch.no_grad():
    occ = torch.zeros(B, 17)
    for tok in range(17):
        keep = ones.clone()
        keep[:, tok] = 0
        occ[:, tok] = base - probs_keep(keep)
print("occlusion: mean drop", round(occ.mean().item(), 4),
      " largest token", int(occ.mean(0).argmax()),
      " class-token drop", round(occ[:, 0].mean().item(), 4))


# %%
# Gradient saliency: the projected token vector is the input to the decoder, so
# the gradient is taken there. The vision encoder is frozen and bypassed; this
# is a token-level attribution, not a pixel attribution.
img = model.image_tokens(imgs).detach().clone().requires_grad_(True)
logits = logits_from_tokens(model, img, q)
logits.gather(1, ans[:, None]).sum().backward()
grad = img.grad.abs().sum(-1)
grad = grad / (grad.sum(1, keepdim=True) + 1e-9)
print("gradient: mass on class token", round(grad[:, 0].mean().item(), 3),
      " largest patch", int(grad[:, 1:].mean(0).argmax()) + 1)


# %%
# Attention: the final query's mass over image tokens, averaged over heads.
# `attention_maps` returns one scene at a time (it keeps the head axis), so we
# loop over the batch rather than averaging the scenes together.
attn = torch.zeros(B, 17)
with torch.no_grad():
    for i in range(B):
        a = attention_maps(model, imgs[i:i+1], q[i:i+1])[0][:, -1, :17].mean(0)
        attn[i] = a / a.sum()
print("attention: class-token mass", round(attn[:, 0].mean().item(), 3),
      " entropy", round(-(attn * attn.clamp_min(1e-9).log()).sum(1).mean().item(), 3))


# %%
# Deletion faithfulness. Rank tokens by each map, keep the top-k, and average
# the correct probability over k. A faithful map makes the answer fall early.
def deletion_curve(score):
    order = score.argsort(dim=1, descending=True)
    curve = []
    for k in range(0, 17):
        keep = torch.ones(B, 17)
        if k:
            keep.scatter_(1, order[:, :k], 0)
        curve.append(probs_keep(keep).mean().item())
    return curve, sum(curve) / len(curve)


curves = {}
for name, score in (("occlusion", occ), ("gradient", grad), ("attention", attn)):
    curve, auc = deletion_curve(score)
    curves[name] = (curve, auc)
    print(f"{name:10s} AUC={auc:.3f}  p@k=1 {curve[1]:.3f}  k=5 {curve[5]:.3f}  k=10 {curve[10]:.3f}")

rand = torch.rand(B, 17, generator=torch.Generator().manual_seed(2))
_, auc = deletion_curve(rand)
print(f"{'random':10s} AUC={auc:.3f}")


# %%
# Rank correlation. The maps answer different questions, so a low correlation
# is expected; a high one would be suspicious, not reassuring.
def spearman(a, b):
    ra = a.argsort(1).argsort(1).float()
    rb = b.argsort(1).argsort(1).float()
    ra = ra - ra.mean(1, keepdim=True)
    rb = rb - rb.mean(1, keepdim=True)
    return (ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1) + 1e-9)


maps = {"occlusion": occ, "gradient": grad, "attention": attn}
for a, b in (("occlusion", "gradient"), ("occlusion", "attention"),
             ("gradient", "attention")):
    r = spearman(maps[a], maps[b]).mean().item()
    print(f"rank corr {a:9s} vs {b:9s}: {r:+.3f}")


# %%
# Non-additivity: is the total effect of the top four tokens equal to the sum
# of their individual effects? If the tokens carry the same colour, deleting
# them together can hurt less than the sum of four separate deletions. If they
# carry different parts of one computation, it can hurt more. Either answer
# means the attribution is not a separable credit assignment.
order = occ.argsort(1, descending=True)
single = occ.gather(1, order[:, :4]).sum(1)
keep = torch.ones(B, 17)
keep.scatter_(1, order[:, :4], 0)
joint = base - probs_keep(keep)
print("sum of four single drops:", round(single.mean().item(), 3))
print("joint drop of the same four:", round(joint.mean().item(), 3))
print("non-additivity (sum - joint):", round((single - joint).mean().item(), 3))


# %%
# The class token is a summary, but it is not the whole image. Compare deleting
# it with deleting the four best patch tokens.
keep = torch.ones(B, 17)
keep[:, 0] = 0
class_only = base - probs_keep(keep)
keep = torch.ones(B, 17)
keep[:, 1:] = 0
patches_only = base - probs_keep(keep)
print("class token alone:", round(class_only.mean().item(), 3))
print("all 16 patch tokens:", round(patches_only.mean().item(), 3))
