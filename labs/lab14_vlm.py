# %% [markdown]
# # Lab 14 — Build a tiny VLM
#
# A vision encoder, a projector, and a frozen language model. The image becomes
# a prefix of tokens; the question follows; the answer is read at the final
# position. This is the LLaVA pattern at a scale where every activation fits in
# memory.
#
# **Milestone:** report the VLM's accuracy by question type, then locate the
# answer in depth with the logit lens.

# %%
from pathlib import Path
import torch

from uml.multimodal import (load_clip, TinyVLM, train_vlm, save_vlm, load_vlm,
                            vlm_accuracy, vlm_logit_lens, attention_maps,
                            image_token_ablation, question_batch, scene,
                            QUESTIONS, WORD_ID, WORDS, BOS)

torch.set_num_threads(4)
clip = load_clip("checkpoints/clip.pt")
CKPT = Path("checkpoints/vlm.pt")

if CKPT.exists():
    model = load_vlm(clip.vision, CKPT)
    print("loaded", CKPT)
else:
    model = TinyVLM(clip.vision, n_layers=4)
    print("trainable parameters:", model.n_params())
    train_vlm(model, steps=1800, batch=64, lr=2e-3)
    save_vlm(model, CKPT)

print("trainable parameters:", model.n_params())
print("accuracy:", vlm_accuracy(model, n=512))
print("by question:", vlm_accuracy(model, n=512, by_type=True))

# %%
# Read the answer at every layer.  The correct colour should become decodable
# before the final block if the model computes it early.
q = torch.tensor([[BOS] + [WORD_ID[w] for w in QUESTIONS["color"]] + [WORD_ID["="]]])
img = scene("circle", "red", "left")[None]
for row in vlm_logit_lens(model, img, q, topk=4):
    print("layer", row["layer"], [(WORDS[i], p) for i, p in zip(row["ids"], row["probs"])])

# %%
# Remove one image token at a time.  A token that carries the colour should
# hurt the colour answer when it is removed.
g = torch.Generator().manual_seed(3)
imgs, qs, ans, meta = question_batch(256, g, types=("color",))
base = (model(imgs, qs)[0][:, -1].argmax(-1) == ans).float().mean().item()
rows = []
for tok in range(17):
    logits = image_token_ablation(model, imgs, qs, tok)
    acc = (logits.argmax(-1) == ans).float().mean().item()
    rows.append((tok, round(acc, 3)))
print("base colour accuracy:", round(base, 3))
print("largest drops:", sorted(rows, key=lambda r: r[1])[:5])

# %%
# Where does the final query read?  The image tokens are positions 0..16, the
# text starts at 17.  Report the mass and the top patch for each position.
for pos in ("left", "center", "right"):
    img = scene("circle", "red", pos)[None]
    att = attention_maps(model, img, q)
    mass = [round(att[l][:, -1, :17].sum(-1).mean().item(), 3) for l in range(len(att))]
    top = att[0][:, -1, :17].mean(0).topk(3).indices.tolist()
    print(pos, "image mass by layer:", mass, "top patches:", top)
