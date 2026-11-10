# %% [markdown]
# # Lab 8 — Find the induction circuit
#
# A head table is a list of suspects. We measure previous-token, same-token,
# successor, and final-query attention for every head, then ablate the two
# strongest induction candidates. The ablation is the causal test.
#
# **Milestone:** name one head whose removal costs accuracy and one whose removal does not.

# %%
import torch
from uml.model import load
from uml.data import induction_gap_prompt, induction_gap_batch
from uml.interp import head_table_many, ablate_head, loss_on, logit_lens

torch.set_num_threads(4)
model, _ = load("checkpoints/induction.pt")
L, gap = 10, 5
g = torch.Generator().manual_seed(31)
prompts = []
for _ in range(16):
    t = torch.randint(1, 16, (L,), generator=g)
    f = torch.randint(1, 16, (gap,), generator=g)
    prompts.append(induction_gap_prompt(t.tolist(), f.tolist()))

rows = head_table_many(model, prompts, L)
print("layer head   prev   self    bos   same   succ  final")
for r in rows:
    print(f"{int(r['layer'])}     {int(r['head'])}   {r['prev']:.3f}  {r['self']:.3f}  {r['bos']:.3f}  {r['same']:.3f}  {r['succ']:.3f}  {r['final']:.3f}")

# %%
x, y = induction_gap_batch(256, L, gap, 16, torch.Generator().manual_seed(4))
base = (model(x)[0].argmax(-1)[y != -100] == y[y != -100]).float().mean().item()
print("base accuracy:", round(base, 3), " loss:", round(loss_on(model, x, y), 4))
for layer, head in [(1, 0), (1, 3), (0, 3), (0, 2)]:
    with ablate_head(model, layer, head):
        acc = (model(x)[0].argmax(-1)[y != -100] == y[y != -100]).float().mean().item()
        lo = loss_on(model, x, y)
    print(f"ablate L{layer}H{head}: accuracy={acc:.3f}  loss={lo:.4f}")

# %%
seq = [3, 7, 11, 5, 9, 2, 14, 6, 8, 4, 10, 13]
filler = [12, 1, 15, 6, 9]
lens = logit_lens(model, induction_gap_prompt(seq, filler), topk=3)
print("logit lens, final position:")
for row in lens:
    print(" layer", row["layer"], " tokens", row["tokens"], " probs", row["probs"])

# %% [markdown]
# The two layer-1 heads with high `succ` and `final` scores are the induction
# heads. Ablating either costs accuracy, but not all of it: the model has
# redundant paths and the layer-0 heads contribute. The logit lens is already
# confident at layer 2, which locates the copy in the first two layers.
