# %% [markdown]
# # Lab 12 — Capstone: the circuit card
#
# One model, one task, four kinds of evidence. This lab runs the whole protocol
# and prints a card: head roles, necessity, sufficiency, and a dictionary check.
# Use it as the template for your own investigation.
#
# **Milestone:** write a two-sentence circuit claim and one falsification condition.

# %%
import torch
from uml.model import load
from uml.data import induction_gap_prompt, induction_gap_batch
from uml.interp import head_table_many, ablate_head, loss_on, residual_cache, train_probe, patch_residual
from uml.sae import train_sae

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
top = sorted(rows, key=lambda r: -(r["succ"] + r["final"]))[:3]
print("top heads by successor + final attention:")
for r in top:
    print(f"  L{int(r['layer'])}H{int(r['head'])}  succ={r['succ']:.3f}  final={r['final']:.3f}")

# %%
x, y = induction_gap_batch(256, L, gap, 16, torch.Generator().manual_seed(4))
base = (model(x)[0].argmax(-1)[y != -100] == y[y != -100]).float().mean().item()
print("behaviour: accuracy", round(base, 3), " loss", round(loss_on(model, x, y), 4))
for r in top[:2]:
    with ablate_head(model, int(r["layer"]), int(r["head"])):
        acc = (model(x)[0].argmax(-1)[y != -100] == y[y != -100]).float().mean().item()
    print(f"necessity: ablate L{int(r['layer'])}H{int(r['head'])} -> accuracy {acc:.3f}")

# %%
t = torch.randint(1, 16, (L,), generator=torch.Generator().manual_seed(8))
f = torch.randint(1, 16, (gap,), generator=torch.Generator().manual_seed(9))
clean = induction_gap_prompt(t.tolist(), f.tolist())
t2 = t.clone(); t2[-1] = (t[-1] % 15) + 1
corrupt = induction_gap_prompt(t2.tolist(), f.tolist())
res = patch_residual(model, clean, corrupt, layer=1, positions=[L], answer=(int(t[-1]), int(t2[-1])))
print("sufficiency: patch layer 1 at the copy target")
print("  restored fraction:", res["restored"])
X, yy = [], []
for _ in range(200):
    tt = torch.randint(1, 16, (L,), generator=g)
    ff = torch.randint(1, 16, (gap,), generator=g)
    X.append(residual_cache(model, induction_gap_prompt(tt.tolist(), ff.tolist()))[1][-1])
    yy.append(int(tt[-2]))
accp, _, _ = train_probe(torch.stack(X), torch.tensor(yy), steps=300)
print("decodability: probe on layer-1 residual:", accp)

# %%
store = {}
acts = []
with torch.no_grad():
    for _ in range(16):
        tt = torch.randint(1, 16, (L,), generator=g)
        ff = torch.randint(1, 16, (gap,), generator=g)
        model(induction_gap_prompt(tt.tolist(), ff.tolist()), store=store)
        acts.append(store[(1, "mlp")][0])
acts = torch.cat(acts, 0).reshape(-1, 256)
_, s = train_sae(acts, d_hidden=256, l1=1.0, steps=400)
print("dictionary: FVU", s["fvu"], " L0", s["l0"])

# %% [markdown]
# **Circuit claim (template).** In this two-layer model the induction behaviour is
# carried mainly by the layer-1 heads with high successor attention; ablating
# them costs accuracy, and patching their residual site restores most of the
# clean logit difference. **Falsification:** if a model with those heads ablated
# still matches baseline accuracy on a gap-varied test set, the claim is false.
