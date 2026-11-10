# %% [markdown]
# # Lab 9 — Probe and patch
#
# A probe asks whether a property is decodable. A patch asks whether an
# activation is sufficient for the behaviour. They are different questions, and
# this lab answers both with numbers.
#
# **Milestone:** report the probe accuracy, the logit difference before and after the patch, and the fraction restored.

# %%
import torch
from uml.model import load
from uml.data import induction_gap_prompt
from uml.interp import residual_cache, train_probe, patch_residual

torch.set_num_threads(4)
model, _ = load("checkpoints/induction.pt")
L, gap = 10, 5

# Probe: is the final token's identity linearly decodable from layer 1?
g = torch.Generator().manual_seed(5)
X, y = [], []
for _ in range(300):
    t = torch.randint(1, 16, (L,), generator=g)
    f = torch.randint(1, 16, (gap,), generator=g)
    p = induction_gap_prompt(t.tolist(), f.tolist())
    X.append(residual_cache(model, p)[1][-1])
    y.append(int(t[-2]))
X, y = torch.stack(X), torch.tensor(y)
acc, W, b = train_probe(X, y, steps=400)
print("probe: token identity at layer 1, final position")
print("  accuracy:", acc, " classes:", int(y.max()) + 1)

# %%
# Patch: replace the corrupted residual at the copy target with the clean one.
t = torch.randint(1, 16, (L,), generator=torch.Generator().manual_seed(8))
f = torch.randint(1, 16, (gap,), generator=torch.Generator().manual_seed(9))
clean = induction_gap_prompt(t.tolist(), f.tolist())
t2 = t.clone(); t2[-1] = (t[-1] % 15) + 1
corrupt = induction_gap_prompt(t2.tolist(), f.tolist())
answer = (int(t[-1]), int(t2[-1]))
for layer in (0, 1):
    res = patch_residual(model, clean, corrupt, layer=layer, positions=[L], answer=answer)
    print(f"patch layer {layer}: before={res['before']:+.2f}  after={res['after']:+.2f}  "
          f"clean={res['clean']:+.2f}  restored={res['restored']:.3f}")

# %% [markdown]
# The probe is perfect, which is expected: the residual stream must carry the
# token identity to the unembedding. The patch is the interesting result. At the
# layer where the induction head writes the copied token, replacing the corrupted
# activation with the clean one restores most of the logit difference. That is
# sufficiency at a site, not a proof that the site is the only path.
