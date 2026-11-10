# %% [markdown]
# # Lab 7 — Train a tiny transformer
#
# The model every later lab dissects. It is a two-layer decoder with four heads
# and a 64-dimensional residual stream. The task is a string, a filler, and the
# same string again: at each second-copy position the model must copy the next
# token from the first copy.
#
# **Milestone:** report the parameter count and the held-out accuracy.

# %%
from pathlib import Path
import torch
from uml.model import Config, TinyGPT, load
from uml.train import fresh, train_lm, accuracy, save
from uml.data import induction_gap_batch

torch.set_num_threads(4)
CKPT = Path("checkpoints/induction.pt")

if CKPT.exists():
    model, ckpt = load(CKPT)
    print("loaded", CKPT, "task:", ckpt.get("extra", {}).get("task"))
else:
    cfg = Config(vocab_size=17, n_layers=2, d_model=64, n_heads=4, d_mlp=256, max_len=40)
    model = fresh(cfg, 5)
    g = torch.Generator().manual_seed(21)
    def gen():
        while True:
            L = int(torch.randint(6, 13, (1,), generator=g))
            gap = int(torch.randint(2, 8, (1,), generator=g))
            yield induction_gap_batch(128, L, gap, 16, g)
    train_lm(model, gen(), steps=2500, lr=3e-3)
    save(model, CKPT, {"task": "induction-gap"})

print("config:", model.cfg.to_dict())
print("parameters:", model.n_params())

# %%
for L, gap in [(8, 4), (10, 5), (12, 3)]:
    x, y = induction_gap_batch(256, L, gap, 16, torch.Generator().manual_seed(5))
    print(f"L={L:2d} gap={gap}  accuracy={accuracy(model, x, y):.3f}")

# %%
seq = [3, 7, 11, 5, 9, 2, 14, 6, 8, 4, 10, 13]
filler = [12, 1, 15, 6, 9]
from uml.data import induction_gap_prompt
p = induction_gap_prompt(seq, filler)
with torch.no_grad():
    logits, _ = model(p)
print("prompt length:", p.shape[1], " next-token prediction:", int(logits[0, -1].argmax()))
print("expected next token:", seq[-1])

# %% [markdown]
# The model is small enough to train on a laptop CPU in about a minute. It is
# also large enough to contain a real circuit: two induction heads, a previous
# token head, and a set of layer-0 heads that support them. The next lab finds
# them.
