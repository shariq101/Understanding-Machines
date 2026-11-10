# %% [markdown]
# # Lab 11 — Reasoning and test-time compute
#
# Parity cannot be computed in one pass by a fixed-depth model, but it can be
# written down step by step. We load two models trained on length-10 parity —
# one direct, one with a running-scratchpad target — and test both at the trained
# length and at longer lengths.
#
# **Milestone:** explain why the chain-of-thought model is perfect at length 10 and near chance at length 14.

# %%
import torch
from uml.model import load
from uml.data import parity_prompt

torch.set_num_threads(4)
direct, _ = load("checkpoints/parity_2l_direct.pt")
cot, _ = load("checkpoints/parity_2l_cot.pt")
print("direct parameters:", direct.n_params(), " CoT parameters:", cot.n_params())


def evaluate(model, bits, cot_mode):
    x = parity_prompt(bits, cot_mode)
    with torch.no_grad():
        if cot_mode:
            for _ in range(len(bits)):
                logits, _ = model(x)
                nxt = logits[0, -1].argmax(-1, keepdim=True)
                x = torch.cat([x, nxt[None]], 1)
            return int(x[0, -1].item() - 2)
        return int(model(x)[0][0, -1].argmax().item() - 2)


for L in (10, 14, 18):
    g = torch.Generator().manual_seed(100 + L)
    cd = cc = 0
    n = 80
    for _ in range(n):
        bits = torch.randint(0, 2, (L,), generator=g).tolist()
        target = sum(bits) % 2
        cd += evaluate(direct, bits, False) == target
        cc += evaluate(cot, bits, True) == target
    print(f"length {L:2d}  direct {cd/n:.3f}  chain-of-thought {cc/n:.3f}")

# %% [markdown]
# The direct model is at chance at every length: the computation does not fit in
# one forward pass. The chain-of-thought model is perfect at length 10, the only
# length it saw during training, and near chance at 14 and 18. It learned a
# length-specific unrolling of the algorithm, not the algorithm itself.

# %%
# The scratchpad is a system: the same model can be sampled repeatedly.
# At length 10 the model is deterministic, so voting changes nothing; the
# useful quantity is the compute cost.
print("forward passes for a length-10 CoT answer:", 1 + 10)
print("forward passes for a length-18 CoT answer:", 1 + 18, "(and it is still wrong)")
