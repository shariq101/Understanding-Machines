# %% [markdown]
# # Lab 20 — The agent loop, and the arithmetic of reliability
#
# An agent is a model plus a loop. This lab makes the loop explicit with a
# hand-written policy and a calculator, then measures what compounding does to
# success. The model is deliberately simple: the lesson is the loop, not the
# policy.
#
# **Milestone:** report success versus chain length for three per-step
# reliabilities, the gain from re-deriving at a checkpoint, and the cost of
# replacing an exact tool with an approximate model.

# %%
import random

OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
}


def make_chain(n, rng, start=3):
    """A deterministic arithmetic chain and its correct final value."""
    ops, x = [], start
    for _ in range(n):
        name = rng.choice(list(OPS))
        arg = rng.randint(1, 4)
        ops.append((name, arg))
        x = OPS[name](x, arg)
    return ops, x


def choose(op, p, rng):
    """The policy picks the intended operation with probability p."""
    if rng.random() < p:
        return op
    wrong = [o for o in (("+", 1), ("-", 1), ("*", 2)) if o != op]
    return rng.choice(wrong)


def run(ops, p, tool=True, checkpoint=0, noise=0.5, rng=None):
    """Run one episode. Returns success, extra tool calls, and the final state."""
    rng = rng or random.Random(0)
    state = 3.0
    ckpt, ckpt_i = state, 0
    extra = 0
    for i, op in enumerate(ops):
        name, arg = choose(op, p, rng)
        state = OPS[name](state, arg)
        if not tool:
            state += rng.gauss(0, noise)
        if checkpoint and (i + 1) % checkpoint == 0:
            # Re-derive the last block from the checkpoint and compare. This is
            # a self-consistency check, not an oracle: a repeated error can
            # survive a second run.
            redo, s = 0.0, ckpt
            for j in range(ckpt_i, i + 1):
                n2, a2 = choose(ops[j], p, rng)
                s = OPS[n2](s, a2)
                if not tool:
                    s += rng.gauss(0, noise)
                redo += 1
            extra += redo
            if abs(s - state) > 1e-6:
                state = s
            ckpt, ckpt_i = state, i + 1
    return state, extra


def correct_final(ops, start=3.0):
    x = start
    for name, arg in ops:
        x = OPS[name](x, arg)
    return x


# %%
# Compounding: with independent per-step errors, success falls like p^n. The
# simulation is a check on the arithmetic, not a new result.
print("length  p=0.99  p=0.97  p=0.90")
for n in (4, 8, 16, 32):
    row = []
    for p in (0.99, 0.97, 0.90):
        ok = 0
        for t in range(300):
            rng = random.Random(1000 + t)
            ops, _ = make_chain(n, rng)
            goal = correct_final(ops)
            state, _ = run(ops, p, rng=rng)
            ok += abs(state - goal) < 1e-6
        row.append(ok / 300)
    print(f"{n:6d}  " + "  ".join(f"{v:6.3f}" for v in row))


# %%
# Checkpointing: every k steps, re-derive the block and keep the re-run if the
# two disagree. It costs tool calls; does it buy success?
print("\ncheckpoint  success  extra calls")
for k in (0, 4, 2):
    ok, extra = 0, 0
    for t in range(300):
        rng = random.Random(2000 + t)
        ops, _ = make_chain(16, rng)
        goal = correct_final(ops)
        state, ex = run(ops, 0.97, checkpoint=k, rng=rng)
        ok += abs(state - goal) < 1e-6
        extra += ex
    print(f"{k:10d}  {ok / 300:7.3f}  {extra / 300:11.1f}")

# A self-check is not an oracle. If a verifier knows the correct prefix with
# probability q and resets at each checkpoint, the gain depends on q. Report
# the upper bound so the self-check result has a scale.
def run_oracle(ops, p, q, checkpoint, rng):
    state = 3.0
    for i, op in enumerate(ops):
        name, arg = choose(op, p, rng)
        state = OPS[name](state, arg)
        if checkpoint and (i + 1) % checkpoint == 0 and rng.random() < q:
            prefix = correct_final(ops[: i + 1])
            if abs(state - prefix) > 1e-6:
                state = prefix
    return state

print("\nverifier q  success")
for q in (0.9, 0.99):
    ok = 0
    for t in range(300):
        rng = random.Random(5000 + t)
        ops, _ = make_chain(16, rng)
        goal = correct_final(ops)
        state = run_oracle(ops, 0.97, q, 4, rng)
        ok += abs(state - goal) < 1e-6
    print(f"{q:10.2f}  {ok / 300:7.3f}")


# %%
# Tool versus model: the calculator is exact conditional on choosing the right
# operation. A model that approximates each result accumulates noise.
print("\nmode   success")
for mode, tool in (("tool", True), ("model", False)):
    ok = 0
    for t in range(300):
        rng = random.Random(3000 + t)
        ops, _ = make_chain(16, rng)
        goal = correct_final(ops)
        state, _ = run(ops, 0.97, tool=tool, noise=0.2, rng=rng)
        ok += abs(state - goal) < 0.5
    print(f"{mode:5s}  {ok / 300:7.3f}")


# %%
# A wrong tool call is not the same as a wrong arithmetic step. If the policy
# selects the right operation but passes a wrong argument 2% of the time, the
# failure is silent: the trace still looks plausible.
print("\nargument error rate  success")
for arg_p in (0.0, 0.02, 0.10):
    ok = 0
    for t in range(300):
        rng = random.Random(4000 + t)
        ops, _ = make_chain(12, rng)
        goal = correct_final(ops)
        state = 3.0
        for name, arg in ops:
            if rng.random() < arg_p:
                arg = arg + rng.choice((-1, 1))
            state = OPS[name](state, arg)
        ok += abs(state - goal) < 1e-6
    print(f"{arg_p:19.2f}  {ok / 300:7.3f}")
