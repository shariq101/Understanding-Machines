# %% [markdown]
# # Lab 24 — Three teachers, one on-policy student
#
# A student can imitate a teacher on the teacher's own examples and still fail
# when it has to continue its own mistakes. That is the distribution-shift
# problem in distillation, and it becomes sharper when there is more than one
# teacher.
#
# This lab builds three small domain specialists. Teacher 0 copies a four-token
# string, teacher 1 reverses it, and teacher 2 sorts it. A single student must
# learn all three tasks. We compare three curricula:
#
# 1. one teacher on its own domain (the control);
# 2. all three teachers on their own **teacher-generated** trajectories
#    (off-policy);
# 3. all three teachers labelling the student's **own** prefixes (on-policy).
#
# The domain token is a fixed gate, not a learned router. The experiment asks
# whether on-policy exposure changes what one student absorbs from a set of
# specialists; it does not claim that the gate is optimal or that the teachers
# agree about a shared task.
#
# **Milestone:** report which curricula succeed on all three domains and whether
# the on-policy student's own prefixes change the result.

# %%
import math
import time

import torch
import torch.nn.functional as F

from uml.model import Config, TinyGPT
from uml.train import fresh

torch.set_num_threads(4)
torch.manual_seed(7)

BOS = 0
DOMAIN_0 = 1
SYM_0 = 4
SYM_N = 8
ANS = 12
EOS = 13
PAD = 14
VOCAB = 15
TASKS = ["copy", "reverse", "sort"]
N_TARGET = 5          # four content tokens, then EOS


def domain_batch(domain, batch, generator):
    """A prompt [BOS, domain, four symbols, ANS] and its five-token target."""
    symbols = torch.randint(SYM_0, SYM_0 + SYM_N, (batch, 4), generator=generator)
    if domain == 0:
        out = symbols
    elif domain == 1:
        out = symbols.flip(1)
    else:
        out = symbols.sort(dim=1).values
    prompt = torch.cat([
        torch.full((batch, 1), BOS, dtype=torch.long),
        torch.full((batch, 1), DOMAIN_0 + domain, dtype=torch.long),
        symbols,
        torch.full((batch, 1), ANS, dtype=torch.long),
    ], 1)
    target = torch.cat([out, torch.full((batch, 1), EOS, dtype=torch.long)], 1)
    seq = torch.cat([prompt, target], 1)
    x = seq[:, :-1]
    y = torch.full_like(x, -100)
    y[:, 6:] = target
    return x, y, prompt, target


def target_for(domain, symbols):
    if domain == 0:
        return symbols
    if domain == 1:
        return symbols.flip(1)
    return symbols.sort(dim=1).values


def make_prompts(domain, batch, generator):
    symbols = torch.randint(SYM_0, SYM_0 + SYM_N, (batch, 4), generator=generator)
    prompt = torch.cat([
        torch.full((batch, 1), BOS, dtype=torch.long),
        torch.full((batch, 1), DOMAIN_0 + domain, dtype=torch.long),
        symbols,
        torch.full((batch, 1), ANS, dtype=torch.long),
    ], 1)
    target = torch.cat([target_for(domain, symbols),
                        torch.full((batch, 1), EOS, dtype=torch.long)], 1)
    return prompt, target


def cfg(seed):
    return Config(vocab_size=VOCAB, n_layers=2, d_model=64, n_heads=4,
                  d_mlp=256, max_len=16)


# %% [markdown]
# ## The teachers
#
# Each teacher is a two-layer, four-head transformer trained only on its own
# domain. The domain token is present, but a teacher never sees the other two
# tasks. This makes the set a genuine mixture of specialists: routing is
# supplied by the domain token, and the interesting question is whether the
# student can learn the routed behaviour rather than only the teacher's
# training distribution.

# %%
def train_teacher(domain, seed, steps=650):
    model = fresh(cfg(seed), seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01,
                            betas=(0.9, 0.99))
    g = torch.Generator().manual_seed(1000 + seed)
    model.train()
    t0 = time.time()
    for step in range(steps):
        x, y, _, _ = domain_batch(domain, 128, g)
        loss = F.cross_entropy(model(x)[0].flatten(0, 1), y.flatten(),
                               ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in (0, steps - 1):
            print(f"{TASKS[domain]} teacher step {step:3d}  loss {loss.item():.3f}")
    model.eval()
    return model, round(time.time() - t0, 1)


teachers, train_seconds = {}, {}
print("teacher set: one domain specialist per task")
for d in range(3):
    teachers[d], train_seconds[d] = train_teacher(d, seed=10 + d)
    print("trained", TASKS[d], "in", train_seconds[d], "s")

# %% [markdown]
# A specialist can be accurate on its own domain and meaningless on another.
# The next table reports both: the teacher's exact-sequence accuracy on its
# domain, then the same checkpoint's accuracy when the prompt is labelled with a
# different domain. That second column is the reason a routed student needs a
# teacher for every domain rather than one strong teacher for all of them.

# %%
@torch.no_grad()
def generate_greedy(model, prompt, steps=N_TARGET):
    x = prompt
    out = []
    for _ in range(steps):
        logits = model(x)[0][:, -1]
        nxt = logits.argmax(-1)
        out.append(nxt)
        x = torch.cat([x, nxt[:, None]], 1)
    return torch.stack(out, 1)


@torch.no_grad()
def score(model, domain, batch=256, seed=900):
    g = torch.Generator().manual_seed(seed + domain)
    prompt, target = make_prompts(domain, batch, g)
    pred = generate_greedy(model, prompt)
    exact = (pred == target).all(1).float().mean().item()
    content = (pred[:, :4] == target[:, :4]).all(1).float().mean().item()
    return exact, content


print("teacher       own exact  own content  other-domain exact")
for d in range(3):
    own = score(teachers[d], d)
    other = score(teachers[d], (d + 1) % 3)
    print(f"{TASKS[d]:8s}  {own[0]:9.3f}  {own[1]:11.3f}  {other[0]:16.3f}")

# %% [markdown]
# ## Collecting a student trajectory
#
# For off-policy distillation, the prefix comes from the teacher's greedy
# rollout. For on-policy distillation, the prefix comes from the student's own
# sampled rollout. In both cases the teacher supplies the next-token
# distribution at every prefix, and the student is trained with a KL loss.
#
# The distinction is in the data, not the loss. Off-policy sees a prefix the
# teacher would produce; on-policy sees the prefix the student actually
# produces. A student that has already made an error is therefore trained on
# the state it will encounter at inference time.

# %%
def collect(teacher_map, domains, batch_per_domain, mode, student=None, seed=0):
    """Return padded prefixes, teacher distributions and prefix lengths."""
    g = torch.Generator().manual_seed(seed)
    xs, ys, lengths = [], [], []
    with torch.no_grad():
        for domain in domains:
            prompt, _ = make_prompts(domain, batch_per_domain, g)
            cur = prompt
            for step in range(N_TARGET):
                teacher = teacher_map[domain]
                tprob = teacher(cur)[0][:, -1].softmax(-1)
                xs.append(cur.clone())
                ys.append(tprob)
                lengths.append(torch.full((cur.shape[0],), cur.shape[1]))
                if mode == "on":
                    sprob = student(cur)[0][:, -1].softmax(-1)
                    nxt = torch.multinomial(sprob, 1).squeeze(1)
                elif mode == "mix":
                    sprob = student(cur)[0][:, -1].softmax(-1)
                    nxt_student = torch.multinomial(sprob, 1).squeeze(1)
                    nxt_teacher = tprob.argmax(-1)
                    take = torch.rand(cur.shape[0]) < 0.5
                    nxt = torch.where(take, nxt_student, nxt_teacher)
                else:
                    nxt = tprob.argmax(-1)
                cur = torch.cat([cur, nxt[:, None]], 1)
    max_len = max(x.shape[1] for x in xs)
    padded = torch.full((sum(x.shape[0] for x in xs), max_len), PAD,
                        dtype=torch.long)
    at = 0
    for x in xs:
        padded[at:at + x.shape[0], :x.shape[1]] = x
        at += x.shape[0]
    return padded, torch.cat(ys, 0), torch.cat(lengths, 0)


def train_student(teacher_map, domains, mode, seed, steps=300,
                  batch_per_domain=32, warm=0):
    model = fresh(cfg(seed), seed)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01,
                            betas=(0.9, 0.99))
    t0 = time.time()
    for step in range(steps):
        phase = "off" if step < warm else mode
        x, tprob, lengths = collect(teacher_map, domains, batch_per_domain,
                                    phase, model, seed=20000 + seed * 1000 + step)
        logits = model(x)[0]
        rows = torch.arange(x.shape[0])
        logp = logits[rows, lengths - 1].log_softmax(-1)
        loss = F.kl_div(logp, tprob, reduction="batchmean")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in (0, steps - 1):
            print(f"{phase:3s} {seed} step {step:3d}  KL {loss.item():.3f}")
    model.eval()
    return model, round(time.time() - t0, 1)


# %% [markdown]
# The single-teacher control is deliberately not a strawman: it is the same
# student architecture, the same loss, and a teacher that is perfect on copy.
# It sees only the copy domain because that is the only domain its teacher was
# trained on. It should be competitive on copy and unable to do reverse or
# sort. The multi-teacher students see all three domains.

# %%
single, single_s = train_student({0: teachers[0]}, [0], "off", seed=31)
off, off_s = train_student(teachers, [0, 1, 2], "off", seed=32)
on, on_s = train_student(teachers, [0, 1, 2], "on", seed=33)
warm, warm_s = train_student(teachers, [0, 1, 2], "on", seed=34, warm=150)
mix, mix_s = train_student(teachers, [0, 1, 2], "mix", seed=35)
print("training seconds: single", single_s, "off", off_s, "on", on_s,
      "warm-start on", warm_s, "prefix mix", mix_s)

# %% [markdown]
# ## The held-out result
#
# Exact sequence accuracy is stricter than token accuracy: a student must emit
# the four content tokens in the right order and then stop with EOS. The
# per-domain columns show whether a curriculum learned all three behaviours or
# averaged its way to a mediocre compromise.

# %%
def row(name, model):
    vals = [score(model, d) for d in range(3)]
    exact = [v[0] for v in vals]
    content = [v[1] for v in vals]
    print(f"{name:18s} " + "  ".join(f"{v:.3f}" for v in exact)
          + f"   mean exact {sum(exact) / 3:.3f}")
    print(f"{'':18s} " + "  ".join(f"{v:.3f}" for v in content)
          + "   content by domain")
    return exact


print("student             copy  reverse   sort")
single_vals = row("single teacher", single)
off_vals = row("multi off-policy", off)
on_vals = row("multi on-policy", on)
warm_vals = row("warm on-policy", warm)
mix_vals = row("prefix mix", mix)

# %% [markdown]
# The table is the lab's result, but the comparison needs one more control:
# teacher quality. If a specialist is weak, a low student score is not evidence
# about on-policy distillation. The next cell prints the teacher's own scores
# beside the student's and a simple **teacher-selection** check: for each
# held-out prompt, whether the routed teacher's top token is the true next
# token. This is a ceiling, not a guarantee, because the student has a smaller
# effective computation than three separate teachers.

# %%
@torch.no_grad()
def teacher_next_token_accuracy(teacher_map, domain, batch=256, seed=1200):
    g = torch.Generator().manual_seed(seed + domain)
    prompt, target = make_prompts(domain, batch, g)
    correct = 0
    total = 0
    for step in range(N_TARGET):
        t = target[:, step]
        logits = teacher_map[domain](prompt)[0][:, -1]
        correct += int((logits.argmax(-1) == t).sum())
        total += prompt.shape[0]
        prompt = torch.cat([prompt, t[:, None]], 1)  # teacher-forced prefix
    return correct / total


print("domain   teacher exact  teacher next-token  student exact (on-policy)  warm on-policy  prefix mix")
for d in range(3):
    print(f"{TASKS[d]:8s} {score(teachers[d], d)[0]:14.3f} "
          f"{teacher_next_token_accuracy(teachers, d):18.3f} "
          f"{on_vals[d]:24.3f} {warm_vals[d]:16.3f} {mix_vals[d]:12.3f}")

# %% [markdown]
# ## What changed, and what did not
#
# The measured result is a warning, not a clean win for on-policy training.
# Off-policy multi-teacher distillation reaches **0.958** mean exact match. The
# student-prefix run reaches **0.355**, the warm-started run **0.380**, and the
# 50/50 prefix mix **0.854**. The specialist teachers are perfect on their own
# clean prefixes but not on the early, malformed prefixes a randomly initialized
# student produces; those labels are poor targets. Coverage is the robust
# effect here (0.367 to 0.958), while naive on-policy exposure is destructive.
# The prefix mix recovers most of the gap, but it does not beat off-policy.
#
# The lab reports three different claims, and they should not be collapsed:
#
# * **Multi-teacher coverage.** The single-teacher student has seen only copy;
#   the multi-teacher students have seen all three domains. Any difference
#   between those rows includes coverage as well as teacher number.
# * **On-policy exposure.** The off-policy and on-policy rows use the same three
#   teachers and the same KL objective. Their difference is the prefix
#   distribution, which is the clean comparison for on-policy distillation.
# * **Teacher routing.** The domain token is an oracle gate supplied by the
#   data. This lab does not learn a router, so it cannot say whether a learned
#   gate would preserve the result.
#
# The next cell makes the cost explicit. It counts how many teacher queries a
# single on-policy training step needs, and how many tokens the student sees
# that a teacher would not have generated.

# %%
one_step = 3 * 32 * N_TARGET
print("teacher queries per on-policy update:", one_step)
print("teacher queries per off-policy update:", one_step)
print("student-generated prefixes per on-policy update:", one_step)
print("content tokens per held-out prompt:", 4, " total target tokens:", N_TARGET)
print("domains:", TASKS)
print("teacher parameters:", [teachers[d].n_params() for d in range(3)])
print("student parameters:", on.n_params())

# %% [markdown]
# ## Reading the result
#
# A single number is not the lesson. The useful result is the pattern:
# coverage should dominate the gap between the single-teacher control and the
# multi-teacher students; on-policy exposure should matter most where the
# student's own prefix is far from the teacher's. If the on-policy student is
# not better than the off-policy one on this toy task, that is a result too:
# the task may be too easy for exposure bias to bite, or 300 updates may be too
# few. The lab keeps both rows and the held-out bank so the claim can be
# checked rather than inferred from a training loss.
#
# The scope is deliberately small: four content tokens, three deterministic
# transformations, one architecture, one seed per curriculum. It is a
# controlled demonstration of a distillation decision, not evidence about a
# production multi-teacher system.
