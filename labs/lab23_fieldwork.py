# %% [markdown]
# # Lab 23 — Audit the fieldwork
#
# Part XIV reports four experiments on Pythia-410M and Pythia-1.4B. This lab
# does to those chapters what Lab 21 does to the atlas: it loads the stored
# JSON, recomputes every headline number, and checks that the claims in the
# prose follow from the data. It also shows the two controls that change the
# story — the bottom-five head ablation and the random steering direction.
#
# **Milestone:** reproduce the four claim checks, report the effect sizes, and
# explain which claim is a null result rather than a confirmation.

# %%
import json
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "experiments").exists():
    ROOT = ROOT.parent
RAW = ROOT / "experiments" / "results" / "kaggle-v4"
R = {tag: json.loads((RAW / f"{tag}.json").read_text()) for tag in ("410m", "1.4b")}

print("model                params   layers heads width dtype")
for tag, r in R.items():
    f = r["facts"]
    print(f"{tag:6s} {f['model']:22s} {f['parameters']/1e9:6.2f}B "
          f"{f['layers']:6d} {f['heads']:5d} {f['hidden']:5d} {f['dtype']}")


# %%
# Claim 1. The induction heads that top the attention table are not
# individually necessary, while the layer is. The test is the damage on the
# repeated task relative to the random-string control.
def abl(r, name):
    return next(x for x in r["experiments"]["induction"]["ablations"] if x["name"] == name)


print("\ninduction: loss damage on repeated strings (nats)")
print("model   top1    top5    random5 bottom5 layer-all")
for tag, r in R.items():
    vals = [abl(r, n)["damage_loss_repeated"] for n in
            ("top_1", "top_5", "random_5", "bottom_5", "best_layer_all")]
    print(f"{tag:6s} " + " ".join(f"{v:7.3f}" for v in vals))

for tag, r in R.items():
    top5, allh = abl(r, "top_5"), abl(r, "best_layer_all")
    assert top5["damage_loss_repeated"] < allh["damage_loss_repeated"], tag
print("claim 1: the five top heads cost less than the whole layer in both models")


# %%
# Claim 2. IOI is distributed: the best single head removes only a small
# fraction of the clean-minus-corrupt gap, and the top five together remove
# more but still not the behaviour.
print("\nIOI: share of the clean-minus-corrupt gap removed")
print("model   gap    best head  top five  all heads  top five by attention")
for tag, r in R.items():
    e = r["experiments"]["ioi"]
    gap = e["baseline"]["gap"]
    best = max(h["drop"] for h in e["top_heads_by_ablation"])
    comb = {c["name"]: c["drop"] for c in e["combined_ablations"]}
    print(f"{tag:6s} {gap:6.2f} {best/gap:9.1%} {comb['top5_by_ablation']/gap:9.1%} "
          f"{comb['all_heads_at_best_layer']/gap:10.1%} {comb['top5_by_attention']/gap:20.1%}")
    assert best / gap < 0.10
    assert comb["all_heads_at_best_layer"] / gap < 0.25
print("claim 2: no single head, and not all heads at one layer, is the circuit")


# %%
# Claim 3. The lens reads a state, not a computation. The true token rises in
# rank, but the final layer is still a distribution. The numbers to check are
# the endpoints and the final top-1 rate.
print("\nlogit lens: mean rank of the true token")
print("model   embedding  final   top-1 final")
for tag, r in R.items():
    ls = r["experiments"]["lens"]["layers"]
    print(f"{tag:6s} {ls[0]['rank_mean']:10.0f} {ls[-1]['rank_mean']:7.1f} {ls[-1]['top1']:10.1%}")
    assert ls[-1]["rank_mean"] < ls[0]["rank_mean"] / 10
    assert ls[-1]["top1"] < 0.5
print("claim 3: prediction is built across depth and does not become certainty")


# %%
# Claim 4. The probe is decodable but not causal. Two comparisons matter: the
# probe against the model's own readout, and the probe direction against a
# random direction at the same norm. The second is the one that usually gets
# left out.
print("\nprobe vs the model's own readout")
for tag, r in R.items():
    p, b = r["experiments"]["probe"], r["experiments"]["probe"]["baseline"]
    print(f"{tag:6s} best L{p['config']['best_layer']:<2d} probe {p['config']['best_accuracy']:6.1%} "
          f" readout {b['accuracy']:6.1%}  delta {b['accuracy']-p['config']['best_accuracy']:+.1%}")
    assert p["config"]["best_accuracy"] <= b["accuracy"]

print("\nsteering accuracy at alpha = +0.5 (same norm)")
print("model   probe   random  orthogonal")
for tag, r in R.items():
    p = r["experiments"]["probe"]
    acc = {s["direction"]: s["accuracy"] for s in p["steering"] if s["alpha"] == 0.5}
    print(f"{tag:6s} {acc['probe']:6.1%} {acc['random']:7.1%} {acc['orthogonal']:10.1%}")
    assert acc["probe"] <= acc["random"]
print("claim 4: the probe direction does not beat the random control; decodability is not use")


# %%
# The ledger. A claim is only as good as the control that could have broken
# it. Two of these survived as weaker claims, one is descriptive, and one is a
# null result. The distinction is the point of the lab.
ledger = [
    ("induction is layer-level and head-redundant", "supported", "top5 < whole layer"),
    ("IOI is distributed across heads", "supported", "best head < 10% of gap"),
    ("prediction is built across depth", "descriptive", "readout, not mechanism"),
    ("sentiment is readable but not a knob", "null", "probe worse than random"),
]
print("\nclaim ledger")
for claim, status, reason in ledger:
    print(f"  {status:11s} {claim:46s} ({reason})")
print("\nlimits: two models, one family, one behavioural seed, synthetic IOI template,")
print("no MLP ablation, no held-out sentence frame. The controls are the contribution.")
