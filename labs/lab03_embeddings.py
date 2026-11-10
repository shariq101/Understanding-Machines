# %% [markdown]
# # Lab 3 — Meaning from co-occurrence
#
# We build a corpus, a co-occurrence matrix, PPMI, and an SVD embedding. Then we
# ask what the geometry captures and what it misses. The corpus is small on
# purpose: every neighbour can be traced to a count.
#
# **Milestone:** report one neighbour that is semantically right and one analogy that fails.

# %%
import torch
from uml.embeddings import corpus, cooccurrence, ppmi, svd_embed, nearest, analogy
from uml.embeddings import ANIMALS, COLORS, COUNTRIES, CAPITALS, FOODS

torch.set_num_threads(4)
sents = corpus(2400, seed=0)
vocab, idx, C = cooccurrence(sents, window=2)
M = ppmi(C)
E = svd_embed(M, 30)
print("sentences:", len(sents), " vocabulary:", len(vocab), " nonzero cells:", int((C > 0).sum()))
print("PPMI is", round(float((M > 0).float().mean()), 3), "dense")

# %%
for w in ["cat", "red", "japan", "tokyo", "rice"]:
    print(f"{w:7s}", nearest(vocab, E, w, 4))

# %%
print("france : paris :: japan : ?")
print(analogy(vocab, E, "france", "paris", "japan", 3))

# %% [markdown]
# The animal and colour neighbourhoods are clean: the corpus gives them nearly
# identical contexts. The country–capital analogy is not reliable at this scale.
# That is a property of the data, not a bug: the relational signal is weaker
# than the topical signal, and 44 words cannot support the geometry that a
# billion-token corpus can.

# %%
cats = {w: "animal" for w in ANIMALS}
cats.update({w: "colour" for w in COLORS})
cats.update({w: "country" for w in COUNTRIES})
cats.update({w: "capital" for w in CAPITALS})
cats.update({w: "food" for w in FOODS})
hits = 0; total = 0
for w, cat in cats.items():
    if w not in vocab: continue
    nn = nearest(vocab, E, w, 1)[0][0]
    total += 1
    hits += cats.get(nn) == cat
print(f"nearest-neighbour category match: {hits}/{total} = {hits/total:.2f}")
