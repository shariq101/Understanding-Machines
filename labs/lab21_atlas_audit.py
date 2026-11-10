# %% [markdown]
# # Lab 21 — Audit the atlas
#
# The atlas is a claim about a literature, not a list. This lab reads the
# catalogue, reproduces the instrument table from Chapter 24, assigns a crude
# evidence rung from each abstract, and checks the ten synthesis claims from
# Chapter 96 against the numbers the labs actually produced.
#
# **Milestone:** report the instrument and year distributions, the rung
# distribution under a stated heuristic, and a pass/partial/untested verdict
# for each of the ten claims.

# %%
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "atlas_catalog.json").exists():
    ROOT = ROOT.parent
cat = json.loads((ROOT / "atlas_catalog.json").read_text())
html = (ROOT / "index.html").read_text()
M = json.loads((ROOT / "measurements.json").read_text())

print("papers:", len(cat))


# %%
# The table in Chapter 24. The counts should be generated from the catalogue,
# not typed; a mismatch is a bookkeeping bug, not a finding.
expected = {
    "Input-level Attribution": 7,
    "Sparse Autoencoder": 12,
    "Probing": 5,
    "Beyond or Logit Lens": 4,
    "Causal Tracing": 4,
    "Steering": 5,
    "Representation": 7,
    "Neuron Analysis": 10,
    "Attention": 4,
    "Tool": 3,
}
counts = Counter(p["cat"] for p in cat)
print("\ninstrument                 catalogue  chapter-24  match")
for name, n in expected.items():
    print(f"{name:26s} {counts[name]:9d} {n:11d}  {counts[name] == n}")


# %%
# Years. The list is not a census; the histogram is the shape of the sample.
years = Counter(p["year"] for p in cat)
print("\nyear  papers")
for y in sorted(years):
    print(f"{y:4d}  {years[y]:6d}")


# %%
# Every paper in the catalogue should have exactly one paper chapter, and every
# paper chapter should carry exactly one arXiv identifier.
paper_arts = [a for a in re.findall(r'<article class="chapter".*?</article>', html, re.S)
              if "Paper atlas" in a]
paper_arx = []
for a in paper_arts:
    found = re.findall(r"arXiv:(\d{4}\.\d{4,5})", a)
    paper_arx.extend(found)
catalog_arx = [p["arxiv"] for p in cat]
print("\npaper chapters:", len(paper_arts))
print("unique chapter arXiv ids:", len(set(paper_arx)))
print("catalogue ids missing from chapters:", sorted(set(catalog_arx) - set(paper_arx)))
print("chapter ids absent from catalogue:", sorted(set(paper_arx) - set(catalog_arx)))


# %%
# A transparent, deliberately crude rung heuristic. It reads the abstract for
# intervention and measurement vocabulary; it is a bibliometric description,
# not a judgment of quality.
def rung(p):
    t = (p["title"] + " " + p.get("abstract", "")).lower()
    if "circuit" in t:
        return "circuit"
    if any(w in t for w in ("ablat", "interven", "causal", "patch", "knockout", "steer")):
        return "intervention"
    if any(w in t for w in ("probe", "decod", "linear", "represent")):
        return "decodability"
    return "description"


rungs = Counter(rung(p) for p in cat)
print("\nrung heuristic        papers")
for k in ("circuit", "intervention", "decodability", "description"):
    print(f"{k:20s} {rungs[k]:6d}")
print("examples:")
for k in ("circuit", "intervention"):
    ex = [p["title"][:70] for p in cat if rung(p) == k][:3]
    print(" ", k, "→", ex)


# %%
# The synthesis. Each claim is paired with the measurement that would support
# or undercut it. The verdict is about the book's own tiny-model evidence, not
# about the literature.
def check(name, grade, evidence, verdict):
    print(f"\n{name}  [book grade {grade}]")
    print("  evidence:", evidence)
    print("  lab verdict:", verdict)


check("1. Three interfaces", "A", "architecture; no single scalar", "structural, not a measurement")

stage = M["vlm_patch_l0"] < 0.1 and M["vlm_patch_l1"] > 0.9 and M["vlm_lens_l1"] > 0.9
check("2. Stage-wise integration", "B",
      f"lens@1={M['vlm_lens_l1']}, patch@0={M['vlm_patch_l0']}, patch@1={M['vlm_patch_l1']}",
      "pass in the tiny VLM" if stage else "not supported by these numbers")

check("3. A small subset of tokens carries the transfer", "B",
      f"token 9 restores {M['vlm_imgtok9']}, token 5 restores {M['vlm_imgtok5']}",
      "partial: the map is concentrated, but no concentration statistic is stored")

check("4. Attention mass is not necessity", "A",
      f"head 3 knockout = {M['knockout_l0h3']}, head 2 = {M['knockout_l0h2']}",
      "partial: knockout effects are stored; the mass baseline is in Lab 18's printed output")

check("5. Modality-specific neurons can be necessary", "B",
      f"{M['neuron_strong_l0']} strong neurons at layer 0; ablating the top 32 costs {M['neuron_ablate_l0']}",
      "pass in the tiny VLM" if (M["neuron_strong_l0"] >= 20 and M["neuron_ablate_l0"] > 0.05)
      else "not supported")

check("6. SAE features are descriptions, not ontology", "B",
      f"vision SAE FVU={M['vsae_fvu']}, L0={M['vsae_l0']}; steering colour {M['sae_steer_color']} vs mean shift {M['mean_steer_color']}",
      "consistent: the dictionary reconstructs and is pure, but is not a better handle")

check("7. Contrastive alignment leaves a gap", "A",
      f"within={M['clip_within']}, cross={M['clip_cross']}, centroids={M['clip_gap']}",
      "pass in the tiny CLIP" if (M["clip_gap"] > 0.05 and M["clip_within"] > 0.9)
      else "not supported")

check("8. Binding is not alignment", "B",
      f"binding={M['clip_binding']}, shape-swap chosen={M['clip_shape_swap']}",
      "pass in the tiny CLIP" if (M["clip_binding"] > 0.7 and M["clip_shape_swap"] > 0.1)
      else "not supported")

check("9. Causal tracing localizes the answer", "A",
      f"restored={M['patch_restored']}, VLM layer 0={M['vlm_patch_l0']}, layer 1={M['vlm_patch_l1']}",
      "pass in the tiny VLM" if (M["patch_restored"] > 0.8 and M["vlm_patch_l0"] < 0.2)
      else "not supported")

check("10. Heads can be load-bearing", "A",
      f"layer-0 head 3 costs {M['knockout_l0h3']}; head 2 costs {M['knockout_l0h2']}",
      "pass in the tiny VLM" if M["knockout_l0h3"] > 0.1 else "not supported")


# %%
# The audit's own limitation: the rung labels are lexical, the claims are
# graded by hand, and the measurements come from one tiny model. A scorecard is
# not a meta-analysis.
print("\nlimits: lexical rungs; one model; hand-graded claims; no confidence intervals")
