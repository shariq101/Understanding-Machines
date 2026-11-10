"""A tiny multimodal world: scenes, a CLIP-style dual encoder, and a VLM.

The point is not scale.  It is that every interface in a modern vision-language
model is present and small enough to intervene on: a patch transformer, a
contrastive alignment stage, a projector, a frozen language model, and a
vocabulary that can be read at every layer.
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Block, Config
from .vision import ViT

SHAPES = ["circle", "square", "triangle"]
COLORS = ["red", "green", "blue"]
POSITIONS = ["left", "center", "right"]
RGB = {
    "red": (1.0, 0.12, 0.10),
    "green": (0.10, 0.88, 0.18),
    "blue": (0.12, 0.22, 1.0),
}
SIZE = 16
PATCH = 4
N_PATCHES = (SIZE // PATCH) ** 2
N_IMAGE_TOKENS = N_PATCHES + 1          # class token plus patches

WORDS = ["<pad>", "<bos>", "a", "red", "green", "blue", "circle", "square",
         "triangle", "on", "the", "left", "center", "right", "what", "color",
         "shape", "where", "is", "it", "?", "="]
WORD_ID = {w: i for i, w in enumerate(WORDS)}
BOS = WORD_ID["<bos>"]
PAD = WORD_ID["<pad>"]

QUESTIONS = {
    "color": ["what", "color", "is", "the", "shape", "?"],
    "shape": ["what", "shape", "is", "it", "?"],
    "position": ["where", "is", "the", "shape", "?"],
}
ANSWER_OF = {"color": COLORS, "shape": SHAPES, "position": POSITIONS}


def _draw(img, shape, color, cx, cy, r, shade):
    yy, xx = torch.meshgrid(torch.arange(SIZE), torch.arange(SIZE), indexing="ij")
    dx, dy = xx - cx, yy - cy
    if shape == "circle":
        mask = dx * dx + dy * dy <= r * r
    elif shape == "square":
        mask = (dx.abs() <= r * 0.85) & (dy.abs() <= r * 0.85)
    else:
        half = (dy + r) * 0.55
        mask = (dy >= -r) & (dy <= r) & (dx.abs() <= half)
    for c, v in enumerate(RGB[color]):
        img[c][mask] = v * shade


def _geometry(position, g):
    cx = {"left": 4, "center": 8, "right": 12}[position] + int(torch.randint(-1, 2, (1,), generator=g))
    cy = 8 + int(torch.randint(-1, 2, (1,), generator=g))
    r = 2.8 + 0.4 * float(torch.rand(1, generator=g))
    shade = 0.72 + 0.28 * float(torch.rand(1, generator=g))
    return cx, cy, r, shade


def scene(shape, color, position, generator=None):
    """One 3x16x16 scene with a single object, jittered in position."""
    g = generator or torch.Generator()
    img = torch.full((3, SIZE, SIZE), 0.035)
    cx, cy, r, shade = _geometry(position, g)
    _draw(img, shape, color, cx, cy, r, shade)
    img = img + 0.035 * torch.randn(3, SIZE, SIZE, generator=g)
    return img.clamp(0, 1)


def scene_pair(shape, position, color_a, color_b, seed=0):
    """Two images with identical geometry and noise, different colour."""
    g = torch.Generator().manual_seed(seed)
    cx, cy, r, shade = _geometry(position, g)
    noise = 0.035 * torch.randn(3, SIZE, SIZE, generator=g)
    out = []
    for color in (color_a, color_b):
        img = torch.full((3, SIZE, SIZE), 0.035)
        _draw(img, shape, color, cx, cy, r, shade)
        out.append((img + noise).clamp(0, 1))
    return out[0], out[1]


def caption(shape, color, position):
    return f"a {color} {shape} on the {position}"


def encode_caption(text):
    return torch.tensor([WORD_ID[w] for w in text.split()], dtype=torch.long)


def scene_batch(n=256, generator=None, balance=False):
    """Images, three labels, captions, and ids for CLIP training."""
    g = generator or torch.Generator()
    imgs, shapes, colors, positions, caps = [], [], [], [], []
    for i in range(n):
        if balance:
            s = SHAPES[i % 3]
            c = COLORS[(i // 3) % 3]
            p = POSITIONS[(i // 9) % 3]
        else:
            s = SHAPES[int(torch.randint(0, 3, (1,), generator=g))]
            c = COLORS[int(torch.randint(0, 3, (1,), generator=g))]
            p = POSITIONS[int(torch.randint(0, 3, (1,), generator=g))]
        imgs.append(scene(s, c, p, g))
        shapes.append(SHAPES.index(s))
        colors.append(COLORS.index(c))
        positions.append(POSITIONS.index(p))
        caps.append(encode_caption(caption(s, c, p)))
    L = max(len(c) for c in caps)
    txt = torch.full((n, L), PAD, dtype=torch.long)
    for i, c in enumerate(caps):
        txt[i, :len(c)] = c
    return (torch.stack(imgs), torch.tensor(shapes), torch.tensor(colors),
            torch.tensor(positions), txt)


def one_per_caption(generator=None):
    """The 27 distinct object/attribute combinations, in a fixed order."""
    g = generator or torch.Generator().manual_seed(7)
    imgs, labels = [], []
    for c in COLORS:
        for s in SHAPES:
            for p in POSITIONS:
                imgs.append(scene(s, c, p, g))
                labels.append((SHAPES.index(s), COLORS.index(c), POSITIONS.index(p)))
    return torch.stack(imgs), torch.tensor(labels)


def question_batch(n=256, generator=None, types=("color", "shape", "position")):
    """Images plus a question and the one-word answer it requires."""
    g = generator or torch.Generator()
    imgs, qs, ans, meta = [], [], [], []
    for _ in range(n):
        s = SHAPES[int(torch.randint(0, 3, (1,), generator=g))]
        c = COLORS[int(torch.randint(0, 3, (1,), generator=g))]
        p = POSITIONS[int(torch.randint(0, 3, (1,), generator=g))]
        typ = types[int(torch.randint(0, len(types), (1,), generator=g))]
        answer = {"color": c, "shape": s, "position": p}[typ]
        imgs.append(scene(s, c, p, g))
        qs.append(torch.tensor([BOS] + [WORD_ID[w] for w in QUESTIONS[typ]] + [WORD_ID["="]],
                               dtype=torch.long))
        ans.append(WORD_ID[answer])
        meta.append((SHAPES.index(s), COLORS.index(c), POSITIONS.index(p), typ))
    L = max(len(q) for q in qs)
    q = torch.full((n, L), PAD, dtype=torch.long)
    for i, x in enumerate(qs):
        q[i, :len(x)] = x
    return torch.stack(imgs), q, torch.tensor(ans), meta


class TextEncoder(nn.Module):
    def __init__(self, d_model=48, vocab=len(WORDS)):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                 nn.Linear(d_model, d_model))

    def forward(self, ids):
        mask = (ids != PAD).float().unsqueeze(-1)
        x = self.tok(ids) * mask
        x = x.sum(1) / mask.sum(1).clamp(min=1)
        return self.net(x)


class TinyCLIP(nn.Module):
    """Two towers, one dot product, symmetric InfoNCE."""

    def __init__(self, d_model=48, d_out=32, n_layers=2, n_heads=4, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.vision = ViT(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                          patch=PATCH, size=SIZE, in_chans=3)
        self.text = TextEncoder(d_model)
        self.vproj = nn.Linear(d_model, d_out)
        self.tproj = nn.Linear(d_model, d_out)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, imgs, txt):
        z = self.vision(imgs)[1][:, 0]
        t = self.text(txt)
        vi = F.normalize(self.vproj(z), dim=-1)
        te = F.normalize(self.tproj(t), dim=-1)
        return vi, te, self.logit_scale.exp()

    def image_embed(self, imgs):
        return F.normalize(self.vproj(self.vision(imgs)[1][:, 0]), dim=-1)

    def text_embed(self, txt):
        return F.normalize(self.tproj(self.text(txt)), dim=-1)


def train_clip(model, steps=1500, batch=128, lr=1.2e-3, seed=0, log=True):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed + 1)
    hist = []
    for step in range(steps):
        imgs, _, _, _, txt = scene_batch(batch, g)
        vi, te, scale = model(imgs, txt)
        logits = scale * vi @ te.T
        labels = torch.arange(batch)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log and (step % 300 == 0 or step == steps - 1):
            hist.append((step, round(loss.item(), 3)))
            print(f"clip step {step:4d} loss {loss.item():.3f}")
    return hist


@torch.no_grad()
def clip_retrieval(model, imgs, txt):
    vi, te, _ = model(imgs, txt)
    logits = vi @ te.T
    labels = torch.arange(len(imgs))
    return {
        "i2t": round((logits.argmax(1) == labels).float().mean().item(), 3),
        "t2i": round((logits.argmax(0) == labels).float().mean().item(), 3),
    }


@torch.no_grad()
def modality_gap(model, imgs, txt):
    vi, te, _ = model(imgs, txt)
    within = (vi * te).sum(-1).mean().item()
    cross = (vi @ te.T).mean().item()
    gap = (vi.mean(0) - te.mean(0)).norm().item()
    return {"within": round(within, 3), "cross": round(cross, 3), "gap": round(gap, 3)}


def save_clip(model, path):
    torch.save({"vision": model.vision.state_dict(), "text": model.text.state_dict(),
                "vproj": model.vproj.state_dict(), "tproj": model.tproj.state_dict(),
                "logit_scale": model.logit_scale.detach()}, path)


def load_clip(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyCLIP()
    model.vision.load_state_dict(ck["vision"])
    model.text.load_state_dict(ck["text"])
    model.vproj.load_state_dict(ck["vproj"])
    model.tproj.load_state_dict(ck["tproj"])
    model.logit_scale.data = ck["logit_scale"]
    model.eval()
    return model


class TinyVLM(nn.Module):
    """A frozen ViT, a projector, and a small causal decoder.

    The image tokens are a prefix.  The decoder sees them, then the question,
    and predicts the answer at the final '=' position.  This is the LLaVA
    pattern reduced to its smallest working instance.
    """

    def __init__(self, vision: ViT, d_model=64, n_layers=2, n_heads=4,
                 max_len=28, vocab=len(WORDS), seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.vision = vision
        for p in self.vision.parameters():
            p.requires_grad_(False)
        self.d_model = d_model
        self.projector = nn.Sequential(nn.Linear(48, d_model), nn.GELU(),
                                       nn.Linear(d_model, d_model))
        self.tok = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d_model)
        cfg = Config(vocab_size=vocab, n_layers=n_layers, d_model=d_model,
                     n_heads=n_heads, d_mlp=2 * d_model, max_len=max_len, causal=True)
        self.cfg = cfg
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab, bias=False)
        self.unembed.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def image_tokens(self, imgs):
        with torch.no_grad():
            z = self.vision(imgs)[1]          # B, 1+patches, 48
        return self.projector(z)

    def forward(self, imgs, qids, store=None):
        img = self.image_tokens(imgs)
        txt = self.tok(qids)
        x = torch.cat([img, txt], dim=1)
        T = x.shape[1]
        x = x + self.pos(torch.arange(T, device=x.device))
        for i, block in enumerate(self.blocks):
            x = block(x, store, i)
        h = self.ln_f(x)
        return self.unembed(h), h

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_vlm(model, steps=1800, batch=64, lr=1.5e-3, seed=0, log=True):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed + 2)
    hist = []
    for step in range(steps):
        imgs, q, ans, _ = question_batch(batch, g)
        logits, _ = model(imgs, q)
        loss = F.cross_entropy(logits[:, -1], ans)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log and (step % 300 == 0 or step == steps - 1):
            hist.append((step, round(loss.item(), 3)))
            print(f"vlm step {step:4d} loss {loss.item():.3f}")
    return hist


@torch.no_grad()
def vlm_accuracy(model, n=384, seed=11, by_type=False):
    g = torch.Generator().manual_seed(seed)
    imgs, q, ans, meta = question_batch(n, g)
    pred = model(imgs, q)[0][:, -1].argmax(-1)
    correct = (pred == ans)
    if not by_type:
        return round(correct.float().mean().item(), 3)
    out = {}
    for typ in QUESTIONS:
        idx = [i for i, m in enumerate(meta) if m[3] == typ]
        out[typ] = round(correct[idx].float().mean().item(), 3) if idx else 0.0
    return out


def save_vlm(model, path):
    torch.save({"n_layers": len(model.blocks), "d_model": model.d_model,
                "projector": model.projector.state_dict(), "tok": model.tok.state_dict(),
                "pos": model.pos.state_dict(), "blocks": model.blocks.state_dict(),
                "ln_f": model.ln_f.state_dict()}, path)


def load_vlm(vision, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyVLM(vision, d_model=ck.get("d_model", 64), n_layers=ck.get("n_layers", 2))
    model.projector.load_state_dict(ck["projector"])
    model.tok.load_state_dict(ck["tok"])
    model.pos.load_state_dict(ck["pos"])
    model.blocks.load_state_dict(ck["blocks"])
    model.ln_f.load_state_dict(ck["ln_f"])
    model.eval()
    return model


@torch.no_grad()
def vlm_logit_lens(model, imgs, qids, topk=3):
    """Decode the final position after every block with the output unembedding."""
    img = model.image_tokens(imgs)
    x = torch.cat([img, model.tok(qids)], dim=1)
    x = x + model.pos(torch.arange(x.shape[1], device=x.device))
    rows = []
    for i, block in enumerate(model.blocks):
        x = block(x)
        probs = model.unembed(model.ln_f(x))[0, -1].softmax(-1)
        top = probs.topk(topk)
        rows.append({"layer": i + 1, "ids": top.indices.tolist(),
                     "probs": [round(v, 4) for v in top.values.tolist()]})
    return rows


@torch.no_grad()
def vlm_residual_cache(model, imgs, qids):
    img = model.image_tokens(imgs)
    x = torch.cat([img, model.tok(qids)], dim=1)
    x = x + model.pos(torch.arange(x.shape[1], device=x.device))
    out = [x.clone()]
    for block in model.blocks:
        x = block(x)
        out.append(x.clone())
    return out


@torch.no_grad()
def patch_vlm(model, clean_imgs, corrupt_imgs, qids, layer, positions, answer):
    """Patch clean residuals into a corrupted run at one layer and position set."""
    clean = vlm_residual_cache(model, clean_imgs, qids)
    n_img = model.image_tokens(clean_imgs).shape[1]

    def run(imgs, replacement=None):
        img = model.image_tokens(imgs)
        x = torch.cat([img, model.tok(qids)], dim=1)
        x = x + model.pos(torch.arange(x.shape[1], device=x.device))
        for i, block in enumerate(model.blocks):
            if replacement is not None and i == layer:
                x = x.clone()
                for p in positions:
                    x[:, p] = replacement[:, p]
            x = block(x)
        return model.unembed(model.ln_f(x))

    def diff(logits):
        return (logits[0, -1, answer[0]] - logits[0, -1, answer[1]]).item()

    before = diff(run(corrupt_imgs))
    after = diff(run(corrupt_imgs, clean[layer]))
    full = diff(run(clean_imgs))
    restored = (after - before) / (full - before) if abs(full - before) > 1e-9 else 0.0
    return {"before": round(before, 3), "after": round(after, 3),
            "clean": round(full, 3), "restored": round(restored, 3),
            "n_image_tokens": n_img}


@torch.no_grad()
def vit_residuals(vision, imgs):
    """Class token plus patch tokens after the embedding and every block."""
    z = vision.embed(vision.patches(imgs))
    z = torch.cat([vision.cls.expand(z.shape[0], -1, -1), z], dim=1) + vision.pos
    out = [z.clone()]
    for block in vision.blocks:
        z = block(z)
        out.append(z.clone())
    return out


@torch.no_grad()
def image_token_ablation(model, imgs, qids, token):
    """Zero one image token after the projector and measure answer accuracy."""
    img = model.image_tokens(imgs).clone()
    img[:, token] = 0
    x = torch.cat([img, model.tok(qids)], dim=1)
    x = x + model.pos(torch.arange(x.shape[1], device=x.device))
    for block in model.blocks:
        x = block(x)
    return model.unembed(model.ln_f(x))[:, -1]


@contextmanager
def attention_knockout(model, layer, head, qslice, kslice):
    """Zero one head's attention from query positions to key positions."""
    attn = model.blocks[layer].attn
    old = attn.knockout
    attn.knockout = (head, qslice, kslice)
    try:
        yield
    finally:
        attn.knockout = old


@torch.no_grad()
def attention_maps(model, imgs, qids):
    store = {}
    model(imgs, qids, store)
    return [store[(i, "attn")][0] for i in range(len(model.blocks))]


@torch.no_grad()
def mlp_activations(model, imgs, qids):
    store = {}
    model(imgs, qids, store)
    return [store[(i, "mlp")] for i in range(len(model.blocks))]


def probe_eval(X, y, seed=0, steps=400, train_frac=0.75, shuffle=False):
    """A linear probe with a held-out split and an optional label control."""
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(X), generator=g)
    n_tr = int(len(X) * train_frac)
    tr, te = order[:n_tr], order[n_tr:]
    if shuffle:
        y = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(seed + 1))]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    W = torch.zeros(X.shape[1], int(y.max()) + 1, requires_grad=True)
    b = torch.zeros(int(y.max()) + 1, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.05, weight_decay=1e-3)
    for _ in range(steps):
        loss = F.cross_entropy(Xtr @ W + b, y[tr])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return round(((Xte @ W + b).argmax(-1) == y[te]).float().mean().item(), 3)


@contextmanager
def ablate_mlp_neurons(model, layer, neurons):
    """Remove selected MLP neurons' write to the residual stream."""
    W = model.blocks[layer].mlp.fc2.weight
    saved = W[:, neurons].clone()
    with torch.no_grad():
        W[:, neurons] = 0
    try:
        yield
    finally:
        with torch.no_grad():
            W[:, neurons] = saved


@torch.no_grad()
def modality_selectivity(model, imgs, qids, layer):
    """Mean MLP activation on image-token vs text-token positions."""
    h = mlp_activations(model, imgs, qids)[layer]
    n_img = model.image_tokens(imgs).shape[1]
    img = h[:, :n_img].mean((0, 1))
    txt = h[:, n_img:].mean((0, 1))
    return ((img - txt) / (img.abs() + txt.abs() + 1e-6)), img, txt
