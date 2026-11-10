import json, time, torch
from uml.model import Config
from uml.train import fresh, train_lm, save
from uml.data import parity_batch, parity_prompt

torch.set_num_threads(4)


def make(cot, layers=2, steps=2500, seed=0):
    cfg = Config(vocab_size=4, n_layers=layers, d_model=64, n_heads=4,
                 d_mlp=256, max_len=40)
    m = fresh(cfg, seed)
    g = torch.Generator().manual_seed(5 + seed)
    L = 10

    def gen():
        while True:
            yield parity_batch(64, L, cot, g)

    def ev():
        while True:
            yield parity_batch(64, L, cot, g)

    t = time.time()
    train_lm(m, gen(), steps=steps, lr=3e-3, eval_batches=ev(), eval_every=500)
    return m, round(time.time() - t, 1)


def eval_model(m, bits, cot):
    x = parity_prompt(bits, cot)
    if cot:
        with torch.no_grad():
            for _ in range(len(bits)):
                logits, _ = m(x)
                nxt = logits[0, -1].argmax(-1, keepdim=True)
                x = torch.cat([x, nxt[None]], 1)
        return int(x[0, -1].item() - 2)
    with torch.no_grad():
        return int(m(x)[0][0, -1].argmax().item() - 2)


for layers in (2, 4):
    for cot in (False, True):
        m, secs = make(cot, layers=layers, steps=2000, seed=10 + layers + int(cot))
        res = {}
        for L in (10, 14, 18):
            g = torch.Generator().manual_seed(100 + L)
            c = 0
            for _ in range(80):
                bits = torch.randint(0, 2, (L,), generator=g).tolist()
                c += eval_model(m, bits, cot) == (sum(bits) % 2)
            res[L] = c / 80
        print(layers, cot, secs, res, flush=True)
        save(m, f"checkpoints/parity_{layers}l_{'cot' if cot else 'direct'}.pt",
             {"task": "parity", "layers": layers, "cot": cot, "acc": res})
