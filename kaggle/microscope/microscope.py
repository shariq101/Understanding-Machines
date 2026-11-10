"""Measure a Goodfire-style feature microscope on InceptionV1.

The model is torchvision GoogLeNet, pretrained on ImageNet-1k. The data bank is
the full Imagenette2-320 train+val set. We hook four Inception layers, record
the strongest activation for every channel on every image, and export the top
five real images plus their spatial activation maps for a browsable microscope.

The visualization is not a feature-visualization optimization. It is an index
of what real images drive each channel, with the activation map that produced
the ranking.
"""
from __future__ import annotations

import gc
import heapq
import json
import os
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import GoogLeNet_Weights, googlenet

os.environ.setdefault("HF_HOME", "/tmp/hf")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path("/kaggle/working")
DATA = ROOT / "imagenette2-320"
OUT = ROOT / "microscope"
URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

# Goodfire's mixed* names map onto torchvision's InceptionV1 modules.
LAYERS = {
    "inception3b": {"goodfire": "mixed3b", "label": "early · mixed3b"},
    "inception4c": {"goodfire": "mixed4c", "label": "middle · mixed4c"},
    "inception4e": {"goodfire": "mixed4e", "label": "late middle · mixed4e"},
    "inception5b": {"goodfire": "mixed5b", "label": "late · mixed5b"},
}
# All of 4c, so the page includes the canonical microscope range; a broad
# selection from the other layers keeps the export inspectable.
SELECTED = {"inception3b": 128, "inception4c": 512, "inception4e": 128, "inception5b": 96}
TOPK = 5

SYNSET = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def ensure_data() -> None:
    if DATA.exists():
        return
    archive = ROOT / "imagenette2-320.tgz"
    if not archive.exists():
        print("downloading Imagenette…", flush=True)
        urllib.request.urlretrieve(URL, archive)
    print("extracting…", flush=True)
    with tarfile.open(archive) as tf:
        tf.extractall(ROOT)
    archive.unlink(missing_ok=True)


class Bank(Dataset):
    def __init__(self, paths, tf):
        self.paths = paths
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        with Image.open(path) as im:
            image = self.tf(im.convert("RGB"))
        return image, i, str(path)


class TopK:
    def __init__(self, k=TOPK):
        self.k = k
        self.heap = []
        self.serial = 0

    def add(self, value, y, x, act_map, image_index):
        if len(self.heap) >= self.k and float(value) <= self.heap[0][0]:
            return
        self.serial += 1
        item = (float(value), self.serial, int(image_index), int(y), int(x), act_map.copy())
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, item)
        else:
            heapq.heapreplace(self.heap, item)

    def sorted(self):
        return sorted(self.heap, key=lambda z: z[0], reverse=True)


def run() -> None:
    ensure_data()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)

    paths = sorted(DATA.glob("train/*/*.JPEG")) + sorted(DATA.glob("val/*/*.JPEG"))
    if not paths:
        raise RuntimeError(f"no JPEGs under {DATA}")
    print(f"images {len(paths)}", flush=True)

    weights = GoogLeNet_Weights.IMAGENET1K_V1
    model = googlenet(weights=weights).eval().to(DEV)
    tf = weights.transforms()
    bank = Bank(paths, tf)
    loader = DataLoader(
        bank, batch_size=24, shuffle=False, num_workers=2, pin_memory=(DEV == "cuda")
    )

    captured = {}
    hooks = []

    def make_hook(name):
        def hook(_module, _inputs, output):
            captured[name] = output
        return hook

    for name in LAYERS:
        module = getattr(model, name)
        hooks.append(module.register_forward_hook(make_hook(name)))

    channel_count = {}
    max_values = {name: [] for name in LAYERS}
    heaps = {}
    labels_by_index = [SYNSET.get(p.parent.name, p.parent.name) for p in paths]
    started = time.time()

    with torch.no_grad():
        for step, (images, indices, _paths) in enumerate(loader):
            images = images.to(DEV, non_blocking=True)
            model(images)
            for name in LAYERS:
                act = captured[name]
                if act.ndim != 4:
                    raise RuntimeError(f"{name} returned {tuple(act.shape)}")
                if name not in heaps:
                    channel_count[name] = int(act.shape[1])
                    heaps[name] = [TopK() for _ in range(act.shape[1])]
                maxima = act.amax(dim=(2, 3)).float().cpu().numpy()
                max_values[name].append(maxima)
                maps = act.float().cpu().numpy()
                for b in range(act.shape[0]):
                    image_index = int(indices[b])
                    for c in range(act.shape[1]):
                        plane = maps[b, c]
                        y, x = np.unravel_index(int(np.argmax(plane)), plane.shape)
                        heaps[name][c].add(
                            float(maxima[b, c]), y, x, plane, image_index
                        )
            if step % 40 == 0:
                print(f"step {step} / {len(loader)}", flush=True)

    for hook in hooks:
        hook.remove()
    del model
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()

    manifest = {
        "model": "InceptionV1 (torchvision GoogLeNet, ImageNet-1k weights)",
        "model_note": (
            "This is standard torchvision GoogLeNet, not the architecture-modified "
            "checkpoint in Goodfire's essay. Layer names are mapped to the mixed* names."
        ),
        "dataset": "Imagenette2-320 train+val (10 classes; real photographs)",
        "images": len(paths),
        "method": (
            "Forward hooks record each selected layer. Images are ranked by the maximum "
            "post-ReLU response anywhere in the spatial map. Heatmaps are that channel's "
            "spatial activation at the same image."
        ),
        "limitations": [
            "Imagenette has 10 coarse classes, not the full 1,000-class ImageNet bank.",
            "This is torchvision GoogLeNet, not Goodfire's modified InceptionV1 checkpoint.",
            "The top images are selected from this image bank; they are not a claim that the channel is monosemantic.",
            "A high maximum is a correlation with the image, not a causal circuit result.",
        ],
        "layers": [],
    }

    for name, info in LAYERS.items():
        vals = np.concatenate(max_values[name], axis=0)
        mx = vals.max(axis=1)
        count = channel_count[name]
        if SELECTED[name] >= count:
            selected = list(range(count))
        else:
            selected = sorted(
                np.argsort(mx)[-SELECTED[name]:].tolist()
            )
        if name == "inception4c" and 447 in range(count) and 447 not in selected:
            selected.append(447)
        selected = sorted(set(selected), key=lambda c: (-float(mx[c]), c))

        neurons = {}
        layer_dir = OUT / "images" / name
        layer_dir.mkdir(parents=True, exist_ok=True)
        for c in selected:
            values = vals[:, c]
            entries = heaps[name][c].sorted()
            images = []
            class_counts = Counter()
            for rank, (value, _serial, image_index, y, x, plane) in enumerate(entries):
                path = paths[image_index]
                label = labels_by_index[image_index]
                class_counts[label] += 1
                filename = f"{name}/{c:04d}_{rank}.jpg"
                target = OUT / "images" / filename
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    im = transforms.Resize(256)(im)
                    im = transforms.CenterCrop(224)(im)
                    im.save(target, quality=82, optimize=True)
                images.append({
                    "file": f"images/{filename}",
                    "label": label,
                    "activation": round(float(value), 3),
                    "position": [int(y), int(x)],
                    "map": np.round(plane, 3).tolist(),
                })
            neurons[str(c)] = {
                "channel": c,
                "top_classes": [
                    {"label": label, "count": n}
                    for label, n in class_counts.most_common(3)
                ],
                "stats": {
                    "max": round(float(values.max()), 3),
                    "p99_9": round(float(np.quantile(values, 0.999)), 3),
                    "p99": round(float(np.quantile(values, 0.99)), 3),
                    "median": round(float(np.median(values)), 3),
                    "mean": round(float(values.mean()), 3),
                    "active_fraction": round(float((values > 0).mean()), 4),
                    "images": int(values.size),
                },
                "images": images,
            }
        payload = {
            "layer": name,
            "goodfire": info["goodfire"],
            "label": info["label"],
            "channels": count,
            "selected": len(selected),
            "spatial": int(heaps[name][0].heap[0][5].shape[0]) if heaps[name] else None,
            "neurons": neurons,
        }
        (OUT / f"{name}.json").write_text(json.dumps(payload, separators=(",", ":")))
        manifest["layers"].append({
            "id": name,
            "goodfire": info["goodfire"],
            "label": info["label"],
            "channels": count,
            "selected": len(selected),
            "spatial": payload["spatial"],
        })
        print(f"{name}: {count} channels, {len(selected)} selected", flush=True)

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", OUT, "in", round(time.time() - started, 1), "s", flush=True)


if __name__ == "__main__":
    run()
