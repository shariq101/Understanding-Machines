"""Turn the lab scripts into a page section and notebooks, from one source.

A lab is a Python file with `# %% [markdown]` and `# %%` cells.  The same file
produces (a) the HTML the site shows, (b) an ipynb Colab can open, and (c) a
JSON of expected outputs that `check_labs.py` compares against a re-run.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import markdown

HERE = Path(__file__).parent
ROOT = HERE.parent
EXPECTED = HERE / "expected"
NOTEBOOKS = HERE / "notebooks"
EXPECTED.mkdir(exist_ok=True)
NOTEBOOKS.mkdir(exist_ok=True)

LABS = sorted(HERE.glob("lab*.py"))


def parse(path):
    cells, cur = [], None
    for line in path.read_text().splitlines():
        if line.startswith("# %% [markdown]"):
            cur = {"type": "markdown", "lines": []}
            cells.append(cur)
        elif line.startswith("# %%"):
            cur = {"type": "code", "lines": []}
            cells.append(cur)
        elif cur is not None:
            if cur["type"] == "markdown":
                if line == "#":
                    line = ""
                elif line.startswith("# "):
                    line = line[2:]
            cur["lines"].append(line)
    for c in cells:
        c["source"] = "\n".join(c["lines"]).strip("\n")
    return [c for c in cells if c["source"].strip()]


def title_of(cells, default):
    for c in cells:
        if c["type"] == "markdown":
            for line in c["source"].splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                    while title.startswith("#"):
                        title = title.lstrip("#").strip()
                    return title
    return default


def run_cells(cells, name):
    ns = {"__name__": "__lab__"}
    outputs = []
    old = os.getcwd()
    os.chdir(HERE)
    sys.path.insert(0, str(HERE))
    try:
        for i, c in enumerate(cells):
            if c["type"] != "code":
                outputs.append(None)
                continue
            buf = io.StringIO()
            try:
                with redirect_stdout(buf):
                    exec(compile(c["source"], f"<{name}:cell{i}>", "exec"), ns)
            except Exception as exc:  # surface the failure in the capture
                buf.write(f"\n[error] {type(exc).__name__}: {exc}\n")
            outputs.append(buf.getvalue().strip("\n"))
    finally:
        os.chdir(old)
        sys.path.pop(0)
    return outputs


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def html_for(cells, outputs, num, title):
    out = [f'<section class="lab" id="lab{num:02d}">',
           '<div class="wrap">',
           f'<p class="kicker">Lab {num:02d}</p>',
           f'<h3>{esc(title)}</h3>']
    goal = ""
    for c, o in zip(cells, outputs):
        if c["type"] == "markdown":
            body = markdown.markdown(c["source"], extensions=["extra"])
            # strip the leading h1, already used as the title
            body = re.sub(r"<h1>.*?</h1>", "", body, flags=re.S).strip()
            if not goal:
                m = re.search(r"<p>(.*?)</p>", body, flags=re.S)
                goal = m.group(1) if m else ""
                if goal:
                    out.append(f'<p class="goal">{goal}</p>')
                    body = body.replace(f"<p>{goal}</p>", "", 1)
            out.append(f'<div class="lab-step">{body}</div>')
        else:
            out.append('<div class="lab-step lab-code">')
            out.append(f'<pre><code>{esc(c["source"])}</code></pre>')
            if o:
                out.append(f'<div class="output">{esc(o)}</div>')
            out.append("</div>")
    out.append("</div></section>")
    return "\n".join(out), goal


def notebook_for(cells, outputs, title):
    nb = {"cells": [], "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                    "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    first = True
    for c, o in zip(cells, outputs):
        if c["type"] == "markdown":
            src = c["source"] if not first else c["source"]
            nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})
        else:
            cell = {"cell_type": "code", "metadata": {}, "execution_count": None,
                    "source": c["source"].splitlines(keepends=True), "outputs": []}
            if o:
                cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": o.splitlines(keepends=True)}]
            nb["cells"].append(cell)
        first = False
    return nb


def main():
    force = "--force" in sys.argv
    sections = []
    index_cards = []
    for path in LABS:
        m = re.search(r"lab(\d+)", path.stem)
        if not m:
            continue
        num = int(m.group(1))
        cells = parse(path)
        title = title_of(cells, path.stem)
        exp = EXPECTED / f"lab{num:02d}.json"
        if exp.exists() and not force:
            outputs = json.loads(exp.read_text())["outputs"]
        else:
            print("running", path.name, "…", flush=True)
            outputs = run_cells(cells, path.stem)
            exp.write_text(json.dumps({"outputs": outputs}, indent=1))
        html, goal = html_for(cells, outputs, num, title)
        sections.append(html)
        index_cards.append((num, title, goal))
        NOTEBOOKS.mkdir(exist_ok=True)
        (NOTEBOOKS / f"lab{num:02d}.ipynb").write_text(json.dumps(notebook_for(cells, outputs, title), indent=1))
        print("ok", path.name, f"({len(outputs)} cells)", flush=True)
    cards = "\n".join(
        f'<div class="card"><span class="num">{n:02d}</span><h4><a href="#lab{n:02d}">{esc(t)}</a></h4>'
        f'<p>{g}</p></div>' for n, t, g in index_cards)
    intro = ('<section class="band" id="labs"><div class="wrap">'
             '<p class="kicker">Laboratory</p>'
             f'<h2 class="big">{len(LABS)} labs that build the evidence.</h2>'
             '<p class="lede">Each lab is a runnable Python program with a milestone. Run them in order: '
             'the later labs load the checkpoints the earlier ones produce. The printed output on these pages '
             'is captured from a real run, not written by hand.</p>'
             f'<div class="grid3">{cards}</div></div></section>')
    (ROOT / "src" / "65-labs.html").write_text(intro + "\n" + "\n".join(sections))
    print("wrote src/65-labs.html and", len(LABS), "notebooks")


if __name__ == "__main__":
    main()
