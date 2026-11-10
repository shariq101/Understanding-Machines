/* The activation atlas. Draws one clean forward pass per model: patch-token
   dimensions for vision transformers, per-head attention for language models.
   Every cell is a lookup in atlas-data.js; no model runs here. */

(async () => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };
  const SVGNS = "http://www.w3.org/2000/svg";
  const sv = (tag, attrs = {}) => {
    const n = document.createElementNS(SVGNS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  };
  const PAPER = [247, 244, 238];
  const BLUE = [33, 84, 160];
  const RED = [184, 54, 42];
  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
  const ramp = t => Math.pow(Math.max(0, Math.min(1, t)), 0.78);
  const divColor = v => (v < 0.5 ? mix(PAPER, BLUE, ramp((0.5 - v) * 2))
                                 : mix(PAPER, RED, ramp((v - 0.5) * 2)));
  const seqColor = v => mix(PAPER, BLUE, ramp(v));
  const bytesOf = s => {
    const bin = atob(s);
    const u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return u;
  };

  let DATA = window.ATLAS || null;
  if (!DATA) {
    try { DATA = await fetch("atlas-data.json", { cache: "no-store" }).then(r => r.json()); }
    catch (err) { DATA = null; }
  }

  const figure = $("#figure");
  const state = { family: "vision", idx: 0, img: 0 };

  const themeBtn = $("#themebtn");
  const saved = localStorage.getItem("um-theme");
  if (saved) document.documentElement.dataset.theme = saved;
  themeBtn?.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("um-theme", next);
  });
  addEventListener("scroll", () => {
    const h = document.documentElement;
    const max = h.scrollHeight - h.clientHeight;
    const bar = $("#progress");
    if (bar) bar.style.width = `${max > 0 ? (h.scrollTop / max) * 100 : 0}%`;
  }, { passive: true });

  if (!DATA || !(DATA.vision?.length || DATA.text?.length)) {
    figure.append(el("p", "pending",
      "The atlas data file is not here yet. Run the Kaggle kernel, then `bash kaggle/atlas/fetch.sh` — the page reads atlas-data.js, it does not measure anything itself."));
    return;
  }

  const list = () => DATA[state.family] || [];

  function pills() {
    const rail = $("#models");
    rail.textContent = "";
    list().forEach((d, i) => {
      const b = el("button", "pill" + (i === state.idx ? " on" : ""), d.name || d.tag);
      b.type = "button";
      b.setAttribute("role", "tab");
      b.setAttribute("aria-selected", i === state.idx ? "true" : "false");
      b.onclick = () => { state.idx = i; state.img = 0; render(); };
      rail.append(b);
    });
  }

  function cellsColumn(mapBytes, side, cmap, tip) {
    const c = el("canvas");
    c.width = side;
    c.height = side;
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(side, side);
    for (let i = 0; i < side * side; i++) {
      const [r, g, b] = cmap(mapBytes[i] / 255);
      const o = i * 4;
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    const cell = el("div", "cell tip");
    cell.dataset.tip = tip;
    cell.append(c);
    return cell;
  }

  function channelFigure(img) {
    const box = el("div", "chan");
    const c = el("canvas");
    c.width = c.height = 32;
    const ctx = c.getContext("2d");
    ctx.drawImage(img, 0, 0, 32, 32);
    const src = ctx.getImageData(0, 0, 32, 32).data;
    ["R", "G", "B"].forEach((name, ch) => {
      const cc = el("canvas");
      cc.width = cc.height = 32;
      const cx = cc.getContext("2d");
      const im = cx.createImageData(32, 32);
      for (let i = 0; i < 32 * 32; i++) {
        const o = i * 4;
        im.data[o] = ch === 0 ? src[o] : 0;
        im.data[o + 1] = ch === 1 ? src[o + 1] : 0;
        im.data[o + 2] = ch === 2 ? src[o + 2] : 0;
        im.data[o + 3] = 255;
      }
      cx.putImageData(im, 0, 0);
      const f = el("figure");
      f.append(cc, el("figcaption", null, name));
      box.append(f);
    });
    return box;
  }

  function outList(rows) {
    const box = el("div", "out-list");
    const max = Math.max(...rows.map(r => r.p), 0.0001);
    rows.forEach(r => {
      const row = el("div", "out-row");
      const label = r.label ?? (r.tok ?? "").replace(/^ /, "␣");
      row.append(el("span", null, label), el("span", "p", `${(r.p * 100).toFixed(1)}%`));
      const bar = el("div", "out-bar");
      const fill = el("i");
      fill.style.width = `${Math.max(2, (r.p / max) * 100)}%`;
      bar.append(fill);
      row.append(bar);
      box.append(row);
    });
    return box;
  }

  function links(inner) {
    const svg = sv("svg", { class: "links" });
    inner.prepend(svg);
    const base = inner.getBoundingClientRect();
    svg.setAttribute("width", inner.scrollWidth);
    svg.setAttribute("height", inner.scrollHeight);
    const cols = [...inner.querySelectorAll(".col")];
    for (let i = 0; i < cols.length - 1; i++) {
      const a = [...cols[i].querySelectorAll(".cell")];
      const b = [...cols[i + 1].querySelectorAll(".cell")];
      const n = Math.min(a.length, b.length);
      for (let r = 0; r < n; r++) {
        const ra = a[r].getBoundingClientRect();
        const rb = b[r].getBoundingClientRect();
        const x1 = ra.right - base.left;
        const y1 = ra.top - base.top + ra.height / 2;
        const x2 = rb.left - base.left;
        const y2 = rb.top - base.top + rb.height / 2;
        const mx = (x1 + x2) / 2;
        svg.append(sv("path", { d: `M${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}` }));
      }
    }
  }

  function imageButtons(d) {
    const box = $("#imgpick");
    if (!box) return;
    box.textContent = "";
    const imgs = d.images || [];
    if (state.family !== "vision" || imgs.length < 2) return;
    imgs.forEach((im, i) => {
      const b = el("button", "ipick" + (i === state.img ? " on" : ""), im.cls);
      b.type = "button";
      b.onclick = () => { state.img = i; render(); };
      box.append(b);
    });
    box.append(el("span", "ipick-note", "same weights, second image"));
  }

  function renderVision(d) {
    figure.style.setProperty("--cell", "54px");
    const imgs = d.images || [{ cls: d.image?.cls, src: d.image?.src, dims: d.dims,
                                layers: d.layers, classes: d.classes }];
    const I = imgs[state.img] || imgs[0];
    imageButtons(d);
    const inner = el("div", "fig-inner");
    const inp = el("aside", "fig-input");
    const img = el("img");
    img.alt = `input image: ${I.cls}`;
    img.src = I.src;
    inp.append(img, el("div", "cap", `input · ${I.cls}`));
    (img.decode ? img.decode() : Promise.resolve())
      .then(() => inp.append(channelFigure(img))).catch(() => {});
    inp.append(el("div", "cap", `rows · 6 of ${d.facts.hidden} dims`));

    const cols = el("div", "cols");
    I.layers.forEach((L, li) => {
      const col = el("section", "col");
      col.append(el("h4", null, li === 0 ? "patch" : `block ${li}`));
      const cellBox = el("div", "cells");
      L.maps.forEach((m, j) => {
        cellBox.append(cellsColumn(bytesOf(m), d.facts.grid, divColor,
          `${li === 0 ? "patch" : "block " + li} · dim ${I.dims[j]}`));
      });
      col.append(cellBox);
      const cb = el("div", "cbar");
      cb.append(el("i"), el("span", null, `±${L.vmax.toFixed(1)}`));
      col.append(cb);
      cols.append(col);
    });

    const out = el("aside", "fig-out");
    out.append(el("h4", null, "readout"));
    if (I.classes.length) out.append(outList(I.classes.slice(0, 8)));
    out.append(el("p", "out-note", d.note));

    inner.append(inp, cols, out);
    figure.textContent = "";
    figure.append(inner);
    links(inner);

    $("#facts").textContent = `${d.name} · ${d.facts.layers} blocks · ${d.facts.grid}×${d.facts.grid} patches · ${I.cls} · one clean pass, ${d.facts.seconds}s on ${d.facts.device}`;
    $("#note").textContent = `Each cell is one hidden dimension at every patch position, scaled to that layer’s largest absolute value. The six rows are chosen by variance at the last layer, so the same row means the same dimension all the way down. ${d.note}.`;
    legend(false);
  }

  function renderText(d) {
    const imgbox = $("#imgpick");
    if (imgbox) imgbox.textContent = "";
    figure.style.setProperty("--cell", "34px");
    const inner = el("div", "fig-inner");
    const inp = el("aside", "fig-input");
    const p = el("div", "prompt", d.prompt);
    inp.append(p, el("div", "cap",
      `${d.facts.tokens} tokens · ${d.facts.layers} layers × ${d.facts.heads} heads`));
    if (d.chat) inp.append(el("div", "cap", "forced prefix · “The answer is”"));

    const cols = el("div", "cols");
    d.layers.forEach((heads, li) => {
      const col = el("section", "col");
      col.append(el("h4", null, `L${li + 1}`));
      const cellBox = el("div", "cells");
      heads.forEach((m, h) => {
        cellBox.append(cellsColumn(bytesOf(m), d.facts.tokens, seqColor,
          `layer ${li + 1} · head ${h + 1}`));
      });
      col.append(cellBox);
      cols.append(col);
    });

    const out = el("aside", "fig-out");
    out.append(el("h4", null, "next token"));
    out.append(outList(d.next));
    out.append(el("p", "out-note", "the model’s own distribution after the prompt"));

    inner.append(inp, cols, out);
    figure.textContent = "";
    figure.append(inner);
    links(inner);

    $("#facts").textContent = `${d.tag} · ${d.facts.layers} layers × ${d.facts.heads} heads · ${d.facts.tokens} prompt tokens · one clean pass, ${d.facts.seconds}s on ${d.facts.device}`;
    $("#note").textContent = `One cell is one head’s attention: rows are queries, columns keys, darker is more attention. Only the last ${d.facts.tokens} prompt tokens are shown, so the grid stays readable. ${d.chat ? "Chat models were given the question and the forced prefix “The answer is”." : "GPT-2 was given the plain completion prompt."}`;
    legend(true);
  }

  function legend(seq) {
    const box = $("#legend");
    box.textContent = "";
    const ramp = el("span", "ramp" + (seq ? " seq" : ""));
    if (seq) {
      box.append(ramp, el("span", null, "attention 0 → 1"));
      const keys = el("span", "keys");
      keys.append(el("i"), el("span", null, "row = query · column = key"));
      box.append(keys);
    } else {
      box.append(el("span", null, "−max"), ramp, el("span", null, "+max"));
      box.append(el("span", null, "scaled per layer · blue and red are sign, not class"));
    }
  }

  function render() {
    const d = list()[state.idx];
    if (!d) return;
    pills();
    if (state.family === "vision") renderVision(d); else renderText(d);
    addEventListener("resize", () => {
      const svg = figure.querySelector("svg.links");
      if (svg) { svg.remove(); const inner = figure.querySelector(".fig-inner"); if (inner) links(inner); }
    }, { once: true });
  }

  $("#family").addEventListener("click", e => {
    const b = e.target.closest("button[data-family]");
    if (!b) return;
    state.family = b.dataset.family;
    state.idx = 0;
    state.img = 0;
    [...$("#family").querySelectorAll("button")].forEach(x => {
      x.classList.toggle("on", x === b);
      x.setAttribute("aria-selected", x === b ? "true" : "false");
    });
    render();
  });

  state.family = DATA.vision?.length ? "vision" : "text";
  [...$("#family").querySelectorAll("button")].forEach(x => {
    const on = x.dataset.family === state.family;
    x.classList.toggle("on", on);
    x.setAttribute("aria-selected", on ? "true" : "false");
  });
  render();
})();
