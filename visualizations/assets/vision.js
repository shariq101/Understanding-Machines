/* The vision lab. Reads captured window.VISION_DATA measurements.
   and builds every panel. No model runs here: the controls select among
   conditions that were measured on a GPU and stored in the data file. */
/* The data file is rewritten in place whenever a sweep lands, so the same URL
   can hold different numbers.  The script tag makes the page work from file://;
   when the page is served, refetch the same file without the HTTP cache, because
   otherwise a plain reload keeps executing the copy the browser already had. */
async function loadVisionData() {
  try {
    const res = await fetch("vision-data.js?ts=" + Date.now(), { cache: "no-store" });
    if (!res.ok) throw new Error(String(res.status));
    const text = await res.text();
    const m = text.match(/^\s*window\.VISION_DATA\s*=\s*([\s\S]*?);?\s*$/);
    if (!m) throw new Error("unexpected data format");
    return JSON.parse(m[1]);
  } catch (err) {
    return window.VISION_DATA || null;
  }
}

(async () => {
  "use strict";
  const D = await loadVisionData();
  const $ = (s, r = document) => r.querySelector(s);
  const nf = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(+x)) ? "—" : (+x).toFixed(d);
  const pct = (x, d = 0) => (x === null || x === undefined || !Number.isFinite(+x)) ? "—" : (+x * 100).toFixed(d) + "%";
  const sci = x => !Number.isFinite(+x) ? "—" : (Math.abs(x) < 1e-3 ? (+x).toExponential(1) : nf(x, 4));
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };
  /* the palette interpolates against the current theme, so a theme switch needs
     no repaint: the custom properties are read at paint time */
  const ramp = t => {
    t = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0));
    return `color-mix(in srgb, var(--ox) ${Math.round(t * 100)}%, var(--paper3))`;
  };
  const metricOf = m => (m.facts.metric === "kl" ? "KL" : m.facts.metric === "mse" ? "MSE" : m.facts.metric);

  if (!D || !D.models || !D.models.length) {
    const main = $(".lab");
    if (main) main.insertAdjacentHTML("beforeend",
      '<p class="pending"><b>The data file is missing.</b> Run <code>python experiments/vacdc/web.py</code>, then reload.</p>');
    return;
  }

  const state = { mi: 0, node: null, mode: "damage", dir: "ablate", i: 0, j: 0, off: new Set() };
  const model = () => D.models[state.mi];

  /* ---------- shared chrome: progress, theme ---------- */
  const bar = $("#progress");
  const onScroll = () => {
    const h = document.documentElement;
    const max = h.scrollHeight - h.clientHeight;
    if (bar) bar.style.width = `${max > 0 ? (h.scrollTop / max) * 100 : 0}%`;
  };
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  const saved = localStorage.getItem("um-theme");
  if (saved) document.documentElement.dataset.theme = saved;
  $("#themebtn")?.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("um-theme", next);
  });

  /* ---------- facts ---------- */
  function renderFacts() {
    const m = model(), f = m.facts, b = m.baseline, c = m.circuit;
    const box = $("#facts");
    box.textContent = "";
    const items = [
      [`${nf(f.parameters / 1e6, 1)} M`, "parameters"],
      [`${f.depth} × ${f.heads}`, "blocks × heads"],
      [`${f.hidden}`, "hidden width"],
      [`${f.patch} px · ${f.tokens}`, "patch · tokens"],
    ];
    if (b.clean_acc !== null && b.clean_acc !== undefined) items.push([pct(b.clean_acc), "clean accuracy"]);
    items.push([nf(b.gap, 2), `clean → corrupt · ${metricOf(m)}`]);
    items.push([`${c.size}/${c.total}`, "edges kept"]);
    items.push([sci(Math.abs(m.exactness.all_edges_gap)), "rebuild error"]);
    for (const [v, k] of items) {
      const d = el("div", "fact");
      d.append(el("b", "", v), el("span", "", k));
      box.append(d);
    }
  }

  /* ---------- atlas ---------- */
  function renderAtlas() {
    const m = model(), f = m.facts, depth = f.depth, heads = f.heads;
    const gap = m.baseline.gap, nodes = m.atlas;
    let maxd = 0;
    for (const k in nodes) maxd = Math.max(maxd, nodes[k].d || 0);
    const box = $("#atlas");
    box.textContent = "";

    const g = el("div", "mx");
    g.style.gridTemplateColumns = `var(--head) repeat(${heads + 1}, var(--cell))`;
    g.append(el("div", "corner"));
    for (let h = 0; h <= heads; h++) g.append(el("div", "hd", h === heads ? "M" : String(h)));
    for (let l = 0; l < depth; l++) {
      g.append(el("div", "rh", String(l)));
      for (let h = 0; h <= heads; h++) {
        const key = h === heads ? `L${l}M` : `L${l}H${h}`;
        const rec = nodes[key];
        const b = el("button", "cell");
        b.type = "button";
        if (!rec) {
          b.classList.add("none");
          b.disabled = true;
          b.title = `${key}: not measured`;
        } else {
          const t = state.mode === "acc" && rec.acc !== null && rec.acc !== undefined
            ? 1 - rec.acc
            : Math.log1p(rec.d) / Math.log1p(maxd);
          b.style.background = ramp(t);
          const bits = [`${key} · ${metricOf(m)} ${nf(rec.d)}`, `${pct(rec.d / gap)} of the gap`];
          if (rec.acc !== null && rec.acc !== undefined) bits.push(`accuracy ${pct(rec.acc)}`);
          b.title = bits.join(" · ");
          b.setAttribute("aria-label", bits.join(", "));
          if (state.node === key) b.classList.add("lit");
          b.addEventListener("pointerenter", () => atlasRead(key, rec));
          b.addEventListener("focus", () => atlasRead(key, rec));
          b.addEventListener("click", () => { state.node = key; renderAtlas(); atlasRead(key, rec); });
        }
        g.append(b);
      }
    }
    box.append(g);
    box.append(el("div", "axlabel", `rows: layer · columns: heads 0–${heads - 1}, then the MLP`));

    const ctl = $("#atlas-ctl");
    ctl.textContent = "";
    if (m.kind !== "mae") {
      const seg = el("div", "seg");
      for (const [mode, label] of [["damage", "Damage"], ["acc", "Accuracy"]]) {
        const b = el("button", "", label);
        b.type = "button";
        b.setAttribute("aria-pressed", String(state.mode === mode));
        b.addEventListener("click", () => { state.mode = mode; renderAtlas(); });
        seg.append(b);
      }
      ctl.append(seg);
    }

    $("#atlas-legend").innerHTML =
      `<span class="sw" style="background:${ramp(0)}"></span>no effect` +
      `<span class="sw" style="background:${ramp(0.5)};margin-left:12px"></span>half the gap` +
      `<span class="sw" style="background:${ramp(1)};margin-left:12px"></span>the whole gap` +
      (state.mode === "acc" ? " · colour is 1 − accuracy after the replacement" : " · colour is log-scaled distance from the clean answer");
    if (!state.node) {
      $("#atlas-read").innerHTML = '<span class="q">Hover a cell to read what that node alone contributes. Click to pin it.</span>';
    }
  }

  function atlasRead(key, rec) {
    const m = model(), gap = m.baseline.gap;
    const parts = [`<b>${key}</b>`, `distance ${metricOf(m)} ${nf(rec.d)} (${pct(rec.d / gap)} of the gap)`];
    if (rec.acc !== null && rec.acc !== undefined) parts.push(`accuracy after this one replacement ${pct(rec.acc)}`);
    const layer = key.match(/^L(\d+)/);
    if (layer) parts.push(`<span class="q">only layer ${layer[1]} is altered; every other node is clean</span>`);
    $("#atlas-read").innerHTML = parts.join(" · ");
  }

  /* ---------- the layer playground ---------- */
  function renderLayer() {
    const m = model(), L = m.layers;
    const box = $("#layer-matrix"), ctl = $("#layer-ctl"), read = $("#layer-read"), recon = $("#layer-recon");
    box.textContent = ""; ctl.textContent = ""; recon.textContent = "";
    if (!L) return renderLayerPending(m);

    const depth = L.depth, gap = L.gap, metric = metricOf(m);
    const cells = new Map(L.cells.map(c => [`${c.dir}-${c.i}-${c.j}`, c]));
    const cur = () => cells.get(`${state.dir}-${state.i}-${state.j}`);
    const isMae = m.kind === "mae";

    /* direction */
    const seg = el("div", "seg");
    for (const [dir, label] of [["ablate", "Remove the block"], ["keep", "Keep only the block"]]) {
      const b = el("button", "", label);
      b.type = "button";
      b.setAttribute("aria-pressed", String(state.dir === dir));
      b.addEventListener("click", () => { state.dir = dir; renderLayer(); });
      seg.append(b);
    }
    ctl.append(seg);

    /* two handles on the same axis */
    const slider = (which, label) => {
      const wrap = el("label", "", label);
      const input = document.createElement("input");
      input.type = "range";
      input.min = "0";
      input.max = String(depth - 1);
      input.step = "1";
      input.value = String(state[which]);
      input.id = `layer-${which}`;
      const out = el("output", "", String(state[which]));
      input.addEventListener("input", () => {
        const v = +input.value;
        state[which] = v;
        if (state.i > state.j) {
          if (which === "i") state.j = v; else state.i = v;
          $("#layer-i").value = String(state.i);
          $("#layer-j").value = String(state.j);
          $("#layer-i").nextElementSibling.textContent = String(state.i);
          $("#layer-j").nextElementSibling.textContent = String(state.j);
        }
        out.textContent = String(v);
        paint();
      });
      wrap.append(input, out);
      return wrap;
    };
    ctl.append(slider("i", "first layer"), slider("j", "last layer"));

    /* the grid: rows are the first layer, columns the last */
    const g = el("div", "mx");
    g.style.gridTemplateColumns = `var(--head) repeat(${depth}, var(--cell))`;
    g.append(el("div", "corner"));
    for (let j = 0; j < depth; j++) g.append(el("div", "hd", String(j)));
    const buttons = new Map();
    for (let i = 0; i < depth; i++) {
      g.append(el("div", "rh", String(i)));
      for (let j = 0; j < depth; j++) {
        const b = el("button", "cell");
        b.type = "button";
        if (j < i) {
          b.classList.add("none");
          b.disabled = true;
          b.setAttribute("aria-hidden", "true");
          b.tabIndex = -1;
        } else {
          const rec = cells.get(`${state.dir}-${i}-${j}`);
          if (!rec || rec.frac === null || rec.frac === undefined) {
            b.classList.add("none");
            b.disabled = true;
          } else {
            const t = Math.pow(Math.max(0, Math.min(1, rec.frac)), 0.75);
            b.style.background = ramp(t);
            b.title = `layers ${i}–${j} · ${state.dir === "ablate" ? "removed" : "kept alone"} · ` +
              `${metric} ${nf(rec.d)} (${pct(rec.frac)} of the gap)` +
              (rec.p !== null && rec.p !== undefined ? ` · true-class probability ${nf(rec.p)}` : "") +
              ` · ${rec.edges} edges in the block`;
            b.setAttribute("aria-label", b.title);
            b.addEventListener("click", () => {
              state.i = i; state.j = j;
              $("#layer-i").value = String(i); $("#layer-j").value = String(j);
              $("#layer-i").nextElementSibling.textContent = String(i);
              $("#layer-j").nextElementSibling.textContent = String(j);
              paint();
            });
            b.addEventListener("pointerenter", () => {
              $("#layer-read").innerHTML = blockSentence(rec, { i, j, dir: state.dir, metric, gap, total: L.edge_total });
            });
          }
          buttons.set(`${i}-${j}`, b);
        }
        g.append(b);
      }
    }
    box.append(g);
    const ax = el("div", "axlabel", "rows: first layer · columns: last layer · a cell is one measured block");
    box.append(ax);

    function paint() {
      for (const [key, b] of buttons) {
        const [i, j] = key.split("-").map(Number);
        b.classList.toggle("lit", i === state.i && j === state.j);
      }
      read.innerHTML = blockSentence(cur(), { i: state.i, j: state.j, dir: state.dir, metric, gap, total: L.edge_total });
      paintRecon();
    }

    function paintRecon() {
      const rec = cur(), sheet = L.sprites && L.sprites[state.dir];
      recon.textContent = "";
      if (!isMae) return;
      if (sheet && rec) {
        const fig = el("figure");
        const px = el("span", "px");
        /* cell (i, j) sits at row i, column j - i of the sheet */
        px.style.backgroundImage = `url("${sheet.url}")`;
        px.style.backgroundSize = `${depth * sheet.cell}px ${depth * sheet.cell}px`;
        px.style.backgroundPosition = `${-(state.j - state.i) * sheet.cell}px ${-state.i * sheet.cell}px`;
        fig.append(px, el("figcaption", "", `reconstruction · ${state.dir === "ablate" ? "block removed" : "block kept"} ${state.i}–${state.j}`));
        recon.append(fig);
      }
      for (const [key, label] of [["clean_0", "clean"], ["input_0", "input"]]) {
        const url = m.study.images[key];
        if (!url) continue;
        const fig = el("figure");
        const px = el("span", "px");
        px.style.backgroundImage = `url("${url}")`;
        px.style.backgroundSize = "cover";
        fig.append(px, el("figcaption", "", label));
        recon.append(fig);
      }
    }

    paint();
  }

  function blockSentence(rec, { i, j, dir, metric, gap, total }) {
    if (!rec || rec.d === null || rec.d === undefined) {
      return `<span class="q">Layers ${i}–${j} are not in the measured grid.</span>`;
    }
    const span = i === j ? `layer ${i}` : `layers ${i}–${j}`;
    let s;
    if (dir === "ablate") {
      s = `<b>${span} removed</b> — the answer moves <b>${pct(rec.frac)}</b> of the way to the corrupted image ` +
        `(${metric} ${nf(rec.d)} of ${nf(gap, 2)})`;
    } else {
      s = `<b>${span} kept alone</b> — the answer sits <b>${pct(rec.frac)}</b> of the gap from clean ` +
        `(${metric} ${nf(rec.d)} of ${nf(gap, 2)})`;
    }
    if (rec.p !== null && rec.p !== undefined) s += `; the true class keeps probability <b>${nf(rec.p)}</b>`;
    s += total ? `. That block sends ${rec.edges} of the ${total} edges.` : `.`;
    return s;
  }

  function renderLayerPending(m) {
    const box = $("#layer-matrix"), read = $("#layer-read");
    box.textContent = "";
    read.textContent = "";
    const note = el("div", "pending");
    note.innerHTML = "<b>The block grid is still being measured.</b> A sweep over every contiguous range of layers is " +
      "queued for this model; the panel fills in when the data file is rebuilt. What the run has already measured is the " +
      "depth curve: keeping the circuit's edges through layer k.";
    box.append(note);
    const rows = m.depth_curve || [];
    if (!rows.length) return;
    const tbl = el("table", "tbl");
    const cap = el("caption", "", "keep the circuit's edges in layers 0…k−1");
    tbl.append(cap);
    const head = el("tr");
    for (const h of ["k", "edges kept", metricOf(m), "% of gap"]) head.append(el("th", "", h));
    tbl.append(head);
    for (const r of rows) {
      const tr = el("tr");
      tr.append(el("td", "", String(r.keep_layers)), el("td", "", String(r.size)),
        el("td", "", nf(r.distance)), el("td", "", pct(r.distance / m.baseline.gap)));
      tbl.append(tr);
    }
    box.append(tbl);
  }

  /* ---------- ablate by hand ----------
     A click replaces one node with the write it produced for a different image.
     Only two shapes of selection were ever measured: a single node, and every
     node of a contiguous run of layers (the block grid). Anything else is a
     sum of separate measurements, and is printed as one. */
  const maxAtlas = m => { let x = 0; for (const k in m.atlas) x = Math.max(x, m.atlas[k].d || 0); return x; };

  function blockSelection(m) {
    if (!state.off.size) return null;
    const depth = m.facts.depth, heads = m.facts.heads;
    const full = [], partial = [];
    for (let l = 0; l < depth; l++) {
      let on = 0;
      for (let h = 0; h < heads; h++) if (state.off.has(`L${l}H${h}`)) on++;
      if (state.off.has(`L${l}M`)) on++;
      if (on === heads + 1) full.push(l);
      else if (on > 0) partial.push(l);
    }
    if (partial.length || !full.length) return null;
    for (let k = 1; k < full.length; k++) if (full[k] !== full[k - 1] + 1) return null;
    return { i: full[0], j: full[full.length - 1] };
  }

  function renderAblate() {
    const m = model(), heads = m.facts.heads, depth = m.facts.depth;
    const metric = metricOf(m), gap = m.baseline.gap, top = maxAtlas(m);
    const box = $("#abl-grid");
    box.textContent = "";

    const g = el("div", "mx");
    g.style.setProperty("--cell", "19px");
    g.style.gridTemplateColumns = `var(--head) repeat(${heads + 1}, var(--cell))`;
    g.append(el("div", "corner"));
    for (let h = 0; h <= heads; h++) g.append(el("div", "hd", h === heads ? "M" : String(h)));
    for (let l = 0; l < depth; l++) {
      g.append(el("div", "rh", String(l)));
      for (let h = 0; h <= heads; h++) {
        const key = h === heads ? `L${l}M` : `L${l}H${h}`;
        const rec = m.atlas[key];
        const b = el("button", "cell");
        b.type = "button";
        if (!rec) {
          b.classList.add("none");
          b.disabled = true;
        } else {
          b.style.background = ramp(Math.log1p(rec.d) / Math.log1p(top));
          b.classList.toggle("off", state.off.has(key));
          b.setAttribute("aria-pressed", String(state.off.has(key)));
          b.title = `${key} · ${metric} ${nf(rec.d)}` +
            (rec.acc !== null && rec.acc !== undefined ? ` · accuracy ${pct(rec.acc)}` : "") +
            (state.off.has(key) ? " · replaced, click to restore" : " · click to replace");
          b.setAttribute("aria-label", b.title);
          b.addEventListener("click", () => {
            if (state.off.has(key)) state.off.delete(key); else state.off.add(key);
            renderAblate();
          });
        }
        g.append(b);
      }
    }
    box.append(g);
    box.append(el("div", "axlabel", "rows: layer · columns: heads, then the MLP · a cross means replaced"));

    const ctl = $("#abl-ctl");
    ctl.textContent = "";
    const clear = el("button", "tab", "Clear");
    clear.type = "button";
    clear.disabled = !state.off.size;
    clear.addEventListener("click", () => { state.off.clear(); renderAblate(); });
    ctl.append(clear);
    ctl.append(el("span", "hint", state.off.size
      ? `${state.off.size} node${state.off.size === 1 ? "" : "s"} replaced`
      : "nothing replaced yet"));

    renderAblRead(m, metric, gap);
  }

  function renderAblRead(m, metric, gap) {
    const read = $("#abl-read"), ledger = $("#abl-ledger");
    read.textContent = "";
    ledger.textContent = "";
    const keys = [...state.off];
    if (!keys.length) {
      read.innerHTML = '<span class="q">Click a cell to replace that node. The cost you get back is the measured single-node cost, not an estimate.</span>';
      return;
    }
    const nodes = keys.map(k => ({ key: k, rec: m.atlas[k] })).filter(x => x.rec);
    const sum = nodes.reduce((a, x) => a + (x.rec.d || 0), 0);
    const block = blockSelection(m);
    const cell = block && m.layers
      ? m.layers.cells.find(c => c.dir === "ablate" && c.i === block.i && c.j === block.j)
      : null;

    if (nodes.length === 1) {
      const n = nodes[0];
      const acc = n.rec.acc !== null && n.rec.acc !== undefined
        ? `; accuracy after that one replacement <b>${pct(n.rec.acc)}</b>` : "";
      read.innerHTML = `<span class="badge ok">measured</span><b>${n.key}</b> replaced by its corrupt write — ` +
        `${metric} <b>${nf(n.rec.d)}</b>, ${pct((n.rec.d || 0) / gap)} of the gap${acc}.`;
    } else if (cell) {
      const p = cell.p !== null && cell.p !== undefined
        ? `; the true class keeps probability <b>${nf(cell.p)}</b>` : "";
      const span = block.i === block.j ? `layer ${block.i}` : `layers ${block.i}–${block.j}`;
      read.innerHTML = `<span class="badge ok">measured</span>Every node in <b>${span}</b> replaced — ` +
        `${metric} <b>${nf(cell.d)}</b>, ${pct(cell.frac)} of the gap${p}. ` +
        '<span class="q">That is the block grid\u2019s own cell, not a sum.</span>';
    } else {
      const layers = new Set(keys.map(k => k.slice(1).split("H")[0].split("M")[0])).size;
      const pending = block && !cell
        ? ' <span class="q">This range is exactly what the block grid measures; it is still running for this model, so the panel cannot quote it yet.</span>'
        : "";
      read.innerHTML = `<span class="badge est">not measured</span><b>${nodes.length} nodes</b> across ${layers} layer${layers === 1 ? "" : "s"} — ` +
        "this exact set was never run. The separate measurements sum to " +
        `${metric} <b>${nf(sum)}</b>, ${pct(sum / gap)} of the gap; the joint effect is not that sum.${pending}`;
    }
    renderLedger(ledger, nodes, metric, gap);
  }

  function renderLedger(box, nodes, metric, gap) {
    if (!nodes.length) return;
    const sorted = [...nodes].sort((a, b) => (b.rec.d || 0) - (a.rec.d || 0));
    const hasAcc = sorted[0].rec.acc !== null && sorted[0].rec.acc !== undefined;
    const tbl = el("table", "tbl");
    tbl.append(el("caption", "", "each replaced node, measured on its own"));
    const head = el("tr");
    for (const h of ["node", metric, "gap", ...(hasAcc ? ["accuracy"] : [])]) head.append(el("th", "", h));
    tbl.append(head);
    for (const n of sorted.slice(0, 10)) {
      const tr = el("tr");
      tr.append(el("td", "", n.key), el("td", "", nf(n.rec.d, 3)), el("td", "", pct((n.rec.d || 0) / gap)));
      if (hasAcc) tr.append(el("td", "", pct(n.rec.acc)));
      tbl.append(tr);
    }
    box.append(tbl);
    if (sorted.length > 10) {
      box.append(el("p", "legend", `and ${sorted.length - 10} more, all counted in the total above.`));
    }
  }

  /* ---------- circuit ---------- */
  function renderCircuit() {
    const m = model(), c = m.circuit, gap = m.baseline.gap;
    const box = $("#circuit");
    box.textContent = "";

    const read = el("p", "readout");
    read.innerHTML =
      `Removing the circuit costs <b>${pct(c.necessity)}</b> of the gap; removing a random set of the same size costs ` +
      `<b>${pct(c.random_remove / gap)}</b> (six draws, ${pct(c.random_remove_min / gap)}–${pct(c.random_remove_max / gap)}). ` +
      `Keeping only the circuit leaves <b>${pct(c.sufficiency)}</b>; keeping only a random set leaves ` +
      `<b>${pct(c.random_keep_only / gap)}</b>.`;
    box.append(read);

    const tbl = el("table", "tbl");
    tbl.append(el("caption", "", `distance from the clean answer · percent of the ${nf(gap, 2)} ${metricOf(m)} clean-to-corrupt gap`));
    const head = el("tr");
    for (const h of ["condition", metricOf(m), "gap"]) head.append(el("th", "", h));
    tbl.append(head);
    const rows = [
      ["the circuit, removed", c.remove],
      ["a random set of the same size, removed", c.random_remove],
      ["only the circuit, kept", c.keep_only],
      ["only a random set, kept", c.random_keep_only],
      [`every edge kept (${c.size} vs ${c.total} is the circuit)`, c.all_edges],
      ["every edge corrupt (the gap itself)", c.corrupt],
    ];
    for (const [label, v] of rows) {
      const tr = el("tr");
      tr.append(el("td", "", label), el("td", "", nf(v, 3)), el("td", "", pct(v / gap)));
      tbl.append(tr);
    }
    box.append(tbl);

    if (c.node_size) {
      box.append(el("p", "legend", `${c.node_size} nodes survive head-level ACDC in this model. Threshold on the edge run: ${nf(c.threshold, 4)}.`));
    }
    renderOcclusion(m);
  }

  function renderOcclusion(m) {
    const box = $("#occlusion");
    box.textContent = "";
    const p = m.study && m.study.patch;
    if (!p) return;
    const ks = Object.keys(p).filter(k => /^mask\d+$/.test(k)).sort((a, b) => +a.slice(4) - +b.slice(4));
    if (!ks.length) return;
    const clean = m.study.conditions && m.study.conditions.clean;
    const base = clean && clean.prob ? clean.prob.reduce((a, b) => a + b, 0) / clean.prob.length : null;
    const tbl = el("table", "tbl");
    tbl.append(el("caption", "", "occluding the k patches the circuit's residual difference favours, against the same number at random · probability of the true class"));
    const head = el("tr");
    for (const h of ["patches", "circuit-guided", "random", "none"]) head.append(el("th", "", h));
    tbl.append(head);
    for (const k of ks) {
      const r = p[k], tr = el("tr");
      tr.append(el("td", "", k.slice(4)),
        el("td", "", nf(r.circuit, 3)), el("td", "", nf(r.random, 3)),
        el("td", "", base === null ? "—" : nf(base, 3)));
      tbl.append(tr);
    }
    box.append(tbl);
  }

  /* ---------- where the circuit sits ---------- */
  function renderCircuitMatrix() {
    const m = model(), c = m.circuit, grid = c.matrix, depth = m.facts.depth;
    const box = $("#circuit-matrix");
    box.textContent = "";
    let max = 0;
    for (const row of grid) for (const v of row) max = Math.max(max, v);
    const g = el("div", "mx");
    g.style.setProperty("--cell", "18px");
    g.style.gridTemplateColumns = `var(--head) repeat(${depth + 1}, var(--cell))`;
    g.append(el("div", "corner"));
    for (let col = 0; col <= depth; col++) {
      g.append(el("div", "hd", col === depth ? "out" : String(col)));
    }
    for (let row = 0; row <= depth; row++) {
      g.append(el("div", "rh", row === 0 ? "E" : String(row - 1)));
      for (let col = 0; col <= depth; col++) {
        const v = grid[row][col];
        const b = el("button", "cell");
        b.type = "button";
        if (!v) {
          b.classList.add("none");
          b.disabled = false;
          b.title = `${row === 0 ? "the input" : "layer " + (row - 1)} → ${col === depth ? "the output" : "layer " + col}: no kept edge`;
        } else {
          b.style.background = ramp(max ? Math.log1p(v) / Math.log1p(max) : 0);
          b.title = `${row === 0 ? "the input" : "layer " + (row - 1)} → ${col === depth ? "the output" : "layer " + col}: ${v} kept edge${v === 1 ? "" : "s"}`;
          b.setAttribute("aria-label", b.title);
        }
        g.append(b);
      }
    }
    box.append(g);
    box.append(el("div", "axlabel", "rows: sender · columns: receiver · the last column is the output head"));
    $("#circuit-legend").innerHTML =
      `<span class="sw" style="background:${ramp(0.08)}"></span>one kept edge` +
      `<span class="sw" style="background:${ramp(0.6)};margin-left:12px"></span>the most kept edges in any cell` +
      `<span class="sw" style="background:repeating-linear-gradient(45deg,transparent,transparent 3px,var(--line) 3px,var(--line) 4px);margin-left:12px"></span>none`;
  }

  /* ---------- wiring ---------- */
  function selectModel(idx) {
    state.mi = idx;
    state.node = null;
    state.dir = "ablate";
    state.off.clear();
    const depth = model().facts.depth;
    state.i = Math.floor(depth / 2);
    state.j = depth - 1;
    for (const [i, tab] of [...$("#tabs").children].entries()) tab.setAttribute("aria-selected", String(i === idx));
    renderFacts(); renderAtlas(); renderAblate(); renderLayer(); renderCircuit(); renderCircuitMatrix();
  }

  function renderTabs() {
    const tabs = $("#tabs");
    tabs.textContent = "";
    D.models.forEach((m, idx) => {
      const b = el("button", "tab", m.name);
      b.type = "button";
      b.setAttribute("role", "tab");
      b.title = `${m.model} — ${m.note}`;
      b.addEventListener("click", () => selectModel(idx));
      tabs.append(b);
    });
  }

  renderTabs();
  selectModel(0);
})();
