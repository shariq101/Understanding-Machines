/* The ablation studio. Reads the same measured file as the vision lab and
   presents four views of one experiment: the layer board, the damage cascade,
   the map of every measured run, and the family of deletion curves. No model
   runs here. Every reading is a lookup, and the panel says when a set was not
   measured as a set. */

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
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
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
  const metricOf = m => m.facts.metric === "kl" ? "KL" : m.facts.metric === "mse" ? "MSE" : m.facts.metric;

  if (!D || !D.models || !D.models.length) {
    const main = $(".studio");
    if (main) main.insertAdjacentHTML("beforeend",
      '<p class="pending"><b>The data file is missing.</b> Run <code>python experiments/vacdc/web.py</code>, then reload.</p>');
    return;
  }

  const state = { mi: 0, view: "board", off: new Set() };
  const model = () => D.models[state.mi];
  const depthOf = m => m.facts.depth;
  const headsOf = m => m.facts.heads;
  const keyOf = (l, h, m) => h === headsOf(m) ? `L${l}M` : `L${l}H${h}`;
  const layerOf = k => +k.slice(1).split(/[HM]/)[0];
  const layerKeys = (l, m) => Array.from({ length: headsOf(m) + 1 }, (_, h) => keyOf(l, h, m));
  const cellOf = (m, i, j) => (m.layers ? m.layers.cells.find(c => c.dir === "ablate" && c.i === i && c.j === j) || null : null);
  const maxNode = m => Object.values(m.atlas).reduce((a, r) => Math.max(a, r.d || 0), 0);
  const ramp = t => `color-mix(in srgb, var(--ox) ${Math.round(clamp(t, 0, 1) * 100)}%, var(--paper3))`;

  /* ---------- chrome ---------- */
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

  $("#clear").addEventListener("click", () => {
    state.off.clear();
    renderAll();
  });

  /* ---------- selection model ---------- */
  function blockSelection(m) {
    if (!state.off.size) return null;
    const layers = [...new Set([...state.off].map(layerOf))].sort((a, b) => a - b);
    if (state.off.size !== layers.length * (headsOf(m) + 1)) return null;
    for (let k = 1; k < layers.length; k++) if (layers[k] !== layers[k - 1] + 1) return null;
    for (const l of layers) if (!layerKeys(l, m).every(k => state.off.has(k))) return null;
    const i = layers[0], j = layers[layers.length - 1];
    return { i, j, cell: cellOf(m, i, j) };
  }

  function measure(m) {
    const keys = [...state.off];
    if (!keys.length) return { kind: "none" };
    const gap = m.baseline.gap;
    if (keys.length === 1) {
      const rec = m.atlas[keys[0]];
      if (!rec) return { kind: "none" };
      return { kind: "node", key: keys[0], d: rec.d, frac: rec.d / gap, acc: rec.acc };
    }
    const b = blockSelection(m);
    if (b && b.cell) {
      const c = b.cell;
      return { kind: "block", i: b.i, j: b.j, cell: c, d: c.d, frac: c.frac, p: c.p, pending: false };
    }
    const byLayer = new Map();
    for (const k of keys) {
      const l = layerOf(k);
      if (!byLayer.has(l)) byLayer.set(l, []);
      byLayer.get(l).push(k);
    }
    const parts = [];
    let sum = 0;
    for (const [l, ks] of [...byLayer.entries()].sort((a, b) => a[0] - b[0])) {
      const all = layerKeys(l, m);
      if (ks.length === all.length && all.every(k => state.off.has(k))) {
        const c = cellOf(m, l, l);
        if (c) {
          parts.push({ label: `layer ${l}`, d: c.d, frac: c.frac, exact: true });
          sum += c.d;
          continue;
        }
      }
      const recs = ks.map(k => m.atlas[k]).filter(Boolean);
      if (!recs.length) continue;
      const d = recs.reduce((a, r) => a + (r.d || 0), 0);
      parts.push({
        label: ks.length === 1 ? ks[0] : `${ks.length} nodes · layer ${l}`,
        d,
        frac: d / gap,
        exact: ks.length === 1,
      });
      sum += d;
    }
    return { kind: "parts", parts, sum, frac: sum / gap, pending: !!(b && !b.cell) };
  }

  const allOff = m => state.off.size === Object.keys(m.atlas).size;
  function toggleNode(key) {
    if (state.off.has(key)) state.off.delete(key);
    else state.off.add(key);
    renderAll();
  }
  function toggleLayer(l, m) {
    const keys = layerKeys(l, m);
    const on = keys.every(k => state.off.has(k));
    for (const k of keys) {
      if (on) state.off.delete(k);
      else state.off.add(k);
    }
    renderAll();
  }
  function selectRun(i, j, m) {
    state.off = new Set();
    for (let l = i; l <= j; l++) for (const k of layerKeys(l, m)) state.off.add(k);
    renderAll();
  }

  /* ---------- model rail and facts ---------- */
  function renderModels() {
    const box = $("#models");
    box.textContent = "";
    D.models.forEach((m, i) => {
      const b = el("button", "pill" + (i === state.mi ? " on" : ""));
      b.type = "button";
      b.setAttribute("role", "tab");
      b.setAttribute("aria-selected", String(i === state.mi));
      b.append(el("b", "", m.name), el("span", "", `${m.facts.depth} layers · ${m.facts.heads} heads`));
      b.addEventListener("click", () => {
        state.mi = i;
        state.off.clear();
        renderAll();
      });
      box.append(b);
    });
  }

  function renderFacts() {
    const m = model(), f = m.facts, b = m.baseline, c = m.circuit;
    const box = $("#facts");
    box.textContent = "";
    const items = [
      [nf(f.parameters / 1e6, 1) + " M", "parameters"],
      [`${f.depth} × ${f.heads}`, "blocks × heads"],
      [nf(f.hidden, 0), "hidden width"],
      [b.clean_acc === null || b.clean_acc === undefined ? "—" : pct(b.clean_acc), "clean accuracy"],
      [nf(b.gap, 2), `gap · ${metricOf(m)}`],
      [`${c.size}/${c.total}`, "edges kept"],
      [m.layers ? String(m.layers.cells.length) : "—", "block conditions"],
    ];
    for (const [v, k] of items) {
      const d = el("div", "fact");
      d.append(el("b", "", v), el("span", "", k));
      box.append(d);
    }
  }

  /* ---------- views ---------- */
  const VIEWS = [
    ["board", "Board", "One row per layer. Click a head, the MLP, or the layer number."],
    ["cascade", "Cascade", "How much each node costs when it is the only thing replaced."],
    ["map", "Map", "Every contiguous run that was measured."],
    ["curves", "Curves", "All runs at once, one line per starting layer."],
  ];
  function renderViews() {
    const box = $("#views");
    box.textContent = "";
    for (const [id, label, title] of VIEWS) {
      const b = el("button", state.view === id ? "on" : "", label);
      b.type = "button";
      b.setAttribute("role", "tab");
      b.setAttribute("aria-selected", String(state.view === id));
      b.title = title;
      b.addEventListener("click", () => {
        state.view = id;
        renderViews();
        renderStage();
      });
      box.append(b);
    }
  }

  function setHead(title, hint) {
    $("#stage-title").textContent = title;
    $("#stage-hint").textContent = hint;
  }

  function renderStage() {
    const m = model(), stage = $("#stage");
    stage.textContent = "";
    const cells = m.layers ? m.layers.cells.length : 0;
    const n = Object.keys(m.atlas).length;
    if (state.view === "board") {
      setHead("The layer board",
        "Each row is a block. A square is one attention head; the wide square is the MLP, outlined in teal. Click any of them to replace its write, click the layer number to take the whole layer, and click again to restore.");
      renderBoard(m, stage);
      $("#stage-note").textContent =
        `${n} nodes, each measured alone · ${cells} block conditions in the file · a whole layer is one of the grid's cells, not a sum.`;
    } else if (state.view === "cascade") {
      setHead("The damage cascade",
        "The same 156 single-node measurements arranged by layer. Bar height is the cost of replacing that node alone; the split bar at the top says whether the damage lives in the heads or the MLPs.");
      renderCascade(m, stage);
      $("#stage-note").textContent =
        "Heights are separate measurements. They do not add to the joint effect of replacing several nodes, and the split bar is a share of the summed singles, not of the model's gap.";
    } else if (state.view === "map") {
      setHead("Every measured run",
        "Row is the first layer, column is the run length. Each cell is a contiguous block that was run on the GPU; click one to remove that entire run at once.");
      renderMap(m, stage);
      $("#stage-note").textContent =
        `${m.layers ? m.layers.cells.filter(c => c.dir === "ablate").length : 0} measured removals · the diagonal is a single layer · both directions are stored, this view shows the removals.`;
    } else {
      setHead("The whole family",
        "One line per starting layer; moving right grows the run downward. Height is the share of the clean-to-corrupt gap. Click any point to select that run.");
      renderCurves(m, stage);
      $("#stage-note").textContent =
        "Every point is a measured block. Lines that rise early say the shallow layers matter; flat lines that jump at the end say the opposite.";
    }
    $("#clear").disabled = !state.off.size;
  }

  /* board */
  function renderBoard(m, stage) {
    const maxd = maxNode(m) || 1;
    const board = el("div", "board");
    for (let l = 0; l < depthOf(m); l++) {
      const row = el("div", "brow");
      const keys = layerKeys(l, m);
      const off = keys.every(k => state.off.has(k));
      if (off) row.classList.add("off");
      const lab = el("button", "blabel", `L${l}`);
      lab.type = "button";
      lab.title = `${off ? "restore" : "remove"} layer ${l}`;
      lab.setAttribute("aria-pressed", String(off));
      lab.setAttribute("aria-label", `${off ? "Restore" : "Remove"} layer ${l}`);
      lab.addEventListener("click", () => toggleLayer(l, m));
      const chips = el("div", "bchips");
      for (let h = 0; h <= headsOf(m); h++) {
        const key = keyOf(l, h, m), rec = m.atlas[key];
        const b = el("button", "chip" + (h === headsOf(m) ? " mlp" : ""));
        b.type = "button";
        if (!rec) {
          b.classList.add("missing");
          b.disabled = true;
          b.title = `${key}: not measured`;
        } else {
          const isOff = state.off.has(key);
          b.style.background = isOff ? "transparent" : ramp(Math.log1p(rec.d) / Math.log1p(maxd));
          if (isOff) b.classList.add("off");
          const bits = [`${key} · ${metricOf(m)} ${nf(rec.d)}`, `${pct(rec.d / m.baseline.gap)} of the gap`];
          if (rec.acc !== null && rec.acc !== undefined) bits.push(`accuracy ${pct(rec.acc)}`);
          b.title = bits.join(" · ") + ` — click to ${isOff ? "restore" : "replace"}`;
          b.setAttribute("aria-pressed", String(isOff));
          b.setAttribute("aria-label", bits.join(", "));
          b.addEventListener("click", () => toggleNode(key));
        }
        chips.append(b);
      }
      const c = cellOf(m, l, l);
      const stat = el("span", "bstat", c ? pct(c.frac) : "—");
      stat.title = c ? `removing the whole of layer ${l} alone: ${metricOf(m)} ${nf(c.d)}` : "not measured";
      row.append(lab, chips, stat);
      board.append(row);
    }
    stage.append(board);
    const legend = el("div", "blegend");
    legend.innerHTML =
      '<span><i style="background:' + ramp(0.08) + '"></i>cheap to replace</span>' +
      '<span><i style="background:' + ramp(0.55) + '"></i>expensive</span>' +
      '<span><i style="background:' + ramp(0.95) + '"></i>close to the whole gap</span>' +
      '<span><i class="x"></i>removed</span>' +
      '<span>wide square = the MLP</span>';
    stage.append(legend);
  }

  /* cascade */
  function renderCascade(m, stage) {
    const maxd = maxNode(m) || 1;
    let att = 0, mlp = 0;
    for (const [k, r] of Object.entries(m.atlas)) {
      if (k.endsWith("M")) mlp += r.d || 0;
      else att += r.d || 0;
    }
    const total = att + mlp || 1;
    const split = el("div", "split");
    split.innerHTML =
      `<div class="split-bar"><span class="s-att" style="width:${(att / total * 100).toFixed(1)}%"></span>` +
      `<span class="s-mlp" style="width:${(mlp / total * 100).toFixed(1)}%"></span></div>` +
      `<div class="split-key"><span><i class="k-att"></i>attention heads · ${pct(att / total)}</span>` +
      `<span><i class="k-mlp"></i>MLPs · ${pct(mlp / total)}</span></div>`;
    stage.append(split);

    const box = el("div", "cascade");
    for (let l = 0; l < depthOf(m); l++) {
      const row = el("div", "crow");
      const keys = layerKeys(l, m);
      const off = keys.every(k => state.off.has(k));
      const lab = el("button", "clabel", `L${l}`);
      lab.type = "button";
      lab.setAttribute("aria-pressed", String(off));
      lab.title = `${off ? "restore" : "remove"} layer ${l}`;
      lab.addEventListener("click", () => toggleLayer(l, m));
      const bars = el("div", "cbars");
      for (let h = 0; h <= headsOf(m); h++) {
        const key = keyOf(l, h, m), rec = m.atlas[key];
        const b = el("button", "cbar" + (h === headsOf(m) ? " mlp" : ""));
        b.type = "button";
        if (!rec) {
          b.disabled = true;
          b.style.height = "2px";
          b.style.background = "var(--line)";
        } else {
          const d = rec.d || 0;
          b.style.height = `${Math.round(4 + 31 * Math.sqrt(d / maxd))}px`;
          b.style.background = ramp(Math.log1p(d) / Math.log1p(maxd));
          if (state.off.has(key)) b.classList.add("off");
          b.title = `${key} · ${metricOf(m)} ${nf(d)} alone · ${pct(d / m.baseline.gap)} of the gap — click to ${state.off.has(key) ? "restore" : "replace"}`;
          b.setAttribute("aria-pressed", String(state.off.has(key)));
          b.addEventListener("click", () => toggleNode(key));
        }
        bars.append(b);
      }
      const c = cellOf(m, l, l);
      row.append(lab, bars, el("span", "cstat", c ? pct(c.frac) : "—"));
      box.append(row);
    }
    stage.append(box);
    stage.append(el("p", "cnote",
      "Bars are single-node costs. The number at the right is the measured cost of replacing the whole layer, which is not the sum of its bars."));
  }

  /* map */
  function renderMap(m, stage) {
    const depth = depthOf(m);
    const grid = el("div", "map");
    grid.style.gridTemplateColumns = `34px repeat(${depth}, 26px)`;
    grid.append(el("div", "mh", ""));
    for (let len = 1; len <= depth; len++) grid.append(el("div", "mh", String(len)));
    const sel = blockSelection(m);
    for (let i = 0; i < depth; i++) {
      grid.append(el("div", "mr", `L${i}`));
      for (let len = 1; len <= depth; len++) {
        if (len > depth - i) {
          grid.append(el("div", "blk empty"));
          continue;
        }
        const j = i + len - 1;
        const c = cellOf(m, i, j);
        const b = el("button", "blk");
        b.type = "button";
        if (!c) {
          b.classList.add("pend");
          b.disabled = true;
          b.title = `layers ${i}–${j}: not measured`;
        } else {
          b.style.background = ramp(Math.pow(c.frac, 0.65));
          const title = `layers ${i}–${j} removed · ${metricOf(m)} ${nf(c.d)} · ${pct(c.frac)} of the gap` +
            (c.p !== null && c.p !== undefined ? ` · true class ${nf(c.p)}` : "");
          b.title = title;
          b.setAttribute("aria-label", title);
          if (sel && sel.i === i && sel.j === j) b.classList.add("sel");
          b.addEventListener("click", () => selectRun(i, j, m));
        }
        grid.append(b);
      }
    }
    stage.append(grid);
    const scale = el("div", "map-scale");
    scale.innerHTML = '<span>0</span><span class="bar"></span><span>100% of the gap</span>';
    stage.append(scale);
    stage.append(el("div", "map-axis", "rows: first layer removed · columns: how many consecutive layers · the diagonal is a single layer"));
  }

  /* curves */
  function renderCurves(m, stage) {
    const depth = depthOf(m);
    const W = 660, H = 320, P = { l: 46, r: 18, t: 16, b: 40 };
    const x = j => P.l + (depth <= 1 ? 0 : (j / (depth - 1)) * (W - P.l - P.r));
    const y = f => H - P.b - clamp(f, 0, 1.02) * (H - P.t - P.b);
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "cost of removing each contiguous run of layers" });
    for (const f of [0, 0.25, 0.5, 0.75, 1]) {
      const yy = y(f);
      svg.append(sv("line", { x1: P.l, x2: W - P.r, y1: yy, y2: yy, class: f === 1 ? "axis-line" : "grid-line" }));
      const t = sv("text", { x: P.l - 8, y: yy + 3, class: "tick", "text-anchor": "end" });
      t.textContent = Math.round(f * 100) + "%";
      svg.append(t);
    }
    for (let j = 0; j < depth; j += (depth > 16 ? 2 : 1)) {
      const t = sv("text", { x: x(j), y: H - P.b + 16, class: "tick", "text-anchor": "middle" });
      t.textContent = j;
      svg.append(t);
    }
    const xn = sv("text", { x: (P.l + W - P.r) / 2, y: H - 6, class: "axis-name", "text-anchor": "middle" });
    xn.textContent = "last layer in the run";
    svg.append(xn);
    const sel = blockSelection(m);
    const hot = sel ? `${sel.i}:${sel.j}` : null;
    for (let i = 0; i < depth; i++) {
      const pts = [];
      for (let j = i; j < depth; j++) {
        const c = cellOf(m, i, j);
        if (c) pts.push([j, c.frac]);
      }
      if (pts.length < 2) continue;
      const d = pts.map(([j, f], k) => `${k ? "L" : "M"}${x(j).toFixed(1)},${y(f).toFixed(1)}`).join(" ");
      const lineHot = sel && sel.i === i;
      svg.append(sv("path", { d, class: "series" + (lineHot ? " hot" : "") }));
    }
    for (let i = 0; i < depth; i++) {
      for (let j = i; j < depth; j++) {
        const c = cellOf(m, i, j);
        if (!c) continue;
        const isHot = hot === `${i}:${j}`;
        const dot = sv("circle", { cx: x(j), cy: y(c.frac), r: isHot ? 3.6 : 2.4, class: "dot" + (isHot ? " hot" : "") });
        const t = sv("title");
        t.textContent = `layers ${i}–${j} · ${pct(c.frac)} of the gap`;
        dot.append(t);
        dot.addEventListener("click", () => selectRun(i, j, m));
        svg.append(dot);
      }
    }
    const cbox = el("div", "curves");
    cbox.append(svg);
    stage.append(cbox);
    const key = el("div", "curve-key");
    key.innerHTML = '<span>each line: fixed first layer, growing run</span><span>red: your selection</span><span>100% = the corrupt answer</span>';
    stage.append(key);
  }

  /* ---------- output ---------- */
  function renderOutput() {
    const m = model(), box = $("#output"), gap = m.baseline.gap, met = metricOf(m);
    const M = measure(m);
    box.textContent = "";

    const head = el("div", "out-head");
    const badge = el("span", "badge " + (M.kind === "none" ? "quiet" : M.kind === "parts" ? "warn" : "ok"),
      M.kind === "none" ? "nothing selected" : M.kind === "parts" ? "separate measurements" : "measured");
    const count = state.off.size;
    head.append(badge, el("span", "out-kicker", count ? `${count} node${count === 1 ? "" : "s"}` : m.name));
    box.append(head);

    if (M.kind === "none") {
      box.append(el("p", "out-prompt", "Click a head or the MLP in the board. Click a layer number to take the whole layer. The measured reading appears here, and the same click puts it back."));
      const facts = el("dl", "out-facts");
      const add = (k, v) => {
        const d = el("div");
        d.append(el("dt", "", k), el("dd", "", v));
        facts.append(d);
      };
      add("gap", `${nf(gap)} ${met}`);
      add("conditions", m.layers ? String(m.layers.cells.length) : "—");
      add("nodes", String(Object.keys(m.atlas).length));
      add("model", m.model.split("·")[0].trim());
      box.append(facts);
      box.append(el("p", "out-foot", "The gap is the distance between the clean answer and the answer for a different image. Everything on this page is read against it."));
      return;
    }

    if (M.kind === "node") {
      box.append(bigNumber(M.d, met, `${pct(M.frac)} of the gap`));
      box.append(gauge(M.frac));
      const facts = el("dl", "out-facts");
      const add = (k, v) => {
        const d = el("div");
        d.append(el("dt", "", k), el("dd", "", v));
        facts.append(d);
      };
      add("node", M.key);
      add("share", pct(M.frac));
      if (M.acc !== null && M.acc !== undefined) add("accuracy after", pct(M.acc));
      add("gap", `${nf(gap)} ${met}`);
      box.append(facts);
      const read = el("p", "out-read");
      read.innerHTML = `<b>${M.key}</b> replaced by the write it produced for a different image. ` +
        `${met} <b>${nf(M.d)}</b>, ${pct(M.frac)} of the gap` +
        (M.acc !== null && M.acc !== undefined ? `; the true class keeps probability only through the rest of the network, and accuracy falls to <b>${pct(M.acc)}</b>` : "") + ".";
      box.append(read);
      box.append(el("p", "out-foot", "One node alone. This is a measurement, not a sum."));
      return;
    }

    if (M.kind === "block") {
      const span = M.i === M.j ? `layer ${M.i}` : `layers ${M.i}–${M.j}`;
      box.append(bigNumber(M.d, met, `${pct(M.frac)} of the gap`));
      box.append(gauge(M.frac));
      const facts = el("dl", "out-facts");
      const add = (k, v) => {
        const d = el("div");
        d.append(el("dt", "", k), el("dd", "", v));
        facts.append(d);
      };
      add("selection", span);
      if (M.p !== null && M.p !== undefined) add("true class", nf(M.p));
      if (M.cell.edges !== undefined && M.cell.edges !== null) add("edges", `${M.cell.edges}/${m.circuit.total}`);
      add("gap", `${nf(gap)} ${met}`);
      box.append(facts);
      const read = el("p", "out-read");
      read.innerHTML = `Every node in <b>${span}</b> replaced at once — ${met} <b>${nf(M.d)}</b>, ${pct(M.frac)} of the gap` +
        (M.p !== null && M.p !== undefined ? `; the true class keeps probability <b>${nf(M.p)}</b>` : "") + ". " +
        '<span class="q">That is the block grid’s own cell, not a sum.</span>';
      box.append(read);
      if (m.kind === "mae" && m.layers && m.layers.sprites && m.layers.sprites.ablate) {
        const sp = m.layers.sprites.ablate;
        const fig = el("figure", "out-recon");
        const img = el("div", "img");
        img.style.backgroundImage = `url("${sp.url}")`;
        img.style.backgroundSize = `${sp.cell * sp.depth}px ${sp.cell * sp.depth}px`;
        img.style.backgroundPosition = `${-(M.j - M.i) * sp.cell}px ${-M.i * sp.cell}px`;
        fig.append(img, el("figcaption", "", `the model's reconstruction with ${span} replaced — this is the final output, not a damage score`));
        box.append(fig);
      }
      box.append(el("p", "out-foot", "The reconstruction, where the model has one, is the sprite cell for exactly this block."));
      return;
    }

    // parts: exact pieces, no joint number
    const parts = M.parts;
    box.append(bigNumber(M.sum, met, "sum of the pieces — not the union"));
    const list = el("div", "pieces");
    for (const p of parts) {
      const row = el("div", "piece");
      const bar = el("span", "p-bar");
      const fill = el("i");
      fill.style.width = `${clamp(p.frac, 0, 1) * 100}%`;
      bar.append(fill);
      row.append(el("span", "", p.label), bar, el("span", "p-val", nf(p.d)));
      row.title = `${p.label}: ${met} ${nf(p.d)} (${pct(p.frac)} of the gap)`;
      list.append(row);
    }
    box.append(list);
    const sum = el("div", "out-sum");
    sum.innerHTML = `<span>these pieces, added</span><span><b>${pct(M.frac)}</b> of the gap</span>`;
    box.append(sum);
    const read = el("p", "out-read");
    read.innerHTML = parts.length === 1
      ? `One piece: <b>${parts[0].label}</b> at ${met} <b>${nf(parts[0].d)}</b>.`
      : `${parts.length} pieces, each measured on its own. <b>The union was never run</b>, so the joint effect is not the sum; ` +
        (M.pending ? "and this model's grid is still missing the range that would settle it." : "the pieces are exact, their total is not.");
    box.append(read);
    box.append(el("p", "out-foot", "Split a selection into measured pieces to get exact readings; anything else is reported as unmeasured."));
  }

  function bigNumber(value, unit, sub) {
    const d = el("div", "out-big");
    d.append(el("b", "", nf(value)), el("span", "", unit));
    const wrap = el("div");
    wrap.append(d, el("p", "out-sub", sub));
    return wrap;
  }

  function gauge(frac) {
    const wrap = el("div");
    const g = el("div", "gauge");
    const mark = el("span", "mark");
    mark.style.left = `${clamp(frac, 0, 1) * 100}%`;
    g.append(mark);
    const labels = el("div", "gauge-labels");
    labels.append(el("span", "", "clean answer"), el("span", "", pct(frac)), el("span", "", "corrupt answer"));
    wrap.append(g, labels);
    return wrap;
  }

  /* ---------- published results ---------- */
  const BERT = [-0.01, 0.10, -0.14, -0.53, -0.29, -0.52, 0.05, -0.72, -0.96, 0.07, -0.19, -0.12];
  function renderBert() {
    const box = $("#lit-bert");
    if (!box) return;
    BERT.forEach((v, i) => {
      const row = el("div", "bert-row");
      row.append(el("span", "bert-l", "L" + (i + 1)));
      const track = el("div", "bert-track");
      track.append(el("i", "bert-zero"));
      const bar = el("span", "bert-bar " + (v < 0 ? "neg" : "pos"));
      bar.style.width = `${(Math.abs(v) / 1.0) * 50}%`;
      if (v < 0) bar.style.right = "50%";
      else bar.style.left = "50%";
      track.append(bar);
      row.append(track, el("span", "bert-v", (v > 0 ? "+" : "") + v.toFixed(2)));
      row.title = `layer ${i + 1}: ${v > 0 ? "+" : ""}${v.toFixed(2)} points`;
      box.append(row);
    });
    const scale = el("div", "bert-scale");
    scale.innerHTML = "<span>−1 pp</span><span>0</span><span>+1 pp</span>";
    box.append(scale);
    $("#lit-bert-read").innerHTML =
      "The worst layer costs <b>0.96 points</b> of accuracy; two layers come out slightly better. " +
      "Kept as one head out of twelve, a whole BERT layer is close to free — which is the same shape as the head atlas here, where single heads are about one percent of the gap.";
  }

  const LPARTS = [
    ["nothing removed", 7.60, true],
    ["attention only", 7.65, false],
    ["the FFN only", 12.35, false],
    ["the whole last layer", 13.37, false],
  ];
  function renderLlamaParts() {
    const box = $("#lit-llama-parts");
    if (!box) return;
    for (const [label, v, base] of LPARTS) {
      const row = el("div", "lpart");
      const track = el("div", "lpart-track");
      const bar = el("span", "lpart-bar" + (base ? " base" : ""));
      bar.style.width = `${(v / 14) * 100}%`;
      track.append(bar);
      track.append(el("i", "lpart-base"));
      row.append(el("span", "lpart-l", label), track, el("span", "lpart-v", v.toFixed(2)));
      row.title = `${label}: perplexity ${v.toFixed(2)}`;
      box.append(row);
    }
    const note = el("div", "lpart-note");
    note.innerHTML = "<span>7.60 = the dense model</span><span>PPL · lower is better</span>";
    box.append(note);
    $("#lit-llama-parts-read").innerHTML =
      "Taking the whole block costs <b>5.77 perplexity</b>; taking its attention costs <b>0.05</b>. " +
      "The feed-forward half is the layer. The same asymmetry shows up in the vision atlas, where the MLP column carries far more of the gap than any single head.";
  }

  const PRUNE = {
    ratio: [0, 3.1, 9.4, 12.5, 15.6, 25.0, 27.1],
    ppl: [8.03, 8.37, 9.44, 10.24, 11.42, 22.29, 40.78],
    mmlu: [43.17, 42.88, 42.31, 41.62, 43.17, 41.68, 43.35],
  };
  function renderLlamaPrune() {
    const box = $("#lit-llama-prune");
    if (!box) return;
    const W = 660, H = 250, P = { l: 48, r: 54, t: 18, b: 42 };
    const xr = r => P.l + (r / 28) * (W - P.l - P.r);
    const yP = v => {
      const lo = Math.log(8), hi = Math.log(45);
      return H - P.b - ((Math.log(v) - lo) / (hi - lo)) * (H - P.t - P.b);
    };
    const yM = v => H - P.b - ((v - 39) / 6) * (H - P.t - P.b);
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "perplexity and MMLU as layers are removed from Llama-2-7B" });
    for (const v of [8, 10, 20, 40]) {
      const y = yP(v);
      svg.append(sv("line", { x1: P.l, x2: W - P.r, y1: y, y2: y, class: "gl" }));
      const t = sv("text", { x: P.l - 8, y: y + 3, class: "tk", "text-anchor": "end" });
      t.textContent = v;
      svg.append(t);
    }
    for (const v of [40, 42, 44]) {
      const t = sv("text", { x: W - P.r + 8, y: yM(v) + 3, class: "tk" });
      t.textContent = v;
      svg.append(t);
    }
    for (const r of [0, 5, 10, 15, 20, 25]) {
      const t = sv("text", { x: xr(r), y: H - P.b + 16, class: "tk", "text-anchor": "middle" });
      t.textContent = r + "%";
      svg.append(t);
    }
    const path = (vals, y) => vals.map((v, i) => `${i ? "L" : "M"}${xr(PRUNE.ratio[i]).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
    svg.append(sv("path", { d: path(PRUNE.ppl, yP), class: "ppl" }));
    svg.append(sv("path", { d: path(PRUNE.mmlu, yM), class: "mmlu" }));
    PRUNE.ratio.forEach((r, i) => {
      const c1 = sv("circle", { cx: xr(r), cy: yP(PRUNE.ppl[i]), r: 3.2, class: "pt", fill: "var(--ox)" });
      const t1 = sv("title");
      t1.textContent = `${r}% removed · PPL ${PRUNE.ppl[i].toFixed(2)}`;
      c1.append(t1);
      const c2 = sv("circle", { cx: xr(r), cy: yM(PRUNE.mmlu[i]), r: 3.2, class: "pt", fill: "var(--teal)" });
      const t2 = sv("title");
      t2.textContent = `${r}% removed · MMLU ${PRUNE.mmlu[i].toFixed(2)}`;
      c2.append(t2);
      svg.append(c1, c2);
    });
    const ax = sv("text", { x: (P.l + W - P.r) / 2, y: H - 6, class: "an", "text-anchor": "middle" });
    ax.textContent = "layers removed";
    svg.append(ax);
    const yl = sv("text", { x: 12, y: P.t - 4, class: "an" });
    yl.textContent = "PPL";
    const yr = sv("text", { x: W - P.r + 6, y: P.t - 4, class: "an" });
    yr.textContent = "MMLU";
    svg.append(yl, yr);
    box.append(svg);
    const key = el("div", "key");
    key.innerHTML = '<span><i style="background:var(--ox)"></i>perplexity</span><span><i style="background:var(--teal)"></i>MMLU accuracy</span>';
    box.append(key);
    $("#lit-llama-prune-read").innerHTML =
      "Between zero and 15.6% removed, MMLU never leaves <b>41.6–43.2</b>; at 15.6% it is back at 43.17, the dense value. " +
      "Perplexity has already risen from 8.03 to 11.42 by then. At 27.1% the perplexity is <b>40.78</b> and MMLU is 43.35. " +
      "A benchmark can be nearly blind to a change a likelihood can see.";
  }

  const CLIFFS = [
    ["Llama-2 family", 45, 55],
    ["Mistral-7B", 35, 35],
    ["Phi-2", 25, 25],
    ["Qwen", 20, 20],
  ];
  function renderCliffs() {
    const box = $("#lit-cliffs");
    if (!box) return;
    for (const [name, lo, hi] of CLIFFS) {
      const row = el("div", "cliff");
      const track = el("div", "cliff-track");
      const range = el("span", "cliff-range" + (hi === lo ? " point" : ""));
      range.style.left = `${(lo / 60) * 100}%`;
      if (hi !== lo) range.style.width = `${((hi - lo) / 60) * 100}%`;
      track.append(range);
      const v = el("span", "cliff-v", hi === lo ? `${lo}%` : `${lo}–${hi}%`);
      track.append(v);
      row.append(el("span", "cliff-l", name), track);
      box.append(row);
    }
    const axis = el("div", "cliff-axis");
    axis.innerHTML = "<span>0%</span><span>30%</span><span>60% of layers</span>";
    box.append(axis);
  }

  /* ---------- wiring ---------- */
  function renderAll() {
    renderModels();
    renderFacts();
    renderViews();
    renderStage();
    renderOutput();
    $("#clear").disabled = !state.off.size;
  }

  renderBert();
  renderLlamaParts();
  renderLlamaPrune();
  renderCliffs();
  renderAll();
})();
