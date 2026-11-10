/* The Understanding Machine — navigation, search, and mechanism widgets. */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  let DATA = {};
  try { DATA = JSON.parse($("#um-data")?.textContent || "{}"); } catch (e) { DATA = {}; }

  /* ---------- progress + active section ---------- */
  const bar = $("#progress");
  const onScroll = () => {
    const h = document.documentElement;
    const max = h.scrollHeight - h.clientHeight;
    if (bar) bar.style.width = `${max > 0 ? (h.scrollTop / max) * 100 : 0}%`;
  };
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  const navLinks = $$("#topnav a");
  const targets = navLinks.map(a => ({ a, el: $(a.getAttribute("href")) })).filter(x => x.el);
  const io = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      navLinks.forEach(a => a.classList.remove("active"));
      const hit = targets.find(t => t.el === e.target);
      if (hit) hit.a.classList.add("active");
    });
  }, { rootMargin: "-20% 0px -70% 0px" });
  targets.forEach(t => io.observe(t.el));

  /* ---------- theme ---------- */
  const themeBtn = $("#themebtn");
  const saved = localStorage.getItem("um-theme");
  if (saved) document.documentElement.dataset.theme = saved;
  themeBtn?.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("um-theme", next);
    drawHero();
  });

  /* ---------- search ---------- */
  const searchBtn = $("#searchbtn"), box = $("#searchbox"), q = $("#q"), results = $("#results");
  const INDEX = $$(".chapter, .lab, .part-open, .problemset").map(el => ({
    id: el.id,
    title: el.querySelector("h2, h3")?.textContent?.trim() || el.id,
    text: el.textContent.replace(/\s+/g, " ").trim()
  })).filter(x => x.id);
  const openSearch = () => {
    box.classList.toggle("on");
    searchBtn.setAttribute("aria-expanded", box.classList.contains("on"));
    if (box.classList.contains("on")) q.focus();
  };
  searchBtn?.addEventListener("click", openSearch);
  addEventListener("keydown", e => {
    if (e.key === "/" && document.activeElement.tagName !== "INPUT") { e.preventDefault(); openSearch(); }
    if (e.key === "Escape") box?.classList.remove("on");
  });
  q?.addEventListener("input", () => {
    const term = q.value.toLowerCase().trim();
    if (!term) { results.innerHTML = ""; return; }
    const hits = INDEX.filter(x => (x.title + " " + x.text).toLowerCase().includes(term)).slice(0, 12);
    results.innerHTML = hits.length
      ? hits.map(x => `<a href="#${x.id}"><b>${x.title}</b>${x.text.slice(Math.max(0, x.text.toLowerCase().indexOf(term) - 40), 140)}…</a>`).join("")
      : `<div class="no">No section matches “${q.value}”.</div>`;
  });
  results?.addEventListener("click", e => { if (e.target.closest("a")) box.classList.remove("on"); });

  /* ---------- hero: attention matrix ---------- */
  const hero = $("#hero-canvas");
  function drawHero() {
    if (!hero) return;
    const ctx = hero.getContext("2d"), n = 12, w = hero.width, h = hero.height;
    const cw = w / n, ch = h / n;
    ctx.clearRect(0, 0, w, h);
    const dark = document.documentElement.dataset.theme === "dark";
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const d = Math.exp(-Math.abs(i - j) / 3) * 0.55 + 0.04 * Math.sin(i * 2.1 + j * 1.3);
        ctx.fillStyle = dark
          ? `rgba(224,113,94,${0.05 + d * 0.55})`
          : `rgba(156,53,39,${0.03 + d * 0.38})`;
        ctx.fillRect(j * cw + 1, i * ch + 1, cw - 2, ch - 2);
      }
    }
  }
  drawHero();

  /* ---------- utils ---------- */
  const C = { ox: "#9c3527", teal: "#1c6b66", gold: "#aa8438", blue: "#213a5c", ink: "#171a1c", muted: "#6e6659", line: "#c9bda6" };
  const dark = () => document.documentElement.dataset.theme === "dark";
  const col = k => dark() ? ({ ox: "#e0715e", teal: "#62c2b8", gold: "#d9b46a", blue: "#9bb7de", ink: "#ebe4d6", muted: "#978d7d", line: "#4b4d52" }[k]) : C[k];
  const fit = (cv) => {
    const dpr = devicePixelRatio || 1, r = cv.getBoundingClientRect();
    if (!r.width) return;
    cv.width = r.width * dpr; cv.height = cv.height ? (cv.height / (cv._ratio || 1)) * 1 : 360;
    cv._ratio = dpr;
    const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: r.width, h: cv.height / dpr };
  };

  /* ---------- perceptron ---------- */
  const pCan = $("#w-perceptron");
  if (pCan) {
    const pts = [[0, 0, -1], [0, 1, 1], [1, 0, 1], [1, 1, -1]];
    const w1 = $("#w1"), w2 = $("#w2"), b = $("#pb"), read = $("#p-read");
    const draw = () => {
      const ctx = pCan.getContext("2d"), dpr = devicePixelRatio || 1;
      const r = pCan.getBoundingClientRect(); pCan.width = r.width * dpr; pCan.height = 340 * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const W = r.width, H = 340, pad = 44, sx = x => pad + x * (W - 2 * pad), sy = y => H - pad - y * (H - 2 * pad);
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = col("line"); ctx.lineWidth = 1;
      ctx.strokeRect(pad, pad, W - 2 * pad, H - 2 * pad);
      ctx.fillStyle = col("muted"); ctx.font = "11px ui-sans-serif"; ctx.fillText("x₁", W - pad + 4, sy(0) + 4); ctx.fillText("x₂", sx(0) - 12, pad - 10);
      const a = +w1.value, c = +w2.value, bb = +b.value;
      if (Math.abs(c) > 1e-6) {
        const x0 = -2, x1 = 3;
        ctx.beginPath(); ctx.moveTo(sx(x0), sy((-bb - a * x0) / c)); ctx.lineTo(sx(x1), sy((-bb - a * x1) / c));
        ctx.strokeStyle = col("gold"); ctx.lineWidth = 2; ctx.stroke();
      }
      let correct = 0;
      pts.forEach(([x, y, l]) => {
        const s = a * x + c * y + bb, pred = s > 0 ? 1 : -1;
        if (pred === l) correct++;
        ctx.beginPath(); ctx.arc(sx(x), sy(y), 9, 0, Math.PI * 2);
        ctx.fillStyle = l > 0 ? col("teal") : col("ox"); ctx.fill();
        ctx.strokeStyle = pred === l ? col("ink") : col("gold"); ctx.lineWidth = pred === l ? 1 : 3; ctx.stroke();
        ctx.fillStyle = col("ink"); ctx.font = "11px ui-monospace"; ctx.fillText(`(${x},${y})`, sx(x) + 13, sy(y) + 4);
      });
      read.textContent = `w·x + b = ${a}·x₁ + ${c}·x₂ ${bb >= 0 ? "+" : "−"} ${Math.abs(bb)}   ·   accuracy ${correct}/4`;
    };
    [w1, w2, b].forEach(el => el.addEventListener("input", draw));
    addEventListener("resize", draw); draw();
  }

  /* ---------- attention ---------- */
  const aCan = $("#w-attention");
  if (aCan && DATA.attention) {
    const sel = $("#a-head"), read = $("#a-read"), d = DATA.attention;
    const opts = d.heads.map((h, i) => `<option value="${i}">layer ${h.layer} · head ${h.head}</option>`).join("");
    sel.innerHTML = opts;
    const draw = () => {
      const h = d.heads[+sel.value], M = h.m, n = M.length;
      const ctx = aCan.getContext("2d"), dpr = devicePixelRatio || 1;
      const r = aCan.getBoundingClientRect(); aCan.width = r.width * dpr; aCan.height = r.width * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const W = r.width, pad = 46, cell = (W - 2 * pad) / n;
      ctx.clearRect(0, 0, W, W);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const v = M[i][j];
        ctx.fillStyle = dark() ? `rgba(217,180,106,${v})` : `rgba(170,132,56,${v})`;
        ctx.fillRect(pad + j * cell, pad + i * cell, cell - 1, cell - 1);
      }
      ctx.fillStyle = col("muted"); ctx.font = "10px ui-monospace";
      d.tokens.forEach((t, i) => {
        if (i % 3 === 0) {
          ctx.save(); ctx.translate(pad + i * cell + cell / 2, pad - 6); ctx.rotate(-Math.PI / 3);
          ctx.fillText(t.length > 3 ? t.slice(0, 3) : t, 0, 0); ctx.restore();
        }
      });
      ctx.fillText("query", 6, pad + (W - 2 * pad) / 2);
      ctx.fillText("key", W / 2, W - 8);
      const last = M[n - 1];
      const top = last.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0])[0];
      read.textContent = `final query → position ${top[1]} (${d.tokens[top[1]]}) with weight ${top[0].toFixed(3)}`;
    };
    sel.addEventListener("change", draw); addEventListener("resize", draw); draw();
  }

  /* ---------- superposition ---------- */
  const sCan = $("#w-superposition");
  if (sCan && DATA.superposition) {
    const sl = $("#s-p"), read = $("#s-read"), sw = DATA.superposition.sweep;
    sl.max = sw.length - 1; sl.value = 0;
    const draw = () => {
      const r0 = sw[+sl.value];
      const ctx = sCan.getContext("2d"), dpr = devicePixelRatio || 1;
      const r = sCan.getBoundingClientRect(); sCan.width = r.width * dpr; sCan.height = 340 * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const W = r.width, H = 340, cx = W / 2, cy = H / 2, R = 120;
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = col("line"); ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
      const arrow = (x, y, color, width, label) => {
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + x * R, cy - y * R);
        ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
        ctx.beginPath(); ctx.arc(cx + x * R, cy - y * R, 3.5, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
        if (label) { ctx.fillStyle = color; ctx.font = "11px ui-monospace"; ctx.fillText(label, cx + x * R + 5, cy - y * R - 5); }
      };
      r0.D.forEach(([x, y], i) => arrow(x, y, col("gold"), 1.2, `f${i}`));
      r0.W.forEach(([x, y]) => { const n = Math.hypot(x, y) || 1; arrow(x / n, y / n, col("teal"), 2.2, ""); });
      read.textContent = `p = ${r0.p} · loss ${r0.loss} · |cos| > 0.7: ${r0.represented} · diag ${r0.diag} · off-diagonal ${r0.off}`;
    };
    sl.addEventListener("input", draw); addEventListener("resize", draw); draw();
  }

  /* ---------- logit lens ---------- */
  const lCan = $("#w-logit");
  if (lCan && DATA.logit_lens) {
    const draw = () => {
      const ctx = lCan.getContext("2d"), dpr = devicePixelRatio || 1;
      const r = lCan.getBoundingClientRect(); lCan.width = r.width * dpr; lCan.height = 230 * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const W = r.width, H = 230, rows = DATA.logit_lens.rows, pad = 70, bw = W - pad - 30, rh = 64;
      ctx.clearRect(0, 0, W, H);
      ctx.font = "12px ui-sans-serif"; ctx.fillStyle = col("muted");
      ctx.fillText(`expected token: ${DATA.logit_lens.expected}`, pad, 16);
      rows.forEach((row, i) => {
        const y = 32 + i * rh;
        ctx.fillStyle = col("ink"); ctx.font = "11px ui-monospace";
        ctx.fillText(`layer ${row.layer}`, 4, y + 18);
        row.tokens.forEach((t, j) => {
          const w = row.probs[j] * bw;
          ctx.fillStyle = t === DATA.logit_lens.expected ? col("teal") : col("ox");
          ctx.globalAlpha = 0.25 + 0.75 * row.probs[j];
          ctx.fillRect(pad, y + j * 16, Math.max(w, 2), 12);
          ctx.globalAlpha = 1;
          ctx.fillStyle = col("ink"); ctx.font = "10px ui-monospace";
          ctx.fillText(`${t} ${(row.probs[j] * 100).toFixed(1)}%`, pad + 4, y + j * 16 + 10);
        });
      });
    };
    addEventListener("resize", draw); draw();
  }

  /* ---------- embeddings ---------- */
  const eCan = $("#w-embeddings");
  if (eCan && DATA.embeddings) {
    const draw = () => {
      const ctx = eCan.getContext("2d"), dpr = devicePixelRatio || 1;
      const r = eCan.getBoundingClientRect(); eCan.width = r.width * dpr; eCan.height = 400 * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const W = r.width, H = 400, pad = 34;
      const pts = DATA.embeddings.points;
      const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
      const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
      const sx = x => pad + (x - xmin) / (xmax - xmin) * (W - 2 * pad);
      const sy = y => H - pad - (y - ymin) / (ymax - ymin) * (H - 2 * pad);
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = col("line"); ctx.strokeRect(pad, pad, W - 2 * pad, H - 2 * pad);
      const colors = { animal: col("teal"), colour: col("ox"), country: col("gold"), capital: col("blue"), food: col("muted") };
      const seen = {};
      pts.forEach(p => {
        ctx.beginPath(); ctx.arc(sx(p.x), sy(p.y), 3.5, 0, Math.PI * 2); ctx.fillStyle = colors[p.c] || col("ink"); ctx.fill();
        if (!seen[p.c]) {
          ctx.font = "10px ui-sans-serif"; ctx.fillText(p.c, pad + 8, pad + 14 + Object.keys(seen).length * 16); seen[p.c] = 1;
        }
        if (["cat", "dog", "red", "japan", "tokyo", "paris"].includes(p.w)) {
          ctx.font = "10px ui-monospace"; ctx.fillStyle = colors[p.c]; ctx.fillText(p.w, sx(p.x) + 5, sy(p.y) - 5);
        }
      });
    };
    addEventListener("resize", draw); draw();
  }

  /* ---------- vision ---------- */
  const vCan = $("#w-vision");
  if (vCan && DATA.vision) {
    const sel = $("#v-head"), read = $("#v-read"), d = DATA.vision;
    sel.innerHTML = d.maps.map((m, i) => `<option value="${i}">layer ${m.layer} · head ${m.head}</option>`).join("");
    const draw = () => {
      const m = d.maps[+sel.value];
      const ctx = vCan.getContext("2d"), dpr = devicePixelRatio || 1;
      const r = vCan.getBoundingClientRect(); vCan.width = r.width * dpr; vCan.height = 300 * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const W = r.width, H = 300, side = Math.min(W * .44, H - 40), ox = 20, oy = 30;
      ctx.clearRect(0, 0, W, H);
      for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
        const v = d.image[y][x];
        ctx.fillStyle = `rgb(${255 - v * 220},${255 - v * 220},${255 - v * 225})`;
        ctx.fillRect(ox + x * side / 16, oy + y * side / 16, side / 16 + .5, side / 16 + .5);
      }
      ctx.strokeStyle = col("ox"); ctx.lineWidth = .6;
      for (let i = 0; i <= 4; i++) {
        ctx.beginPath(); ctx.moveTo(ox + i * side / 4, oy); ctx.lineTo(ox + i * side / 4, oy + side); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(ox, oy + i * side / 4); ctx.lineTo(ox + side, oy + i * side / 4); ctx.stroke();
      }
      const ax = ox + side + 55, cell = side / 4;
      for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
        ctx.fillStyle = dark() ? `rgba(217,180,106,${m.m[i][j]})` : `rgba(170,132,56,${m.m[i][j]})`;
        ctx.fillRect(ax + j * cell, oy + i * cell, cell - 1, cell - 1);
      }
      read.textContent = `class-token attention, layer ${m.layer}, head ${m.head}; mean ${(m.m.flat().reduce((a, b) => a + b, 0) / 16).toFixed(3)}`;
      ctx.fillStyle = col("muted"); ctx.font = "11px ui-sans-serif";
      ctx.fillText("image", ox, 18); ctx.fillText("patch attention", ax, 18);
    };
    sel.addEventListener("change", draw); addEventListener("resize", draw); draw();
  }

  /* ---------- ACDC explorer ---------- */
  const acdcCan = $("#w-acdc");
  if (acdcCan && DATA.acdc) {
    const d = DATA.acdc, detail = $("#a-detail"), read = $("#a-read");
    const modes = $$("#acdc-widget [data-mode]");
    const nL = d.layers, nH = d.heads;
    const kept = new Set(d.kept);
    const headRank = new Map(), mlpRank = new Map();
    const flat = [];
    d.head_damage.forEach((row, l) => row.forEach((v, h) => flat.push({ id: `L${l}H${h}`, v })));
    flat.sort((a, b) => b.v - a.v).forEach((x, i) => headRank.set(x.id, i + 1));
    d.mlp_damage.map((v, l) => ({ id: `L${l}M`, v })).sort((a, b) => b.v - a.v)
      .forEach((x, i) => mlpRank.set(x.id, i + 1));
    let mode = "both", pinned = null, hover = null, view = null;

    const paper = () => dark() ? "#1b1d20" : "#fbf7ee";
    const shade = () => dark() ? "#23262a" : "#e9e1d1";
    const heat = v => Math.pow(Math.log1p(Math.max(0, v) * 60) / Math.log1p(60 * 8), 0.7);
    const label = id => id.endsWith("M")
      ? `${id.slice(0, -1)} · MLP`
      : `${id.split("H")[0]} · H${id.split("H")[1]}`;

    function cellAt(px, py) {
      const c = view.cell, x0 = view.x0, y0 = view.y0;
      if (py < y0 || py >= y0 + nL * c) return null;
      const l = Math.floor((py - y0) / c);
      const xr = (px - x0) / c;
      if (xr >= 0 && xr < nH) {
        const h = Math.floor(xr);
        return { kind: "head", l, h, id: `L${l}H${h}`, v: d.head_damage[l][h] };
      }
      if (px >= view.mlpX && px < view.mlpX + c) {
        return { kind: "mlp", l, id: `L${l}M`, v: d.mlp_damage[l] };
      }
      return null;
    }

    function detailFor(c) {
      if (!c) {
        const total = nL * (nH + 1);
        return `Hover a cell to inspect it. <b>${kept.size}</b> of ${total} nodes are in the circuit; the complement leaves ${(100 * d.complement.accuracy).toFixed(1)}% accuracy and ${d.complement.kl.toFixed(2)} nats of KL.`;
      }
      const r = c.kind === "head" ? headRank.get(c.id) : mlpRank.get(c.id);
      const total = c.kind === "head" ? nL * nH : nL;
      return `<b>${label(c.id)}</b> · ${c.v.toFixed(3)} nats · rank ${r} of ${total} · ${kept.has(c.id) ? "kept by ACDC" : "not in the circuit"}`;
    }

    function drawCell(ctx, x, y, s, v, on) {
      if (mode === "circuit") {
        ctx.fillStyle = on ? col("teal") : shade();
        ctx.globalAlpha = on ? 0.9 : 0.65;
      } else {
        ctx.fillStyle = col("ox");
        ctx.globalAlpha = mode === "damage" ? 0.06 + 0.94 * heat(v) : 0.04 + 0.96 * heat(v);
      }
      ctx.fillRect(x + 0.6, y + 0.6, s - 1.2, s - 1.2);
      ctx.globalAlpha = 1;
      if (on && mode !== "damage") {
        ctx.strokeStyle = col("teal");
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x + 1.1, y + 1.1, s - 2.2, s - 2.2);
      }
    }

    function draw() {
      const r = acdcCan.getBoundingClientRect();
      if (!r.width) return;
      const W = r.width;
      const cell = Math.max(9, Math.min(22, (W - 150) / (nH + 2.4)));
      const H = 24 + nL * cell + 34;
      const dpr = devicePixelRatio || 1;
      acdcCan.width = W * dpr;
      acdcCan.height = H * dpr;
      acdcCan.style.height = `${H}px`;
      const ctx = acdcCan.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const x0 = 36, y0 = 24, mlpX = x0 + (nH + 0.62) * cell;
      const barsX = x0 + (nH + 1.65) * cell + 26;
      view = { W, H, cell, x0, y0, mlpX };

      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = paper();
      ctx.fillRect(0, 0, W, H);
      ctx.font = "10px ui-monospace";
      ctx.textAlign = "center";
      ctx.fillStyle = col("muted");
      for (let h = 0; h < nH; h += 4) ctx.fillText(`H${h}`, x0 + (h + 0.5) * cell, 15);
      ctx.fillText("MLP", mlpX + cell / 2, 15);

      for (let l = 0; l < nL; l++) {
        const y = y0 + l * cell;
        if (l % 2 === 0) {
          ctx.textAlign = "right";
          ctx.fillStyle = col("muted");
          ctx.fillText(`L${l}`, x0 - 6, y + cell * 0.72);
        }
        for (let h = 0; h < nH; h++) {
          const id = `L${l}H${h}`;
          drawCell(ctx, x0 + h * cell, y, cell, d.head_damage[l][h], kept.has(id));
        }
        drawCell(ctx, mlpX, y, cell, d.mlp_damage[l], kept.has(`L${l}M`));
      }

      ctx.strokeStyle = col("line");
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(mlpX - cell * 0.22, y0);
      ctx.lineTo(mlpX - cell * 0.22, y0 + nL * cell);
      ctx.stroke();

      if (barsX < W - 30) {
        const bw = W - barsX - 16;
        const means = d.head_damage.map(row => row.reduce((a, b) => a + b, 0) / nH);
        const mx = Math.max(...means) || 1;
        ctx.textAlign = "left";
        ctx.fillStyle = col("muted");
        ctx.fillText("mean head KL", barsX, 15);
        ctx.strokeStyle = col("line");
        ctx.lineWidth = 0.7;
        ctx.beginPath();
        ctx.moveTo(barsX, y0);
        ctx.lineTo(barsX, y0 + nL * cell);
        ctx.stroke();
        const pts = means.map((v, l) => [barsX + Math.sqrt(v / mx) * bw, y0 + (l + 0.5) * cell]);
        ctx.strokeStyle = col("gold");
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
        ctx.stroke();
        pts.forEach(([x, y], l) => {
          const any = [...Array(nH).keys()].some(h => kept.has(`L${l}H${h}`));
          ctx.beginPath();
          ctx.arc(x, y, 2.1, 0, Math.PI * 2);
          ctx.fillStyle = any ? col("teal") : col("gold");
          ctx.fill();
        });
      }

      const cur = hover || pinned;
      if (cur) {
        const x = cur.kind === "head" ? x0 + cur.h * cell : mlpX;
        const y = y0 + cur.l * cell;
        ctx.strokeStyle = col("ink");
        ctx.lineWidth = 2;
        ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
        if (pinned && hover && pinned.id !== hover.id) {
          const px = pinned.kind === "head" ? x0 + pinned.h * cell : mlpX;
          const py = y0 + pinned.l * cell;
          ctx.strokeStyle = col("gold");
          ctx.lineWidth = 1.6;
          ctx.strokeRect(px + 0.5, py + 0.5, cell - 1, cell - 1);
        }
      }
      detail.innerHTML = detailFor(hover || pinned);
    }

    acdcCan.addEventListener("mousemove", e => {
      if (!view) return;
      const r = acdcCan.getBoundingClientRect();
      hover = cellAt((e.clientX - r.left) * (view.W / r.width),
                     (e.clientY - r.top) * (view.H / r.height));
      acdcCan.style.cursor = hover ? "pointer" : "default";
      draw();
    });
    acdcCan.addEventListener("mouseleave", () => { hover = null; draw(); });
    acdcCan.addEventListener("click", e => {
      if (!view) return;
      const r = acdcCan.getBoundingClientRect();
      const hit = cellAt((e.clientX - r.left) * (view.W / r.width),
                         (e.clientY - r.top) * (view.H / r.height));
      if (hit) pinned = pinned && pinned.id === hit.id ? null : hit;
      draw();
    });
    modes.forEach(b => b.addEventListener("click", () => {
      mode = b.dataset.mode;
      modes.forEach(x => x.classList.toggle("on", x === b));
      draw();
    }));
    read.textContent = `${kept.size} of ${nL * (nH + 1)} nodes kept · complement ${d.complement.kl.toFixed(2)} KL, ${(100 * d.complement.accuracy).toFixed(1)}% accuracy · circuit ablation ${d.circuit_ablated.kl.toFixed(2)} KL, ${(100 * d.circuit_ablated.accuracy).toFixed(1)}%`;
    addEventListener("resize", draw);
    draw();
  }

  /* ---------- anatomy explorer ---------- */
  const anatStack = $("#anat-stack");
  if (anatStack && DATA.acdc) {
    const d = DATA.acdc, F = d.facts || {};
    const nL = d.layers, nH = d.heads;
    const kept = new Set(d.kept);
    const score = d.head_scores || [];
    const panel = $("#anat-panel"), read = $("#anat-read");
    const buttons = $$("#anatomy-widget [data-amode]");
    let mode = "bare", pinned = Math.min(11, nL - 1), hover = null;

    const flat = [];
    d.head_damage.forEach((row, l) => row.forEach((v, h) => flat.push({ id: `L${l}H${h}`, v })));
    flat.sort((a, b) => b.v - a.v).forEach((x, i) => { x.rank = i + 1; });
    const headRank = new Map(flat.map(x => [x.id, x.rank]));
    const mlpRank = new Map(d.mlp_damage.map((v, l) => ({ id: `L${l}M`, v }))
      .sort((a, b) => b.v - a.v).map((x, i) => [x.id, i + 1]));
    const hmax = Math.max(...d.head_damage.flat(), 1e-6);
    const smax = Math.max(1e-6, ...score.flat());
    const heat = v => Math.pow(Math.log1p(Math.max(0, v) * 60) / Math.log1p(60 * 8), 0.7);
    const pct = x => `${(100 * x).toFixed(1)}%`;

    const capTop = document.createElement("div");
    capTop.className = "anat-cap"; capTop.textContent = "residual out · final norm → unembedding";
    anatStack.appendChild(capTop);
    for (let l = nL - 1; l >= 0; l--) {
      const row = document.createElement("div");
      row.className = "anat-row"; row.dataset.l = l; row.tabIndex = 0;
      row.setAttribute("role", "button");
      row.setAttribute("aria-label", `layer ${l}`);
      const mean = d.head_damage[l].reduce((a, b) => a + b, 0) / nH;
      row.innerHTML = `<span class="anat-n">L${l}</span>` +
        `<span class="anat-cells">${Array.from({ length: nH }, (_, h) => `<i class="anat-pip" data-id="L${l}H${h}"></i>`).join("")}</span>` +
        `<span class="anat-mlp"><i class="anat-pip" data-id="L${l}M"></i></span>` +
        `<span class="anat-mean"><i style="width:${Math.max(2, 100 * mean / hmax).toFixed(1)}%"></i></span>`;
      anatStack.appendChild(row);
    }
    const capBot = document.createElement("div");
    capBot.className = "anat-cap"; capBot.textContent = "token embedding → block 0";
    anatStack.appendChild(capBot);

    const rows = $$(".anat-row", anatStack);
    const pips = $$(".anat-pip", anatStack);

    function paint() {
      pips.forEach(el => {
        const id = el.dataset.id, l = +id.slice(1).split("H")[0];
        const mlp = id.endsWith("M");
        const v = mlp ? d.mlp_damage[l] : d.head_damage[l][+id.split("H")[1]];
        el.style.borderColor = ""; el.style.background = ""; el.style.opacity = "";
        if (mode === "bare") {
          el.style.opacity = mlp ? ".95" : ".72";
        } else if (mode === "damage") {
          el.style.background = col("ox"); el.style.borderColor = "transparent";
          el.style.opacity = (0.05 + 0.95 * heat(v)).toFixed(3);
        } else if (mode === "circuit") {
          const on = kept.has(id);
          el.style.background = on ? col(mlp ? "ox" : "teal") : col("line");
          el.style.borderColor = "transparent";
          el.style.opacity = on ? ".95" : ".35";
        } else {
          if (mlp) {
            el.style.background = col("line"); el.style.borderColor = "transparent";
            el.style.opacity = ".45";
          } else {
            const s = (score[l] || [])[+id.split("H")[1]] || 0;
            el.style.background = col("gold"); el.style.borderColor = "transparent";
            el.style.opacity = (0.06 + 0.94 * s / smax).toFixed(3);
          }
        }
      });
      rows.forEach(r => r.classList.toggle("on", +r.dataset.l === pinned));
    }

    function layerCard(l) {
      const hs = d.head_damage[l];
      const mean = hs.reduce((a, b) => a + b, 0) / nH;
      const mx = Math.max(...hs), mh = hs.indexOf(mx);
      const kn = hs.filter((_, h) => kept.has(`L${l}H${h}`)).length;
      const bars = hs.map((v, h) =>
        `<i class="${kept.has(`L${l}H${h}`) ? "on" : ""}" style="height:${Math.max(2, 100 * heat(v) / heat(hmax)).toFixed(1)}%"></i>`).join("");
      panel.innerHTML = `<div class="big">Layer ${l}</div>
        <div class="sub">${l} of ${nL - 1} · ${Math.round(100 * l / (nL - 1))}% of depth · ${l < 8 ? "early" : l < 16 ? "middle" : "late"}</div>
        <dl>
          <dt>heads kept</dt><dd>${kn} / ${nH}</dd>
          <dt>mean head KL</dt><dd>${mean.toFixed(4)} nats</dd>
          <dt>strongest head</dt><dd>H${mh} · ${mx.toFixed(3)}</dd>
          <dt>MLP KL</dt><dd>${d.mlp_damage[l].toFixed(3)}${kept.has(`L${l}M`) ? " · kept" : ""}</dd>
          <dt>parameters</dt><dd>${F.block_params ? (F.block_params / 1e6).toFixed(2) + " M" : "12.60 M"}</dd>
        </dl>
        <div class="anat-bars">${bars}</div>
        <div class="anat-scale"><span>H0</span><span>head KL in this layer</span><span>H15</span></div>`;
    }

    function nodeCard(id) {
      const mlp = id.endsWith("M"), l = +id.slice(1).split("H")[0];
      const h = mlp ? -1 : +id.split("H")[1];
      const v = mlp ? d.mlp_damage[l] : d.head_damage[l][h];
      const rank = mlp ? mlpRank.get(id) : headRank.get(id);
      const total = mlp ? nL : nL * nH;
      const s = mlp ? null : (score[l] || [])[h];
      const m = (d.marginal || {})[id];
      const hd = F.head_dim || 64;
      panel.innerHTML = `<div class="big">${mlp ? `L${l} · MLP` : `L${l} · H${h}`}</div>
        <div class="sub">${mlp ? "the block's 1,024-wide MLP" : `dims ${h * hd}–${(h + 1) * hd - 1}, before W_O`}</div>
        <dl>
          <dt>ablation KL</dt><dd>${v.toFixed(4)} nats</dd>
          <dt>rank</dt><dd>${rank} of ${total}</dd>
          <dt>in circuit</dt><dd>${kept.has(id) ? "yes" : "no"}</dd>
          ${mlp ? "" : `<dt>induction score</dt><dd>${s == null ? "–" : s.toFixed(3)}</dd>`}
          ${m == null ? "" : `<dt>marginal KL</dt><dd>${m.toFixed(4)} nats</dd>`}
        </dl>`;
    }

    anatStack.addEventListener("mousemove", e => {
      const pip = e.target.closest(".anat-pip");
      const id = pip ? pip.dataset.id : null;
      if (id !== hover) {
        hover = id;
        if (id) nodeCard(id); else layerCard(pinned);
      }
    });
    anatStack.addEventListener("mouseleave", () => { hover = null; layerCard(pinned); });
    anatStack.addEventListener("click", e => {
      const row = e.target.closest(".anat-row");
      if (!row) return;
      pinned = +row.dataset.l; hover = null; paint(); layerCard(pinned);
    });
    anatStack.addEventListener("keydown", e => {
      if (e.key !== "Enter" && e.key !== " ") return;
      const row = e.target.closest(".anat-row");
      if (!row) return;
      e.preventDefault(); pinned = +row.dataset.l; paint(); layerCard(pinned);
    });
    buttons.forEach(b => b.addEventListener("click", () => {
      mode = b.dataset.amode;
      buttons.forEach(x => x.classList.toggle("on", x === b));
      paint();
    }));

    read.textContent = `${nL * (nH + 1)} places · ${kept.size} kept · complement ${d.complement.kl.toFixed(2)} KL, ${pct(d.complement.accuracy)} · circuit ablated ${d.circuit_ablated.kl.toFixed(2)} KL, ${pct(d.circuit_ablated.accuracy)}`;
    addEventListener("um-theme", paint);
    paint(); layerCard(pinned);
  }

  /* ---------- redraw widgets on theme change ---------- */
  themeBtn?.addEventListener("click", () => setTimeout(() => {
    for (const id of ["#w-perceptron", "#w-attention", "#w-superposition", "#w-logit", "#w-embeddings", "#w-vision"]) {
      const el = $(id); if (el) { const ev = new Event("resize"); dispatchEvent(ev); }
    }
    drawHero();
    dispatchEvent(new Event("um-theme"));
  }, 0));
})();
