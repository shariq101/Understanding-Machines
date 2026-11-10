/* Teacher vs distilled student. Draws two stacks from the measured JSON:
   whole-block KL on one shared scale, component costs beside it, the CKA
   correspondence, and the output divergence. No model runs here. */

(async () => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };

  let DATA = window.DISTILL || window.DISTILL2 || window.DISTILL_QWEN || window.DISTILL_TINY || window.DISTILL_VISION || null;
  if (!DATA) {
    try { DATA = await fetch("distill-data.json", { cache: "no-store" }).then(r => r.json()); }
    catch (err) { DATA = null; }
  }

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

  if (!DATA || !DATA.teacher || !DATA.student) {
    $("#stack-teacher").append(el("p", "pending",
      "The distillation data file is not here yet. Run the Kaggle kernel, then `bash kaggle/distill/fetch.sh` — the page reads distill-data.js and measures nothing itself."));
    return;
  }

  const T = DATA.teacher;
  const S = DATA.student;
  const roleT = DATA.roles?.teacher || "teacher";
  const roleS = DATA.roles?.student || "student";
  const title = s => s.charAt(0).toUpperCase() + s.slice(1);

  function comp(m, layer, kind) {
    return m.components.find(c => c.layer === layer && c.kind === kind);
  }
  function meanKl(c) {
    return c ? c.runs.reduce((a, r) => a + r.kl, 0) / c.runs.length : 0;
  }

  const wholeMeans = [];
  let maxWhere = { v: 0, who: roleT, layer: 0 };
  for (const [who, m] of [[roleT, T], [roleS, S]]) {
    for (let l = 0; l < m.facts.layers; l++) {
      const v = meanKl(comp(m, l, "whole"));
      wholeMeans.push(v);
      if (v > maxWhere.v) maxWhere = { v, who, layer: l };
    }
  }
  const maxWhole = Math.max(...wholeMeans, 0.001);

  $("#facts").textContent = `${T.facts.params_m}M ${roleT} → ${S.facts.params_m}M ${roleS} · ${T.facts.layers} → ${S.facts.layers} blocks · ${T.facts.hidden} wide · ${T.facts.prompts} ${DATA.readout || "masked"} prompts · measured in ${T.facts.seconds}s + ${S.facts.seconds}s on ${T.facts.device}`;

  if (DATA.probe && $("#probe")) {
    const p = DATA.probe;
    const table = el("table", "causal");
    const head = el("tr");
    ["model", "probe accuracy", "error", "train / test"].forEach(h => head.append(el("th", null, h)));
    table.append(head);
    for (const [who, m] of [[roleT, p.teacher], [roleS, p.student]]) {
      const tr = el("tr");
      tr.append(el("td", null, title(who)));
      tr.append(el("td", null, `${(m.accuracy * 100).toFixed(1)}%`));
      tr.append(el("td", null, `${((1 - m.accuracy) * 100).toFixed(1)}%`));
      tr.append(el("td", null, `${m.train} / ${m.test}`));
      table.append(tr);
    }
    $("#probe").append(table);
    const chance = (p.chance * 100).toFixed(0);
    const agree = Math.round(p.agreement * p.test);
    $("#probe-read").innerHTML =
      `The frozen features support a <b>${p.labels.length}-way</b> readout (${p.labels.join(" / ")}): ` +
      `<b>${(p.teacher.accuracy * 100).toFixed(1)}%</b> for the ${roleT} and <b>${(p.student.accuracy * 100).toFixed(1)}%</b> for the ${roleS} on ${p.test} held-out headlines; chance is ${chance}%. ` +
      `Their top-1 choices agree on <b>${agree} of ${p.test}</b> (${(p.agreement * 100).toFixed(1)}%); mean total variation is <b>${p.mean_tv.toFixed(3)}</b> and mean KL is <b>${p.mean_kl.toFixed(3)}</b>. ` +
      `These are independent linear probes on frozen CLS features, not a comparison of the pretrained language heads; the probe weights and test split are fixed in the kernel.`;
  }

  function renderStack(root, m, title, sub) {
    root.textContent = "";
    root.append(el("h2", null, title), el("p", "stack-sub", sub));
    const rows = el("div", "rows");
    for (let l = 0; l < m.facts.layers; l++) {
      const whole = meanKl(comp(m, l, "whole"));
      const attn = meanKl(comp(m, l, "attn"));
      const mlp = meanKl(comp(m, l, "mlp"));
      const ln = (meanKl(comp(m, l, "ln1")) + meanKl(comp(m, l, "ln2"))) / 2;
      const row = el("div", "drow");
      row.append(el("span", "dl", `L${l + 1}`));
      const bar = el("div", "dbar");
      const fill = el("i");
      const w = Math.log1p(whole) / Math.log1p(maxWhole);
      fill.style.width = `${Math.max(1.5, w * 100)}%`;
      bar.append(fill);
      bar.title = `whole block KL ${whole.toFixed(3)}`;
      row.append(bar, el("span", "dk", whole.toFixed(3)));
      const parts = el("div", "dparts");
      parts.innerHTML = `attn <b>${attn.toFixed(3)}</b> · mlp <b>${mlp.toFixed(3)}</b> · ln <b>${ln.toFixed(3)}</b> · entropy ${(m.entropy[l] ?? 0).toFixed(2)}`;
      row.append(parts);
      rows.append(row);
    }
    root.append(rows);
  }

  renderStack($("#stack-teacher"), T, `${title(roleT)} · ${T.facts.label || "BERT-base"}`,
    `${T.facts.layers} blocks · ${T.facts.heads} heads · ${T.facts.hidden} wide`);
  renderStack($("#stack-student"), S, `${title(roleS)} · ${S.facts.label || "DistilBERT"}`,
    `${S.facts.layers} blocks · ${S.facts.heads} heads · ${S.facts.hidden} wide`);

  $("#scale").textContent = `Bars share one log scale (log1p), so a single dominant block does not flatten the rest; the printed numbers are raw mean KL, and the largest is ${maxWhole.toFixed(3)} (${maxWhere.who} L${maxWhere.layer + 1}). Component numbers are separate interventions, not addends.`;

  /* ---- CKA matrix ---- */
  const cka = DATA.cka;
  const cols = cka[0].length;
  const table = el("table", "cka");
  const head = el("tr");
  head.append(el("th", "rl", `${roleS} \\ ${roleT}`));
  for (let j = 0; j < cols; j++) head.append(el("th", null, `L${j + 1}`));
  table.append(head);

  const sameDepth = T.facts.layers === S.facts.layers;
  const argmax = cka.map(row => row.indexOf(Math.max(...row)));
  const best = sameDepth ? cka.map((_, i) => i) : argmax;

  cka.forEach((row, i) => {
    const tr = el("tr");
    tr.append(el("th", "rl", `${roleS} L${i + 1}`));
    row.forEach((v, j) => {
      const td = el("td", j === best[i] ? "best" : "");
      td.style.background = `color-mix(in srgb, var(--blue) ${Math.round(v * 88)}%, var(--paper))`;
      if (v > 0.62) td.style.color = "var(--paper)";
      const span = el("span", "v", v.toFixed(2));
      td.append(span);
      td.title = `CKA(${roleS} L${i + 1}, ${roleT} L${j + 1}) = ${v.toFixed(4)}`;
      tr.append(td);
    });
    table.append(tr);
  });
  const wrap = el("div", "cka-wrap");
  wrap.append(table);
  $("#cka").append(wrap);

  const pairs = best.map((j, i) => `${roleS} L${i + 1} ↔ ${roleT} L${j + 1} (${cka[i][j].toFixed(2)})`);
  const monotone = argmax.every((j, i) => i === 0 || j >= argmax[i - 1]);
  const diag = cka.map((row, i) => row[i]);
  const diagMean = diag.reduce((a, b) => a + b, 0) / diag.length;
  const diagMin = Math.min(...diag);
  const diagMax = Math.max(...diag);
  const diagMinAt = diag.indexOf(diagMin) + 1;
  const argmaxOnDiag = argmax.filter((j, i) => j === i).length;
  const pairText = sameDepth
    ? `Same-index CKA runs from ${diagMin.toFixed(2)} (L${diagMinAt}) to ${diagMax.toFixed(2)}. `
    : `Closest pairs: <b>${pairs.join(" · ")}</b>. `;
  $("#cka-read").innerHTML =
    pairText +
    (sameDepth
      ? `The diagonal averages ${diagMean.toFixed(3)}; the row-wise residual argmax is on the diagonal for only ${argmaxOnDiag} of ${S.facts.layers} layers. The residual stream accumulates, so neighboring layers can be nearly collinear; for an equal-depth pair the same-index block is the comparison used below, not the argmax.`
      : (monotone
        ? `The best matches rise monotonically with depth: the ${S.facts.layers} ${roleS} blocks line up with a non-decreasing sequence of ${roleT} blocks, which is what a depth-compressed copy of the same pipeline would look like.`
        : `The best matches do not rise monotonically: at least one ${roleS} block is closest to a ${roleT} block shallower than the one before it, so the mapping is not a clean subsequence of the ${roleT}’s depth.`)) +
    ` The deepest ${roleS} block still has a measurable match — no ${roleS} layer is left without a counterpart.`;

  /* ---- intervention fingerprints ---- */
  const wholeVec = (m, l) => comp(m, l, "whole").runs.map(r => r.kl);
  function pearson(a, b) {
    const n = a.length;
    const ma = a.reduce((x, y) => x + y, 0) / n;
    const mb = b.reduce((x, y) => x + y, 0) / n;
    let va = 0, vb = 0, cov = 0;
    for (let k = 0; k < n; k++) {
      va += (a[k] - ma) ** 2;
      vb += (b[k] - mb) ** 2;
      cov += (a[k] - ma) * (b[k] - mb);
    }
    return va === 0 || vb === 0 ? 0 : cov / Math.sqrt(va * vb);
  }
  function residualMatrix(rows) {
    const L = rows.length, P = rows[0].length;
    const grand = rows.flat().reduce((x, y) => x + y, 0) / (L * P);
    const rm = rows.map(r => r.reduce((x, y) => x + y, 0) / P);
    const cm = Array.from({ length: P }, (_, p) => rows.reduce((s, r) => s + r[p], 0) / L);
    return rows.map((row, l) => row.map((v, p) => v - rm[l] - cm[p] + grand));
  }
  const RT = residualMatrix(Array.from({ length: T.facts.layers }, (_, l) => wholeVec(T, l)));
  const RS = residualMatrix(Array.from({ length: S.facts.layers }, (_, i) => wholeVec(S, i)));

  const causal = el("table", "causal");
  const chead = el("tr");
  [roleS, "CKA match", "r at CKA", "damage match", "r", "verdict"].forEach(h => chead.append(el("th", null, h)));
  causal.append(chead);

  const rows = [];
  for (let i = 0; i < S.facts.layers; i++) {
    const ckaBest = best[i];
    const cors = RT.map((row, j) => pearson(RS[i], row));
    const damageBest = cors.indexOf(Math.max(...cors));
    const rCka = pearson(RS[i], RT[ckaBest]);
    const rBest = cors[damageBest];
    const same = ckaBest === damageBest;
    rows.push({ i, ckaBest, damageBest, rCka, rBest, same });
    const tr = el("tr");
    tr.append(el("td", null, `L${i + 1}`));
    tr.append(el("td", null, `T${ckaBest + 1}`));
    tr.append(el("td", null, rCka.toFixed(2)));
    tr.append(el("td", null, `T${damageBest + 1}`));
    tr.append(el("td", null, rBest.toFixed(2)));
    tr.append(el("td", same ? "ok" : "no", same ? "same" : "elsewhere"));
    causal.append(tr);
  }
  $("#causal").append(causal);

  const sameCount = rows.filter(r => r.same).length;
  const diverging = rows.filter(r => !r.same);
  const strongest = diverging.reduce((a, r) => r.rBest > a.rBest ? r : a, diverging[0] || rows[0]);
  $("#causal-read").innerHTML =
    `<b>${sameCount} of ${S.facts.layers}</b> ${roleS} blocks ${sameCount === 1 ? "keeps" : "keep"} the same partner under both measures. ` +
    (diverging.length
      ? `The strongest disagreement is ${roleS} L${strongest.i + 1}: its damage profile is closest to ${roleT} L${strongest.damageBest + 1} (r = ${strongest.rBest.toFixed(2)}), while its CKA partner is ${roleT} L${strongest.ckaBest + 1} (r = ${strongest.rCka.toFixed(2)}). `
      : "") +
    `With ${(DATA.prompts || []).length} prompts and ${rows.length * T.facts.layers} comparisons, individual correlations are descriptive rather than confirmatory; the permutation null below shows how large the best match can look by chance. A disagreement does not refute the CKA mapping: representational geometry and the pattern of intervention damage are different measurements, and this section reports where they part.`;

  /* ---- attention vs MLP fingerprints ---- */
  const profiles = (m, kind) => Array.from({ length: m.facts.layers }, (_, l) =>
    comp(m, l, kind).runs.map(r => r.kl));

  function componentMatch(kind) {
    const RT = residualMatrix(profiles(T, kind));
    const RS = residualMatrix(profiles(S, kind));
    const matches = [];
    for (let i = 0; i < S.facts.layers; i++) {
      const cors = RT.map(row => pearson(RS[i], row));
      const j = cors.indexOf(Math.max(...cors));
      matches.push({ j, r: cors[j] });
    }
    return { RT, RS, matches };
  }

  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  function nullMax(RT, RS, seed, iters = 1000) {
    const rnd = mulberry32(seed);
    const vals = [];
    for (let it = 0; it < iters; it++) {
      const shuffled = RS.map(row => {
        const c = row.slice();
        for (let k = c.length - 1; k > 0; k--) {
          const j = Math.floor(rnd() * (k + 1));
          [c[k], c[j]] = [c[j], c[k]];
        }
        return c;
      });
      let m = 0;
      for (let i = 0; i < shuffled.length; i++) {
        for (let j = 0; j < RT.length; j++) m = Math.max(m, Math.abs(pearson(shuffled[i], RT[j])));
      }
      vals.push(m);
    }
    vals.sort((a, b) => a - b);
    return vals[Math.floor(vals.length * 0.95)];
  }

  const kinds = ["whole", "attn", "mlp"];
  const nullIters = S.facts.layers * T.facts.layers > 200 ? 400 : 1000;
  const comps = kinds.map(k => ({ k, ...componentMatch(k) }));
  const nulls = Object.fromEntries(comps.map((c, i) => [c.k, nullMax(c.RT, c.RS, 17 + i * 31, nullIters)]));

  const ptable = el("table", "causal");
  const phead = el("tr");
  [roleS, "CKA partner", "whole", "attention", "MLP"].forEach(h => phead.append(el("th", null, h)));
  ptable.append(phead);
  for (let i = 0; i < S.facts.layers; i++) {
    const tr = el("tr");
    tr.append(el("td", null, `L${i + 1}`));
    tr.append(el("td", null, `T${best[i] + 1} (${cka[i][best[i]].toFixed(2)})`));
    for (const c of comps) {
      const m = c.matches[i];
      tr.append(el("td", null, `T${m.j + 1} (${m.r.toFixed(2)})`));
    }
    ptable.append(tr);
  }
  $("#parts").append(ptable);

  const agree = k => comps.find(c => c.k === k).matches.filter((m, i) => m.j === best[i]).length;
  const ckaName = sameDepth ? "same-index CKA partner" : "residual-CKA argmax";
  const attnMlpSplit = comps[1].matches.filter((m, i) => m.j !== comps[2].matches[i].j).length;
  const strongestPart = comps.map(c => ({ k: c.k, m: Math.max(...c.matches.map(x => x.r)) }))
    .sort((a, b) => b.m - a.m)[0];
  const nPrompts = (DATA.prompts || []).length;
  const overNull = strongestPart.m - nulls[strongestPart.k];
  $("#parts-read").innerHTML =
    `The damage partner equals the ${ckaName} for <b>${agree("attn")} of ${S.facts.layers}</b> ${roleS} blocks under attention, <b>${agree("mlp")}</b> under the MLP, and <b>${agree("whole")}</b> under the whole block. ` +
    `Attention and MLP choose different ${roleT} partners for <b>${attnMlpSplit} of ${S.facts.layers}</b> blocks, so the two components are not interchangeable fingerprints. ` +
    `Family-wise 95% permutation nulls (best of all ${S.facts.layers} × ${T.facts.layers} pairs, ${nullIters} shuffles): whole ${nulls.whole.toFixed(2)}, attention ${nulls.attn.toFixed(2)}, MLP ${nulls.mlp.toFixed(2)}. ` +
    `The strongest observed match in any component is ${strongestPart.m.toFixed(2)} (${strongestPart.k}); ` +
    (overNull > 0
      ? `it ${overNull > 0.05 ? "exceeds" : "sits at the edge of"} its 95% null, so with ${nPrompts} prompts the result is ${overNull > 0.05 ? "unlikely under the permutation model, though still one prompt bank" : "suggestive, not decisive"}.`
      : `it falls inside its null, so this prompt bank cannot distinguish it from chance.`) +
    (DATA.next ? ` ${DATA.next}` : (nPrompts >= 20
      ? ` This is the larger bank; the null above already reflects its extra power.`
      : ` The 25-prompt RoBERTa run is the replication that tests whether either component holds up.`));

  if (DATA.cka_parts) {
    const diagAgree = k => DATA.cka_parts[k].filter((row, i) => row.indexOf(Math.max(...row)) === i).length;
    const diagLine = ` By component CKA, the diagonal is the argmax for attention ${diagAgree("attn")}/${S.facts.layers}, MLP ${diagAgree("mlp")}/${S.facts.layers}, and residual ${diagAgree("resid")}/${S.facts.layers}.`;
    $("#parts-read").innerHTML += diagLine;
    const cparts = ["resid", "attn", "mlp"];
    const ctable = el("table", "causal");
    const chead2 = el("tr");
    [roleS, ...cparts.map(k => `${k} CKA`)].forEach(h => chead2.append(el("th", null, h)));
    ctable.append(chead2);
    for (let i = 0; i < S.facts.layers; i++) {
      const tr = el("tr");
      tr.append(el("td", null, `L${i + 1}`));
      for (const k of cparts) {
        const row = DATA.cka_parts[k][i];
        const j = sameDepth ? i : row.indexOf(Math.max(...row));
        tr.append(el("td", null, `T${j + 1} (${row[j].toFixed(2)})`));
      }
      ctable.append(tr);
    }
    const cap = el("p", "sub");
    cap.style.marginTop = "16px";
    cap.textContent = sameDepth
      ? `Same-index ${roleT} partner by CKA on the residual stream, the attention write and the MLP write. Agreement between these columns is a component-level representation check; it is not an ablation effect.`
      : `Best ${roleT} partner by CKA on the residual stream, the attention write and the MLP write. Agreement between these columns is a component-level representation check; it is not an ablation effect.`;
    const box = el("div");
    box.append(ctable, cap);
    $("#cka-parts").append(box);
  }

  /* ---- which component of the reference each part resembles ---- */
  if (DATA.cka_cross && $("#borrow")) {
    const cross = DATA.cka_cross;
    const borrowRows = [];
    const pick = (a, b, i) => {
      let bestPick = { j: 0, v: -1, kind: "attn" };
      for (let j = 0; j < T.facts.layers; j++) {
        if (a[i][j] > bestPick.v) bestPick = { j, v: a[i][j], kind: "attn" };
        if (b[i][j] > bestPick.v) bestPick = { j, v: b[i][j], kind: "MLP" };
      }
      return bestPick;
    };
    for (let i = 0; i < S.facts.layers; i++) {
      const residual = best[i];
      const attn = pick(cross.sa_ta, cross.sa_tm, i);
      const mlp = pick(cross.sm_ta, cross.sm_tm, i);
      borrowRows.push({ i, residual, attn, mlp });
    }

    const btable = el("table", "causal");
    const bhead = el("tr");
    [roleS, "residual", "attention write is closest to", "MLP write is closest to"].forEach(h => bhead.append(el("th", null, h)));
    btable.append(bhead);
    for (const r of borrowRows) {
      const tr = el("tr");
      tr.append(el("td", null, `L${r.i + 1}`));
      tr.append(el("td", null, `T${r.residual + 1} (${cka[r.i][r.residual].toFixed(2)})`));
      const ac = el("td", r.attn.kind === "attn" ? "ok" : "no",
        `T${r.attn.j + 1} ${r.attn.kind} (${r.attn.v.toFixed(2)})`);
      const mc = el("td", r.mlp.kind === "MLP" ? "ok" : "no",
        `T${r.mlp.j + 1} ${r.mlp.kind} (${r.mlp.v.toFixed(2)})`);
      tr.append(ac, mc);
      btable.append(tr);
    }
    $("#borrow").append(btable);

    const attnCross = borrowRows.filter(r => r.attn.kind !== "attn").length;
    const mlpCross = borrowRows.filter(r => r.mlp.kind !== "MLP").length;
    const own = (r, k) => r[k].kind === (k === "attn" ? "attn" : "MLP");
    const strongestAttn = borrowRows.reduce((a, r) => r.attn.v > a.attn.v ? r : a);
    const strongestMlp = borrowRows.reduce((a, r) => r.mlp.v > a.mlp.v ? r : a);
    $("#borrow-read").innerHTML =
      `The student’s attention write is closest to the ${roleT}’s attention write for <b>${S.facts.layers - attnCross} of ${S.facts.layers}</b> blocks and to its MLP write for <b>${attnCross}</b>; the student’s MLP is closest to the ${roleT} MLP for <b>${S.facts.layers - mlpCross}</b> and to ${roleT} attention for <b>${mlpCross}</b>. ` +
      `The strongest attention match is ${roleS} L${strongestAttn.i + 1} ↔ ${roleT} L${strongestAttn.attn.j + 1} ${strongestAttn.attn.kind} (${strongestAttn.attn.v.toFixed(2)}); the strongest MLP match is ${roleS} L${strongestMlp.i + 1} ↔ ${roleT} L${strongestMlp.mlp.j + 1} ${strongestMlp.mlp.kind} (${strongestMlp.mlp.v.toFixed(2)}). ` +
      `A cross-kind win means the student component’s geometry resembles the other teacher component more closely. It is not proof that the computation was copied: CKA is measured on the clean pass, the two depths differ, and a block can absorb work from more than one ${roleT} block. The same-component wins are the ones that support a borrowing claim; the cross wins are the places where “layer <i>i</i> learned layer <i>j</i>’s attention” is too simple.`;
  }

  /* ---- divergence ---- */
  const gapBox = $("#gap");
  const prompts = DATA.prompts || [];
  (DATA.top_gap || []).forEach((g, i) => {
    const row = el("div", "grow");
    row.append(el("span", "q", prompts[i] || `prompt ${i + 1}`));
    const t = el("span", "t");
    t.innerHTML = `${roleT} <b>${(g.teacher || "").replace(/^ /, "␣")}</b>`;
    const s = el("span", "s");
    s.innerHTML = `${roleS} <b>${(g.student || "").replace(/^ /, "␣")}</b>`;
    const m = el("div", "m");
    const bar = el("div", "bar");
    const fill = el("i");
    fill.style.width = `${Math.max(1.5, Math.min(1, g.tv) * 100)}%`;
    bar.append(fill);
    m.append(bar, el("span", null, g.tv.toFixed(2)));
    row.append(t, s, m);
    gapBox.append(row);
  });

  const meanTv = (DATA.top_gap || []).reduce((a, g) => a + g.tv, 0) / Math.max(1, DATA.top_gap.length);
  const worst = (DATA.top_gap || []).reduce((a, g, i) => g.tv > a.tv ? { tv: g.tv, i } : a, { tv: -1, i: 0 });
  if (DATA.top_gap && DATA.top_gap.length) {
    const note = el("p", "sub");
    note.style.marginTop = "16px";
    note.textContent = `Mean total variation across the ${(DATA.top_gap || []).length} prompts: ${meanTv.toFixed(3)}. Largest divergence: prompt ${worst.i + 1}, where the ${roleT} says “${DATA.top_gap[worst.i].teacher.trim() || "—"}” and the ${roleS} says “${DATA.top_gap[worst.i].student.trim() || "—"}”.`;
    gapBox.append(note);
  }

  /* ---- the actual RegNet teacher ---- */
  const teacherBox = $("#teacher-check");
  const R = DATA.regnet_teacher;
  if (teacherBox && R) {
    const Tt = R.teacher;
    const summary = Tt.summary || {};
    const pct = v => `${(100 * Number(v || 0)).toFixed(1)}%`;
    const num = (v, n = 3) => Number(v || 0).toFixed(n);
    const lead = el("p", "sub");
    lead.innerHTML = `<b>${Tt.label}</b> (${Tt.model}) is run on the same twenty images. ` +
      `Ten-way accuracy <b>${pct(summary.accuracy_10)}</b>; mean candidate confidence ` +
      `<b>${num(summary.mean_confidence)}</b>. The teacher is not retrained or ablated.`;
    teacherBox.append(lead);

    const table = el("table", "causal");
    const thead = el("thead");
    const hr = el("tr");
    ["model", "10-way acc", "agree w/ teacher", "KL 1000", "TV 1000",
     "KL 10", "TV 10", "feature CKA"].forEach(h => hr.append(el("th", null, h)));
    thead.append(hr);
    table.append(thead);
    const tbody = el("tbody");

    const trT = el("tr");
    trT.append(el("td", null, Tt.label));
    trT.append(el("td", null, pct(summary.accuracy_10)));
    ["—", "—", "—", "—", "—"].forEach(v => trT.append(el("td", null, v)));
    tbody.append(trT);

    ["reference", "distilled"].forEach(tag => {
      const s = R.students?.[tag];
      if (!s) return;
      const m = s.summary || {};
      const c = R.cka?.[tag] || {};
      const tr = el("tr");
      tr.append(el("td", null, `${s.label} (${tag})`));
      tr.append(el("td", null, pct(m.accuracy_10)));
      tr.append(el("td", null, pct(m.agreement_full)));
      tr.append(el("td", null, num(m.mean_kl_full)));
      tr.append(el("td", null, num(m.mean_tv_full)));
      tr.append(el("td", null, num(m.mean_kl_10)));
      tr.append(el("td", null, num(m.mean_tv_10)));
      tr.append(el("td", null, c.best === undefined ? "—" : `${num(c.best)} (L${c.layer})`));
      tbody.append(tr);
    });
    table.append(tbody);
    teacherBox.append(table);

    const ckaNote = el("p", "cka-read");
    const cr = R.cka?.reference || {};
    const cd = R.cka?.distilled || {};
    ckaNote.innerHTML = `The teacher’s pooled feature has its strongest image-level CKA with the ` +
      `reference class token at <b>L${cr.layer ?? "?"} (${num(cr.best)})</b> and with the ` +
      `distilled class token at <b>L${cd.layer ?? "?"} (${num(cd.best)})</b>. ` +
      `The 200-permutation null for the best layer reaches p95 <b>${num(cr.null_p95)}</b> / ` +
      `<b>${num(cd.null_p95)}</b>. This is twenty-image geometry, not token-level correspondence ` +
      `and not a causal transfer test.`;
    teacherBox.append(ckaNote);

    const rows = R.students?.distilled?.rows || [];
    if (rows.length) {
      const label = el("p", "sub");
      label.style.marginTop = "24px";
      label.textContent = "Per image: the teacher’s ten-way bet, the distilled student’s bet, and their ten-way total variation.";
      teacherBox.append(label);
      const list = el("div", "gap-list");
      rows.forEach(r => {
        const row = el("div", "grow");
        row.append(el("span", "q", r.prompt));
        const t = el("span", "t");
        t.innerHTML = `teacher <b>${r.teacher_candidate}</b>`;
        const s = el("span", "s");
        s.innerHTML = `${roleS} <b>${r.student_candidate}</b>`;
        const m = el("div", "m");
        const bar = el("div", "bar");
        const fill = el("i");
        fill.style.width = `${Math.max(1.5, Math.min(1, r.tv_10) * 100)}%`;
        bar.append(fill);
        m.append(bar, el("span", null, num(r.tv_10, 2)));
        row.append(t, s, m);
        list.append(row);
      });
      teacherBox.append(list);
    }
  } else if (teacherBox) {
    teacherBox.append(el("p", "pending", "The RegNet teacher output is not in this data file yet. Run kaggle/distill-vision-teacher/fetch.sh; the page reads the measured JSON and does not run a model."));
  }
})();
