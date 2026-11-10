/* the ripple: reads measured residual deltas; never runs a model */
const BASE = 'data/textablate/';
let MANIFEST = [], D = null, pi = 0, cut = 0, kind = 'block', head = null, MAXR = 1, timer = null;

const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const vis = t => esc(String(t).replace(/\n/g, '↵').replace(/^ /, '␣')).slice(0, 11) || '∅';
const pct = v => (v * 100).toFixed(1) + '%';

const themeBtn = $('themebtn');
const saved = localStorage.getItem('um-theme');
if (saved) document.documentElement.dataset.theme = saved;
themeBtn.onclick = () => {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  localStorage.setItem('um-theme', next);
};
window.addEventListener('scroll', () => {
  const h = document.documentElement;
  $('progress').style.width = h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight) * 100 + '%';
}, { passive: true });

const travelOf = l => (D.travel ? D.travel.layers.find(x => x.layer === l) : null);
const partOf = (l, k, h) => (D.parts || []).find(p => p.layer === l && p.kind === k && (k !== 'head' || p.head === h));

function condition() {
  if (kind === 'block') {
    const t = travelOf(cut);
    return t ? { runs: D.layers[cut].runs, resid: t.runs[pi].resid } : null;
  }
  const p = partOf(cut, kind, head);
  return p ? { runs: p.runs, resid: p.runs[pi].resid } : null;
}

function cutLabel() {
  if (kind === 'head') return `Layer ${cut} · head ${head}`;
  if (kind === 'ln1') return `Layer ${cut} · LN 1`;
  if (kind === 'ln2') return `Layer ${cut} · LN 2`;
  if (kind === 'attn') return `Layer ${cut} · all heads`;
  if (kind === 'mlp') return `Layer ${cut} · MLP`;
  return `Layer ${cut} · whole block`;
}

function whatCut() {
  if (kind === 'head') return `head ${head}’s slice of the concatenated write`;
  if (kind === 'ln1') return 'the first layer norm’s output';
  if (kind === 'ln2') return 'the second layer norm’s output';
  if (kind === 'attn') return 'the whole concatenated attention write';
  if (kind === 'mlp') return 'the MLP’s write';
  return `all ${D.facts.heads} heads and the MLP`;
}

function cutShort() {
  if (kind === 'head') return 'head ' + head;
  if (kind === 'ln1') return 'ln 1';
  if (kind === 'ln2') return 'ln 2';
  if (kind === 'attn') return 'heads';
  if (kind === 'mlp') return 'mlp';
  return 'cut';
}

function scale() {
  let m = 0;
  for (const L of D.travel.layers) for (const r of L.runs) for (const v of r.resid) m = Math.max(m, v);
  return m || 1;
}

function barsHTML(list, max, cls) {
  return list.map(t =>
    `<div class="bar"><span class="t">${vis(t.tok)}</span><span class="p">${t.p.toFixed(3)}</span>` +
    `<span class="rail2"><i style="width:${Math.max(1, t.p / max * 100).toFixed(1)}%"></i></span></div>`
  ).join('');
}

function renderFacts() {
  const f = D.facts;
  $('lede').textContent =
    `${D.hf_name || f.family}${f.chat ? ' (chat)' : ''}, ${f.params_m}M parameters. ` +
    'Cut one of its layers, then follow what that cut did at every later depth and at the next token. ' +
    'Every bar and every cell is a measured lookup.';
  $('facts').textContent =
    `${f.params_m}M params · ${f.layers} layers · ${f.heads} heads · ${f.prompts} prompts · ` +
    `${f.device} ${f.dtype}`;
}

function renderPrompts() {
  const row = $('prompts');
  row.replaceChildren();
  D.prompts.forEach((p, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'q' + (i === pi ? ' on' : '');
    b.textContent = p;
    b.onclick = () => { pi = i; stopTrace(); render(); };
    row.append(b);
  });
}

function renderArch() {
  const cond = condition();
  if (!cond) { $('arch').innerHTML = '<p class="note">No measurement for that piece yet.</p>'; return; }
  const resid = cond.resid;
  const RM = Math.max(...resid, 1e-9);
  const n = D.facts.layers;
  const W = 720, TOP = 58, STEP = 40, BH = 30, H = TOP + n * STEP + 64;
  const r = D.layers[cut].runs[pi];
  const out = [];
  out.push(`<svg viewBox="0 0 ${W} ${H}" class="arch-svg" role="img" aria-label="GPT-2 architecture with layer ${cut} cut and the measured downstream change">`);
  out.push(`<line class="rail" x1="26" y1="44" x2="26" y2="${TOP + n * STEP - 4}"/>`);
  out.push('<g class="node"><rect x="52" y="10" width="330" height="34" rx="4"/>' +
    '<text class="lbl" x="64" y="25">token ids</text>' +
    '<text class="lbl dim" x="64" y="37">embedding lookup</text></g>');
  for (let l = 0; l < n; l++) {
    const y = TOP + l * STEP;
    const a = Math.min(1, Math.pow(resid[l] / RM, 0.6));
    const isCut = l === cut;
    out.push(`<g class="ablock${isCut ? ' cut' : ''}" data-layer="${l}" tabindex="0" role="button" aria-label="cut layer ${l}">`);
    out.push(`<title>${isCut ? 'layer ' + l + ' is cut' : 'cut layer ' + l}</title>`);
    out.push(`<line class="tap" x1="26" y1="${y + BH / 2}" x2="52" y2="${y + BH / 2}"/>`);
    out.push(`<circle class="tapdot" cx="26" cy="${y + BH / 2}" r="2.5"/>`);
    out.push(`<rect class="box" x="52" y="${y}" width="330" height="${BH}" rx="3"/>`);
    out.push(`<rect class="tint" x="52" y="${y}" width="330" height="${BH}" rx="3" style="fill:var(--ox);fill-opacity:${(0.04 + 0.4 * a).toFixed(3)}"/>`);
    out.push(`<text class="lbl" x="62" y="${y + 19}">block ${l}</text>`);
    out.push(`<rect class="mini" x="128" y="${y + 7}" width="54" height="16" rx="2"/><text class="mini-t" x="155" y="${y + 18}" text-anchor="middle">ATTN</text>`);
    out.push(`<rect class="mini m" x="188" y="${y + 7}" width="46" height="16" rx="2"/><text class="mini-t" x="211" y="${y + 18}" text-anchor="middle">MLP</text>`);
    out.push(`<text class="plus" x="242" y="${y + 19}">+ residual</text>`);
    out.push(`<rect class="bar-bg" x="400" y="${y + 10}" width="230" height="10" rx="2"/>`);
    out.push(`<rect class="bar-fill" x="400" y="${y + 10}" width="${(resid[l] / RM * 230).toFixed(1)}" height="10" rx="2"/>`);
    out.push(`<text class="val" x="638" y="${y + 19}">${pct(resid[l])}</text>`);
    if (isCut) out.push(`<text class="cutnote" x="392" y="${y + 19}" text-anchor="end">${cutShort()}</text>`);
    out.push('</g>');
  }
  const fy = TOP + n * STEP + 4;
  out.push(`<line class="rail" x1="26" y1="${TOP + n * STEP - 4}" x2="26" y2="${fy + 23}"/>`);
  out.push(`<g class="node out"><rect x="52" y="${fy}" width="330" height="46" rx="4"/>`);
  out.push(`<text class="lbl" x="62" y="${fy + 17}">final layer norm → logits</text>`);
  out.push(`<text class="out-t" x="62" y="${fy + 35}">next token ${vis(r.top[0].tok)} at ${pct(r.top[0].p)}</text></g>`);
  out.push(`<circle id="pulse" cx="26" cy="${TOP + cut * STEP + BH / 2}" r="4" class="pulse" visibility="hidden"/>`);
  out.push('</svg>');
  const arch = $('arch');
  arch.innerHTML = out.join('');
  arch.querySelectorAll('.ablock').forEach(g => {
    g.onclick = () => { cut = +g.dataset.layer; kind = 'block'; head = null; stopTrace(); render(); };
    g.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); g.onclick(); } };
  });
}

function renderPicker() {
  const box = $('picker');
  box.replaceChildren();
  const row = document.createElement('div');
  row.className = 'prow';
  row.append(el('span', 'plabel', 'inside block ' + cut));
  const mk = (k, h, text) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'pchip' + (kind === k && head === h ? ' on' : '');
    b.textContent = text;
    b.onclick = () => { kind = k; head = h; stopTrace(); render(); };
    row.append(b);
  };
  mk('ln1', null, 'LN 1');
  if (D.parts) {
    for (let h = 0; h < D.facts.heads; h++) mk('head', h, 'H' + h);
  } else {
    const n = document.createElement('span');
    n.className = 'pnote';
    n.textContent = 'per-head ripple needs a re-run';
    row.append(n);
  }
  mk('attn', null, 'all heads');
  mk('mlp', null, 'MLP');
  mk('ln2', null, 'LN 2');
  mk('block', null, 'whole block');
  box.append(row);
  if (!D.parts) {
    const n = document.createElement('p');
    n.className = 'note';
    n.textContent = 'This results file has block-level ripple only; LN, head and attention cuts need the newer measurement.';
    box.append(n);
  }
}

function renderStack() {
  const cond = condition();
  const stack = $('stack');
  if (!cond || !stack) { if (stack) stack.replaceChildren(); return; }
  const resid = cond.resid;
  const RM = Math.max(...resid, 1e-9);
  stack.replaceChildren();
  for (let l = 0; l < D.facts.layers; l++) {
    const row = document.createElement('div');
    row.className = 'srow' + (l === cut ? ' cut' : '');
    row.innerHTML =
      `<span class="idx">L${l}</span>` +
      '<span class="blocks"><i class="blk a">ATTN</i><i class="blk m">MLP</i></span>' +
      `<span class="track"><i style="width:${(resid[l] / RM * 100).toFixed(1)}%"></i></span>` +
      `<span class="val">${pct(resid[l])}</span>`;
    row.onclick = () => { cut = l; kind = 'block'; head = null; stopTrace(); render(); };
    stack.append(row);
  }
}

function renderReadout() {
  const cond = condition();
  if (!cond) return;
  const r = cond.runs[pi];
  const base = D.baseline[pi];
  const top = base.top;
  const after = r.top;
  const drop = (top[0].p - r.p_top1) * 100;
  $('readout').innerHTML =
    '<p class="tag">cut</p>' +
    `<h3>${cutLabel()}</h3>` +
    `<p class="sub">${whatCut()}, mean-ablated at every token</p>` +
    `<p class="metric">KL from the clean output<b>${r.kl.toFixed(2)}</b></p>` +
    `<p class="sub">clean top-1 ${vis(top[0].tok)} ${pct(top[0].p)} → ${pct(r.p_top1)} ` +
    `(${drop >= 0 ? '−' : '+'}${Math.abs(drop).toFixed(1)} points)</p>` +
    '<button class="btn" id="trace" type="button">Trace the ripple</button>' +
    '<div class="cmp"><div><h4>clean</h4>' + barsHTML(top, Math.max(top[0].p, 0.01)) + '</div>' +
    '<div class="cmp after"><div><h4>after the cut</h4>' + barsHTML(after, Math.max(after[0].p, 0.01)) + '</div></div>';
  $('trace').onclick = trace;
}

/* the matrix: one row per cut, one column per measured depth */
function renderMatrix() {
  const n = D.facts.layers;
  const BM = scale();
  const grid = $('matrix');
  grid.style.gridTemplateColumns = `2.1rem repeat(${n}, minmax(14px,1fr))`;
  grid.replaceChildren();
  grid.append(el('div', 'mhead'));
  for (let c = 0; c < n; c++) grid.append(el('div', 'mhead', 'L' + c));
  for (let r = 0; r < n; r++) {
    const lab = el('div', 'mrow', 'L' + r);
    lab.title = `cut layer ${r}`;
    lab.onclick = () => { cut = r; stopTrace(); render(); };
    grid.append(lab);
    const run = travelOf(r).runs[pi].resid;
    for (let c = 0; c < n; c++) {
      if (c < r) {
        const z = el('div', 'mcell zero');
        z.title = `cut L${r}, measured after L${c}: before the cut, so no change by construction`;
        grid.append(z);
      } else {
        const v = run[c];
        const cell = el('div', 'mcell' + (r === cut ? ' on' : ''));
        cell.style.setProperty('--v', Math.min(1, Math.pow(v / BM, 0.6)).toFixed(3));
        cell.title = `cut L${r} · measured after L${c} · ${pct(v)} relative change`;
        cell.onclick = () => { cut = r; stopTrace(); render(); };
        grid.append(cell);
      }
    }
  }
  const leak = D.travel.leak ?? 0;
  $('matrix-note').textContent =
    `Upper triangle: the stream cannot change before the block is cut. Largest measured leak: ${pct(leak)}. ` +
    (leak < 1e-6 ? 'Exactly zero, as it must be.' : 'Non-zero only by floating-point noise.');
}

function el(tag, cls, text) {
  const n = document.createElement(tag);
  n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

function stopTrace() {
  if (timer) { clearInterval(timer); timer = null; }
  document.querySelectorAll('.ablock.lit').forEach(b => b.classList.remove('lit'));
  const pulse = document.getElementById('pulse');
  if (pulse) pulse.setAttribute('visibility', 'hidden');
}

function trace() {
  stopTrace();
  const blocks = [...document.querySelectorAll('.ablock')];
  if (!blocks.length) return;
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
    blocks.forEach((b, i) => { if (i >= cut) b.classList.add('lit'); });
    return;
  }
  const pulse = document.getElementById('pulse');
  const TOP = 58, STEP = 40, BH = 30;
  let i = cut;
  if (pulse) pulse.setAttribute('visibility', 'visible');
  timer = setInterval(() => {
    if (i >= blocks.length) { stopTrace(); return; }
    if (pulse) pulse.setAttribute('cy', TOP + i * STEP + BH / 2);
    blocks[i].classList.add('lit');
    i++;
  }, 170);
}

function render() {
  if (!D.travel) {
    $('lede').textContent = 'This model was measured before the ripple existed.';
    $('arch').innerHTML = '<p class="note">No per-depth residual data in this results file. Re-run the measurement with <code>--travel</code> to fill it in.</p>';
    $('readout').innerHTML = '';
    $('matrix').replaceChildren();
    $('matrix-note').textContent = '';
    return;
  }
  if (cut >= D.facts.layers) cut = D.facts.layers - 1;
  if (!D.parts && kind !== 'block') { kind = 'block'; head = null; }
  renderFacts(); renderPrompts(); renderPicker(); renderArch(); renderStack(); renderReadout(); renderMatrix();
}

async function load(slug) {
  const entry = MANIFEST.find(m => m.slug === slug);
  const res = await fetch(BASE + (entry ? entry.file : slug + '.json'), { cache: 'no-store' });
  if (!res.ok) throw new Error('HTTP ' + res.status);
  D = await res.json();
  pi = 0; cut = Math.min(Math.floor(D.facts.layers / 2), D.facts.layers - 1);
  kind = 'block'; head = null;
  document.querySelectorAll('#models .pill')
    .forEach(b => b.classList.toggle('on', b.dataset.slug === slug));
  render();
}

(async function start() {
  try {
    const res = await fetch(BASE + 'index.json', { cache: 'no-store' });
    if (!res.ok) throw new Error('no index');
    MANIFEST = await res.json();
    if (!Array.isArray(MANIFEST) || !MANIFEST.length) throw new Error('empty index');
    // only models with the per-depth pass can be drawn here
    const deep = MANIFEST.filter(m => m.travel);
    if (deep.length) MANIFEST = deep;
    for (const m of MANIFEST) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'pill';
      b.dataset.slug = m.slug;
      b.textContent = m.slug;
      b.onclick = () => load(m.slug).catch(e => { $('facts').textContent = 'Could not load ' + m.slug + ': ' + e.message; });
      $('models').append(b);
    }
    await load(MANIFEST[0].slug);
  } catch (e) {
    $('facts').textContent = 'No measurements yet — ' + e.message;
    $('arch').innerHTML = '<p class="note">Run <code>python experiments/textablate/run.py</code> to measure GPT-2, then reload.</p>';
  }
})();
