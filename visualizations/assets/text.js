/* the language-model ablation panel: reads the measured JSON, never a model */
const BASE = 'data/textablate/';
let MANIFEST = [], D = null, pi = 0, sel = null, MAXKL = 1;

const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const vis = t => esc(String(t).replace(/\n/g, '↵').replace(/^ /, '␣')).slice(0, 12) || '∅';

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
  const pct = h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight) * 100;
  $('progress').style.width = pct + '%';
}, { passive: true });

function bars(list, maxP) {
  return '<div class="bars">' + list.map(t =>
    `<div class="bar"><span class="t">${vis(t.tok)}</span>` +
    `<span class="track"><i style="width:${Math.max(1, t.p / maxP * 100).toFixed(1)}%"></i></span>` +
    `<span class="p">${t.p.toFixed(3)}</span></div>`).join('') + '</div>';
}

function renderFacts() {
  const f = D.facts;
  $('lede').textContent =
    `${D.hf_name || f.family}${f.chat ? ' (chat)' : ''}, ${f.params_m}M parameters. ` +
    'Every head, every MLP, and every layer removed on its own — read what happened to the next ' +
    'token, then put it back. Nothing runs in the browser; each chip is a lookup.';
  $('facts').textContent =
    `${f.params_m}M params · ${f.layers} layers · ${f.heads} heads × ${f.head_dim} · ` +
    `${f.prompts} prompts · ${f.device} ${f.dtype} · measured in ${D.elapsed_s}s`;
  $('stage-note').textContent = f.operation_note || '';
}

function renderPrompts() {
  const row = $('prompts');
  row.replaceChildren();
  D.prompts.forEach((p, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'q' + (i === pi ? ' on' : '');
    b.textContent = p;
    b.onclick = () => { pi = i; sel = null; board(); renderPrompts(); output(); };
    row.append(b);
  });
}

function chip(n) {
  const r = n.runs[pi];
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'chip ' + n.kind;
  if (sel && sel.layer === n.layer && sel.kind === n.kind && sel.head === n.head) b.classList.add('on');
  b.style.setProperty('--a', Math.min(1, Math.pow(r.kl / MAXKL, 0.6)).toFixed(3));
  b.textContent = n.kind === 'head' ? 'H' + n.head : n.kind === 'mlp' ? 'MLP' : 'ALL';
  b.title = `L${n.layer} ${n.kind === 'head' ? 'head ' + n.head : n.kind} · KL ${r.kl.toFixed(3)} · top-1 ${r.p_top1.toFixed(3)}`;
  b.onclick = () => {
    const same = sel && sel.layer === n.layer && sel.kind === n.kind && sel.head === n.head;
    sel = same ? null : { layer: n.layer, kind: n.kind, head: n.head };
    board(); output();
  };
  return b;
}

function board() {
  const f = D.facts;
  MAXKL = Math.max(...D.nodes.map(n => n.runs[pi].kl), 1e-9);
  const frag = document.createDocumentFragment();
  for (let l = 0; l < f.layers; l++) {
    const row = document.createElement('div');
    row.className = 'row';
    const lab = document.createElement('div');
    lab.className = 'lab';
    lab.textContent = 'L' + l;
    const chips = document.createElement('div');
    chips.className = 'chips';
    for (const n of D.nodes) if (n.layer === l) chips.append(chip(n));
    const whole = D.layers.find(x => x.layer === l);
    if (whole) chips.append(chip({ layer: l, kind: 'all', head: null, runs: whole.runs }));
    row.append(lab, chips);
    frag.append(row);
  }
  $('stage').replaceChildren(frag);
}

function output() {
  const o = $('output');
  const base = D.baseline[pi];
  const head = base.top;
  const maxC = Math.max(...head.map(t => t.p), 0.01);
  let html =
    '<p class="tag">clean model</p>' +
    `<h3 class="prompt-title">${esc(D.prompts[pi])}</h3>` +
    '<p class="sub">Its next token, before anything is removed.</p>' +
    bars(head, maxC) +
    `<p class="sub">Entropy ${base.entropy.toFixed(2)} nats. Click a chip to remove that piece; ` +
    'click it again to put it back.</p>';

  if (!sel) { o.innerHTML = html + '<p class="sub">Nothing removed.</p>'; return; }

  let runs;
  if (sel.kind === 'head') {
    runs = D.nodes.find(n => n.layer === sel.layer && n.kind === 'head' && n.head === sel.head).runs;
  } else if (sel.kind === 'mlp') {
    runs = D.nodes.find(n => n.layer === sel.layer && n.kind === 'mlp').runs;
  } else {
    runs = D.layers.find(x => x.layer === sel.layer).runs;
  }
  const r = runs[pi];
  const pointDrop = (head[0].p - r.p_top1) * 100;
  const maxA = Math.max(...r.top.map(t => t.p), 0.01);
  const label = sel.kind === 'head' ? `Layer ${sel.layer} · head ${sel.head}`
    : sel.kind === 'mlp' ? `Layer ${sel.layer} · MLP`
      : `Layer ${sel.layer} · whole block`;

  o.innerHTML = html + '<hr>' +
    '<p class="tag removed">removed</p>' +
    `<h3>${label}</h3>` +
    `<p class="metric">KL from clean<b>${r.kl.toFixed(2)}</b></p>` +
    `<p class="sub">The clean top-1, ${esc(vis(head[0].tok))}, was ${(head[0].p * 100).toFixed(1)}%; ` +
    `with this piece removed it is ${(r.p_top1 * 100).toFixed(1)}% ` +
    `(${pointDrop >= 0 ? '−' : '+'}${Math.abs(pointDrop).toFixed(1)} points).</p>` +
    bars(r.top, maxA) +
    '<button class="btn" id="restore" type="button">Put it back</button>';

  $('restore').onclick = () => { sel = null; board(); output(); };
}

function render() { renderFacts(); renderPrompts(); board(); output(); }

async function load(slug) {
  const entry = MANIFEST.find(m => m.slug === slug);
  const res = await fetch(BASE + (entry ? entry.file : slug + '.json'), { cache: 'no-store' });
  if (!res.ok) throw new Error('HTTP ' + res.status);
  D = await res.json();
  pi = 0; sel = null;
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
    for (const m of MANIFEST) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'pill';
      b.dataset.slug = m.slug;
      b.textContent = m.slug;
      b.onclick = () => load(m.slug).catch(err => { $('facts').textContent = 'Could not load ' + m.slug + ': ' + err.message; });
      $('models').append(b);
    }
    await load(MANIFEST[0].slug);
  } catch (e) {
    $('facts').textContent = 'No measurements yet — ' + e.message;
    $('stage').innerHTML =
      '<p class="note">Run <code>python experiments/textablate/run.py</code> to measure GPT-2 on this ' +
      'machine, then reload. The Kaggle notebook writes the same schema for the larger models.</p>';
    $('output').innerHTML = '<p class="sub">Waiting for a results file.</p>';
  }
})();
