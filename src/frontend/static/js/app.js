import { fetchHealth, fetchDocuments, uploadFile, askStream, documentUrl } from './api.js';
import { renderHighlightedPage } from './pdfViewer.js';

let busy = false;

// Visitors just get answers; the router picks pipeline or agent on the server.
// Add ?debug=1 to the URL to see which path ran, the live graph and the steps.
const DEBUG = new URLSearchParams(location.search).has('debug');

document.addEventListener('DOMContentLoaded', () => {
  applyTheme(localStorage.getItem('theme') || 'light');
  bindEvents();
  loadDocs();
  el('qi').focus();
});

function bindEvents() {
  el('upload-btn').addEventListener('click', () => el('file-in').click());
  el('file-in').addEventListener('change', (e) => upload(e.target.files[0]));

  el('theme-btn').addEventListener('click', toggleTheme);

  el('qi').addEventListener('keydown', onKey);
  el('qi').addEventListener('input', () => grow(el('qi')));
  el('send-btn').addEventListener('click', send);

  el('welcome').addEventListener('click', (e) => {
    if (e.target.classList.contains('eg')) fill(e.target.textContent);
  });

  el('doc-list').addEventListener('click', (e) => {
    const item = e.target.closest('.doc-item');
    if (item) window.open(documentUrl(item.dataset.file), '_blank');
  });

  el('modal').addEventListener('click', closeModal);
  el('modal-close').addEventListener('click', () => closeModal());
}

// ── Documents ──────────────────────────────────────────────────────
async function loadDocs() {
  // The hosted API scales to zero; a cold start takes ~30 s of loading models.
  const waking = setTimeout(() => {
    el('hdr-stat').textContent = 'Waking up the server (up to ~30 s on first visit)…';
  }, 3000);
  try {
    const [docs, health] = await Promise.all([fetchDocuments(), fetchHealth()]);
    clearTimeout(waking);

    el('doc-list').innerHTML = docs.documents.length
      ? docs.documents.map(f => `<button class="doc-item" title="${x(f)}" data-file="${x(f)}">${x(f)}</button>`).join('')
      : '<div class="doc-empty">No documents yet</div>';

    el('hdr-stat').innerHTML = `<b>${health.chunks_indexed}</b> chunks indexed`;
    el('foot-stat').textContent =
      `${docs.total} document${docs.total !== 1 ? 's' : ''} loaded`;
  } catch {
    clearTimeout(waking);
    el('hdr-stat').textContent = '';
    el('doc-list').innerHTML = '<div class="doc-empty">Cannot reach server</div>';
  }
}

// ── Send ───────────────────────────────────────────────────────────
async function send() {
  const input = el('qi');
  const q = input.value.trim();
  if (!q || busy) return;

  input.value = '';
  input.style.height = 'auto';
  setLock(true);

  const welcome = el('welcome');
  if (welcome) welcome.remove();

  addHtml('chat', `<div class="msg msg-u"><div class="bubble-u">${x(q)}</div></div>`);

  const id = 'r' + Date.now();
  addHtml('chat', `
    <div class="msg" id="${id}">
      <div class="answer-card">
        <div class="answer-top">
          <span class="answer-lbl" id="${id}-l">ClinicalRAG</span>
          <span class="answer-time" id="${id}-t">retrieving...</span>
        </div>
        ${DEBUG ? graphHtml(id) : ''}
        <div class="answer-body" id="${id}-b"></div>
      </div>
    </div>`);

  scrollEnd();

  const t0 = Date.now();
  let text = '';
  let srcs = [];
  let trace = null;
  let error = null;
  let finished = false;

  try {
    for await (const ev of askStream(q, 3, 'auto')) {
      if (ev.type === 'token') {
        text += ev.content;
        el(id + '-b').innerHTML = fmt(stripSrc(text)) + '<span class="cursor"></span>';
        scrollEnd();

      } else if (ev.type === 'sources') {
        srcs = ev.sources;

      } else if (ev.type === 'node') {
        graphEnter(id, ev.node);

      } else if (ev.type === 'step') {
        if (DEBUG) {
          // Current step where "retrieving..." normally sits, and the path taken.
          el(id + '-t').textContent = ev.content;
          if (ev.route) el(id + '-l').textContent = `ClinicalRAG · ${ev.route === 'simple' ? 'pipeline' : 'agent'}`;
        } else if (ev.route === 'complex') {
          el(id + '-t').textContent = 'checking several reports...';
        }
        scrollEnd();

      } else if (ev.type === 'trace') {
        trace = ev;

      } else if (ev.type === 'error') {
        // Replaces any partial answer: half an answer reads as a complete one.
        error = ev.message;

      } else if (ev.type === 'done') {
        finished = true;
        const secs = ((Date.now() - t0) / 1000).toFixed(1);
        el(id + '-t').textContent = `${secs}s`;
        el(id + '-b').innerHTML = error ? errorHtml(error) : fmt(stripSrc(text));
        if (srcs.length && !error) renderSources(id, srcs);
        if (DEBUG && trace && trace.trace.length) renderSteps(id, trace.trace);
        graphDone(id, !error);
      }
    }
    if (!finished) {
      el(id + '-t').textContent = '';
      el(id + '-b').innerHTML = errorHtml(error || 'The answer stream stopped unexpectedly. Please try again.');
    }
  } catch (e) {
    el(id + '-t').textContent = '';
    el(id + '-b').innerHTML = errorHtml(`Error: ${String(e)}. Is the server running?`);
  }

  setLock(false);
  scrollEnd();
}

// ── Source references + tooltips ───────────────────────────────────
function renderSources(id, srcs) {
  const card = document.querySelector(`#${id} .answer-card`);
  const row = document.createElement('div');
  row.className = 'sources-row';

  srcs.forEach((s, i) => {
    const preview = (s.text || '').slice(0, 220).replace(/\n+/g, ' ');

    const btn = document.createElement('div');
    btn.className = 'src-ref';
    btn.innerHTML = `
      <span class="src-n">[${i + 1}]</span>
      <span class="src-sec">${x(s.section)}</span>
      <span class="src-file">${x(s.source_file)}</span>
      <div class="src-tip">${x(preview)}${s.text && s.text.length > 220 ? '...' : ''}</div>`;
    btn.addEventListener('click', () => openModal(s));
    row.appendChild(btn);
  });

  card.appendChild(row);
}

// Collapsed by default: the steps are for checking how an answer was
// reached, not something to read every time.
function renderSteps(id, steps) {
  const card = document.querySelector(`#${id} .answer-card`);
  const items = steps.map(s =>
    `<li>${x(s.step)}<div class="step-sum">${x(s.summary)}</div></li>`).join('');
  card.insertAdjacentHTML('beforeend', `
    <details class="steps">
      <summary>${steps.length} agent step${steps.length !== 1 ? 's' : ''}</summary>
      <ol>${items}</ol>
    </details>`);
}

// ── Live agent graph ───────────────────────────────────────────────
// A hand-drawn copy of the nodes in src/agent/graph.py. The server sends a
// "node" event as each node starts: the running node is filled, visited
// ones are outlined, and a node that runs again gets a count, which is how
// the agent <-> tools loop shows up. Labels are fixed strings, never model
// text. If the graph changes, update this and scripts/draw_agent_graph.py.
const GNODES = {
  route:  { label: 'route',           x: 8,   y: 40, w: 72,  tip: 'Sorts the question: simple or complex' },
  simple: { label: 'simple pipeline', x: 110, y: 6,  w: 110, tip: 'One retrieval + one answer (the normal pipeline)' },
  agent:  { label: 'agent',           x: 110, y: 40, w: 72,  tip: 'The model plans: call a tool, or stop' },
  tools:  { label: 'tools',           x: 110, y: 78, w: 72,  tip: 'Runs the tool calls the agent asked for' },
  answer: { label: 'answer',          x: 230, y: 40, w: 72,  tip: 'Writes the answer from the gathered passages only' },
  ground: { label: 'ground',          x: 340, y: 40, w: 72,  tip: 'Checks every citation; asks for one rewrite if any fail' },
};
const GEDGES = {
  'route>simple':  'M80 46 L110 24',
  'route>agent':   'M80 52 L110 52',
  'agent>tools':   'M140 64 L140 78',
  'tools>agent':   'M152 78 L152 64',
  'agent>answer':  'M182 52 L230 52',
  'tools>answer':  'M182 86 L230 60',
  'answer>ground': 'M302 52 L340 52',
  'ground>answer': 'M376 40 Q321 18 266 40',
  'ground>end':    'M412 52 L442 52',
  'simple>end':    'M220 18 L448 18 L448 46',
};

function graphHtml(id) {
  // Open on wide screens, a one-line summary on phones.
  const open = window.matchMedia('(min-width: 720px)').matches ? ' open' : '';
  const edges = Object.entries(GEDGES).map(([k, d]) =>
    `<path class="ge" data-edge="${k}" d="${d}" marker-end="url(#gm)"/>`).join('');
  const nodes = Object.entries(GNODES).map(([k, n]) => `
    <g class="gn" data-node="${k}"><title>${n.tip}</title>
      <rect x="${n.x}" y="${n.y}" width="${n.w}" height="24" rx="3"/>
      <text x="${n.x + n.w / 2}" y="${n.y + 12}">${n.label}</text></g>`).join('');
  return `
    <details class="glive" id="${id}-g"${open}>
      <summary>Agent graph <span class="glive-now" id="${id}-gn"></span></summary>
      <svg viewBox="0 0 460 104" role="img" aria-label="Agent graph, current step highlighted">
        <defs>
          <marker id="gm" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path class="gm" d="M0 0 L6 3 L0 6z"/></marker>
          <marker id="gm-on" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path class="gm-on" d="M0 0 L6 3 L0 6z"/></marker>
        </defs>
        ${edges}${nodes}
        <g class="gn" data-node="end"><title>Finished</title><circle cx="448" cy="52" r="6"/></g>
      </svg>
    </details>`;
}

function graphMove(svg, from, to) {
  const edge = svg.querySelector(`[data-edge="${from}>${to}"]`);
  if (edge) {
    edge.classList.add('seen');
    edge.setAttribute('marker-end', 'url(#gm-on)');
  }
}

function graphEnter(id, node) {
  const box = el(id + '-g');
  // Only names we drew are looked up, so the selector never sees odd input.
  if (!box || !Object.hasOwn(GNODES, node)) return;
  const prev = box.dataset.cur;
  if (prev) {
    box.querySelector(`[data-node="${prev}"]`).classList.remove('on');
    graphMove(box, prev, node);
  }
  const g = box.querySelector(`[data-node="${node}"]`);
  const n = Number(g.dataset.n || 0) + 1;
  g.dataset.n = n;
  g.classList.add('seen', 'on');
  g.querySelector('text').textContent = n > 1 ? `${GNODES[node].label} ×${n}` : GNODES[node].label;
  box.dataset.cur = node;
  el(id + '-gn').textContent = GNODES[node].label;
}

function graphDone(id, ok) {
  const box = el(id + '-g');
  if (!box || !box.dataset.cur) return;
  const cur = box.dataset.cur;
  box.querySelector(`[data-node="${cur}"]`).classList.remove('on');
  if (ok) {
    graphMove(box, cur, 'end');
    box.querySelector('[data-node="end"]').classList.add('seen');
  }
  el(id + '-gn').textContent = ok ? 'done' : `stopped at ${GNODES[cur].label}`;
}

// ── Modal ──────────────────────────────────────────────────────────
function openModal(src) {
  el('m-sec').textContent = src.section;
  el('m-src').textContent = `${src.source_file} — page ${src.page_number}`;
  el('modal').classList.add('open');

  const pdfUrl = documentUrl(src.source_file);
  renderHighlightedPage(el('m-body'), pdfUrl, src.page_number, src.text || '')
    .catch((e) => {
      el('m-body').innerHTML = `<div class="pdf-status" style="color:var(--hi)">Could not render PDF: ${x(String(e))}</div>`;
    });
}

function closeModal(e) {
  if (!e || e.target === el('modal')) el('modal').classList.remove('open');
}

// ── Upload ─────────────────────────────────────────────────────────
async function upload(file) {
  if (!file) return;
  const msg = el('upload-msg');
  msg.textContent = `Uploading ${file.name}...`;

  try {
    const { ok, data } = await uploadFile(file);
    msg.textContent = ok ? `${data.chunks_added} chunks added` : data.detail || 'Upload failed';
    if (ok) await loadDocs();
  } catch {
    msg.textContent = 'Upload failed';
  }

  setTimeout(() => { msg.textContent = ''; }, 4000);
  el('file-in').value = '';
}

// ── Theme ──────────────────────────────────────────────────────────
// Light is always the default for a new visitor — no OS prefers-color-scheme
// fallback. Dark is opt-in only, remembered via localStorage once toggled.
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  el('theme-btn').textContent = theme === 'dark' ? '☀' : '☾';
}

function toggleTheme() {
  const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  localStorage.setItem('theme', next);
  applyTheme(next);
}

// ── Utils ──────────────────────────────────────────────────────────
// Remove [Source: ...] and [Source: ..., ...] patterns the LLM inlines
function stripSrc(t) {
  return t.replace(/\[Source:[^\]]+\]/gi, '').replace(/\s{2,}/g, ' ').trim();
}

// Colour lab flags
function fmt(t) {
  return x(t)
    .replace(/\[CRITICALLY HIGH\]/g, '<span class="hi">[CRITICALLY HIGH]</span>')
    .replace(/\[HIGH\]/g,   '<span class="hi">[HIGH]</span>')
    .replace(/\[LOW\]/g,    '<span class="lo">[LOW]</span>')
    .replace(/\[Normal\]/g, '<span class="ok">[Normal]</span>');
}

function errorHtml(msg) {
  return `<span style="color:var(--hi)">${x(msg)}</span>`;
}

function x(s) {
  return String(s ?? '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function el(id) { return document.getElementById(id) }
function addHtml(id, html) { el(id).insertAdjacentHTML('beforeend', html) }
function scrollEnd() { const c = el('chat'); c.scrollTop = c.scrollHeight }
function setLock(b) { busy = b; el('send-btn').disabled = b; el('qi').disabled = b }
function fill(t) { const i = el('qi'); i.value = t; grow(i); i.focus() }
function onKey(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }
function grow(el) { el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 120) + 'px' }
