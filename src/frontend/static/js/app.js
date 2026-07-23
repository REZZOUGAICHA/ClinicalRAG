import { fetchHealth, fetchDocuments, uploadFile, askStream } from './api.js';
import { renderHighlightedPage } from './pdfViewer.js';

let busy = false;

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
    if (item) window.open(`/documents/${encodeURIComponent(item.dataset.file)}`, '_blank');
  });

  el('modal').addEventListener('click', closeModal);
  el('modal-close').addEventListener('click', () => closeModal());
}

// ── Documents ──────────────────────────────────────────────────────
async function loadDocs() {
  try {
    const [docs, health] = await Promise.all([fetchDocuments(), fetchHealth()]);

    el('doc-list').innerHTML = docs.documents.length
      ? docs.documents.map(f => `<button class="doc-item" title="${x(f)}" data-file="${x(f)}">${x(f)}</button>`).join('')
      : '<div class="doc-empty">No documents yet</div>';

    el('hdr-stat').innerHTML = `<b>${health.chunks_indexed}</b> chunks indexed`;
    el('foot-stat').textContent =
      `${docs.total} document${docs.total !== 1 ? 's' : ''} loaded`;
  } catch {
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
          <span class="answer-lbl">ClinicalRAG</span>
          <span class="answer-time" id="${id}-t">retrieving...</span>
        </div>
        <div class="answer-body" id="${id}-b"></div>
      </div>
    </div>`);

  scrollEnd();

  const t0 = Date.now();
  let text = '';
  let srcs = [];

  try {
    for await (const ev of askStream(q, 3)) {
      if (ev.type === 'token') {
        text += ev.content;
        el(id + '-b').innerHTML = fmt(stripSrc(text)) + '<span class="cursor"></span>';
        scrollEnd();

      } else if (ev.type === 'sources') {
        srcs = ev.sources;

      } else if (ev.type === 'done') {
        const secs = ((Date.now() - t0) / 1000).toFixed(1);
        el(id + '-t').textContent = `${secs}s`;
        el(id + '-b').innerHTML = fmt(stripSrc(text));
        if (srcs.length) renderSources(id, srcs);
      }
    }
  } catch (e) {
    el(id + '-b').innerHTML = `<span style="color:var(--hi)">Error: ${x(String(e))}. Is the server running?</span>`;
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

// ── Modal ──────────────────────────────────────────────────────────
function openModal(src) {
  el('m-sec').textContent = src.section;
  el('m-src').textContent = `${src.source_file} — page ${src.page_number}`;
  el('modal').classList.add('open');

  const pdfUrl = `/documents/${encodeURIComponent(src.source_file)}`;
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
