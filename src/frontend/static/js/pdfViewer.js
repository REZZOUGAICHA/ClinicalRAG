// Renders one page of a source PDF and highlights the text that the
// retriever actually used to answer the question — the "prove it" view.
import * as pdfjsLib from './vendor/pdfjs/pdf.min.mjs';

pdfjsLib.GlobalWorkerOptions.workerSrc = '/static/js/vendor/pdfjs/pdf.worker.min.mjs';

const MIN_MATCH_LEN = 12; // ignore short runs ("the", "of") to avoid false-positive highlights
const norm = (s) => s.replace(/\s+/g, ' ').trim().toLowerCase();

// item.width/item.height are already in page (unscaled user-space) units —
// the same space item.transform's translation (e,f) lives in. So only the
// viewport's own scale converts them to CSS pixels; the item's transform is
// used purely to place the run's origin (tx[4], tx[5]), not to rescale it.
// (An earlier version multiplied width by the *combined* transform's scale,
// double-counting the run's own font scaling — that's why highlights were
// rendering far too wide, spilling past the page edge.)
function itemRect(item, viewport) {
  const tx = pdfjsLib.Util.transform(viewport.transform, item.transform);
  const scale = Math.hypot(viewport.transform[0], viewport.transform[1]);
  const width = item.width * scale;
  const height = item.height * scale;
  return {
    left: tx[4],
    top: tx[5] - height,
    width,
    height,
  };
}

export async function renderHighlightedPage(container, pdfUrl, pageNumber, snippet) {
  container.innerHTML = '<div class="pdf-status">Loading page…</div>';

  const pdf = await pdfjsLib.getDocument(pdfUrl).promise;
  const page = await pdf.getPage(pageNumber);
  const viewport = page.getViewport({ scale: 1.5 });

  const canvas = document.createElement('canvas');
  canvas.width = viewport.width;
  canvas.height = viewport.height;

  const wrap = document.createElement('div');
  wrap.className = 'pdf-page-wrap';
  wrap.style.width = viewport.width + 'px';
  wrap.style.height = viewport.height + 'px';
  wrap.appendChild(canvas);

  container.innerHTML = '';
  container.appendChild(wrap);

  await page.render({ canvasContext: canvas.getContext('2d'), viewport }).promise;

  const textContent = await page.getTextContent();
  const target = norm(snippet);

  let firstHighlight = null;
  for (const item of textContent.items) {
    const t = norm(item.str);
    if (t.length < MIN_MATCH_LEN || !target.includes(t)) continue;

    const rect = itemRect(item, viewport);
    const hl = document.createElement('div');
    hl.className = 'pdf-highlight';
    hl.style.left = rect.left + 'px';
    hl.style.top = rect.top + 'px';
    hl.style.width = rect.width + 'px';
    hl.style.height = rect.height + 'px';
    wrap.appendChild(hl);
    firstHighlight ??= hl;
  }

  if (firstHighlight) {
    firstHighlight.scrollIntoView({ block: 'center' });
  }
}
