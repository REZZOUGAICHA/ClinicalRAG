// Thin fetch layer — every network call the frontend makes lives here.
// app.js never touches fetch() directly.

export async function fetchHealth() {
  const r = await fetch('/health');
  return r.json();
}

export async function fetchDocuments() {
  const r = await fetch('/documents');
  return r.json();
}

export async function uploadFile(file) {
  const fd = new FormData();
  fd.append('file', file);
  const r = await fetch('/upload', { method: 'POST', body: fd });
  const data = await r.json();
  return { ok: r.ok, data };
}

// Streams /ask as an async generator of parsed SSE events ({type, ...}).
export async function* askStream(query, topK = 3) {
  const res = await fetch('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK }),
  });

  if (!res.ok) throw new Error(String(res.status));

  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buf += dec.decode(value, { stream: true });
    const lines = buf.split('\n');
    buf = lines.pop();

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      try {
        yield JSON.parse(line.slice(6));
      } catch {
        // ignore malformed SSE lines
      }
    }
  }
}
