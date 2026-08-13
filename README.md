---
title: ClinicalRAG
emoji: 🏥
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# ClinicalRAG

A clinical document question-answering system: upload clinical reports (typed, scanned, or handwritten), ask questions in plain English, and get answers grounded in the actual source documents — every claim traceable to a specific passage, with the cited PDF page rendered and highlighted inline.

Built as a personal project to explore real-world RAG engineering: not just "call an LLM," but hybrid retrieval, real OCR (tested against genuine scanned/handwritten datasets, not synthetic stand-ins), measured retrieval/generation quality via RAGAS, and the debugging that real integration work actually involves.

## What it does

- **Ask a question, get a grounded answer.** Every answer cites the exact source file, section, and page it came from — click a citation to see the actual PDF page with the cited text highlighted.
- **Upload your own PDFs.** Typed reports extract instantly via PyMuPDF; scanned or photographed documents fall back to real OCR (docTR).
- **Handles messy real-world documents**, not just clean synthetic ones — see [Real-data testing](#real-data-testing) below.
- **Refuses to guess.** If the answer isn't in the documents, it says so — no hallucinated lab values or diagnoses.

## Architecture

```
PDF/image upload
      │
      ▼
┌─────────────┐   text layer present?  ┌──────────────────┐
│  PyMuPDF     │ ─────── no ──────────▶│  docTR OCR        │
│  extraction  │                       │  (detect + read)  │
└──────┬───────┘                       └─────────┬─────────┘
       │                          low page confidence?
       │                                          │ yes
       │                                          ▼
       │                              ┌────────────────────────┐
       │                              │ fine-tuned TrOCR        │
       │                              │ (handwriting specialist)│
       │                              └────────────┬────────────┘
       ▼                                            ▼
                  section-aware chunking
                            │
                            ▼
              BAAI/bge-small-en embeddings → ChromaDB
                            │
              ┌─────────────┴─────────────┐
              │   hybrid retrieval          │
              │   BM25 + dense + RRF fusion │
              │   → cross-encoder rerank    │
              │   → parent-document expand  │
              └─────────────┬───────────────┘
                            ▼
              Groq (Llama 3.3 70B) — cited, grounded answer
```

**Retrieval robustness, added after real testing surfaced real gaps:**
- **Query rewriting** — typos ("whats the id of lymphoma patien") no longer silently break retrieval; a fast LLM pass normalizes the query before search.
- **Parent-document expansion** — a question needing two sections of the same report (e.g. "what medications is the lymphoma patient on?" — diagnosis and medication list live in different chunks) pulls in the right document's full context, not just the single top-scoring chunk.
- **Conversational messages** ("thanks!") get a natural reply instead of a forced, awkward "not found in documents."

## Real-data testing

Everything below was measured against real external datasets, not just synthetic test cases:

| Test | Dataset | Result |
|---|---|---|
| OCR — printed/scanned text | FUNSD (real scanned forms) | **90.7%** word-level recall |
| OCR — handwritten text (baseline) | RxHandBD (real handwritten prescriptions) | 0% exact match — a genuine, documented hard-OCR-problem finding |
| OCR — handwritten, fine-tuned model | RxHandBD (held-out test set) | **47.5%** exact match after fine-tuning TrOCR on real labeled data |
| OCR — handwritten, generalization check | IAM (general handwriting) | Confirmed a real trade-off: fine-tuning for medical vocabulary cost general-handwriting accuracy (53.8%→7.7%) — documented, not hidden |

The handwriting-recognition fine-tuning (`notebooks/finetune_trocr_rxhandbd.ipynb`) and the full generalization analysis are part of this repo — including the honest negative result, because that's the actually-useful finding.

## Data

37 real clinical documents: 10 synthetic (LLM-generated, for controlled testing) + 27 real, de-identified transcriptions from **MTSamples** across 9 medical specialties. Real dictated notes use a completely different section-header format than synthetic reports — the chunker (`src/chunking/chunker.py`) handles both.

## Evaluation

`scripts/evaluate.py` — a RAGAS-based harness scoring faithfulness, context precision, context recall, and answer relevancy against hand-written ground truth pulled directly from the source documents.

## Stack

| Layer | Choice |
|---|---|
| Backend | FastAPI |
| Frontend | Vanilla HTML/CSS/JS (ES modules), PDF.js for citation rendering |
| LLM | Groq (Llama 3.3 70B) |
| Embeddings | BAAI/bge-small-en |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector store | ChromaDB |
| OCR | PyMuPDF (text PDFs) + docTR (scanned/image) + fine-tuned TrOCR (handwriting) |
| Eval | RAGAS |

## Running locally

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
pip install -r requirements.txt
# create a .env file with GROQ_API_KEY=your-key-here

python scripts/run_pipeline.py      # chunk the source PDFs
python scripts/ingest.py            # embed + index into ChromaDB

uvicorn src.api.main:app --reload --port 8000
```

Then open `http://localhost:8000`.

## Running with Docker

```bash
docker build -t clinicalrag .
docker run -p 7860:7860 -e GROQ_API_KEY=your-key-here clinicalrag
```

## Known limitations

- Handwriting OCR remains genuinely hard — see the real, measured numbers above rather than a claim that it "works."
- Vector store is ChromaDB (local/embedded) — a Qdrant Cloud migration was planned but not completed.
- The fine-tuned handwriting model (1.3GB) isn't bundled in this repo/image — see `notebooks/finetune_trocr_rxhandbd.ipynb` to reproduce it.
