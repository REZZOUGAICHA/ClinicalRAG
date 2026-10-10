# Retrieval experiments (Phase A)

Run 2026-10-08. All rows: `scripts/eval_retrieval.py` on the 44 answerable items of
`eval/questions.json`, same cached query rewrites (`data/eval/rewrite_cache.json`),
top_k=3. Per-run JSON is in `data/eval/retrieval_<config>.json`; per-question
diffs come from `scripts/compare_retrieval.py baseline <config>`.

n = 44, so one question moves doc_hit by 0.023. Lookup and mtsamples are at 1.0 in
every config except the alternative rerankers; the differences are in
cross_section (n=9), typo (n=4) and multi_hop (n=12). Read the flip lists, not
just the means.

Latency: `scripts/bench_retrieval.py`, median / p90 ms for one `retrieve()` call
(top_k=3, parent expansion), torch limited to 2 threads, after warm-up, on an
Intel i7-8665U laptop (15 W, 1.9 GHz) with nothing else running. Query rewriting
(Groq) and generation are not included. Absolute numbers on a Spaces 2-vCPU box
will differ; the ratios between rows are the useful part.

## Settings

All switches are env vars read in `src/config.py`; unset means the original
pipeline. `baseline-repro` and `baseline-repro2` (all defaults, after the code
changes) reproduce `baseline` exactly: same ranked lists, same reranker scores
to the last float.

| env var | values (default first) |
|---|---|
| RAG_EMBEDDER | bge-small-en, bge-small-en-v1.5, medcpt |
| RAG_VECTORSTORE_DIR | data/vectorstore; experiments in data/vectorstore_exp/<name> |
| RAG_RERANKER | ms-marco-minilm, medcpt, bge-reranker-base, bge-reranker-v2-m3 |
| RAG_RERANK_POOL | 5 |
| RAG_FIRST_STAGE_K | 10 |
| RAG_RERANK_BATCH | 32 |
| RAG_CONTEXTUAL | 0 |
| RAG_RERANK_CONTEXT | 0 |

Config names: `poolN` = RAG_RERANK_POOL=N, `-k20` = RAG_FIRST_STAGE_K=20,
`emb-*` = embedder, `rr-*` = reranker, `ctx` = contextual chunks (BM25 + dense
index), `rrctx` = reranker also sees the context, `-b1` = RAG_RERANK_BATCH=1.

## Results

| config | hit@1 | hit@3 | recall@3 | MRR | sec_recall | ctx chars | typo hit@1 | cross_sec hit@1 | multi_hop sec_recall | latency ms |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.932 | 0.932 | 0.886 | 0.932 | 0.862 | 3303 | 0.75 | 0.89 | 0.660 | 622 / 937 |
| pool10 | 0.932 | 0.977 | 0.943 | 0.951 | 0.894 | 3035 | 0.75 | 0.78 | 0.778 | 1662 / 1871 |
| pool20 | 0.932 | 0.977 | 0.943 | 0.960 | 0.894 | 3150 | 0.75 | 0.78 | 0.778 | 2694 / 3449 |
| pool20-k20 | 0.932 | 0.977 | 0.955 | 0.960 | 0.886 | 3113 | 0.75 | 0.78 | 0.750 | - |
| emb-bge-v1.5 | 0.932 | 0.932 | 0.898 | 0.932 | 0.867 | 3167 | 0.75 | 0.89 | 0.681 | 611 / 868 |
| emb-medcpt | 0.932 | 0.955 | 0.909 | 0.939 | 0.866 | 3309 | 0.75 | 0.89 | 0.674 | 755 / 959 |
| emb-medcpt-pool20 | 0.955 | 0.955 | 0.913 | 0.955 | 0.890 | 3277 | 0.75 | 0.89 | 0.764 | - |
| rr-medcpt | 0.886 | 0.932 | 0.871 | 0.905 | 0.828 | 2902 | 0.50 | 0.89 | 0.660 | 3168 / 5274 |
| rr-medcpt-pool20 | 0.909 | 0.955 | 0.905 | 0.938 | 0.883 | 2528 | 0.50 | 0.89 | 0.778 | 18485 / 22574 |
| rr-bge-base | 0.886 | 0.932 | 0.875 | 0.905 | 0.835 | 3132 | 0.75 | 0.67 | 0.646 | 3464 / 5628 |
| rr-bge-base-pool20 | 0.750 | 0.955 | 0.894 | 0.846 | 0.794 | 3177 | 0.25 | 0.67 | 0.576 | - |
| rr-bge-v2-m3 | not run | | | | | | | | | 8289-11208 (8 queries) |
| ctx | 0.955 | 0.955 | 0.920 | 0.955 | 0.896 | 3091 | 0.75 | 1.00 | 0.701 | 558 / 936 |
| ctx-rrctx | 0.955 | 0.955 | 0.909 | 0.955 | 0.886 | 2907 | 0.75 | 1.00 | 0.667 | - |
| ctx-pool10 | 0.955 | 0.955 | 0.920 | 0.957 | 0.900 | 3200 | 0.75 | 1.00 | 0.715 | - |
| **ctx-pool20** | 0.977 | 1.000 | 0.977 | 0.985 | 0.920 | 3110 | 0.75 | 1.00 | 0.750 | 2620 / 3270 |
| ctx-pool20-k20 | 0.955 | 1.000 | 0.977 | 0.974 | 0.909 | 3177 | 0.75 | 0.89 | 0.750 | - |
| ctx-rrctx-pool20 | 0.977 | 0.977 | 0.928 | 0.981 | 0.879 | 2750 | 0.75 | 1.00 | 0.639 | - |
| ctx-bge-v1.5 | 0.955 | 0.977 | 0.943 | 0.966 | 0.907 | 3032 | 0.75 | 1.00 | 0.701 | - |
| ctx-bge-v1.5-pool20 | 0.977 | 0.977 | 0.966 | 0.982 | 0.915 | 3048 | 0.75 | 1.00 | 0.771 | - |
| ctx-medcpt | 0.955 | 0.955 | 0.913 | 0.955 | 0.892 | 3027 | 0.75 | 1.00 | 0.688 | - |
| ctx-medcpt-pool20 | 0.977 | 0.977 | 0.955 | 0.983 | 0.922 | 3145 | 0.75 | 1.00 | 0.799 | 2825 / 3214 |
| **ctx-pool20-b1** | 0.977 | 1.000 | 0.977 | 0.985 | 0.920 | 3110 | 0.75 | 1.00 | 0.750 | 1161 / 1637 |
| ctx-b1 | 0.955 | 0.955 | 0.920 | 0.955 | 0.896 | 3091 | 0.75 | 1.00 | 0.701 | 312 / 494 |
| baseline-b1 (latency only) | = baseline | | | | | | | | | 401 |
| pool10-b1 (latency only) | = pool10 | | | | | | | | | 776 |

The MRR and hit@3 columns are computed over the reranked list, which is as long as
the pool (5 by default, up to 10 with a bigger pool), so a deeper pool can raise
MRR just by ranking a gold chunk at 6-10.

## Model sizes and costs

| model | params | on disk | notes |
|---|---|---|---|
| BAAI/bge-small-en (current) | 33M | ~130 MB | |
| BAAI/bge-small-en-v1.5 | 33M | ~130 MB | same query instruction as v1 |
| ncbi/MedCPT Query + Article encoders | 219M (2 x 110M) | ~880 MB | both needed at runtime (article encoder for uploads) |
| cross-encoder/ms-marco-MiniLM-L-6-v2 (current) | 23M | ~90 MB | |
| ncbi/MedCPT-Cross-Encoder | 110M | ~440 MB | |
| BAAI/bge-reranker-base | 278M | 1.1 GB | |
| BAAI/bge-reranker-v2-m3 | 568M | 2.3 GB | 8-11 s per query at pool 5; dropped |
| contexts (gpt-oss-20b) | - | 242 entries, 100 KB JSON | 37 calls, 39k tokens, about 10 min on the free tier |

## What moved (vs baseline)

Almost every config with a bigger pool, MedCPT or contexts gains `mh-pth-abnormal`
(multi_hop section_recall 0.50 -> 1.00). It shows up under unrelated methods, so it's a
side effect of pulling a different candidate set, not something any one method fixes.

**pool20** (6 changed): + lymphoma-id-cross-section (hit@3), + mh-enoxaparin (hit@1),
+ mh-bisoprolol / mh-pth-abnormal (section_recall), + meds-acute-mi-paraphrase (MRR only);
**- depression-sertraline** (hit@1 1 -> 0: report_006 MEDICATIONS outranks the
depression report once it's in the pool). pool20-k20 makes the same 6 changes.

**ctx** (5 changed): + lymphoma-id-cross-section (hit@1 0 -> 1), + mh-enoxaparin (hit@1 0 -> 1),
+ mh-pth-abnormal; **- mh-bisoprolol** (hit@1 1 -> 0: the report_003 MEDICATIONS chunk
drops out of the top-5 fused candidates; the context makes the chunk longer and dilutes the rare
term "bisoprolol" for BM25 and the dense vector), - mh-female-ages (section_recall 0.50 -> 0.33).

**ctx-pool20** (5 changed, no losses): + lymphoma-id-cross-section (hit@1 0 -> 1),
+ mh-enoxaparin (hit@1 0 -> 1), + meds-acute-mi-paraphrase (hit@3 0 -> 1),
+ mh-pth-abnormal and mh-hba1c-all (section_recall). The bigger pool gets
bisoprolol back, and the contexts keep depression-sertraline on top.
`ctx-pool20-b1` is identical question by question.

Caveat on ctx-pool20: with RAG_FIRST_STAGE_K=20 (ctx-pool20-k20),
lymphoma-id-cross-section drops back to hit@1 = 0 (gold at rank 2). That win
depends on which 20 candidates make the pool, so it's less solid than it looks.

**emb-bge-v1.5**: 1 change (mh-bisoprolol section_recall up). Equivalent to v1 here.

**emb-medcpt** (6 changed): + lymphoma-id-cross-section, **- meds-acute-mi-cross-section**
(hit@1 1 -> 0; with pool 20 it loses hit@3 too), section_recall up on 2 and down on 2.
Net zero on hit@1.

**ctx-medcpt-pool20 vs ctx-pool20** (3 changed): - meds-acute-mi-paraphrase hit@3,
+ section_recall on mh-bisoprolol and mh-enoxaparin. A wash, at 7x the
embedder size.

**rr-medcpt** (2 changed, both losses): lymphoma-id-typo and mts-ob-us-afi-weight
lose hit@1. **rr-bge-base**: wbc-lymphoma and renal-stopped-meds lose hit@1.
**rr-bge-base-pool20**: 14 questions change, 10 of them losses. Neither reranker
beats ms-marco-MiniLM on this set, and both are 5-30x slower.

**ctx-rrctx** (reranker sees context + chunk): same hits as ctx, lower
section_recall; with pool 20 it loses 5 questions against ctx-pool20.
The reranker does better on the raw chunk.

Items no config fixes: meds-acute-mi-paraphrase never reaches hit@1 (typo
hit@1 is 0.75 everywhere except rr-medcpt / rr-medcpt-pool20 at 0.50 and
rr-bge-base-pool20 at 0.25).

## Reranker batch size

sentence-transformers' CrossEncoder pads every pair in a batch to the longest
one. Pool pairs are 112 tokens at the median but up to 512 (long MTSamples
chunks), so a single batch of 20 is mostly padding. With batch size 1:
pool 20 goes from 2.6 s to 1.16 s, and the default pool 5 from 622 ms to 401 ms.
Scores change by at most 4e-6 (float noise), and every eval run with `-b1`
matches its batch-32 counterpart question by question. It's a switch with the
default left at 32, so the baseline stays bit-identical.

## Contextual retrieval: implementation notes

- `scripts/build_contexts.py` builds `data/contexts/chunk_contexts.json` (not
  ignored by git or docker), keyed by chunk_id with a hash of the chunk text.
  Stale entries are ignored and regenerated. Resumable, paced to 7000 TPM.
- One gpt-oss-20b call per document, not one per chunk as in Anthropic's
  recipe: Groq's free tier has no prompt caching, and per-chunk calls would
  cost about 6x the tokens. The model returns one sentence identifying the
  document (type, patient ID, diagnosis/procedure) plus one sentence per chunk,
  and we join them. When asked for full per-chunk contexts, gpt-oss-20b named
  the patient only in the first chunk's context, which defeats the purpose.
- Context goes into the BM25 tokens and the dense vector only. The user and the
  answer LLM still see the original chunk text, because the context is
  model-written and must not be cited as if it were in the PDF. Feeding it to the
  reranker too (rrctx) was measured and is worse.
- Uploads: `src/retrieval/contextual.embed_for_index(chunks)` generates
  contexts (falling back to none if the LLM call fails), caches them, and
  returns embeddings for the configured embedder.
