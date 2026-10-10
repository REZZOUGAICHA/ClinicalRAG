"""
Embeddings module — converts text into vectors using bge-small-en.

WHY DO WE NEED THIS?
When a user asks "what was the WBC count?", we can't compare that sentence
to 80 chunks using string matching — the words won't overlap.
Embeddings solve this: both the question and the chunks are converted into
vectors (lists of 384 numbers), and we find chunks whose vectors are
mathematically close to the question's vector. "WBC count" and
"Complete Blood Count: WBC 11.2" end up close in vector space.

WHY bge-small-en SPECIFICALLY?
- "bge" = Beijing Academy of AI General Embeddings — consistently top-ranked
  on the MTEB benchmark (the standard leaderboard for embedding quality)
- "small" = 384-dimensional vectors, runs on CPU in under 1 second per batch
- "en" = English
- Free, local, no API call, no cost, works offline

WHY NOT OpenAI embeddings?
OpenAI's text-embedding-ada-002 is paid (though cheap). bge-small-en
matches or beats it on most retrieval benchmarks. For a portfolio project
that runs locally and deploys for free, this is the right call.

HOW THE MODEL WORKS (conceptual):
The model is a transformer (same family as BERT) fine-tuned specifically
to produce vectors where semantically similar sentences are close together.
"Fine-tuned for retrieval" means it was trained on pairs of (question, answer)
so it learned what "similar for the purpose of answering" means, not just
what "grammatically similar" means.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDERS, settings

# Model name on HuggingFace Hub (default BAAI/bge-small-en; RAG_EMBEDDER in
# src/config.py switches it). First run downloads it, then it's cached.
MODEL_NAME = EMBEDDERS[settings.embedder]
MEDCPT_ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"

# Same instruction for bge-small-en and bge-small-en-v1.5; the v1.5 model card
# says it's optional but still recommended for short queries against passages.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Module-level singleton — we load the model once and reuse it.
# Loading a transformer model takes ~2 seconds. If we reloaded it on every
# function call, a batch of 80 chunks would take 160 seconds instead of 2.
_model = None


class MedCPTEncoder:
    """
    MedCPT is an asymmetric pair: a query encoder for short questions and an
    article encoder for (title, text) pairs, trained together so their [CLS]
    vectors share one space. Usage follows the model cards: CLS pooling, no
    normalisation (it was trained with dot-product scores), max 64 query
    tokens and 512 article tokens.
    """

    def __init__(self):
        from transformers import AutoModel, AutoTokenizer

        self.query_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.query_model = AutoModel.from_pretrained(MODEL_NAME).eval()
        self.article_tokenizer = AutoTokenizer.from_pretrained(MEDCPT_ARTICLE_MODEL)
        self.article_model = AutoModel.from_pretrained(MEDCPT_ARTICLE_MODEL).eval()

    @staticmethod
    def _cls(model, encoded) -> np.ndarray:
        import torch

        with torch.no_grad():
            return model(**encoded).last_hidden_state[:, 0, :].numpy()

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        encoded = self.query_tokenizer(queries, truncation=True, padding=True,
                                       return_tensors="pt", max_length=64)
        return self._cls(self.query_model, encoded)

    def encode_articles(self, titles: list[str], texts: list[str], batch_size: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            pairs = [[t, x] for t, x in zip(titles[i:i + batch_size], texts[i:i + batch_size])]
            encoded = self.article_tokenizer(pairs, truncation=True, padding=True,
                                             return_tensors="pt", max_length=512)
            out.append(self._cls(self.article_model, encoded))
        return np.concatenate(out)


def get_model():
    global _model
    if _model is None:
        if settings.embedder == "medcpt":
            print(f"  Loading embedding models: {MODEL_NAME} + {MEDCPT_ARTICLE_MODEL} ...")
            _model = MedCPTEncoder()
        else:
            print(f"  Loading embedding model: {MODEL_NAME} (first run downloads ~130MB)...")
            _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(
    texts: list[str],
    *,
    titles: list[str] | None = None,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """
    Convert a list of text strings into a 2D numpy array of vectors.

    Input:  ["Hodgkin Lymphoma stage IIB", "WBC 11.2 HIGH", ...]
    Output: array of shape (n_texts, 384) — one 384-dim vector per text

    WHY batch instead of one at a time?
    The model processes text in parallel on the GPU/CPU in batches.
    Embedding 80 chunks at once is ~10x faster than embedding them one by one.

    show_progress_bar=True is useful during the initial ingest of all 80 chunks
    so you can see it's working and not frozen.

    titles (the chunks' section names) are only used by MedCPT, whose article
    encoder takes (title, text) pairs; bge ignores them.
    """
    model = get_model()
    if isinstance(model, MedCPTEncoder):
        return model.encode_articles(titles or [""] * len(texts), texts)
    return model.encode(
        texts,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=True,  # normalise to unit length — required for cosine similarity
    )


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string. Used at retrieval time.

    WHY a separate function from embed_texts?
    bge models are designed with a distinction between "document" and "query"
    embeddings. For queries, you prefix with "Represent this sentence for
    searching relevant passages:" — this is documented in the bge paper and
    improves retrieval quality by ~3-5% on benchmarks.
    We handle that distinction here so callers don't need to know about it.
    """
    model = get_model()
    if isinstance(model, MedCPTEncoder):
        return model.encode_queries([query])[0]  # trained without an instruction
    prefixed = f"{QUERY_INSTRUCTION}{query}"
    vector = model.encode(
        prefixed,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vector
