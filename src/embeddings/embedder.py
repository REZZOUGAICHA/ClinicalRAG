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

from sentence_transformers import SentenceTransformer
import numpy as np

# Model name on HuggingFace Hub.
# First time this runs it downloads ~130MB — after that it's cached locally.
MODEL_NAME = "BAAI/bge-small-en"

# Module-level singleton — we load the model once and reuse it.
# Loading a transformer model takes ~2 seconds. If we reloaded it on every
# function call, a batch of 80 chunks would take 160 seconds instead of 2.
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"  Loading embedding model: {MODEL_NAME} (first run downloads ~130MB)...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Convert a list of text strings into a 2D numpy array of vectors.

    Input:  ["Hodgkin Lymphoma stage IIB", "WBC 11.2 HIGH", ...]
    Output: array of shape (n_texts, 384) — one 384-dim vector per text

    WHY batch instead of one at a time?
    The model processes text in parallel on the GPU/CPU in batches.
    Embedding 80 chunks at once is ~10x faster than embedding them one by one.

    show_progress_bar=True is useful during the initial ingest of all 80 chunks
    so you can see it's working and not frozen.
    """
    model = get_model()
    return model.encode(
        texts,
        show_progress_bar=True,
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
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    vector = model.encode(
        prefixed,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vector
