FROM python:3.13-slim

# build-essential: chromadb depends on chroma-hnswlib, a C++ extension with
# no prebuilt wheel for this platform/Python combo — pip falls back to
# compiling it from source, which fails outright on a "slim" image (that's
# exactly what "slim" means: no compiler included at all).
# libgl1 etc.: opencv-python (a docTR dependency) needs these present at
# runtime on Debian-based images, or it fails with "libGL.so.1: cannot open
# shared object file" the moment anything imports it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch FIRST, from PyTorch's own CPU index — pip's default PyPI
# wheel for torch bundles full CUDA/GPU runtime libraries (nvidia-cudnn,
# nvidia-cublas, etc.), adding 2GB+ of libraries that a CPU-only deployment
# (this one) will never use. Installing the CPU build first means the later
# `pip install -r requirements.txt` sees torch/torchvision already satisfied
# and doesn't pull the GPU variant in via python-doctr[torch]'s dependency.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies in their own layer — this only gets
# rebuilt when requirements.txt actually changes, not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Spaces runs the container as uid 1000, not root. Everything from here on runs
# as that user, so the model caches land in /home/user/.cache (readable at
# runtime) and data/ stays writable for uploads. Built as root, the caches sat
# in /root, which uid 1000 can't read, and startup failed in offline mode.
RUN useradd -m -u 1000 user && chown user /app
USER user
ENV HOME=/home/user

# Now the actual application code and the source PDFs the pipeline needs.
COPY --chown=user src/ src/
COPY --chown=user scripts/ scripts/
COPY --chown=user data/raw/ data/raw/
# LLM-written chunk contexts (scripts/build_contexts.py). Contextual retrieval
# is on by default, so ingest below embeds context + chunk and BM25 indexes the
# same. Entries are keyed by a hash of the chunk text: ingest prints
# "N applied, N stale, N missing"; stale entries mean extraction here produced
# different chunk text than the cache was built on (those chunks get no context).
COPY --chown=user data/contexts/ data/contexts/

# Bake a ready-to-query vector store into the image at build time, so the
# container starts up with real data already indexed instead of empty.
# All 37 source PDFs here are born-digital (not scanned), so this doesn't
# trigger docTR — that model is pre-warmed separately below, since the
# OCR path (used when a user uploads a scanned document at runtime) would
# otherwise download its weights on that user's first request instead.
RUN python scripts/run_pipeline.py && python scripts/ingest.py
RUN python -c "from src.ocr.extractor import get_ocr_model; get_ocr_model()"
# Same for the reranker: the server loads it at startup, so without this every
# cold start of the Space (they sleep when idle) re-downloads it first.
RUN python -c "from src.retrieval.retriever import get_reranker; get_reranker()"

# Every model is in the image now. Without this, transformers still pings
# huggingface.co on each load to check for newer revisions, which slows cold
# starts and fails outright if the Hub is unreachable.
ENV HF_HUB_OFFLINE=1

# Hugging Face Spaces' Docker SDK expects the app on port 7860 by default.
EXPOSE 7860

# 0.0.0.0, not 127.0.0.1 — otherwise the server only accepts connections
# from inside the container itself, unreachable from outside it.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
