FROM python:3.13-slim

# opencv-python (a docTR dependency) needs these system libraries present at
# runtime on Debian-based images, or it fails with "libGL.so.1: cannot open
# shared object file" the moment anything imports it.
RUN apt-get update && apt-get install -y --no-install-recommends \
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

# Now the actual application code and the source PDFs the pipeline needs.
COPY src/ src/
COPY scripts/ scripts/
COPY data/raw/ data/raw/

# Bake a ready-to-query vector store into the image at build time, so the
# container starts up with real data already indexed instead of empty.
# All 37 source PDFs here are born-digital (not scanned), so this doesn't
# trigger docTR — that model is pre-warmed separately below, since the
# OCR path (used when a user uploads a scanned document at runtime) would
# otherwise download its weights on that user's first request instead.
RUN python scripts/run_pipeline.py && python scripts/ingest.py
RUN python -c "from src.ocr.extractor import get_ocr_model; get_ocr_model()"

# Hugging Face Spaces' Docker SDK expects the app on port 7860 by default.
EXPOSE 7860

# 0.0.0.0, not 127.0.0.1 — otherwise the server only accepts connections
# from inside the container itself, unreachable from outside it.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
