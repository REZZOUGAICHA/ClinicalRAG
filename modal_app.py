"""
Serves the FastAPI backend on Modal. The frontend is hosted separately
(Netlify, see netlify.toml) and calls this API.

    modal secret create clinicalrag-groq GROQ_API_KEY=...   # once
    set CLINICALRAG_CORS_ORIGINS=https://<your-site>.netlify.app
    modal deploy modal_app.py

The image is built from the same Dockerfile used for local Docker runs, so the
vector index, contexts and models are baked in exactly as tested there.
"""

import os
from pathlib import Path

import modal

ROOT = Path(__file__).parent

# Read at deploy time and passed into the container as an env var.
CORS_ORIGINS = os.environ.get("CLINICALRAG_CORS_ORIGINS", "*")

image = (
    modal.Image.from_dockerfile(ROOT / "Dockerfile", context_dir=ROOT)
    # The Dockerfile ends as uid 1000 (what Hugging Face Spaces expects), but
    # Modal mounts this file and its runtime under /root, which that user can't
    # read. The model caches stay where they were built (HOME=/home/user).
    .dockerfile_commands(["USER root"])
)

app = modal.App("clinicalrag")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("clinicalrag-groq")],
    env={"CLINICALRAG_CORS_ORIGINS": CORS_ORIGINS},
    cpu=2,
    memory=4096,
    # Scale to zero when idle so a portfolio demo costs almost nothing; one
    # container is plenty and keeps uploads and the index in one place.
    min_containers=0,
    max_containers=1,
    scaledown_window=300,
    # Agent-mode questions can take a minute on Groq's free tier.
    timeout=600,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def web():
    import sys

    # The app uses paths relative to the repo root (data/, src/frontend/).
    os.chdir("/app")
    sys.path.insert(0, "/app")
    from src.api.main import app as fastapi_app

    return fastapi_app
