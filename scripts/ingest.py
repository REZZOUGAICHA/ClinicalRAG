"""
Ingest script — embeds all 80 chunks and stores them in ChromaDB.

Run this ONCE after Phase 1 (or whenever documents change).
After this, the vector store persists on disk in data/vectorstore/
and doesn't need to be rebuilt unless you add new documents.

Usage: python scripts/ingest.py

Variants (see src/config.py) build their own index, e.g.
    RAG_EMBEDDER=medcpt RAG_VECTORSTORE_DIR=data/vectorstore_exp/medcpt python scripts/ingest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.config import settings
from src.vectorstore.store import VECTORSTORE_DIR, ingest_chunks

CHUNKS_FILE = Path("data/processed/chunks.json")

if __name__ == "__main__":
    if not CHUNKS_FILE.exists():
        print("chunks.json not found. Run: python scripts/run_pipeline.py first.")
        sys.exit(1)

    print(f"Starting ingestion into {VECTORSTORE_DIR} "
          f"(embedder={settings.embedder}, contextual={settings.contextual})...")
    count = ingest_chunks(CHUNKS_FILE)
    print(f"\nDone — {count} chunks embedded and stored in ChromaDB ({VECTORSTORE_DIR}/).")
    print("You can now run: python scripts/ask.py")
