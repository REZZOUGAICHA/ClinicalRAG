"""
Ingest script — embeds all 80 chunks and stores them in ChromaDB.

Run this ONCE after Phase 1 (or whenever documents change).
After this, the vector store persists on disk in data/vectorstore/
and doesn't need to be rebuilt unless you add new documents.

Usage: python scripts/ingest.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.vectorstore.store import ingest_chunks

CHUNKS_FILE = Path("data/processed/chunks.json")

if __name__ == "__main__":
    if not CHUNKS_FILE.exists():
        print("chunks.json not found. Run: python scripts/run_pipeline.py first.")
        sys.exit(1)

    print("Starting ingestion...")
    count = ingest_chunks(CHUNKS_FILE)
    print(f"\nDone — {count} chunks embedded and stored in ChromaDB (data/vectorstore/).")
    print("You can now run: python scripts/ask.py")
