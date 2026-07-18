"""
Runs the full Phase 1 pipeline end-to-end and prints a spot-check report.
Usage: python scripts/run_pipeline.py

This is your manual verification step — the "spend real time here before
touching the LLM layer" advice made concrete and runnable.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.ocr.extractor import extract
from src.chunking.chunker import chunk_document


DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def run():
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in data/raw/. Run: python scripts/generate_reports.py")
        return

    print(f"Found {len(pdfs)} PDF(s). Running pipeline...\n")
    all_chunks = []

    for pdf_path in pdfs:
        print(f"Processing: {pdf_path.name}")
        doc = extract(pdf_path)
        chunks = chunk_document(doc)

        print(f"  Pages: {len(doc.pages)}  |  Chars: {doc.total_chars}  |  Chunks: {len(chunks)}")
        for chunk in chunks:
            print(f"    [{chunk.section}] {chunk.char_count} chars")
        print()

        all_chunks.extend([c.to_dict() for c in chunks])

    # Save all chunks to a JSON file for inspection
    output_file = PROCESSED_DIR / "chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Total chunks across all documents: {len(all_chunks)}")
    print(f"Saved to: {output_file}")
    print("\n--- SPOT CHECK (first 3 chunks) ---")
    for chunk in all_chunks[:3]:
        print(f"\nID: {chunk['chunk_id']}")
        print(f"Section: {chunk['section']}")
        print(f"Text preview: {chunk['text'][:200]}...")


if __name__ == "__main__":
    run()
