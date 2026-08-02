"""
Selects a diverse subset of real clinical transcriptions from the MTSamples
dataset and writes each one as a PDF into data/raw/, alongside the synthetic
reports, so they flow through the exact same pipeline (extract -> chunk ->
embed -> store) and get the same PDF-viewer/citation-highlight treatment.

DATA SOURCE:
MTSamples (mtsamples.com) — de-identified, publicly available medical
transcription samples, widely used in NLP/clinical-research contexts. This
script expects the Kaggle mirror CSV at data/mtsamples_raw/mtsamples.csv
(columns: description, medical_specialty, sample_name, transcription,
keywords). Not committed to source control — download it yourself.

WHY REAL DATA MATTERS HERE:
The synthetic reports (Week 1) are clean and LLM-generated, with predictable
section headers. Real transcriptions are messier: inconsistent formatting,
abbreviations, dictation artifacts. Testing retrieval/chunking against real
data is what actually validates the system beyond its own synthetic test set.

WHY THESE ARE KEPT VERBATIM:
The transcription text is written to PDF as-is, not "cleaned up" — an
authentic representation of real clinical documentation is the point.

Usage: python scripts/prepare_mtsamples.py
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import fitz  # PyMuPDF

CSV_PATH = Path("data/mtsamples_raw/mtsamples.csv")
OUT_DIR = Path("data/raw")
MANIFEST_PATH = Path("data/mtsamples_raw/manifest.csv")

# Length window: long enough to be a real multi-section note, short enough
# to keep PDFs and downstream chunk counts reasonable for a portfolio demo.
MIN_LEN, MAX_LEN = 800, 4000

# "Umbrella" categories that aren't really their own clinical specialty
# (e.g. "Consult - History and Phy." spans every specialty) — skip them in
# favor of genuine specialty diversity.
EXCLUDE_SPECIALTIES = {
    "Consult - History and Phy.", "SOAP / Chart / Progress Notes",
    "Discharge Summary", "Office Notes", "Letters",
    "IME-QME-Work Comp etc.", " Consult - History and Phy.",
}

RECORDS_PER_SPECIALTY = 3
NUM_SPECIALTIES = 9  # 3 x 9 = 27 documents


def _slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    return re.sub(r"-+", "-", text).strip("-")[:50]


def select_records() -> list[dict]:
    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_specialty = defaultdict(list)
    for r in rows:
        specialty = (r["medical_specialty"] or "").strip()
        transcription = (r["transcription"] or "").strip()
        if specialty in EXCLUDE_SPECIALTIES or not transcription:
            continue
        if MIN_LEN <= len(transcription) <= MAX_LEN:
            by_specialty[specialty].append(r)

    top_specialties = sorted(by_specialty, key=lambda s: -len(by_specialty[s]))[:NUM_SPECIALTIES]

    selected = []
    for specialty in top_specialties:
        # Prefer distinct sample_names (MTSamples has near-duplicate
        # "X - 1", "X - 2" template variants) for genuine content diversity.
        seen_names = set()
        for r in by_specialty[specialty]:
            base_name = re.sub(r"\s*-\s*\d+\s*$", "", r["sample_name"].strip())
            if base_name in seen_names:
                continue
            seen_names.add(base_name)
            selected.append(r)
            if len(seen_names) >= RECORDS_PER_SPECIALTY:
                break

    return selected


PAGE_RECT = fitz.Rect(50, 50, 545, 792)
FONT_SIZE = 10


def _fits(text: str) -> bool:
    """insert_textbox on a throwaway page: >=0 return means it all fit."""
    probe = fitz.open()
    fits = probe.new_page().insert_textbox(PAGE_RECT, text, fontsize=FONT_SIZE) >= 0
    probe.close()
    return fits


def _paginate(text: str) -> list[str]:
    """
    Split text into page-sized pieces. insert_textbox() only reports whether
    text fit (return value's sign) — it doesn't hand back what didn't fit —
    so pagination has to be figured out ourselves: binary-search the largest
    word-aligned prefix that fits, place it, repeat with the remainder.
    """
    pages = []
    remaining = text
    while remaining:
        if _fits(remaining):
            pages.append(remaining)
            break

        words = remaining.split(" ")
        lo, hi = 1, len(words)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _fits(" ".join(words[:mid])):
                lo = mid
            else:
                hi = mid - 1

        pages.append(" ".join(words[:lo]))
        remaining = " ".join(words[lo:])
    return pages


def write_pdf(record: dict, out_path: Path) -> None:
    specialty = record["medical_specialty"].strip()
    sample_name = record["sample_name"].strip()
    transcription = record["transcription"].strip()

    header = f"{sample_name}\nSpecialty: {specialty}\nSource: MTSamples (mtsamples.com)\n\n"
    body = header + transcription

    doc = fitz.open()
    for page_text in _paginate(body):
        page = doc.new_page()
        page.insert_textbox(PAGE_RECT, page_text, fontsize=FONT_SIZE, fontname="helv")

    doc.save(str(out_path))
    doc.close()


def run():
    if not CSV_PATH.exists():
        print(f"MTSamples CSV not found at {CSV_PATH}. Download the Kaggle mirror CSV there first.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = select_records()
    print(f"Selected {len(records)} records across {len(set(r['medical_specialty'].strip() for r in records))} specialties.\n")

    manifest_rows = []
    for i, record in enumerate(records, start=1):
        slug = _slugify(record["sample_name"])
        filename = f"mts_{i:03d}_{slug}.pdf"
        out_path = OUT_DIR / filename
        write_pdf(record, out_path)
        print(f"  [{record['medical_specialty'].strip()}] {filename}")
        manifest_rows.append({
            "filename": filename,
            "specialty": record["medical_specialty"].strip(),
            "sample_name": record["sample_name"].strip(),
            "description": record["description"].strip(),
        })

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "specialty", "sample_name", "description"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nWrote {len(records)} PDFs to {OUT_DIR}/")
    print(f"Manifest saved to {MANIFEST_PATH}")
    print("\nNext: python scripts/run_pipeline.py && python scripts/ingest.py")


if __name__ == "__main__":
    run()
