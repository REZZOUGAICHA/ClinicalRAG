"""
PDF text extraction pipeline.

Architecture decision:
  - Text-based PDFs (e.g. our synthetic reports, most electronic clinical docs):
    use PyMuPDF (fitz) — fast, no ML model required, perfect accuracy on digital text.
  - Scanned / image PDFs (e.g. paper forms photographed and uploaded):
    use docTR (deep learning OCR) — handles degraded scan quality, mixed layouts.

Why two paths? In a real clinical setting you get both. A senior interviewer will ask
"what happens when someone uploads a scanned handwritten referral?" — this design
answers that question. The docTR path is wired in Week 2 when we add the heavy ML deps.

PhD angle: this mirrors the multi-modal ingestion pipeline described in MedRAG and
BioRAG papers — they face the exact same text vs. scan dichotomy.
"""

import fitz  # PyMuPDF
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ExtractedPage:
    page_number: int       # 1-indexed
    raw_text: str          # full page text as extracted
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.raw_text)


@dataclass
class ExtractedDocument:
    source_path: str
    pages: list[ExtractedPage]
    full_text: str = field(init=False)
    total_chars: int = field(init=False)

    def __post_init__(self):
        self.full_text = "\n\n".join(p.raw_text for p in self.pages)
        self.total_chars = len(self.full_text)


def extract_text_pdf(pdf_path: Path) -> ExtractedDocument:
    """
    Extract text from a text-based PDF using PyMuPDF.
    Fast and lossless for digitally generated PDFs.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []
    with fitz.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            pages.append(ExtractedPage(page_number=i + 1, raw_text=text))

    return ExtractedDocument(source_path=str(pdf_path), pages=pages)


def is_likely_scanned(doc: ExtractedDocument, min_chars_per_page: int = 100) -> bool:
    """
    Heuristic: if extracted text is very sparse, the PDF is probably a scan
    and needs the OCR fallback path.

    min_chars_per_page: if average chars/page falls below this, flag as likely scanned.
    A real clinical report page has hundreds of characters; a blank OCR result has near zero.
    """
    if not doc.pages:
        return True
    avg = doc.total_chars / len(doc.pages)
    return avg < min_chars_per_page


def extract(pdf_path: Path) -> ExtractedDocument:
    """
    Main entry point. Tries text extraction first; warns if the document
    looks scanned (docTR fallback will be wired in Week 2).
    """
    doc = extract_text_pdf(pdf_path)

    if is_likely_scanned(doc):
        print(
            f"  [WARN] {Path(pdf_path).name} looks like a scanned PDF "
            f"(avg {doc.total_chars / max(len(doc.pages), 1):.0f} chars/page). "
            "docTR OCR fallback will be added in Phase 2."
        )

    return doc
