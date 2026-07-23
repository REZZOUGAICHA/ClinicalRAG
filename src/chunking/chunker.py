"""
Section-aware clinical document chunker.

Why NOT fixed-size chunking:
  Fixed-size chunking (e.g. every 512 tokens) is the naive default and breaks
  clinical reports badly. A report has a "DIAGNOSIS" section and a "MEDICATIONS"
  section — those are semantically distinct units. If we split mid-section,
  a question like "what is the diagnosis?" might retrieve a chunk that starts
  mid-diagnosis and begins with drug names. The retriever then has the wrong context.

  Section-aware chunking preserves the clinical meaning of each section as a unit.
  Each chunk also carries its section label as metadata — so when we cite sources
  in Week 2, we can say "from the LABORATORY RESULTS section of report_001" instead
  of just a page number.

PhD angle: Biomedical RAG papers (MedRAG, ClinicalBERT) consistently find that
preserving document structure improves retrieval precision. This is a deliberate
design choice we can articulate and cite.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.ocr.extractor import ExtractedDocument


# Section headers we expect in our clinical reports.
# This list is the "schema" we impose on the document.
# In production you'd learn this from a larger corpus or use an NLP classifier.
KNOWN_SECTIONS = [
    "PATIENT INFORMATION",
    "CHIEF COMPLAINT",
    "DIAGNOSIS",
    "LABORATORY RESULTS",
    "IMAGING",
    "MEDICATIONS",
    "TREATMENT PLAN",
    "FOLLOW-UP",
    # Catch-all for sections we haven't seen before
    "NOTES",
    "SUMMARY",
    "HISTORY",
    "PHYSICAL EXAMINATION",
    "ASSESSMENT",
    "PLAN",
    "PROCEDURE",
    "DISCHARGE INSTRUCTIONS",
]

# Regex: a line that IS or CONTAINS one of our known section names (uppercase, standalone)
_SECTION_PATTERN = re.compile(
    r"^(" + "|".join(re.escape(s) for s in KNOWN_SECTIONS) + r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class Chunk:
    chunk_id: str           # e.g. "report_001_lymphoma_staging__DIAGNOSIS__0"
    source_file: str        # filename of the source PDF
    section: str            # section label (e.g. "DIAGNOSIS")
    text: str               # the actual chunk text
    page_number: int        # 1-indexed page this chunk's content starts on
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "section": self.section,
            "text": self.text,
            "page_number": self.page_number,
            "char_count": self.char_count,
        }


def _offset_to_page(pages: list, offset: int) -> int:
    """
    Map a character offset in doc.full_text back to the 1-indexed page it
    falls on. full_text is built as "\\n\\n".join(page.raw_text for page in
    pages), so each page after the first is preceded by 2 extra characters.
    """
    cursor = 0
    for page in pages:
        cursor += len(page.raw_text)
        if offset <= cursor:
            return page.page_number
        cursor += 2  # the "\n\n" joiner
    return pages[-1].page_number if pages else 1


def chunk_document(doc: ExtractedDocument) -> list[Chunk]:
    """
    Split a document's full text into section-aware chunks.

    Returns one Chunk per detected section. If no section headers are found
    (e.g. unstructured note), falls back to a single chunk with section="FULL_DOCUMENT".
    """
    source_name = Path(doc.source_path).stem
    text = doc.full_text

    # Find all section headers and their positions in the text
    matches = list(_SECTION_PATTERN.finditer(text))

    if not matches:
        # Fallback: no structure detected — treat whole document as one chunk
        return [Chunk(
            chunk_id=f"{source_name}__FULL_DOCUMENT__0",
            source_file=Path(doc.source_path).name,
            section="FULL_DOCUMENT",
            text=text.strip(),
            page_number=doc.pages[0].page_number if doc.pages else 1,
        )]

    chunks = []
    for i, match in enumerate(matches):
        section_name = match.group(1).upper().strip()
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()

        if not content:
            continue

        chunk_id = f"{source_name}__{section_name.replace(' ', '_')}__{i}"
        chunks.append(Chunk(
            chunk_id=chunk_id,
            source_file=Path(doc.source_path).name,
            section=section_name,
            text=content,
            page_number=_offset_to_page(doc.pages, content_start),
        ))

    return chunks


def chunk_documents(docs: list[ExtractedDocument]) -> list[Chunk]:
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
