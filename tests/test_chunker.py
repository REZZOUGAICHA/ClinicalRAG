"""
Tests for the section-aware chunker.
"""

import pytest
from pathlib import Path
from src.ocr.extractor import extract
from src.chunking.chunker import chunk_document, Chunk


SAMPLE_PDF = Path("data/raw/report_001_lymphoma_staging.pdf")


@pytest.fixture(scope="module")
def chunks():
    if not SAMPLE_PDF.exists():
        pytest.skip("Synthetic reports not generated yet. Run: python scripts/generate_reports.py")
    doc = extract(SAMPLE_PDF)
    return chunk_document(doc)


def test_returns_list_of_chunks(chunks):
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)


def test_at_least_one_chunk(chunks):
    assert len(chunks) > 0


def test_expected_sections_present(chunks):
    section_names = {c.section for c in chunks}
    expected = {"DIAGNOSIS", "LABORATORY RESULTS", "MEDICATIONS", "TREATMENT PLAN"}
    for s in expected:
        assert s in section_names, (
            f"Expected section '{s}' not found. Found: {section_names}. "
            "Check that the section header regex matches the PDF content."
        )


def test_no_empty_chunks(chunks):
    for chunk in chunks:
        assert chunk.text.strip(), f"Chunk {chunk.chunk_id} has empty text."


def test_chunk_ids_are_unique(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk IDs found."


def test_diagnosis_chunk_contains_diagnosis(chunks):
    diag = next((c for c in chunks if c.section == "DIAGNOSIS"), None)
    assert diag is not None
    assert "Hodgkin" in diag.text, (
        "DIAGNOSIS chunk doesn't mention Hodgkin — chunker may be misaligning sections."
    )
