"""
Tests for the OCR extraction pipeline.

What a senior interviewer asks: "How do you know your extraction is correct?"
Answer: "I have tests that verify extraction quality on known documents."

What a PhD committee asks: "How did you validate the preprocessing step?"
Answer: same — documented, reproducible verification with known ground truth.
"""

import pytest
from pathlib import Path
from src.ocr.extractor import extract, is_likely_scanned, ExtractedDocument


SAMPLE_PDF = Path("data/raw/report_001_lymphoma_staging.pdf")


@pytest.fixture(scope="module")
def extracted_doc():
    if not SAMPLE_PDF.exists():
        pytest.skip("Synthetic reports not generated yet. Run: python scripts/generate_reports.py")
    return extract(SAMPLE_PDF)


def test_extraction_returns_document(extracted_doc):
    assert isinstance(extracted_doc, ExtractedDocument)


def test_extraction_has_pages(extracted_doc):
    assert len(extracted_doc.pages) > 0


def test_extraction_has_text(extracted_doc):
    assert extracted_doc.total_chars > 200, (
        f"Expected >200 chars, got {extracted_doc.total_chars}. "
        "Extraction may have failed."
    )


def test_not_flagged_as_scanned(extracted_doc):
    assert not is_likely_scanned(extracted_doc), (
        "Text-based synthetic PDF was incorrectly flagged as scanned. "
        "Check extraction or threshold."
    )


def test_known_content_present(extracted_doc):
    """Ground truth check: our report_001 must contain these strings."""
    required = ["DIAGNOSIS", "Hodgkin Lymphoma", "LABORATORY RESULTS", "WBC"]
    for term in required:
        assert term in extracted_doc.full_text, (
            f"Expected '{term}' in extracted text but didn't find it. "
            "OCR/extraction may be corrupting content."
        )


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        extract(Path("data/raw/nonexistent.pdf"))
