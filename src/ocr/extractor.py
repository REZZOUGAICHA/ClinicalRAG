"""
PDF / image text extraction pipeline.

Architecture decision:
  - Text-based PDFs (e.g. our synthetic reports, most electronic clinical docs):
    use PyMuPDF (fitz) — fast, no ML model required, perfect accuracy on digital text.
  - Scanned / image PDFs and plain images (e.g. a paper form photographed and
    uploaded): use docTR (deep learning OCR) — a two-stage pipeline (text
    detection, then recognition) that handles degraded scan quality, skew,
    and mixed layouts.

Why two paths? In a real clinical setting you get both. A senior interviewer will ask
"what happens when someone uploads a scanned handwritten referral?" — this design
answers that question.

PhD angle: this mirrors the multi-modal ingestion pipeline described in MedRAG and
BioRAG papers — they face the exact same text vs. scan dichotomy.
"""

import fitz  # PyMuPDF
from pathlib import Path
from dataclasses import dataclass, field

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

_ocr_model = None

# Fine-tuned handwriting recognizer (see notebooks/finetune_trocr_rxhandbd.ipynb).
# Not committed to the repo (1.3GB, exceeds GitHub's file-size limit) — if it's
# not present (e.g. a deploy environment that didn't bundle it), the pipeline
# just skips this enhancement and falls back to docTR's own recognition, which
# still works, just worse on genuine handwriting. See get_trocr_model().
TROCR_MODEL_PATH = Path("models/trocr-finetuned-rxhandbd")
_trocr_processor = None
_trocr_model = None

# docTR's own recognition reports a per-word confidence. Below this, treat the
# word as likely wrong and worth re-reading with the handwriting specialist.
# This is a heuristic default, not a rigorously tuned number — printed text
# recognized correctly tends to score 0.9+; genuinely garbled/illegible
# handwriting scores much lower, so 0.5 is a reasonable dividing line.
LOW_CONFIDENCE_THRESHOLD = 0.5


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


def get_ocr_model():
    """
    Lazy-load docTR's OCR pipeline — same singleton pattern used for the
    embedding model and reranker elsewhere in this project. The model is
    two stages: text DETECTION (find where the text is on the page) then
    RECOGNITION (read what it says) — loading both takes a few seconds,
    worth paying once per process rather than per document.

    detect_orientation / straighten_pages: a phone-photographed document is
    rarely perfectly level the way a flatbed scan is — this compensates.
    """
    global _ocr_model
    if _ocr_model is None:
        from doctr.models import ocr_predictor
        print("  Loading docTR OCR model (detection + recognition)...")
        _ocr_model = ocr_predictor(
            pretrained=True,
            detect_orientation=True,
            straighten_pages=True,
        )
    return _ocr_model


def get_trocr_model():
    """
    Lazy-load the RxHandBD-fine-tuned TrOCR handwriting recognizer, if it's
    present on disk. Returns (None, None) if it isn't, so callers can degrade
    gracefully instead of crashing — this model is a large, optional artifact,
    not a hard dependency of the OCR pipeline.

    IMPORTANT CAVEAT (measured, not assumed — see scripts/test_ocr_generalization.py):
    fine-tuning narrowly on RxHandBD's medical vocabulary made this model
    dramatically better at reading medical handwriting (12.5% -> 47.5% exact
    match) but dramatically WORSE at general handwriting (53.8% -> 7.7%
    exact match) — a textbook case of catastrophic forgetting toward a
    narrow domain. That's exactly why this is used ONLY as a targeted re-read
    of specific low-confidence words (see _apply_handwriting_recognition),
    never as a blanket replacement for docTR's own, more general recognition.
    """
    global _trocr_processor, _trocr_model
    if _trocr_model is None:
        if not TROCR_MODEL_PATH.exists():
            return None, None
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        print(f"  Loading fine-tuned handwriting model from {TROCR_MODEL_PATH}...")
        _trocr_processor = TrOCRProcessor.from_pretrained(str(TROCR_MODEL_PATH))
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(str(TROCR_MODEL_PATH))
    return _trocr_processor, _trocr_model


def _crop_word(image, geometry, page_height: int, page_width: int):
    """
    docTR reports each word's location as relative coordinates (0-1 range):
    ((x0, y0), (x1, y1)). Convert to absolute pixels and crop. Note
    page.dimensions is (height, width), the opposite order from PIL's
    (width, height) — verified directly, easy to get backwards.
    """
    from PIL import Image

    (x0, y0), (x1, y1) = geometry
    left, right = int(x0 * page_width), int(x1 * page_width)
    top, bottom = int(y0 * page_height), int(y1 * page_height)
    return Image.fromarray(image).crop((left, top, right, bottom))


def _apply_handwriting_recognition(images: list, result) -> None:
    """
    For every word docTR itself scored below LOW_CONFIDENCE_THRESHOLD, crop
    that region and re-read it with the fine-tuned handwriting model, then
    mutate the word in place. Mutating and re-rendering (rather than
    reconstructing page text by hand) reuses docTR's own layout/reading-order
    logic in Page.render() — verified directly that this mutation approach
    works (Word objects are plain mutable Python objects, not frozen).

    No-op (fast return) if the fine-tuned model isn't available.
    """
    processor, model = get_trocr_model()
    if model is None:
        return

    for image, page in zip(images, result.pages):
        height, width = page.dimensions
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    if word.confidence >= LOW_CONFIDENCE_THRESHOLD:
                        continue
                    crop = _crop_word(image, word.geometry, height, width)
                    pixel_values = processor(images=crop, return_tensors="pt").pixel_values
                    generated_ids = model.generate(pixel_values)
                    reread = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    if reread:
                        word.value = reread


_MIN_OCR_DIMENSION = 512  # px


def _pad_small_images(images: list) -> list:
    """
    docTR's detection model is trained on document-scale images with real
    margins around the text — fed a tightly-cropped small image (e.g. a
    single word crop, 128x128px), it finds ZERO text regions and returns
    nothing, even though the text itself is perfectly legible. Verified
    directly: detection blocks went from 0 to 1 just by pasting a 128x128
    crop onto a 512x512 white canvas — nothing else changed. This isn't a
    workaround for one dataset; any small/tightly-cropped input would hit
    the same failure, so every image gets this safety margin.
    """
    import numpy as np
    from PIL import Image

    padded = []
    for arr in images:
        h, w = arr.shape[:2]
        if h >= _MIN_OCR_DIMENSION and w >= _MIN_OCR_DIMENSION:
            padded.append(arr)
            continue

        size = max(_MIN_OCR_DIMENSION, h, w) + 2 * min(h, w)
        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        canvas.paste(Image.fromarray(arr).convert("RGB"), ((size - w) // 2, (size - h) // 2))
        padded.append(np.array(canvas))
    return padded


def extract_scanned_document(path: Path) -> ExtractedDocument:
    """
    Extract text from a scanned/image document — a plain image file (JPG,
    PNG, ...) or a PDF whose pages are images with no usable text layer —
    using docTR.

    WHY NOT JUST PyMuPDF FOR EVERYTHING?
    PyMuPDF reads text objects that are already embedded in a PDF; a scanned
    document has no such objects, just pixels. docTR is a real OCR model —
    it looks at the pixels and reads what's actually printed/written there,
    the same way a human would.
    """
    from doctr.io import DocumentFile

    path = Path(path)
    if path.suffix.lower() == ".pdf":
        images = DocumentFile.from_pdf(str(path))
    else:
        images = DocumentFile.from_images(str(path))

    images = _pad_small_images(images)

    model = get_ocr_model()
    result = model(images)

    pages = [
        ExtractedPage(page_number=i + 1, raw_text=page.render())
        for i, page in enumerate(result.pages)
    ]
    return ExtractedDocument(source_path=str(path), pages=pages)


def extract(path: Path) -> ExtractedDocument:
    """
    Main entry point. Routes to the right extraction path:
    - A plain image file always goes straight to OCR (docTR) — there's no
      text layer to even attempt reading with PyMuPDF.
    - A PDF tries PyMuPDF's text-layer extraction first (fast, exact); if
      the result looks too sparse to be real (is_likely_scanned), it's
      almost certainly an image-only PDF, so it falls back to docTR too.
    """
    path = Path(path)

    if path.suffix.lower() in IMAGE_EXTENSIONS:
        print(f"  {path.name} is an image file — extracting with docTR OCR...")
        return extract_scanned_document(path)

    doc = extract_text_pdf(path)

    if is_likely_scanned(doc):
        print(
            f"  {path.name} looks like a scanned PDF "
            f"(avg {doc.total_chars / max(len(doc.pages), 1):.0f} chars/page). "
            "Falling back to docTR OCR..."
        )
        return extract_scanned_document(path)

    return doc
