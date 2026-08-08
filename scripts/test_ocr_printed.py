"""
Measures docTR's OCR accuracy against FUNSD — a public dataset of real
scanned PRINTED business forms (not handwritten). This is the complementary
test to test_ocr_accuracy.py (which tested real handwritten prescriptions
and found recognition performs very poorly there). Printed/typed scanned
text — closer to a real discharge letter, fax, or typed intake form — is
the more common case in an actual clinical setting, and OCR models
generally do far better on it than on handwriting.

Ground truth here is per-word bounding boxes, not a single ordered string,
so accuracy is measured as word-level recall: what fraction of the words
FUNSD says are actually on the page did docTR's OCR output contain,
regardless of exact reading order.

Usage: python scripts/test_ocr_printed.py [n_samples]
"""

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.ocr.extractor import extract_scanned_document

IMAGES_DIR = Path("data/ocr_test_printed/dataset/testing_data/images")
ANNOTATIONS_DIR = Path("data/ocr_test_printed/dataset/testing_data/annotations")
N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def _words(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z0-9]+", text) if len(w) > 1}


def run():
    image_paths = sorted(IMAGES_DIR.glob("*.png"))
    random.seed(42)
    sample = random.sample(image_paths, min(N_SAMPLES, len(image_paths)))

    recalls = []
    print(f"Testing docTR against {len(sample)} real scanned printed forms (FUNSD)...\n")

    for image_path in sample:
        annotation_path = ANNOTATIONS_DIR / f"{image_path.stem}.json"
        ground_truth_text = " ".join(
            entry["text"] for entry in json.loads(annotation_path.read_text(encoding="utf-8"))["form"]
        )
        gt_words = _words(ground_truth_text)

        doc = extract_scanned_document(image_path)
        predicted_words = _words(doc.full_text)

        found = gt_words & predicted_words
        recall = len(found) / len(gt_words) if gt_words else 0.0
        recalls.append(recall)

        print(f"  {image_path.name}: {len(found)}/{len(gt_words)} ground-truth words found ({recall:.1%})")

    print(f"\n{'=' * 60}")
    print(f"Average word-level recall: {sum(recalls) / len(recalls):.1%}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
