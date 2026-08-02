"""
Measures docTR's real OCR accuracy against RxHandBD — a public dataset of
genuine handwritten medical prescription word images (Zenodo, MIT license).

This is the first time this project's OCR path has ever been exercised
against real images — everything before this was PyMuPDF reading text that
was already embedded in a PDF, never actual optical character recognition.

Usage: python scripts/test_ocr_accuracy.py [n_samples]
"""

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from rapidfuzz.distance import Levenshtein

from src.ocr.extractor import extract_scanned_document

TEST_DIR = Path("data/ocr_test/Test_Set")
LABELS_PATH = Path("data/ocr_test/Test_Labels.csv")
N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def run():
    with open(LABELS_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    random.seed(42)  # reproducible sample
    sample = random.sample(rows, min(N_SAMPLES, len(rows)))

    exact_matches = 0
    char_similarities = []

    print(f"Testing docTR against {len(sample)} real handwritten prescription images...\n")

    for row in sample:
        image_path = TEST_DIR / row["Images"]
        ground_truth = row["Text"].strip()

        doc = extract_scanned_document(image_path)
        predicted = doc.full_text.strip()

        is_exact = predicted.lower() == ground_truth.lower()
        exact_matches += is_exact

        # Normalized similarity: 1.0 = identical, 0.0 = completely different
        similarity = Levenshtein.normalized_similarity(predicted.lower(), ground_truth.lower())
        char_similarities.append(similarity)

        status = "OK  " if is_exact else "MISS"
        print(f"  [{status}] truth={ground_truth!r:20s} predicted={predicted!r:20s} similarity={similarity:.2f}")

    n = len(sample)
    print(f"\n{'=' * 60}")
    print(f"Exact match accuracy:     {exact_matches}/{n} ({100 * exact_matches / n:.1f}%)")
    print(f"Avg character similarity: {sum(char_similarities) / n:.3f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
