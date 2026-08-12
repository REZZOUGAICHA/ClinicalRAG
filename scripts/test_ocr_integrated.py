"""
Measures the FULL integrated pipeline's accuracy — extract_scanned_document()
exactly as the real app calls it (docTR detection -> page-level confidence
check -> crop -> fine-tuned TrOCR recognition for low-confidence pages) —
on the same 40-sample RxHandBD test set (seed=42) used by every other OCR
test script in this project, for a direct, fair comparison against all
prior baselines:
  docTR alone:                        0.0% exact,  4.0% char similarity
  TrOCR pretrained (raw image):      12.5% exact, 50.1% char similarity
  TrOCR fine-tuned (raw image):      47.5% exact, 73.5% char similarity
  Integrated pipeline (this script):    ?    exact,    ?   char similarity

The "raw image" baselines fed TrOCR the whole clean original crop directly.
This script instead crops based on docTR's DETECTED bounding box, which may
not perfectly match the original word boundaries — this test is what tells
us whether that detection-then-crop step costs meaningful accuracy.

Usage: python scripts/test_ocr_integrated.py [n_samples]
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

    random.seed(42)  # same seed as every other OCR test script -> same sample
    sample = random.sample(rows, min(N_SAMPLES, len(rows)))

    exact_matches = 0
    char_similarities = []
    no_detection = 0

    print(f"Testing the full integrated pipeline against {len(sample)} real handwritten images...\n")

    for row in sample:
        image_path = TEST_DIR / row["Images"]
        ground_truth = row["Text"].strip()

        doc = extract_scanned_document(image_path)
        predicted = doc.full_text.strip()

        if not predicted:
            no_detection += 1

        is_exact = predicted.lower() == ground_truth.lower()
        exact_matches += is_exact

        similarity = Levenshtein.normalized_similarity(predicted.lower(), ground_truth.lower())
        char_similarities.append(similarity)

        status = "OK  " if is_exact else "MISS"
        print(f"  [{status}] truth={ground_truth!r:20s} predicted={predicted!r:20s} similarity={similarity:.2f}")

    n = len(sample)
    print(f"\n{'=' * 60}")
    print(f"Exact match accuracy:     {exact_matches}/{n} ({100 * exact_matches / n:.1f}%)")
    print(f"Avg character similarity: {sum(char_similarities) / n:.3f}")
    print(f"No detection at all:      {no_detection}/{n}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
