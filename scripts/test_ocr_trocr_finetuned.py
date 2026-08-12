"""
Measures the RxHandBD-fine-tuned TrOCR model's accuracy — same 40-sample
test set (seed=42) as scripts/test_ocr_trocr.py's pretrained-model test,
for a direct, fair three-way comparison: docTR -> pretrained TrOCR ->
fine-tuned TrOCR.

Usage: python scripts/test_ocr_trocr_finetuned.py [n_samples]
"""

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from PIL import Image
from rapidfuzz.distance import Levenshtein
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

TEST_DIR = Path("data/ocr_test/Test_Set")
LABELS_PATH = Path("data/ocr_test/Test_Labels.csv")
N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 40

MODEL_PATH = "models/trocr-finetuned-rxhandbd"


def run():
    with open(LABELS_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    random.seed(42)  # same seed as test_ocr_trocr.py -> same sample
    sample = random.sample(rows, min(N_SAMPLES, len(rows)))

    print(f"Loading fine-tuned model from {MODEL_PATH}...")
    processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)

    exact_matches = 0
    char_similarities = []

    print(f"\nTesting fine-tuned TrOCR against {len(sample)} real handwritten prescription images...\n")

    for row in sample:
        image_path = TEST_DIR / row["Images"]
        ground_truth = row["Text"].strip()

        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        predicted = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

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
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
