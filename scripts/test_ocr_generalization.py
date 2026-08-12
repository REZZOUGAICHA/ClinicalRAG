"""
Generalization check for the RxHandBD fine-tuned TrOCR model: does it still
read GENERAL English handwriting, or did fine-tuning narrowly on medical
vocabulary hurt broader capability ("catastrophic forgetting" toward a
narrow domain)?

Tests both the pretrained (microsoft/trocr-base-handwritten) and the
RxHandBD-fine-tuned model on the SAME sample from IAM_words_text_recognition
(real handwritten general-English words, a completely different dataset
from RxHandBD — different writers, different vocabulary, different source)
for a direct before/after comparison.

Usage: python scripts/test_ocr_generalization.py [n_samples]
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from datasets import load_dataset
from rapidfuzz.distance import Levenshtein
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 40

MODELS = {
    "pretrained": "microsoft/trocr-base-handwritten",
    "fine-tuned (RxHandBD)": "models/trocr-finetuned-rxhandbd",
}


def evaluate(model_path: str, samples: list) -> tuple[float, float]:
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)

    exact_matches = 0
    similarities = []

    for sample in samples:
        image = sample["image"].convert("RGB")
        ground_truth = sample["text"].strip()

        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        predicted = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        exact_matches += predicted.lower() == ground_truth.lower()
        similarities.append(Levenshtein.normalized_similarity(predicted.lower(), ground_truth.lower()))

    n = len(samples)
    return exact_matches / n, sum(similarities) / n


def run():
    print("Loading IAM word-level test set (general English handwriting, NOT medical)...")
    ds = load_dataset("priyank-m/IAM_words_text_recognition", split="test")

    random.seed(42)
    indices = random.sample(range(len(ds)), min(N_SAMPLES, len(ds)))
    samples = [ds[i] for i in indices]
    # Skip empty/whitespace-only ground truths (a few exist in IAM's punctuation-only entries)
    # and degenerate near-zero-size images (a few exist in IAM, e.g. 1x1px crops)
    # that break the image processor's channel-dimension detection.
    samples = [s for s in samples if s["text"].strip() and min(s["image"].size) > 5]

    print(f"Testing {len(samples)} real general-handwriting word images.\n")

    results = {}
    for label, path in MODELS.items():
        print(f"--- {label} ({path}) ---")
        exact, sim = evaluate(path, samples)
        results[label] = (exact, sim)
        print(f"  Exact match:        {exact*100:.1f}%")
        print(f"  Char similarity:    {sim*100:.1f}%\n")

    print("=" * 60)
    print("SUMMARY — general handwriting (IAM), same samples for both models")
    for label, (exact, sim) in results.items():
        print(f"  {label:28s} exact={exact*100:5.1f}%  char_sim={sim*100:5.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    run()
