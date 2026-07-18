"""
CLI to test the full RAG pipeline end-to-end.

Usage: python scripts/ask.py "what was the WBC count in the lymphoma case?"
   or: python scripts/ask.py   (runs a set of demo questions)

This is your manual verification tool — the human-in-the-loop check
that confirms the pipeline produces correct, cited answers before
you build the frontend on top of it.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.pipeline import ask


DEMO_QUESTIONS = [
    "What was the WBC count in the lymphoma patient?",
    "What medications was the stroke patient discharged with?",
    "What is the HbA1c of the diabetic patient and is it within target?",
    "Which patient was admitted to the ICU and why?",
    "What treatment plan was given for the thyroid cancer patient?",
]


def print_result(result: dict):
    print("\n" + "=" * 60)
    print(f"QUESTION: {result['query']}")
    print("-" * 60)
    print(f"ANSWER:\n{result['answer']}")
    print("-" * 60)
    print("RETRIEVED CHUNKS:")
    for i, chunk in enumerate(result["chunks"], 1):
        score = chunk.get("reranker_score", chunk.get("score", "?"))
        print(f"  [{i}] {chunk['source_file']} | {chunk['section']} | score: {score:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        result = ask(query)
        print_result(result)
    else:
        print("Running demo questions...\n")
        for question in DEMO_QUESTIONS:
            result = ask(question)
            print_result(result)
