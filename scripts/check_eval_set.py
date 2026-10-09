"""
Sanity checks for eval/questions.json. Run after editing the eval set:

    python scripts/check_eval_set.py

Fails (exit 1) if an id repeats, a gold chunk doesn't exist in chunks.json,
a key fact can't be found in its own gold text, or an unanswerable item
carries gold/key facts. The point is that every number the eval scripts
produce can be traced back to text that is actually in the corpus.
"""

import json
import sys
from collections import Counter, defaultdict

from eval_common import CHUNKS_PATH, QUESTIONS_PATH, VALID_TYPES, contains_fact

REQUIRED = ("id", "question", "type", "answerable", "reference", "gold", "key_facts")


def check(items: list[dict], chunks: list[dict]) -> list[str]:
    # a (file, section) pair can map to several chunks (mts_010 has two
    # PROCEDURE chunks), so gold text is the concatenation of all of them
    text_by_key = defaultdict(list)
    for c in chunks:
        text_by_key[(c["source_file"], c["section"])].append(c["text"])

    errors = []
    ids = Counter(q.get("id") for q in items)
    errors += [f"duplicate id: {i}" for i, n in ids.items() if n > 1]

    for q in items:
        qid = q.get("id", "<no id>")
        missing = [k for k in REQUIRED if k not in q]
        if missing:
            errors.append(f"{qid}: missing fields {missing}")
            continue
        if q["type"] not in VALID_TYPES:
            errors.append(f"{qid}: invalid type {q['type']!r}")
        if not q["question"].strip():
            errors.append(f"{qid}: empty question")
        if "twin" in q and q["twin"] not in ids:
            errors.append(f"{qid}: twin {q['twin']!r} does not exist")

        if not q["answerable"]:
            if q["type"] != "unanswerable":
                errors.append(f"{qid}: answerable=false but type is {q['type']}")
            if q["gold"] or q["key_facts"]:
                errors.append(f"{qid}: unanswerable item must have empty gold and key_facts")
            continue

        if q["type"] == "unanswerable":
            errors.append(f"{qid}: type unanswerable but answerable=true")
        if not q["gold"]:
            errors.append(f"{qid}: answerable item has no gold chunks")
        if not q["key_facts"]:
            errors.append(f"{qid}: answerable item has no key_facts")

        gold_text = []
        for g in q["gold"]:
            key = (g["source_file"], g["section"])
            if key not in text_by_key:
                errors.append(f"{qid}: gold {key} not in chunks.json")
            gold_text += text_by_key.get(key, [])
        if len({(g["source_file"], g["section"]) for g in q["gold"]}) != len(q["gold"]):
            errors.append(f"{qid}: duplicate gold entries")

        joined = "\n".join(gold_text)
        for fact in q["key_facts"]:
            if not contains_fact(joined, fact):
                errors.append(f"{qid}: key fact {fact!r} not found in gold text")

    return errors


def summarize(items: list[dict]) -> None:
    by_type = Counter(q["type"] for q in items)
    print(f"{len(items)} items")
    for t in sorted(by_type):
        print(f"  {t:<14} {by_type[t]}")

    single = [q for q in items if q["type"] != "multi_hop"]
    docs = {g["source_file"] for q in single for g in q["gold"]}
    reports = sorted(d for d in docs if d.startswith("report_"))
    mts = sorted(d for d in docs if d.startswith("mts_"))
    all_docs = {g["source_file"] for q in items for g in q["gold"]}
    print(f"single-question items cover {len(reports)} synthetic reports, {len(mts)} MTSamples docs")
    print(f"whole set covers {len(all_docs)} distinct documents")
    twins = sum("twin" in q for q in items)
    print(f"typo/paraphrase items with a clean twin: {twins}")


def main() -> int:
    items = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    errors = check(items, chunks)
    summarize(items)
    if errors:
        print(f"\nFAILED with {len(errors)} problem(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
