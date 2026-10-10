"""
Write the contextual-retrieval context for every chunk in chunks.json and
cache it in data/contexts/chunk_contexts.json (see src/retrieval/contextual.py).

    python scripts/build_contexts.py            # fill in whatever is missing
    python scripts/build_contexts.py --dry-run  # just report cache status

Resumable: documents whose chunks all have a fresh cache entry are skipped,
and the cache is saved after every document, so an interrupted run (or one
stopped by Groq's daily limit) picks up where it left off.

Uses openai/gpt-oss-20b, one call per document, paced to stay under the free
tier's 8000 tokens/minute for that model.
"""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from eval_common import CHUNKS_PATH, call_with_groq_retry  # noqa: E402

from src.config import settings  # noqa: E402
from src.retrieval.contextual import (  # noqa: E402
    CONTEXT_MODEL,
    build_prompt,
    cache_entry,
    cached_context,
    generate_document_contexts,
    load_contexts,
    save_contexts,
)

TOKENS_PER_MINUTE = 7000  # the limit is 8000; leave room for estimation error


class TokenPacer:
    """Sliding one-minute window of tokens actually used."""

    def __init__(self, budget: int):
        self.budget = budget
        self.window: deque[tuple[float, int]] = deque()

    def used(self) -> int:
        cutoff = time.monotonic() - 60
        while self.window and self.window[0][0] < cutoff:
            self.window.popleft()
        return sum(t for _, t in self.window)

    def wait_for(self, estimate: int) -> None:
        while self.window and self.used() + estimate > self.budget:
            time.sleep(max(0.5, self.window[0][0] + 60 - time.monotonic()))

    def record(self, tokens: int) -> None:
        self.window.append((time.monotonic(), tokens))


def estimate_tokens(prompt: str, n_chunks: int) -> int:
    # ~3.5 chars per token for this English text, plus output and reasoning
    return int(len(prompt) / 3.5) + 100 * n_chunks + 300


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, help="process at most this many documents")
    args = parser.parse_args()

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    contexts = load_contexts()

    docs: dict[str, list[dict]] = {}
    for c in chunks:
        docs.setdefault(c["source_file"], []).append(c)

    todo = [f for f, cs in docs.items() if any(cached_context(c, contexts) is None for c in cs)]
    stale = sum(1 for c in chunks if c["chunk_id"] in contexts and cached_context(c, contexts) is None)
    print(f"{len(chunks)} chunks in {len(docs)} documents; {len(todo)} documents need contexts "
          f"({stale} stale entries); cache: {settings.contexts_path}")
    if args.dry_run or not todo:
        return

    pacer = TokenPacer(TOKENS_PER_MINUTE)
    total_tokens = 0
    for n, source_file in enumerate(todo[: args.limit] if args.limit else todo, 1):
        # Regenerate the whole document, not just the missing chunks: the
        # model writes better contexts when it numbers the full sequence.
        doc_chunks = docs[source_file]
        pacer.wait_for(estimate_tokens(build_prompt(source_file, doc_chunks), len(doc_chunks)))
        try:
            generated, tokens = call_with_groq_retry(generate_document_contexts, source_file, doc_chunks)
        except Exception as e:
            print(f"  [{n}/{len(todo)}] {source_file}: FAILED ({type(e).__name__}: {e})")
            if "429" in str(e) or "rate_limit" in str(e):
                print("Rate limit that won't clear soon; rerun later to resume.")
                break
            continue
        pacer.record(tokens)
        total_tokens += tokens
        for c in doc_chunks:
            if c["chunk_id"] in generated:
                contexts[c["chunk_id"]] = cache_entry(c, generated[c["chunk_id"]])
        save_contexts(contexts)
        missing = len(doc_chunks) - len(generated)
        print(f"  [{n}/{len(todo)}] {source_file}: {len(generated)}/{len(doc_chunks)} contexts, "
              f"{tokens} tokens" + (f", {missing} missing" if missing else ""))

    print(f"\nDone. {total_tokens} tokens on {CONTEXT_MODEL}. Rerun to fill any gaps.")


if __name__ == "__main__":
    main()
