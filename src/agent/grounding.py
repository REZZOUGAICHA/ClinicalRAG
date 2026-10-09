"""
Deterministic citation check for agent answers.

The pipeline puts three chunks in the prompt and anything the model cites is
one of them. The agent's answer step sees many more passages gathered over
several turns, and a model asked to cite a lot will occasionally cite a
plausible-looking source it never saw. Every [Source: file | SECTION] tag is
therefore checked against the chunks the tools actually returned in this run.
A tag that matches nothing is either fixed by one regeneration or removed;
nothing here ever invents a source.
"""

import re

from src.generation.generator import _BRACKETS

REFUSAL = "The answer to this question is not found in the available clinical documents."

# One bracket may hold several citations: "[Source: a.pdf | LAB; b.pdf | MEDS]".
# gpt-oss-20b also drops the "Source:" label ("[a.pdf | LAB]"); a bracket that
# starts with a .pdf filename and a "|" is unambiguous enough to count.
_BLOCK_RE = re.compile(r"\[\s*Source:\s*([^\]]*)\]|\[\s*([^\[\]|]+?\.pdf\s*\|[^\]]*)\]",
                       re.IGNORECASE)
_LABEL_RE = re.compile(r"^(?:Source:|Section:)\s*", re.IGNORECASE)


_REFUSAL_PHRASE = "not found in the available clinical documents"


def is_refusal(answer: str) -> bool:
    """
    True only if the answer is the refusal and nothing else. A draft like
    "Warfarin 10 mg daily. The INR is not found in the available clinical
    documents." makes a claim, so it has to be cited like any other answer.
    """
    text = _BLOCK_RE.sub(" ", answer.translate(_BRACKETS))
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if re.search(r"\w", s)]
    return bool(sentences) and all(_REFUSAL_PHRASE in s.lower() for s in sentences)


def _parse_block(inner: str) -> list[tuple[str, str]]:
    pairs = []
    for part in re.split(r"\s*;\s*", inner):
        file, _, section = part.partition("|")
        pairs.append((_unlabel(file), _unlabel(section)))
    return pairs


def _unlabel(text: str) -> str:
    return _LABEL_RE.sub("", text.strip()).strip()


def _match(file: str, section: str, chunks: list[dict]) -> list[dict]:
    stem = file.lower().removesuffix(".pdf")
    files = {c["source_file"] for c in chunks}
    # Tolerate "report_003" for "report_003_acute_mi.pdf" when it's unambiguous.
    candidates = [f for f in files if f.lower().removesuffix(".pdf") == stem]
    if not candidates:
        candidates = [f for f in files if f.lower().startswith(stem + "_")]
    if len(candidates) != 1:
        return []
    return [c for c in chunks
            if c["source_file"] == candidates[0] and c["section"].upper() == section.upper()]


def check_citations(answer: str, chunks: list[dict]) -> dict:
    """
    Returns:
        answer:   the answer with known citations rewritten to the canonical
                  "[Source: file.pdf | SECTION]" form and unknown ones removed
        unknown:  the (file, section) pairs that matched no retrieved chunk
        cited:    retrieved chunks the answer cites, in first-cited order
        n_cited:  how many valid citations the answer has
    """
    answer = answer.translate(_BRACKETS)
    unknown: list[tuple[str, str]] = []
    cited: dict[str, dict] = {}
    n_valid = 0

    def replace(m: re.Match) -> str:
        nonlocal n_valid
        kept = []
        for file, section in _parse_block(m.group(1) or m.group(2)):
            hits = _match(file, section, chunks)
            if not hits:
                unknown.append((file, section))
                continue
            n_valid += 1
            for c in hits:
                cited.setdefault(c["chunk_id"], c)
            kept.append(f"[Source: {hits[0]['source_file']} | {hits[0]['section']}]")
        return " ".join(kept)

    fixed = _BLOCK_RE.sub(replace, answer)
    # Tidy the gaps a removed citation leaves behind ("value  ." -> "value.")
    fixed = re.sub(r"[ \t]+([.,;:])", r"\1", fixed)
    fixed = re.sub(r"[ \t]{2,}", " ", fixed).strip()
    return {"answer": fixed, "unknown": unknown, "cited": list(cited.values()), "n_cited": n_valid}
