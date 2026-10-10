"""
Shared bits for the eval scripts: loading eval/questions.json, the
key-fact matcher, refusal detection, Groq retry, and the summary log.
"""

import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "eval" / "questions.json"
SUMMARY_PATH = ROOT / "eval" / "results" / "summary.jsonl"
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.json"
RUNS_DIR = ROOT / "data" / "eval"

VALID_TYPES = {"lookup", "cross_section", "typo", "mtsamples", "unanswerable", "multi_hop"}

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")
_QUOTES = dict.fromkeys(map(ord, "‘’"), "'")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_DASHES).translate(_QUOTES).lower()
    # Answers come back as markdown ("score of **1**") and with thousands
    # separators ("3,289 grams"); the PDFs have neither.
    text = re.sub(r"[*`]+", "", text)
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    # "37weeks" in the PDFs vs "37 weeks" in answers (and "4cm" vs "4 cm")
    text = re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", text)
    text = re.sub(r"\s+%", "%", text)
    return re.sub(r"\s+", " ", text).strip()


# Citation tags in the forms src/agent/grounding.py accepts: "[Source: f.pdf |
# SECTION]", the same in 【】 brackets, and "[f.pdf | SECTION]" without the
# label. Filenames like report_006_stroke.pdf would otherwise count as the key
# fact "stroke".
_CITATION_RE = re.compile(r"\[\s*source:[^\]]*\]|\[[^\[\]|]+?\.pdf\s*\|[^\]]*\]", re.IGNORECASE)
_FILENAME_RE = re.compile(r"[\w.-]+\.pdf\b", re.IGNORECASE)


def strip_citations(text: str) -> str:
    text = text.replace("【", "[").replace("】", "]")
    return _FILENAME_RE.sub(" ", _CITATION_RE.sub(" ", text))


def _fact_pattern(alt: str) -> re.Pattern:
    alt = normalize(alt)
    short = len(alt) <= 3
    # Numeric edges need a boundary or "4.8" would match "14.8" and "5 mg"
    # would match "2.5 mg". Very short words ("AF", "INR") need one too.
    # Longer words don't, so "NSAID" still matches "NSAIDs".
    head = r"(?<![a-z0-9])(?<!\d\.)" if short or alt[0].isdigit() else ""
    tail = r"(?![a-z0-9])(?!\.\d)" if short or alt[-1].isdigit() else ""
    return re.compile(head + re.escape(alt) + tail)


_TOKEN_CHAR = re.compile(r"[\w/-]")


def _inside_date_or_id(text: str, m: re.Match) -> bool:
    # "28" inside "1942-02-28", "28/02/2024" or "syn-2024-028" is part of a
    # date or ID. One separator is still a range or ratio ("70-75", "120/80").
    start, end = m.start(), m.end()
    while start > 0 and _TOKEN_CHAR.match(text[start - 1]):
        start -= 1
    while end < len(text) and _TOKEN_CHAR.match(text[end]):
        end += 1
    token = text[start:end].strip("-/")
    return m.group() != token and len(re.findall(r"[-/]\d", token)) >= 2


def contains_fact(text: str, fact: str) -> bool:
    """A key fact may list alternatives separated by "|"; any one counts."""
    norm = normalize(text)
    for alt in fact.split("|"):
        numeric = any(ch.isdigit() for ch in alt)
        for m in _fact_pattern(alt).finditer(norm):
            if not (numeric and _inside_date_or_id(norm, m)):
                return True
    return False


def key_fact_recall(answer: str, facts: list[str]) -> float | None:
    if not facts:
        return None
    answer = strip_citations(answer)
    return sum(contains_fact(answer, f) for f in facts) / len(facts)


# generator.py and the agent tell the model to answer with exactly "The answer
# to this question is not found in the available clinical documents." The
# model usually complies, but paraphrases often enough that an exact match
# would undercount, so accept the common variants too. Every alternative
# starts and ends on a word boundary: without that "no ... data" matched
# inside "normal", "known" and "nodes".
_NOT = r"(?:\bnot|\bcannot|n't)"
_ABSENT = (r"(?:found|present|mentioned|provided|available|documented|stated|reported|"
           r"specified|included|contained|listed|recorded|given|described)")
_WHERE = r"(?:documents?|passages?|records?|reports?|notes?|sources?|context)"
_REFUSAL_RE = re.compile(
    # "is not found in the available clinical documents", "isn't mentioned anywhere in the notes"
    rf"{_NOT} (?:be )?{_ABSENT}\b(?: [\w-]+){{0,2}}? (?:in|within|from|by|among) (?:[\w-]+ ){{0,5}}?{_WHERE}\b"
    # "there is no information about", "does not contain any data on"
    r"|\b(?:no|(?:does|do|did) not contain|(?:does|do|did)n't contain)"
    r"(?: [\w-]+){0,3}? (?:information|mention|record|records|data|details)\b"
    # "the documents do not state"
    rf"|\b{_WHERE} (?:provided |available )?(?:(?:do|does|did) not|(?:do|does|did)n't) "
    r"(?:say|state|mention|include|specify|list|provide|contain|report|record|give)\b",
)


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL_RE.search(normalize(answer)))


RAGAS_METRICS = ("faithfulness", "context_precision", "context_recall", "answer_relevancy")


def deterministic_scores(item: dict, answer: str) -> dict:
    refused = is_refusal(answer)
    if not item["answerable"]:
        return {"refused": refused, "refused_correctly": refused}
    recall = key_fact_recall(answer, item["key_facts"])
    return {
        "refused": refused,
        "key_fact_recall": recall,
        "missing_facts": [f for f in item["key_facts"] if key_fact_recall(answer, [f]) == 0],
        # a hedge like "the dose isn't stated" next to a correct answer is not a refusal
        "false_refusal": refused and recall == 0,
    }


def aggregate(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    out = {
        "n": len(rows),
        "key_fact_recall": mean(r["scores"]["key_fact_recall"] for r in answerable),
        "all_facts_rate": mean(float(r["scores"]["key_fact_recall"] == 1.0) for r in answerable),
        "false_refusal_rate": mean(float(r["scores"]["false_refusal"]) for r in answerable),
        "refusal_accuracy": mean(float(r["scores"]["refused_correctly"]) for r in unanswerable),
        "mean_chunks": mean(r.get("chunks_retrieved") for r in rows),
        "mean_latency_s": mean(r.get("latency_s") for r in rows),
    }
    scored = [r for r in rows if r.get("ragas")]
    if scored:
        out["ragas_n"] = len(scored)
        for m in RAGAS_METRICS:
            out[m] = mean(r["ragas"].get(m) for r in scored)
    return {k: v for k, v in out.items() if v is not None}


def rescore_row(row: dict, item: dict) -> dict:
    """A stored answer, re-scored against the current eval item."""
    return {**row, "type": item["type"], "answerable": item["answerable"],
            "reference": item["reference"], "key_facts": item["key_facts"],
            "scores": deterministic_scores(item, row["answer"])}


def aggregate_by_type(rows: list[dict]) -> tuple[dict, dict]:
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        grouped.setdefault(r["type"], []).append(r)
    return aggregate(rows), {t: aggregate(rs) for t, rs in sorted(grouped.items())}


def print_report(overall: dict, by_type: dict) -> None:
    cols = [c for c in ("n", "key_fact_recall", "all_facts_rate", "false_refusal_rate",
                        "refusal_accuracy", "ragas_n", *RAGAS_METRICS) if c in overall]
    print(f"{'type':<14}" + "".join(f"{c:>20}" for c in cols))
    for name, agg in list(by_type.items()) + [("ALL", overall)]:
        cells = []
        for c in cols:
            v = agg.get(c)
            cells.append(f"{'-':>20}" if v is None else f"{v:>20}" if c in ("n", "ragas_n")
                         else f"{v:>20.3f}")
        print(f"{name:<14}" + "".join(cells))


def load_questions(types=None, ids=None, limit=None) -> list[dict]:
    items = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if types:
        unknown = set(types) - VALID_TYPES
        if unknown:
            raise SystemExit(f"Unknown --types: {sorted(unknown)}. Valid: {sorted(VALID_TYPES)}")
        items = [q for q in items if q["type"] in types]
    if ids:
        known = {q["id"] for q in items}
        missing = [i for i in ids if i not in known]
        if missing:
            raise SystemExit(f"Unknown --ids (or filtered out by --types): {missing}")
        items = [q for q in items if q["id"] in ids]
    return items[:limit] if limit else items


def split_csv(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def append_summary(record: dict) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    with SUMMARY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# Groq's 429 body says e.g. "Please try again in 7.66s" or "in 1m2.5s"
_GROQ_WAIT_RE = re.compile(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s")


def _groq_wait_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    match = _GROQ_WAIT_RE.search(str(exc))
    if match:
        return int(match.group(1) or 0) * 60 + float(match.group(2))
    return None


def call_with_groq_retry(fn, *args, max_attempts: int = 5, base_delay: float = 20.0, **kwargs):
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            status = getattr(e, "status_code", None)
            transient = status in (429, 500, 502, 503) or "429" in str(e) or "rate_limit" in str(e)
            if attempt == max_attempts or not transient:
                raise
            # Daily-token exhaustion also comes back as 429 but with a wait of
            # many minutes; don't sleep through that, let the run stop and
            # resume from its checkpoint later.
            wait = _groq_wait_seconds(e)
            if wait is not None and wait > 300:
                raise
            wait = (wait + 2) if wait is not None else base_delay * attempt
            print(f"    Groq {status or 'error'}, retrying in {wait:.0f}s (attempt {attempt}/{max_attempts})")
            time.sleep(wait)
