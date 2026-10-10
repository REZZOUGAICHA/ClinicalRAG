"""
Chat model setup for the agent, plus a retry wrapper for Groq rate limits.

The agent makes several calls per question (route, a few tool-planning turns,
the answer), so on the free tier (8000 tokens/minute per model) a multi-step
run can hit a 429 part-way through. Retrying just that call keeps the work
already done; letting it bubble up would restart the whole run.
"""

import logging
import os
import re
import time
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"

# Completion caps include gpt-oss's hidden reasoning tokens, not just the
# visible output, so they're set well above what the output itself needs.
MAX_TOKENS = {
    "router": 256,
    "agent": 1024,
    "answer": 2048,
}

# Longer waits than this mean the daily quota is gone, not the per-minute one.
_MAX_WAIT_S = 90

log = logging.getLogger(__name__)
_GROQ_WAIT_RE = re.compile(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s")


def agent_model() -> str:
    return os.getenv("AGENT_MODEL", DEFAULT_MODEL)


@lru_cache(maxsize=None)
def _groq_chat(model: str, purpose: str):
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=model,
        temperature=0.1,
        max_tokens=MAX_TOKENS[purpose],
        reasoning_effort="low",
        # The groq SDK's own retry gives up on waits over 60s and backs off
        # blindly otherwise; invoke_with_retry below reads the real wait.
        max_retries=0,
        # gpt-oss doesn't take reasoning_format; this is how Groq drops the
        # reasoning text from the response for it.
        model_kwargs={"include_reasoning": False},
    )


def get_chat_model(purpose: str):
    """Chat model for one stage of the graph: "router", "agent" or "answer"."""
    return _groq_chat(agent_model(), purpose)


def _wait_seconds(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        pass
    match = _GROQ_WAIT_RE.search(str(exc))
    if match:
        return int(match.group(1) or 0) * 60 + float(match.group(2))
    return None


def invoke_with_retry(runnable, messages, *, max_attempts: int = 6, sleep=time.sleep):
    for attempt in range(1, max_attempts + 1):
        try:
            return runnable.invoke(messages)
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status not in (429, 500, 502, 503) or attempt == max_attempts:
                raise
            wait = _wait_seconds(e)
            if wait is not None and wait > _MAX_WAIT_S:
                raise
            wait = (wait + 1) if wait is not None else 10 * attempt
            log.warning("Groq %s, retrying in %.0fs (attempt %d/%d)", status, wait, attempt, max_attempts)
            sleep(wait)
