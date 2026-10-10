"""
Optional tracing: every LLM call, tool call and graph node as a timed span,
with prompts, outputs and token counts, sent to a local Arize Phoenix.

Off unless CLINICALRAG_TRACING is set, and when it's off nothing extra is
imported, so the deployed image doesn't need the packages.

    CLINICALRAG_TRACING=phoenix      turn it on
    PHOENIX_COLLECTOR_ENDPOINT       default http://localhost:6006
    PHOENIX_PROJECT_NAME             default clinicalrag

Only LangChain is instrumented, which covers the whole agent graph. The
pipeline's direct groq SDK calls are not traced: adding the groq
instrumentor as well gave every agent LLM call a second, nested span
(ChatGroq calls the groq SDK underneath), which doubled the token totals.
The simple route still shows up as a "simple" node span with its timing.

The packages are in requirements-dev.txt; if they're missing the app logs a
warning and runs untraced.

Spans hold full prompts, which here means patient text. Send them only to a
collector you run yourself.
"""

import logging
import os

log = logging.getLogger(__name__)

_enabled: str | None = None


def setup_tracing() -> str | None:
    """Turn tracing on if CLINICALRAG_TRACING asks for it. Safe to call twice."""
    global _enabled
    mode = os.getenv("CLINICALRAG_TRACING", "").strip().lower()
    if _enabled or mode in ("", "0", "off", "false", "none"):
        return _enabled
    if mode != "phoenix":
        log.warning("CLINICALRAG_TRACING=%s: only 'phoenix' is supported, tracing is off", mode)
        return None
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register
    except ImportError:
        log.warning("CLINICALRAG_TRACING=phoenix but the tracing packages aren't installed "
                    "(pip install -r requirements-dev.txt); tracing is off")
        return None

    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006").rstrip("/")
    provider = register(
        endpoint=f"{endpoint}/v1/traces",
        project_name=os.getenv("PHOENIX_PROJECT_NAME", "clinicalrag"),
        # Batch export runs in a background thread, so a slow or stopped
        # Phoenix doesn't add latency to answers. Leftovers flush at exit.
        batch=True,
        # Our provider only, so libraries that also speak OpenTelemetry
        # (ChromaDB) don't start sending their own spans.
        set_global_tracer_provider=False,
        verbose=False,
    )
    LangChainInstrumentor().instrument(tracer_provider=provider)
    log.warning("Tracing to Phoenix at %s", endpoint)
    _enabled = mode
    return _enabled
