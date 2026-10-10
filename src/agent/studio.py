"""
Entry point for LangGraph Studio (langgraph.json points here).

`langgraph dev` imports this module and serves `graph` with Studio's UI on
top: you type a question, watch each node run, and inspect the state after
every step. It's the same graph the app uses (build_graph() with the real
Groq models), so each Studio run costs real API calls. Set AGENT_MODEL in
.env to try a cheaper model.
"""

from src.agent.graph import build_graph
from src.tracing import setup_tracing

setup_tracing()

graph = build_graph()
