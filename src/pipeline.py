"""
RAG pipeline orchestrator — the single entry point for "question in, answer out."

WHY A SEPARATE ORCHESTRATOR?
Each module (embedder, store, retriever, generator) has one job.
The pipeline's job is to wire them together in the right order.
This separation means:
- You can swap the LLM (Gemini → Claude → Ollama) by changing only generator.py
- You can swap the vector DB (ChromaDB → Qdrant) by changing only store.py
- The pipeline itself never changes

This is the "modular RAG" design pattern described in the literature.
Interviewers ask: "how would you swap in a different LLM?"
Answer: "Change the generator module. The pipeline doesn't care."

WHAT THIS FUNCTION DOES (step by step):
1. Embed the user's question into a vector
2. Search ChromaDB for the 5 most similar chunks
3. Rerank the top-5 with a cross-encoder to get the best 3
4. Build a prompt with those 3 chunks + the question
5. Send to Gemini → get cited answer
6. Return everything (answer + sources + debug info)
"""

from src.retrieval.retriever import retrieve
from src.generation.generator import generate_answer


def ask(query: str, top_k: int = 3, use_reranker: bool = True) -> dict:
    """
    Main RAG pipeline function.

    Args:
        query:        the user's natural language question
        top_k:        number of chunks to send to the LLM
        use_reranker: whether to apply cross-encoder reranking

    Returns:
        {
          "query":       the original question,
          "answer":      the LLM's cited answer,
          "sources":     list of {source_file, section} dicts,
          "chunks":      the actual retrieved passages (for debugging),
          "model":       which LLM produced the answer,
          "chunks_used": how many chunks were sent to the LLM,
        }
    """
    # Step 1 + 2 + 3: retrieve (embed → search → rerank)
    chunks = retrieve(query, top_k=top_k, use_reranker=use_reranker)

    if not chunks:
        return {
            "query": query,
            "answer": "No relevant documents found in the knowledge base.",
            "sources": [],
            "chunks": [],
            "model": None,
            "chunks_used": 0,
        }

    # Step 4 + 5: generate cited answer
    result = generate_answer(query, chunks)

    return {
        "query": query,
        "answer": result["answer"],
        "sources": result["sources"],
        "chunks": chunks,
        "model": result["model"],
        "chunks_used": result["chunks_used"],
    }
