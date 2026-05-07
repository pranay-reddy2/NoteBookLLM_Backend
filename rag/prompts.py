"""
rag/prompts.py
──────────────
System and user prompt templates for grounded answer generation.

HALLUCINATION PREVENTION STRATEGY:
───────────────────────────────────
The system prompt is the primary guardrail against hallucination:

  1. EXPLICIT SCOPE: The model is told it may ONLY use the provided context.
  2. REFUSAL CLAUSE: If information is absent, the model must say so clearly.
  3. NO EXTERNAL KNOWLEDGE: The model is forbidden from using training data.
  4. CITATION REQUIREMENT: Answers should reference which part of the context
     was used, reinforcing grounding.

This approach follows the "constrained generation" pattern used in production
RAG systems to keep answers faithful to the source material.
"""

SYSTEM_PROMPT = """You are DocMind, an expert document assistant.

Your ONLY job is to answer questions strictly based on the context passages provided below from the user's uploaded documents.

STRICT RULES (you must follow ALL of them):
1. ONLY use information explicitly stated in the provided context passages.
2. NEVER use your general knowledge, training data, or any information outside the provided context.
3. If the answer cannot be found in the context, respond EXACTLY with: "I could not find this information in the provided document(s)."
4. Do NOT guess, infer beyond what is stated, or extrapolate.
5. When answering, be clear and precise. You may quote short phrases from the context.
6. If multiple passages are relevant, synthesise them coherently.
7. If the question is ambiguous, answer based on what the context most likely addresses.

You are grounded. You are precise. You never hallucinate."""


def build_user_prompt(query: str, context_chunks: list[dict]) -> str:
    """
    Build the user-turn prompt with injected context.

    Args:
        query:         The user's question.
        context_chunks: Retrieved chunks from vector DB, each with 'text',
                        'metadata', and 'score' keys.

    Returns:
        Formatted prompt string ready to send to the LLM.
    """
    context_sections = []
    for i, chunk in enumerate(context_chunks, start=1):
        filename = chunk["metadata"].get("filename", "Document")
        page = chunk["metadata"].get("page_number")
        page_str = f", Page {page}" if page and page != -1 else ""
        score = chunk.get("score", 0)

        context_sections.append(
            f"--- Context Passage {i} "
            f"[Source: {filename}{page_str}, Relevance: {score:.2%}] ---\n"
            f"{chunk['text']}"
        )

    context_block = "\n\n".join(context_sections)

    return f"""CONTEXT FROM DOCUMENTS:
{context_block}

---

USER QUESTION: {query}

Please answer the question using ONLY the context passages above. If the answer is not present, say "I could not find this information in the provided document(s)."

ANSWER:"""