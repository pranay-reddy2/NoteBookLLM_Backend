"""
rag/query_processor.py
──────────────────────
Advanced Query Processing Pipeline — RAG 2 Techniques

Implements:
  1. Query Translation     — Fix typos, rephrase ambiguous queries
  2. Sub-query Decomposition — Break complex queries into simpler ones
  3. HyDE (Hypothetical Document Embeddings) — Generate a fake answer,
     embed it, and use THAT for retrieval (often beats query embedding)
  4. Multi-query Expansion — Generate N variants of the query for broader recall

WHY THESE MATTER:
  The biggest RAG bottleneck is the gap between how users phrase questions
  and how information is stored in chunks. These techniques bridge that gap
  BEFORE retrieval happens — fixing the "Garbage In, Garbage Out" problem
  at the query level.

ARCHITECTURE:
  Raw Query
     │
     ▼
  [1] Query Translation (fix typos, clarify)
     │
     ▼
  [2] Sub-query Decomposition (split complex → simple)
     │
     ├──► [3] HyDE per sub-query (generate hypothetical doc)
     │
     └──► [4] Multi-query Expansion (N rephrasings)
     │
     ▼
  Merged deduplicated query set → Retrieval
"""

from __future__ import annotations

import logging
import os
import json
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
logger = logging.getLogger(__name__)

# ── Gemini singleton for query processing ──────────────────────────────────────

_qp_model: Any = None


def _get_qp_model():
    """
    Lazy-load a lightweight Gemini model for query processing.
    Uses Flash (fast + cheap) since these are short structured tasks.
    """
    global _qp_model
    if _qp_model is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set.")
        genai.configure(api_key=api_key)
        _qp_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,        # Low temp for deterministic rewrites
                max_output_tokens=512,
            ),
        )
        logger.info("✅ Query processor model ready.")
    return _qp_model


def _call_gemini(prompt: str) -> str:
    """Helper: call Gemini and return stripped text."""
    model = _get_qp_model()
    response = model.generate_content(prompt)
    return (response.text or "").strip()


# ── 1. Query Translation ───────────────────────────────────────────────────────

def translate_query(raw_query: str) -> str:
    """
    Technique: Query Translation / Rewriting

    Fix typos, grammar, ambiguous pronouns, and rephrase for clarity.
    This is the first guardrail against the GIGO problem at query time.

    Example:
      Input:  "wht is the totl revnue of compny in 2023?"
      Output: "What is the total revenue of the company in 2023?"

    Args:
        raw_query: The original user input (may have typos/ambiguity).

    Returns:
        A cleaned, rephrased query string.
    """
    prompt = f"""You are a query rewriter for a document search system.

Task: Fix any typos, grammar errors, and ambiguous phrasing in the user query below.
Make it clear, specific, and well-formed for semantic search.
Return ONLY the rewritten query — no explanation, no quotes, no extra text.

Original query: {raw_query}

Rewritten query:"""

    try:
        rewritten = _call_gemini(prompt)
        logger.info(f"Query translated: '{raw_query}' → '{rewritten}'")
        return rewritten if rewritten else raw_query
    except Exception as e:
        logger.warning(f"Query translation failed: {e}. Using original.")
        return raw_query


# ── 2. Sub-query Decomposition ─────────────────────────────────────────────────

def decompose_query(query: str) -> List[str]:
    """
    Technique: Sub-query Enhancement / Decomposition

    Break a complex multi-part question into focused atomic sub-queries.
    Each sub-query retrieves independently, giving broader and more
    precise coverage.

    Example:
      Input:  "What is the revenue, profit margin, and employee count?"
      Output: [
                "What is the revenue?",
                "What is the profit margin?",
                "What is the employee count?"
              ]

    If the query is already simple, returns [query] unchanged.

    Args:
        query: A potentially complex multi-part question.

    Returns:
        List of atomic sub-queries (at least 1).
    """
    prompt = f"""You are a query decomposer for a document search system.

Task: If the query below contains multiple distinct questions or information needs,
split it into individual focused sub-queries. If it's already simple, return it as-is.

Return a JSON array of strings. No explanation. No markdown. Just the JSON array.

Examples:
  Input: "What is the revenue and who is the CEO?"
  Output: ["What is the revenue?", "Who is the CEO?"]

  Input: "What is the revenue?"
  Output: ["What is the revenue?"]

Query: {query}

JSON array:"""

    try:
        raw = _call_gemini(prompt)
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        sub_queries = json.loads(raw)
        if isinstance(sub_queries, list) and sub_queries:
            logger.info(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
            return [str(q) for q in sub_queries]
    except Exception as e:
        logger.warning(f"Query decomposition failed: {e}. Using original.")

    return [query]


# ── 3. HyDE — Hypothetical Document Embeddings ────────────────────────────────

def generate_hyde_document(query: str) -> str:
    """
    Technique: HyDE (Hypothetical Document Embeddings)

    Instead of embedding the user's question (which may look nothing like
    the stored document text), generate a SHORT hypothetical answer that
    WOULD appear in the document, then embed THAT.

    Why it works:
      Stored chunks are written as facts/statements.
      Queries are written as questions.
      The embedding space gap between question-style and statement-style
      text causes retrieval misses. HyDE bridges this gap.

    Example:
      Query:     "What was the company's revenue in 2023?"
      HyDE Doc:  "The company reported total revenue of $X billion in fiscal year 2023,
                  representing a Y% increase from the previous year."

    This hypothetical document will embed MUCH closer to the actual
    document chunk than the question would.

    Args:
        query: A single focused query or sub-query.

    Returns:
        A short hypothetical document passage (2-4 sentences).
    """
    prompt = f"""You are helping with document retrieval. 

Task: Write a short hypothetical passage (2-4 sentences) that would appear in a 
professional document and would ANSWER the question below. 
Write it as factual document text, NOT as a question or answer.
Use placeholder values like [X], [Y] if specific numbers are unknown.
Return ONLY the passage — no explanation, no quotes.

Question: {query}

Hypothetical document passage:"""

    try:
        hyde_doc = _call_gemini(prompt)
        logger.info(f"HyDE generated ({len(hyde_doc)} chars) for: '{query[:60]}...'")
        return hyde_doc if hyde_doc else query
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}. Using original query.")
        return query


# ── 4. Multi-query Expansion ──────────────────────────────────────────────────

def expand_query(query: str, n: int = 3) -> List[str]:
    """
    Technique: Multi-query Expansion

    Generate N alternative phrasings of the same query.
    Each variant is retrieved independently, results are merged.
    This improves recall by catching chunks that match one phrasing
    but not another.

    Example:
      Input:  "What are the company's earnings?"
      Output: [
                "What is the total revenue of the company?",
                "How much profit did the company make?",
                "What are the financial results of the company?"
              ]

    Args:
        query: The (already translated) user query.
        n:     Number of alternative phrasings to generate.

    Returns:
        List of n alternative query strings.
    """
    prompt = f"""You are a query expansion assistant for document retrieval.

Task: Generate {n} alternative phrasings of the query below.
Each phrasing should ask for the same information but use different words/structure.
Return a JSON array of {n} strings. No explanation. No markdown.

Query: {query}

JSON array of {n} alternatives:"""

    try:
        raw = _call_gemini(prompt)
        raw = raw.replace("```json", "").replace("```", "").strip()
        variants = json.loads(raw)
        if isinstance(variants, list) and variants:
            logger.info(f"Expanded into {len(variants)} query variants.")
            return [str(v) for v in variants[:n]]
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}.")

    return [query]


# ── Master Pipeline ────────────────────────────────────────────────────────────

def process_query_advanced(
    raw_query: str,
    use_hyde: bool = True,
    use_expansion: bool = True,
    use_decomposition: bool = True,
) -> Dict[str, Any]:
    """
    Master query processing pipeline combining all techniques.

    Flow:
      raw_query
        → translate (fix typos/clarity)
        → decompose (split complex into sub-queries)
        → for each sub-query:
            → HyDE document (for embedding)
            → query variants (for multi-query retrieval)

    Args:
        raw_query:        Original user query string.
        use_hyde:         Whether to apply HyDE.
        use_expansion:    Whether to apply multi-query expansion.
        use_decomposition: Whether to apply sub-query decomposition.

    Returns:
        Dict with keys:
          - translated_query: str
          - sub_queries: List[str]
          - hyde_documents: List[str]  (one per sub-query)
          - expanded_queries: List[str] (all variants, deduplicated)
          - all_search_texts: List[str] (everything to embed for retrieval)
    """
    result: Dict[str, Any] = {
        "original_query": raw_query,
        "translated_query": raw_query,
        "sub_queries": [raw_query],
        "hyde_documents": [],
        "expanded_queries": [],
        "all_search_texts": [],
    }

    # Step 1: Translate
    translated = translate_query(raw_query)
    result["translated_query"] = translated

    # Step 2: Decompose
    sub_queries = [translated]
    if use_decomposition:
        sub_queries = decompose_query(translated)
    result["sub_queries"] = sub_queries

    # Step 3 & 4: HyDE + Expansion per sub-query
    all_search_texts: List[str] = list(sub_queries)  # Start with sub-queries themselves
    hyde_docs: List[str] = []
    expanded: List[str] = []

    for sq in sub_queries:
        if use_hyde:
            hyde = generate_hyde_document(sq)
            hyde_docs.append(hyde)
            all_search_texts.append(hyde)

        if use_expansion:
            variants = expand_query(sq, n=2)  # 2 variants per sub-query to control cost
            expanded.extend(variants)
            all_search_texts.extend(variants)

    result["hyde_documents"] = hyde_docs
    result["expanded_queries"] = expanded

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for t in all_search_texts:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    result["all_search_texts"] = deduped

    logger.info(
        f"Query processing complete: {len(deduped)} unique search texts "
        f"from original query '{raw_query[:50]}'"
    )
    return result