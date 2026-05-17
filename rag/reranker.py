"""
rag/reranker.py
───────────────
Post-retrieval quality layer — RAG 2 Techniques

Implements:
  1. Cross-encoder Re-ranking   — Score each (query, chunk) pair jointly
                                   for precision beyond cosine similarity
  2. LLM-as-a-Judge             — Use Gemini to score relevance of chunks
  3. Corrective RAG (CRAG)      — Detect when retrieved chunks are irrelevant
                                   and trigger a fallback strategy

WHY RETRIEVAL ISN'T ENOUGH:
  Cosine similarity between embeddings is approximate. Two chunks may have
  similar embeddings but one may not actually answer the query. Re-ranking
  adds a second, more expensive but accurate scoring pass.

PIPELINE POSITION:
  Retrieval (top-k chunks via cosine)
       │
       ▼
  [Re-ranking] — score each chunk vs query more precisely
       │
       ▼
  [LLM Judge] — filter out irrelevant chunks, flag low confidence
       │
       ▼
  [CRAG check] — if all chunks are bad, trigger corrective action
       │
       ▼
  Generator (Gemini)

SPEED vs ACCURACY TRADEOFF:
  Re-ranking adds latency but dramatically improves precision.
  The tradeoff config allows toggling each layer independently.
"""

from __future__ import annotations

import logging
import os
import json
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

# Minimum relevance score to keep a chunk (0.0 – 1.0)
RELEVANCE_THRESHOLD = 0.5

# If fewer than this many chunks pass the judge, trigger CRAG fallback
MIN_ACCEPTABLE_CHUNKS = 1

# ── Gemini singleton for judging ───────────────────────────────────────────────

_judge_model: Any = None


def _get_judge_model():
    global _judge_model
    if _judge_model is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set.")
        genai.configure(api_key=api_key)
        _judge_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,        # Deterministic scoring
                max_output_tokens=256,
            ),
        )
        logger.info("✅ LLM Judge model ready.")
    return _judge_model


def _call_judge(prompt: str) -> str:
    model = _get_judge_model()
    response = model.generate_content(prompt)
    return (response.text or "").strip()


# ── 1. Cross-encoder Style LLM Re-ranking ─────────────────────────────────────

def llm_rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Technique: Re-ranking with LLM as Cross-encoder

    Instead of using only the cosine similarity from the retrieval step,
    ask Gemini to score each (query, chunk) pair on a 0-10 scale.

    This is the "cross-encoder" approach: the query and document are
    evaluated TOGETHER rather than independently — catching semantic
    relevance that embedding similarity misses.

    Cost vs Accuracy tradeoff:
      - More accurate than cosine-only ranking
      - Slower (one LLM call per chunk, or batched)
      - Use only on the top-k candidates from initial retrieval

    Args:
        query:  The user's (translated) query.
        chunks: List of retrieved chunk dicts (from retriever.py).

    Returns:
        Chunks re-sorted by LLM relevance score (descending).
        Each chunk gets a new key: 'llm_relevance_score' (0.0–1.0)
    """
    if not chunks:
        return chunks

    # Batch all chunks into a single prompt to minimize API calls
    chunks_text = ""
    for i, chunk in enumerate(chunks):
        chunks_text += f"\n[CHUNK {i+1}]:\n{chunk['text'][:400]}\n"

    prompt = f"""You are a relevance scorer for a document retrieval system.

Given a USER QUERY and multiple DOCUMENT CHUNKS, score each chunk's relevance
to the query on a scale of 0-10.

Scoring guide:
  10 = Directly answers the query with precise information
  7-9 = Highly relevant, contains key information
  4-6 = Somewhat relevant, related topic but incomplete
  1-3 = Tangentially related or mostly irrelevant
  0   = Completely irrelevant

Return ONLY a JSON array of numbers (one per chunk, in order).
Example for 3 chunks: [8, 3, 10]

USER QUERY: {query}

DOCUMENT CHUNKS:
{chunks_text}

JSON array of scores (0-10):"""

    try:
        raw = _call_judge(prompt)
        raw = raw.replace("```json", "").replace("```", "").strip()
        scores = json.loads(raw)

        if isinstance(scores, list) and len(scores) == len(chunks):
            for chunk, score in zip(chunks, scores):
                chunk["llm_relevance_score"] = round(float(score) / 10.0, 3)

            # Re-sort by LLM score descending
            reranked = sorted(chunks, key=lambda c: c["llm_relevance_score"], reverse=True)
            logger.info(f"Re-ranked {len(reranked)} chunks. Top score: {reranked[0]['llm_relevance_score']}")
            return reranked

    except Exception as e:
        logger.warning(f"LLM re-ranking failed: {e}. Using original order.")

    # Fallback: tag with cosine score as llm_relevance_score
    for chunk in chunks:
        chunk["llm_relevance_score"] = chunk.get("score", 0.5)

    return chunks


# ── 2. LLM-as-a-Judge: Relevance Filtering ────────────────────────────────────

def judge_chunk_relevance(
    query: str,
    chunks: List[Dict[str, Any]],
    threshold: float = RELEVANCE_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Technique: LLM-as-a-Judge

    After re-ranking, filter out chunks that fall below the relevance
    threshold. The LLM acts as a quality gate — only chunks that are
    genuinely relevant pass through to the generator.

    This prevents the generator from being confused by irrelevant context,
    which is a common cause of hallucination in RAG systems.

    Args:
        query:     The user query.
        chunks:    Re-ranked chunks (each should have 'llm_relevance_score').
        threshold: Minimum score to keep (default 0.5 = score of 5/10).

    Returns:
        Tuple of (accepted_chunks, rejected_chunks).
        Accepted chunks are passed to the generator.
        Rejected chunks are logged and used in CRAG decision.
    """
    accepted = []
    rejected = []

    for chunk in chunks:
        score = chunk.get("llm_relevance_score", chunk.get("score", 0))
        if score >= threshold:
            accepted.append(chunk)
        else:
            rejected.append(chunk)
            logger.debug(
                f"Chunk rejected (score={score:.2f}): "
                f"{chunk['text'][:80]}..."
            )

    logger.info(
        f"LLM Judge: {len(accepted)} accepted, {len(rejected)} rejected "
        f"(threshold={threshold})"
    )
    return accepted, rejected


# ── 3. Corrective RAG (CRAG) ──────────────────────────────────────────────────

class CRAGResult:
    """Result container for a CRAG evaluation."""
    def __init__(
        self,
        status: str,                        # "confident" | "uncertain" | "irrelevant"
        accepted_chunks: List[Dict],
        rejected_chunks: List[Dict],
        fallback_message: Optional[str],
        confidence_score: float,
    ):
        self.status = status
        self.accepted_chunks = accepted_chunks
        self.rejected_chunks = rejected_chunks
        self.fallback_message = fallback_message
        self.confidence_score = confidence_score

    def should_answer(self) -> bool:
        return self.status in ("confident", "uncertain")

    def __repr__(self):
        return (
            f"CRAGResult(status={self.status}, "
            f"accepted={len(self.accepted_chunks)}, "
            f"confidence={self.confidence_score:.2f})"
        )


def corrective_rag_check(
    query: str,
    accepted_chunks: List[Dict[str, Any]],
    rejected_chunks: List[Dict[str, Any]],
) -> CRAGResult:
    """
    Technique: Corrective RAG (CRAG)

    After the LLM Judge filters chunks, decide whether the remaining
    context is sufficient to answer the query:

      CONFIDENT:   Enough high-quality chunks → answer normally
      UNCERTAIN:   Some relevant chunks but low confidence → answer with caveat
      IRRELEVANT:  No useful chunks → return fallback message, do NOT hallucinate

    This is the final safety net against hallucination: if we can't find
    the answer in the documents, we explicitly say so rather than inventing.

    Args:
        query:           The user query.
        accepted_chunks: Chunks that passed the relevance filter.
        rejected_chunks: Chunks that were filtered out.

    Returns:
        CRAGResult with status and appropriate chunks/message.
    """
    total = len(accepted_chunks) + len(rejected_chunks)

    # No chunks at all
    if total == 0:
        return CRAGResult(
            status="irrelevant",
            accepted_chunks=[],
            rejected_chunks=[],
            fallback_message="No documents have been uploaded yet. Please upload a document first.",
            confidence_score=0.0,
        )

    # Not enough chunks passed the judge
    if len(accepted_chunks) < MIN_ACCEPTABLE_CHUNKS:
        logger.warning(
            f"CRAG: Only {len(accepted_chunks)} chunks passed judge "
            f"(need {MIN_ACCEPTABLE_CHUNKS}). Triggering corrective action."
        )
        return CRAGResult(
            status="irrelevant",
            accepted_chunks=[],
            rejected_chunks=rejected_chunks,
            fallback_message=(
                "I could not find relevant information about this in your documents. "
                "The uploaded documents may not contain the answer to this question. "
                "Please ensure you have uploaded the relevant document."
            ),
            confidence_score=0.0,
        )

    # Calculate overall confidence from LLM relevance scores
    scores = [c.get("llm_relevance_score", c.get("score", 0)) for c in accepted_chunks]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    if avg_score >= 0.7:
        status = "confident"
    elif avg_score >= 0.5:
        status = "uncertain"
    else:
        status = "irrelevant"

    fallback = None
    if status == "uncertain":
        fallback = (
            "⚠️ Note: The retrieved information has moderate relevance to your question. "
            "The answer below is based on the best matching content found, but may be incomplete."
        )
    elif status == "irrelevant":
        fallback = (
            "I could not find sufficiently relevant information in the uploaded documents "
            "to answer this question confidently."
        )
        accepted_chunks = []  # Don't pass irrelevant chunks to generator

    logger.info(f"CRAG status: {status} (avg_score={avg_score:.2f})")

    return CRAGResult(
        status=status,
        accepted_chunks=accepted_chunks,
        rejected_chunks=rejected_chunks,
        fallback_message=fallback,
        confidence_score=avg_score,
    )


# ── Master Post-Retrieval Pipeline ────────────────────────────────────────────

def post_retrieval_pipeline(
    query: str,
    raw_chunks: List[Dict[str, Any]],
    use_reranking: bool = True,
    use_judge: bool = True,
    use_crag: bool = True,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
) -> CRAGResult:
    """
    Full post-retrieval quality pipeline.

    Steps:
      1. LLM Re-ranking  (cross-encoder style)
      2. LLM Judge       (filter below threshold)
      3. CRAG            (decide confident / uncertain / irrelevant)

    Args:
        query:               User query (translated).
        raw_chunks:          Chunks from retriever.
        use_reranking:       Toggle re-ranking step.
        use_judge:           Toggle LLM judge filtering.
        use_crag:            Toggle CRAG check.
        relevance_threshold: Minimum score for judge acceptance.

    Returns:
        CRAGResult with final decision and accepted chunks.
    """
    chunks = list(raw_chunks)

    # Step 1: Re-rank
    if use_reranking and chunks:
        chunks = llm_rerank_chunks(query, chunks)

    # Step 2: LLM Judge
    accepted, rejected = chunks, []
    if use_judge and chunks:
        accepted, rejected = judge_chunk_relevance(query, chunks, relevance_threshold)

    # Step 3: CRAG
    if use_crag:
        return corrective_rag_check(query, accepted, rejected)

    # No CRAG: return accepted chunks as-is
    return CRAGResult(
        status="confident",
        accepted_chunks=accepted,
        rejected_chunks=rejected,
        fallback_message=None,
        confidence_score=1.0,
    )