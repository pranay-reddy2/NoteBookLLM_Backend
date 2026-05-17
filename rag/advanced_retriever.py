"""
rag/advanced_retriever.py
─────────────────────────
Multi-query retrieval layer that works with the query processor.

Takes multiple search texts (sub-queries, HyDE docs, expansions) from
query_processor.py, retrieves chunks for each, and merges/deduplicates
results — giving much higher recall than single-query retrieval.

RECALL vs PRECISION:
  - More queries = higher recall (find more relevant chunks)
  - Re-ranking (reranker.py) then improves precision
  - This two-stage approach is the industry standard for production RAG
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from rag.retriever import retrieve_relevant_chunks

logger = logging.getLogger(__name__)


def multi_query_retrieve(
    search_texts: List[str],
    top_k_per_query: int = 4,
    document_ids: Optional[List[str]] = None,
    max_total_chunks: int = 12,
) -> List[Dict[str, Any]]:
    """
    Retrieve chunks for multiple search texts and merge results.

    For each text in search_texts (sub-queries, HyDE docs, variants),
    retrieve top_k_per_query chunks. Merge all results and deduplicate
    by chunk_id, keeping the highest score for each unique chunk.

    Args:
        search_texts:     List of texts to retrieve for (from query_processor).
        top_k_per_query:  Chunks to retrieve per text.
        document_ids:     Optional document filter.
        max_total_chunks: Cap on merged results (for token budget).

    Returns:
        Deduplicated merged list, sorted by score descending.
    """
    if not search_texts:
        return []

    seen: Dict[str, Dict[str, Any]] = {}  # chunk_id → best chunk

    for i, text in enumerate(search_texts):
        try:
            chunks = retrieve_relevant_chunks(
                query=text,
                top_k=top_k_per_query,
                document_ids=document_ids,
            )
            for chunk in chunks:
                cid = chunk["chunk_id"]
                if cid not in seen or chunk["score"] > seen[cid]["score"]:
                    seen[cid] = chunk
                    seen[cid]["_matched_by"] = i  # Track which query found it

            logger.debug(f"Query {i+1}/{len(search_texts)}: retrieved {len(chunks)} chunks")

        except Exception as e:
            logger.warning(f"Retrieval failed for text {i+1}: {e}")
            continue

    merged = sorted(seen.values(), key=lambda c: c["score"], reverse=True)
    result = merged[:max_total_chunks]

    logger.info(
        f"Multi-query retrieval: {len(search_texts)} queries → "
        f"{len(seen)} unique chunks → top {len(result)} selected"
    )
    return result