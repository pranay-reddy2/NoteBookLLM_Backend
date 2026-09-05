"""
rag/chunker.py
──────────────
Text chunking using LangChain's RecursiveCharacterTextSplitter.

WHY CHUNKING?
─────────────
LLMs have a finite context window.  Entire documents (often thousands of
words) cannot be fed in a single prompt.  Chunking splits the document into
overlapping segments so:

  1. Each chunk fits inside the LLM's context window.
  2. Semantically related text stays together.
  3. The retrieval system can pinpoint the exact passage that answers a query.

WHY OVERLAP?
────────────
Without overlap, a sentence that straddles the boundary of two chunks would
be split mid-thought.  An overlap of 100 characters ensures that context near
chunk boundaries is duplicated into the neighbouring chunk, preserving
semantic continuity and improving retrieval recall.

HOW CHUNKING IMPROVES RETRIEVAL QUALITY:
─────────────────────────────────────────
  - Smaller, focused chunks produce more precise embeddings.
  - Semantic search returns the exact passage, not the whole document.
  - Overlap reduces the chance that the answer is cut across a boundary.
"""

from __future__ import annotations

import re
import logging
from typing import List, Dict, Any


logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

CHUNK_SIZE = 500       # Maximum characters per chunk
CHUNK_OVERLAP = 100    # Characters shared between consecutive chunks

# Page marker pattern inserted by loader.py
PAGE_MARKER_RE = re.compile(r"\[PAGE (\d+)\]")


def chunk_text(text: str) -> List[Dict[str, Any]]:
    """
    Split extracted document text into overlapping chunks.

    Args:
        text: Raw extracted text (may contain [PAGE N] markers).

    Returns:
        List of dicts, each with:
          - "text":        chunk content
          - "chunk_index": sequential index (0-based)
          - "page_number": best-guess page number (int or None)
    """
    if not text.strip():
        return []

    # Lazy import: langchain_text_splitters is a ~1.5 s import and only the
    # upload path needs it, so /health never pays for it.
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:  # older layout
        from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Hierarchy of separators: paragraphs → sentences → words → chars
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    raw_chunks = splitter.split_text(text)

    chunks: List[Dict[str, Any]] = []
    for idx, chunk_text in enumerate(raw_chunks):
        page_number = _extract_page_number(chunk_text, text, idx, raw_chunks)

        # Strip the [PAGE N] markers from the stored chunk text
        clean_text = PAGE_MARKER_RE.sub("", chunk_text).strip()

        if not clean_text:
            continue

        chunks.append(
            {
                "text": clean_text,
                "chunk_index": idx,
                "page_number": page_number,
            }
        )

    logger.info(
        f"Chunking complete: {len(chunks)} chunks from {len(text)} characters "
        f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )
    return chunks


def _extract_page_number(
    chunk: str,
    full_text: str,
    chunk_idx: int,
    all_chunks: List[str],
) -> int | None:
    """
    Attempt to infer the page number for a chunk.

    Strategy:
      1. Look for a [PAGE N] marker inside the chunk itself.
      2. Walk backwards through earlier chunks to find the most recent marker.
      3. Return None if no marker is found.
    """
    # Direct match inside current chunk
    match = PAGE_MARKER_RE.search(chunk)
    if match:
        return int(match.group(1))

    # Walk backwards through prior chunks
    for prev_chunk in reversed(all_chunks[:chunk_idx]):
        match = PAGE_MARKER_RE.search(prev_chunk)
        if match:
            return int(match.group(1))

    return None