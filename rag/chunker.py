"""
rag/chunker.py
──────────────
Text chunking using a pure-Python RecursiveCharacterSplitter.
No LangChain dependency — identical logic, zero conflicts.

WHY CHUNKING?
  LLMs have a finite context window. Chunking splits documents into
  segments that fit inside the window and produce focused embeddings.

WHY OVERLAP?
  A 100-character overlap duplicates text near chunk boundaries so a
  sentence split across two chunks is still fully captured in one of them.

HOW IT IMPROVES RETRIEVAL:
  Smaller focused chunks produce more precise embeddings. Semantic search
  returns the exact passage rather than an entire document.
"""

from __future__ import annotations
import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
SEPARATORS    = ["\n\n", "\n", ". ", " ", ""]
PAGE_RE       = re.compile(r"\[PAGE (\d+)\]")


def _split_text(text: str, separators: List[str], chunk_size: int, overlap: int) -> List[str]:
    """Recursively split text by a hierarchy of separators."""
    separator = separators[-1]
    for sep in separators:
        if sep == "" or sep in text:
            separator = sep
            break

    splits = text.split(separator) if separator else list(text)

    chunks: List[str] = []
    current = ""

    for split in splits:
        piece = (separator + split) if current else split
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            if current:
                chunks.append(current)
            # If a single split exceeds chunk_size, recurse with next separator
            if len(split) > chunk_size and len(separators) > 1:
                sub = _split_text(split, separators[1:], chunk_size, overlap)
                chunks.extend(sub)
                current = ""
            else:
                # Start new chunk with overlap from end of previous
                if chunks:
                    prev = chunks[-1]
                    current = prev[-overlap:] + separator + split if overlap else split
                else:
                    current = split

    if current:
        chunks.append(current)

    return chunks


def chunk_text(text: str) -> List[Dict[str, Any]]:
    """
    Split extracted document text into overlapping chunks.

    Returns:
        List of dicts with keys: text, chunk_index, page_number.
    """
    if not text.strip():
        return []

    raw_chunks = _split_text(text, SEPARATORS, CHUNK_SIZE, CHUNK_OVERLAP)

    results: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(raw_chunks):
        # Extract page number from [PAGE N] markers
        match = PAGE_RE.search(chunk)
        if not match:
            for prev in reversed(raw_chunks[:idx]):
                m = PAGE_RE.search(prev)
                if m:
                    match = m
                    break

        page_number = int(match.group(1)) if match else None
        clean = PAGE_RE.sub("", chunk).strip()

        if not clean:
            continue

        results.append({
            "text":        clean,
            "chunk_index": idx,
            "page_number": page_number,
        })

    logger.info(f"Chunked into {len(results)} pieces (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return results