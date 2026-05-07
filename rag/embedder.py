"""
rag/embedder.py
───────────────
Embedding generation using sentence-transformers/all-MiniLM-L6-v2.

MODEL CHOICE:
  all-MiniLM-L6-v2 is a lightweight but powerful sentence transformer that:
  - Produces 384-dimensional dense embeddings.
  - Is optimised for semantic similarity tasks.
  - Runs efficiently on CPU (important for free-tier deployments).
  - Achieves strong benchmark scores on STS and retrieval tasks.

The model is loaded once at startup and reused across all requests
(singleton pattern) to avoid repeated disk I/O.
"""

from __future__ import annotations

import logging
from typing import List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Singleton ──────────────────────────────────────────────────────────────────

_MODEL: SentenceTransformer | None = None
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model() -> SentenceTransformer:
    """
    Return the shared SentenceTransformer instance (lazy initialisation).

    The model is downloaded on first call and cached for the process lifetime.
    """
    global _MODEL
    if _MODEL is None:
        logger.info(f"Loading embedding model '{MODEL_NAME}'…")
        _MODEL = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded.")
    return _MODEL


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings.

    Args:
        texts: List of text passages to embed.

    Returns:
        List of float vectors (one per input text).
    """
    model = get_embedding_model()
    # batch_size controls GPU/CPU memory; 64 is safe for CPU
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,  # Cosine similarity via dot-product
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string.

    The query is treated identically to passage embeddings so that
    cosine similarity between query and passage embeddings is meaningful.

    Args:
        query: User question string.

    Returns:
        Float vector of length 384.
    """
    model = get_embedding_model()
    embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding.tolist()