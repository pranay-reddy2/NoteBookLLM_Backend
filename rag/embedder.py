"""
rag/embedder.py
───────────────
Embeddings via fastembed 0.7.x — no torch, no numpy version conflicts.
Uses the ONNX-based all-MiniLM-L6-v2 model (384 dims, L2-normalised).
"""

from __future__ import annotations
import logging
from typing import List

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def get_embedding_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        logger.info(f"Loading fastembed model '{MODEL_NAME}'...")
        _model = TextEmbedding(model_name=MODEL_NAME)
        logger.info("Embedding model ready.")
    return _model


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate normalised embeddings for a list of texts."""
    model = get_embedding_model()
    return [emb.tolist() for emb in model.embed(texts)]


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    model = get_embedding_model()
    return list(model.embed([query]))[0].tolist()