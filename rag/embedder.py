"""
rag/embedder.py
───────────────
Embedding generation with a hosted-API default and a local fallback.

PROVIDERS (env EMBEDDING_PROVIDER):
  gemini (default)  Gemini `gemini-embedding-001` via REST, 768-dim.
                    Nothing heavy is imported or loaded at boot; the only
                    cost is an HTTPS call per batch. Free tier is enough
                    for a demo.
  local             sentence-transformers/all-MiniLM-L6-v2, 384-dim.
                    The old behaviour. torch + the model are imported and
                    loaded lazily on the FIRST embedding call, never at
                    import time. Needs requirements-local-embeddings.txt.

The two providers produce incompatible vectors, so the retriever keys the
ChromaDB collection name on `embedding_signature()`; switching providers
means a re-index (see rag/retriever.py: migrate_legacy_collection).
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, List

import httpx

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini").strip().lower()

GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_EMBED_DIM = int(os.getenv("GEMINI_EMBED_DIM", "768"))
GEMINI_BATCH_SIZE = 100          # API hard limit per batchEmbedContents call
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

LOCAL_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_DIM = 384

_MODEL: Any = None  # sentence-transformers singleton (local provider only)


def embedding_signature() -> str:
    """Stable identifier for the vector space in use. Part of the collection name."""
    if EMBEDDING_PROVIDER == "local":
        return f"minilm-l6-v2-{LOCAL_DIM}"
    return f"{GEMINI_EMBED_MODEL}-{GEMINI_EMBED_DIM}"


def embedding_dim() -> int:
    return LOCAL_DIM if EMBEDDING_PROVIDER == "local" else GEMINI_EMBED_DIM


# ── Gemini (hosted) ────────────────────────────────────────────────────────────

def _gemini_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set (needed for EMBEDDING_PROVIDER=gemini). "
            "Create a free key at https://aistudio.google.com/apikey"
        )
    return key


def _normalise(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _gemini_batch(texts: List[str], task_type: str) -> List[List[float]]:
    """One batchEmbedContents call (<=100 texts). Retries on 429/5xx."""
    url = f"{GEMINI_API_BASE}/models/{GEMINI_EMBED_MODEL}:batchEmbedContents"
    body = {
        "requests": [
            {
                "model": f"models/{GEMINI_EMBED_MODEL}",
                "content": {"parts": [{"text": t}]},
                "taskType": task_type,
                "outputDimensionality": GEMINI_EMBED_DIM,
            }
            for t in texts
        ]
    }
    headers = {"x-goog-api-key": _gemini_key(), "Content-Type": "application/json"}

    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, json=body, headers=headers)
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = 2.0 * (attempt + 1)
                logger.warning(f"Gemini embeddings {resp.status_code}; retrying in {wait}s")
                time.sleep(wait)
                last_error = RuntimeError(f"Gemini embeddings HTTP {resp.status_code}: {resp.text[:200]}")
                continue
            resp.raise_for_status()
            data = resp.json()
            return [_normalise(e["values"]) for e in data["embeddings"]]
        except httpx.HTTPError as e:
            last_error = e
            time.sleep(1.0)
    raise RuntimeError(f"Gemini embeddings failed after retries: {last_error}")


def _gemini_embed(texts: List[str], task_type: str) -> List[List[float]]:
    out: List[List[float]] = []
    for i in range(0, len(texts), GEMINI_BATCH_SIZE):
        out.extend(_gemini_batch(texts[i : i + GEMINI_BATCH_SIZE], task_type))
    return out


# ── Local (sentence-transformers) ──────────────────────────────────────────────

def get_embedding_model():
    """
    Lazily load the local sentence-transformers model.

    Only used when EMBEDDING_PROVIDER=local. Imported here, inside the
    function, so `import rag.embedder` never pulls in torch.
    """
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer  # heavy: lazy on purpose

        logger.info(f"Loading local embedding model '{LOCAL_MODEL_NAME}'…")
        _MODEL = SentenceTransformer(LOCAL_MODEL_NAME)
        logger.info("Local embedding model loaded.")
    return _MODEL


def _local_embed(texts: List[str]) -> List[List[float]]:
    model = get_embedding_model()
    return model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Embed document chunks (passages)."""
    if not texts:
        return []
    if EMBEDDING_PROVIDER == "local":
        return _local_embed(texts)
    return _gemini_embed(texts, "RETRIEVAL_DOCUMENT")


def embed_query(query: str) -> List[float]:
    """Embed a single user question."""
    if EMBEDDING_PROVIDER == "local":
        return _local_embed([query])[0]
    return _gemini_embed([query], "RETRIEVAL_QUERY")[0]
