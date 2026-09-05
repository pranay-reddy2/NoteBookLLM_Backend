"""
rag/retriever.py
────────────────
ChromaDB vector store integration.

Responsibilities:
  - Lazily open a persistent ChromaDB collection (first real request, never at boot).
  - Store chunk embeddings with rich metadata.
  - Retrieve the top-k most semantically similar chunks for a query.
  - Document management (list / delete).
  - One-time migration of the legacy MiniLM collection to the current
    embedding space (see migrate_legacy_collection).

COLLECTION NAMING:
  Vectors from different embedding models are not comparable, so the
  collection name carries the embedding signature, e.g.
  docmind_gemini-embedding-001-768. The original collection was
  docmind_chunks (MiniLM, 384-dim); it is migrated once and left in place.

CHROMADB PERSISTENCE:
  Embeddings are written to ./chroma_db on disk so they survive server
  restarts. On Render this path is a mounted persistent disk (render.yaml).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional

from rag.embedder import embed_query, embedding_signature, generate_embeddings

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
LEGACY_COLLECTION_NAME = "docmind_chunks"           # MiniLM-L6-v2, 384-dim
COLLECTION_NAME = f"docmind_{embedding_signature()}"

_client: Any = None       # chromadb.PersistentClient
_collection: Any = None   # chromadb.Collection
_lock = threading.Lock()


# ── Initialisation ─────────────────────────────────────────────────────────────

def initialize_vector_store() -> None:
    """
    Open (or create) the persistent ChromaDB collection for the current
    embedding space, migrating the legacy collection if this one is empty.

    chromadb is imported here, not at module level: it is a multi-second
    import and /health must not pay for it.
    """
    global _client, _collection

    import chromadb
    from chromadb.config import Settings

    _client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"ChromaDB ready - collection '{COLLECTION_NAME}' has {_collection.count()} chunks.")

    if _collection.count() == 0 and os.getenv("AUTO_MIGRATE_LEGACY", "1") == "1":
        migrate_legacy_collection()


def _ensure_initialized() -> None:
    if _collection is None:
        with _lock:
            if _collection is None:
                initialize_vector_store()


def is_initialized() -> bool:
    return _collection is not None


# ── Legacy migration ───────────────────────────────────────────────────────────

def migrate_legacy_collection(batch_size: int = 64) -> int:
    """
    Re-embed every chunk from the legacy MiniLM collection into the current
    collection using the active embedding provider. Runs once: afterwards the
    new collection is non-empty and initialize_vector_store() skips this.

    Returns the number of chunks migrated (0 if nothing to do).
    """
    assert _client is not None and _collection is not None
    if COLLECTION_NAME == LEGACY_COLLECTION_NAME:
        return 0

    try:
        legacy = _client.get_collection(LEGACY_COLLECTION_NAME)
    except Exception:  # noqa: BLE001 - chroma raises different types per version
        return 0

    total = legacy.count()
    if total == 0:
        return 0

    logger.info(f"Migrating {total} chunks from '{LEGACY_COLLECTION_NAME}' -> '{COLLECTION_NAME}' ...")
    items = legacy.get(include=["documents", "metadatas"])
    ids, docs, metas = items["ids"], items["documents"], items["metadatas"]

    migrated = 0
    for i in range(0, total, batch_size):
        b_ids, b_docs, b_metas = ids[i:i + batch_size], docs[i:i + batch_size], metas[i:i + batch_size]
        vecs = generate_embeddings(b_docs)
        _collection.upsert(ids=b_ids, documents=b_docs, embeddings=vecs, metadatas=b_metas)
        migrated += len(b_ids)
        logger.info(f"  migrated {migrated}/{total}")

    logger.info("Migration complete.")
    return migrated


# ── Storage ────────────────────────────────────────────────────────────────────

def store_chunks(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    document_id: str,
    filename: str,
) -> None:
    """Persist chunk embeddings and metadata in ChromaDB."""
    _ensure_initialized()

    ids: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    documents: List[str] = []
    vecs: List[List[float]] = []

    for chunk, embedding in zip(chunks, embeddings):
        ids.append(f"{document_id}_{chunk['chunk_index']}")
        documents.append(chunk["text"])
        vecs.append(embedding)
        metadatas.append(
            {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": chunk["chunk_index"],
                "page_number": chunk.get("page_number") or -1,  # ChromaDB requires int, not None
            }
        )

    # ChromaDB upsert is idempotent - safe to re-upload the same document
    _collection.upsert(ids=ids, documents=documents, embeddings=vecs, metadatas=metadatas)
    logger.info(f"Stored {len(ids)} chunks for document '{filename}' ({document_id}).")


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve_relevant_chunks(
    query: str,
    top_k: int = 4,
    document_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k semantically similar chunks for a query.

    Returns a list of dicts with keys: text, chunk_id, metadata, score,
    sorted by descending similarity.
    """
    _ensure_initialized()

    count = _collection.count()
    if count == 0:
        return []

    query_embedding = embed_query(query)

    where_filter = {"document_id": {"$in": document_ids}} if document_ids else None

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    retrieved: List[Dict[str, Any]] = []
    if not results["ids"] or not results["ids"][0]:
        return retrieved

    for chunk_id, doc, meta, dist in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        # ChromaDB returns cosine *distance* (0 = identical, 2 = opposite) -> similarity in [0, 1].
        similarity = max(0.0, 1.0 - dist / 2.0)

        if meta.get("page_number") == -1:
            meta["page_number"] = None

        retrieved.append({"text": doc, "chunk_id": chunk_id, "metadata": meta, "score": similarity})

    return retrieved


# ── Document Management ────────────────────────────────────────────────────────

def list_documents() -> List[Dict[str, Any]]:
    """Return a deduplicated list of documents stored in the collection."""
    _ensure_initialized()

    if _collection.count() == 0:
        return []

    all_items = _collection.get(include=["metadatas"])
    seen: Dict[str, Dict[str, Any]] = {}

    for meta in all_items["metadatas"]:
        doc_id = meta.get("document_id", "unknown")
        if doc_id not in seen:
            seen[doc_id] = {"document_id": doc_id, "filename": meta.get("filename", "Unknown"), "chunk_count": 1}
        else:
            seen[doc_id]["chunk_count"] += 1

    return list(seen.values())


def delete_document(document_id: str) -> int:
    """Delete all chunks belonging to a document. Returns the number deleted."""
    _ensure_initialized()

    results = _collection.get(where={"document_id": document_id}, include=[])
    ids_to_delete = results["ids"]

    if ids_to_delete:
        _collection.delete(ids=ids_to_delete)
        logger.info(f"Deleted {len(ids_to_delete)} chunks for document {document_id}.")

    return len(ids_to_delete)


def get_collection_stats() -> Dict[str, Any]:
    """Basic stats about the vector store. Opens Chroma if needed - not for /health."""
    _ensure_initialized()
    return {
        "total_chunks": _collection.count(),
        "collection_name": COLLECTION_NAME,
        "embedding": embedding_signature(),
        "storage_path": CHROMA_PATH,
    }
