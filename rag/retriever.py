"""
rag/retriever.py
────────────────
ChromaDB vector store integration.

Responsibilities:
  - Initialise (or connect to) a persistent ChromaDB collection.
  - Store chunk embeddings with rich metadata.
  - Retrieve the top-k most semantically similar chunks for a query.
  - Document management (list / delete).

CHROMADB PERSISTENCE:
  Embeddings are written to ./chroma_db on disk so they survive server
  restarts.  On Render (free tier) the disk is ephemeral – for truly
  persistent storage mount a Render Disk or swap ChromaDB for a managed
  service such as Pinecone/Weaviate.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from rag.embedder import embed_query

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "docmind_chunks"

_client: chromadb.Client | None = None
_collection: Any = None  # chromadb.Collection


# ── Initialisation ─────────────────────────────────────────────────────────────

def initialize_vector_store() -> None:
    """
    Create (or connect to) the persistent ChromaDB collection.

    Called once during FastAPI startup.
    """
    global _client, _collection

    _client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )

    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # Use cosine distance
    )

    count = _collection.count()
    logger.info(f"ChromaDB ready – collection '{COLLECTION_NAME}' has {count} chunks.")


def _ensure_initialized() -> None:
    if _collection is None:
        initialize_vector_store()


# ── Storage ────────────────────────────────────────────────────────────────────

def store_chunks(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    document_id: str,
    filename: str,
) -> None:
    """
    Persist chunk embeddings and metadata in ChromaDB.

    Args:
        chunks:      List of dicts from chunker.chunk_text().
        embeddings:  Parallel list of embedding vectors.
        document_id: Unique identifier for the document.
        filename:    Original upload filename (stored in metadata).
    """
    _ensure_initialized()

    ids: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    documents: List[str] = []
    vecs: List[List[float]] = []

    for chunk, embedding in zip(chunks, embeddings):
        chunk_id = f"{document_id}_{chunk['chunk_index']}"
        ids.append(chunk_id)
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

    # ChromaDB upsert is idempotent – safe to re-upload the same document
    _collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=vecs,
        metadatas=metadatas,
    )

    logger.info(f"Stored {len(ids)} chunks for document '{filename}' ({document_id}).")


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve_relevant_chunks(
    query: str,
    top_k: int = 4,
    document_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k semantically similar chunks for a query.

    Uses cosine similarity (via ChromaDB's HNSW index) between the query
    embedding and all stored chunk embeddings.

    Args:
        query:        User question string.
        top_k:        Number of chunks to return.
        document_ids: Optional whitelist of document IDs to restrict search.

    Returns:
        List of dicts with keys: text, chunk_id, metadata, score.
        Sorted by descending similarity (most relevant first).
    """
    _ensure_initialized()

    query_embedding = embed_query(query)

    where_filter = None
    if document_ids:
        where_filter = {"document_id": {"$in": document_ids}}

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, _collection.count() or 1),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    retrieved: List[Dict[str, Any]] = []
    if not results["ids"] or not results["ids"][0]:
        return retrieved

    for chunk_id, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB returns cosine *distance* (0 = identical, 2 = opposite).
        # Convert to similarity score in [0, 1].
        similarity = max(0.0, 1.0 - dist / 2.0)

        # Normalise page_number sentinel back to None
        page_num = meta.get("page_number")
        if page_num == -1:
            page_num = None
        else:
            meta["page_number"] = page_num

        retrieved.append(
            {
                "text": doc,
                "chunk_id": chunk_id,
                "metadata": meta,
                "score": similarity,
            }
        )

    return retrieved


# ── Document Management ────────────────────────────────────────────────────────

def list_documents() -> List[Dict[str, Any]]:
    """
    Return a deduplicated list of documents stored in the collection.
    """
    _ensure_initialized()

    if _collection.count() == 0:
        return []

    all_items = _collection.get(include=["metadatas"])
    seen: Dict[str, Dict[str, Any]] = {}

    for meta in all_items["metadatas"]:
        doc_id = meta.get("document_id", "unknown")
        if doc_id not in seen:
            seen[doc_id] = {
                "document_id": doc_id,
                "filename": meta.get("filename", "Unknown"),
                "chunk_count": 1,
            }
        else:
            seen[doc_id]["chunk_count"] += 1

    return list(seen.values())


def delete_document(document_id: str) -> int:
    """
    Delete all chunks belonging to a document.

    Returns:
        Number of chunks deleted.
    """
    _ensure_initialized()

    # Find all chunk IDs for this document
    results = _collection.get(
        where={"document_id": document_id},
        include=[],
    )
    ids_to_delete = results["ids"]

    if ids_to_delete:
        _collection.delete(ids=ids_to_delete)
        logger.info(f"Deleted {len(ids_to_delete)} chunks for document {document_id}.")

    return len(ids_to_delete)


def get_collection_stats() -> Dict[str, Any]:
    """Return basic stats about the vector store."""
    _ensure_initialized()
    return {
        "total_chunks": _collection.count(),
        "collection_name": COLLECTION_NAME,
        "storage_path": CHROMA_PATH,
    }