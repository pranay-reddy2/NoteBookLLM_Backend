"""
DocMind RAG Application - FastAPI Backend
Main application entry point with all API endpoints.

Boot is deliberately light: nothing touches the embedding provider, the LLM,
or ChromaDB until the first real request, so the process serves /health
within a couple of seconds of starting (important on Render's free tier).
"""

import os
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

load_dotenv()

from rag.loader import extract_text_from_file          # noqa: E402
from rag.chunker import chunk_text                      # noqa: E402
from rag.embedder import generate_embeddings            # noqa: E402
from rag.retriever import (                             # noqa: E402
    store_chunks,
    retrieve_relevant_chunks,
    get_collection_stats,
    delete_document,
    list_documents,
)
from rag.generator import generate_answer, provider_chain   # noqa: E402

# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DocMind RAG API",
    description="RAG application API powered by ChromaDB + hosted embeddings + free-tier LLMs",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

PROCESS_STARTED_AT = time.time()


@app.on_event("startup")
async def startup_event():
    # Intentionally does NOT pre-load embeddings or Chroma: see module docstring.
    print("DocMind RAG API up. Heavy components load lazily on first request.")


# ── Demo guard rails: daily request cap + response cache ──────────────────────

DAILY_REQUEST_CAP = int(os.getenv("DAILY_REQUEST_CAP", "200"))
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "pranayreddy672@gmail.com")
LIMIT_MESSAGE = f"Demo limit reached for today — email {CONTACT_EMAIL} and I'll reset it."

_cap_lock = threading.Lock()
_cap_day: Optional[str] = None
_cap_count = 0


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _consume_daily_quota() -> bool:
    """Count one paid-API request against today's cap. False if the cap is hit."""
    global _cap_day, _cap_count
    with _cap_lock:
        today = _today()
        if _cap_day != today:
            _cap_day, _cap_count = today, 0
        if _cap_count >= DAILY_REQUEST_CAP:
            return False
        _cap_count += 1
        return True


def _quota_snapshot() -> dict:
    with _cap_lock:
        used = _cap_count if _cap_day == _today() else 0
    return {"cap": DAILY_REQUEST_CAP, "used_today": used, "remaining": max(0, DAILY_REQUEST_CAP - used)}


ANSWER_CACHE_SIZE = int(os.getenv("ANSWER_CACHE_SIZE", "500"))
_answer_cache: "OrderedDict[Tuple, dict]" = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(query: str, document_ids: Optional[List[str]], top_k: int) -> Tuple:
    return (tuple(sorted(document_ids or [])), " ".join(query.lower().split()), top_k)


def _cache_get(key: Tuple) -> Optional[dict]:
    with _cache_lock:
        hit = _answer_cache.get(key)
        if hit is not None:
            _answer_cache.move_to_end(key)
        return hit


def _cache_put(key: Tuple, value: dict) -> None:
    with _cache_lock:
        _answer_cache[key] = value
        _answer_cache.move_to_end(key)
        while len(_answer_cache) > ANSWER_CACHE_SIZE:
            _answer_cache.popitem(last=False)


def _cache_invalidate() -> None:
    with _cache_lock:
        _answer_cache.clear()


# ── Request/Response Models ────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None  # Filter by specific documents
    top_k: int = 4


class SourceChunk(BaseModel):
    content: str
    source: str
    chunk_id: str
    page_number: Optional[int] = None
    similarity_score: float


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    query: str
    processing_time_ms: float
    cached: bool = False
    limit_reached: bool = False


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    message: str


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    uploaded_at: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Liveness probe. Touches nothing: no LLM, no embeddings, no Chroma."""
    return {"ok": True}


@app.get("/stats")
async def stats():
    """Deeper status (opens Chroma if it isn't open yet). Not for uptime pingers."""
    return {
        "service": "DocMind RAG API",
        "version": app.version,
        "uptime_s": round(time.time() - PROCESS_STARTED_AT, 1),
        "llm_providers": provider_chain(),
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "gemini"),
        "quota": _quota_snapshot(),
        "cache_entries": len(_answer_cache),
        "vector_store": get_collection_stats(),
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT document.

    Pipeline: save -> extract text -> chunk -> embed (hosted API) -> ChromaDB.
    Counts against the daily cap because embedding uses the same free quota.
    """
    filename = file.filename or "document"
    ext = Path(filename).suffix.lower()

    if ext not in [".pdf", ".txt"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Please upload PDF or TXT files.",
        )

    if not _consume_daily_quota():
        return JSONResponse(status_code=429, content={"detail": LIMIT_MESSAGE, "limit_reached": True})

    document_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{document_id}{ext}"

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(content) > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB.")

        with open(save_path, "wb") as f:
            f.write(content)

        extracted_text = extract_text_from_file(str(save_path), ext)
        if not extracted_text.strip():
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from the document. Ensure it is not scanned/image-only.",
            )

        chunks = chunk_text(extracted_text)
        if not chunks:
            raise HTTPException(status_code=422, detail="Document produced no usable text chunks.")

        embeddings = generate_embeddings([c["text"] for c in chunks])
        store_chunks(chunks=chunks, embeddings=embeddings, document_id=document_id, filename=filename)
        _cache_invalidate()  # answers over "all documents" may change

        return UploadResponse(
            document_id=document_id,
            filename=filename,
            chunks_created=len(chunks),
            message=f"Successfully processed '{filename}' into {len(chunks)} chunks.",
        )

    except HTTPException:
        raise
    except Exception as e:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question against the uploaded documents.

    Pipeline: cache lookup -> embed query -> ChromaDB top-k -> LLM -> answer + sources.
    Repeated (documents, question) pairs are served from the in-memory cache
    and do not count against the daily cap.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    start_time = time.time()
    key = _cache_key(request.query, request.document_ids, request.top_k)

    cached = _cache_get(key)
    if cached is not None:
        return AskResponse(**{**cached, "cached": True,
                              "processing_time_ms": round((time.time() - start_time) * 1000, 2)})

    if not _consume_daily_quota():
        return AskResponse(
            answer=LIMIT_MESSAGE, sources=[], query=request.query,
            processing_time_ms=round((time.time() - start_time) * 1000, 2), limit_reached=True,
        )

    try:
        retrieved = retrieve_relevant_chunks(
            query=request.query, top_k=request.top_k, document_ids=request.document_ids,
        )

        if not retrieved:
            return AskResponse(
                answer="I could not find this information in the document. Please upload a relevant document first.",
                sources=[], query=request.query,
                processing_time_ms=round((time.time() - start_time) * 1000, 2),
            )

        answer = generate_answer(query=request.query, retrieved_chunks=retrieved)

        sources = [
            SourceChunk(
                content=chunk["text"],
                source=chunk["metadata"].get("filename", "Unknown"),
                chunk_id=chunk["chunk_id"],
                page_number=chunk["metadata"].get("page_number"),
                similarity_score=round(chunk["score"], 4),
            )
            for chunk in retrieved
        ]

        payload = {
            "answer": answer,
            "sources": [s.model_dump() for s in sources],
            "query": request.query,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
        }
        _cache_put(key, payload)
        return AskResponse(**payload)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")


@app.get("/documents", response_model=List[DocumentInfo])
async def get_documents():
    """List all uploaded documents stored in the vector database."""
    try:
        return [DocumentInfo(**doc) for doc in list_documents()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def remove_document(document_id: str):
    """Delete a document and its embeddings from the vector store."""
    try:
        deleted_count = delete_document(document_id)
        _cache_invalidate()
        return {"message": f"Deleted document {document_id} ({deleted_count} chunks removed)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "production") == "development",
    )
