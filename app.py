"""
DocMind RAG Application - FastAPI Backend
Advanced RAG 2 Pipeline: Query Translation → HyDE → Multi-query Retrieval
                          → LLM Re-ranking → LLM Judge → Corrective RAG → Generate
"""

import os
import uuid
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from rag.loader import extract_text_from_file
from rag.chunker import chunk_text
from rag.embedder import generate_embeddings, get_embedding_model
from rag.retriever import (
    initialize_vector_store,
    store_chunks,
    get_collection_stats,
    delete_document,
    list_documents,
)
from rag.generator import generate_answer

# ── Advanced RAG 2 imports ─────────────────────────────────────────────────────
from rag.query_processor import process_query_advanced
from rag.advanced_retriever import multi_query_retrieve
from rag.reranker import post_retrieval_pipeline

# ── App Setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DocMind RAG API",
    description="Advanced RAG 2 pipeline: Query Translation + HyDE + Re-ranking + CRAG",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.on_event("startup")
async def startup_event():
    print("🚀 Starting DocMind RAG API v2 (Advanced RAG)...")
    get_embedding_model()
    initialize_vector_store()
    print("✅ Vector store initialized")


# ── Request/Response Models ────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    top_k: int = 4

    # Advanced RAG 2 toggles — all ON by default
    use_query_translation: bool = True      # Fix typos, rephrase
    use_decomposition: bool = True          # Sub-query splitting
    use_hyde: bool = True                   # HyDE embeddings
    use_expansion: bool = True             # Multi-query expansion
    use_reranking: bool = True             # LLM cross-encoder re-ranking
    use_llm_judge: bool = True             # LLM relevance filtering
    use_crag: bool = True                  # Corrective RAG


class SourceChunk(BaseModel):
    content: str
    source: str
    chunk_id: str
    page_number: Optional[int] = None
    similarity_score: float
    llm_relevance_score: Optional[float] = None  # NEW: LLM judge score


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    query: str
    translated_query: Optional[str] = None         # NEW
    sub_queries: Optional[List[str]] = None         # NEW
    crag_status: Optional[str] = None               # NEW: confident/uncertain/irrelevant
    confidence_score: Optional[float] = None        # NEW
    processing_time_ms: float
    pipeline_stats: Optional[dict] = None           # NEW: debug info


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


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    stats = get_collection_stats()
    return {
        "status": "healthy",
        "service": "DocMind RAG API",
        "version": "2.0.0",
        "rag_version": "Advanced RAG 2 (Query Translation + HyDE + Re-ranking + CRAG)",
        "vector_store": stats,
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT document.
    Pipeline: Save → Extract → Chunk → Embed → Store in ChromaDB
    """
    filename = file.filename or "document"
    ext = Path(filename).suffix.lower()

    if ext not in [".pdf", ".txt"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Please upload PDF or TXT files.",
        )

    document_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{document_id}{ext}"

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum 50MB.")

        with open(save_path, "wb") as f:
            f.write(content)

        extracted_text = extract_text_from_file(str(save_path), ext)
        if not extracted_text.strip():
            raise HTTPException(
                status_code=422,
                detail="Could not extract text. Ensure document is not image-only.",
            )

        chunks = chunk_text(extracted_text)
        if not chunks:
            raise HTTPException(status_code=422, detail="Document produced no usable chunks.")

        embeddings = generate_embeddings([c["text"] for c in chunks])
        store_chunks(chunks=chunks, embeddings=embeddings, document_id=document_id, filename=filename)

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
    Advanced RAG 2 Query Pipeline:

    1. Query Translation    → Fix typos, rephrase for clarity
    2. Sub-query Decomposition → Split complex queries into atomic parts
    3. HyDE                 → Generate hypothetical doc for better embedding match
    4. Multi-query Expansion → Generate query variants for higher recall
    5. Multi-query Retrieval → Retrieve + merge chunks from all search texts
    6. LLM Re-ranking       → Cross-encoder style scoring of (query, chunk) pairs
    7. LLM Judge            → Filter irrelevant chunks below threshold
    8. Corrective RAG       → Decide confident / uncertain / irrelevant
    9. Generate             → Grounded answer via Gemini with filtered context
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    start_time = time.time()
    pipeline_stats = {"stages": {}}

    try:
        # ── Stage 1-4: Query Processing ────────────────────────────────────────
        t0 = time.time()

        if any([request.use_query_translation, request.use_decomposition,
                request.use_hyde, request.use_expansion]):
            query_result = process_query_advanced(
                raw_query=request.query,
                use_hyde=request.use_hyde,
                use_expansion=request.use_expansion,
                use_decomposition=request.use_decomposition,
            )
            translated_query = query_result["translated_query"]
            sub_queries = query_result["sub_queries"]
            search_texts = query_result["all_search_texts"]
        else:
            # All advanced features disabled — basic RAG fallback
            translated_query = request.query
            sub_queries = [request.query]
            search_texts = [request.query]

        pipeline_stats["stages"]["query_processing_ms"] = round((time.time() - t0) * 1000, 1)
        pipeline_stats["search_texts_count"] = len(search_texts)

        # ── Stage 5: Multi-query Retrieval ─────────────────────────────────────
        t0 = time.time()

        raw_chunks = multi_query_retrieve(
            search_texts=search_texts,
            top_k_per_query=request.top_k,
            document_ids=request.document_ids,
            max_total_chunks=request.top_k * 3,  # Retrieve more, re-rank down
        )

        pipeline_stats["stages"]["retrieval_ms"] = round((time.time() - t0) * 1000, 1)
        pipeline_stats["raw_chunks_retrieved"] = len(raw_chunks)

        if not raw_chunks:
            return AskResponse(
                answer="I could not find relevant information. Please upload a relevant document first.",
                sources=[],
                query=request.query,
                translated_query=translated_query,
                sub_queries=sub_queries,
                crag_status="irrelevant",
                confidence_score=0.0,
                processing_time_ms=round((time.time() - start_time) * 1000, 2),
                pipeline_stats=pipeline_stats,
            )

        # ── Stages 6-8: Re-ranking + Judge + CRAG ──────────────────────────────
        t0 = time.time()

        crag_result = post_retrieval_pipeline(
            query=translated_query,
            raw_chunks=raw_chunks,
            use_reranking=request.use_reranking,
            use_judge=request.use_llm_judge,
            use_crag=request.use_crag,
        )

        pipeline_stats["stages"]["rerank_judge_crag_ms"] = round((time.time() - t0) * 1000, 1)
        pipeline_stats["crag_status"] = crag_result.status
        pipeline_stats["chunks_after_filtering"] = len(crag_result.accepted_chunks)

        # CRAG says no relevant chunks — return fallback
        if not crag_result.should_answer() or not crag_result.accepted_chunks:
            return AskResponse(
                answer=crag_result.fallback_message or "I could not find relevant information in your documents.",
                sources=[],
                query=request.query,
                translated_query=translated_query,
                sub_queries=sub_queries,
                crag_status=crag_result.status,
                confidence_score=crag_result.confidence_score,
                processing_time_ms=round((time.time() - start_time) * 1000, 2),
                pipeline_stats=pipeline_stats,
            )

        # ── Stage 9: Generate Answer ────────────────────────────────────────────
        t0 = time.time()

        final_chunks = crag_result.accepted_chunks[:request.top_k]  # Cap for token budget

        # Prepend uncertainty note if CRAG is uncertain
        prefix = ""
        if crag_result.status == "uncertain" and crag_result.fallback_message:
            prefix = crag_result.fallback_message + "\n\n"

        answer = generate_answer(
            query=translated_query,  # Use translated (cleaner) query
            retrieved_chunks=final_chunks,
        )

        pipeline_stats["stages"]["generation_ms"] = round((time.time() - t0) * 1000, 1)

        sources = [
            SourceChunk(
                content=chunk["text"],
                source=chunk["metadata"].get("filename", "Unknown"),
                chunk_id=chunk["chunk_id"],
                page_number=chunk["metadata"].get("page_number"),
                similarity_score=round(chunk["score"], 4),
                llm_relevance_score=chunk.get("llm_relevance_score"),
            )
            for chunk in final_chunks
        ]

        return AskResponse(
            answer=prefix + answer,
            sources=sources,
            query=request.query,
            translated_query=translated_query if translated_query != request.query else None,
            sub_queries=sub_queries if len(sub_queries) > 1 else None,
            crag_status=crag_result.status,
            confidence_score=round(crag_result.confidence_score, 3),
            processing_time_ms=round((time.time() - start_time) * 1000, 2),
            pipeline_stats=pipeline_stats,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")


@app.get("/documents", response_model=List[DocumentInfo])
async def get_documents():
    try:
        docs = list_documents()
        return [DocumentInfo(**doc) for doc in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def remove_document(document_id: str):
    try:
        deleted_count = delete_document(document_id)
        return {"message": f"Deleted document {document_id} ({deleted_count} chunks removed)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "production") == "development",
    )