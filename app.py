"""
DocMind RAG Application - FastAPI Backend
Main application entry point with all API endpoints.
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
    retrieve_relevant_chunks,
    get_collection_stats,
    delete_document,
    list_documents,
)
from rag.generator import generate_answer

# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DocMind RAG API",
    description="Production-quality RAG application API powered by LangChain + ChromaDB + Gemini",
    version="1.0.0",
)

# Allow frontend origins (update for production)
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,https://*.vercel.app",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize vector store on startup
@app.on_event("startup")
async def startup_event():
    """Initialize embedding model and vector store on startup."""
    print("🚀 Starting DocMind RAG API...")
    get_embedding_model()   # Pre-load model into memory
    initialize_vector_store()
    print("✅ Vector store initialized")


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
    """Health check endpoint for deployment monitoring."""
    stats = get_collection_stats()
    return {
        "status": "healthy",
        "service": "DocMind RAG API",
        "version": "1.0.0",
        "vector_store": stats,
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT document.
    
    Pipeline:
    1. Save file to disk
    2. Extract text content
    3. Chunk text with overlap
    4. Generate embeddings
    5. Store in ChromaDB
    """
    # Validate file type
    allowed_types = {
        "application/pdf": ".pdf",
        "text/plain": ".txt",
        "application/octet-stream": None,  # Allow generic binary (some clients send this)
    }
    
    filename = file.filename or "document"
    ext = Path(filename).suffix.lower()
    
    if ext not in [".pdf", ".txt"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Please upload PDF or TXT files.",
        )

    # Generate unique document ID
    document_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{document_id}{ext}"

    try:
        # Save file
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(content) > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB.")

        with open(save_path, "wb") as f:
            f.write(content)

        # Extract text
        extracted_text = extract_text_from_file(str(save_path), ext)
        if not extracted_text.strip():
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from the document. Ensure it is not scanned/image-only.",
            )

        # Chunk text
        chunks = chunk_text(extracted_text)
        if not chunks:
            raise HTTPException(status_code=422, detail="Document produced no usable text chunks.")

        # Generate embeddings & store
        embeddings = generate_embeddings([c["text"] for c in chunks])
        store_chunks(
            chunks=chunks,
            embeddings=embeddings,
            document_id=document_id,
            filename=filename,
        )

        return UploadResponse(
            document_id=document_id,
            filename=filename,
            chunks_created=len(chunks),
            message=f"Successfully processed '{filename}' into {len(chunks)} chunks.",
        )

    except HTTPException:
        raise
    except Exception as e:
        # Clean up saved file on error
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question against the uploaded documents.
    
    Pipeline:
    1. Embed the user query
    2. Semantic similarity search in ChromaDB
    3. Retrieve top-k relevant chunks
    4. Inject context into Gemini prompt
    5. Return grounded answer + sources
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    start_time = time.time()

    try:
        # Retrieve relevant chunks via semantic search
        retrieved = retrieve_relevant_chunks(
            query=request.query,
            top_k=request.top_k,
            document_ids=request.document_ids,
        )

        if not retrieved:
            return AskResponse(
                answer="I could not find this information in the document. Please upload a relevant document first.",
                sources=[],
                query=request.query,
                processing_time_ms=round((time.time() - start_time) * 1000, 2),
            )

        # Generate grounded answer via Gemini
        answer = generate_answer(
            query=request.query,
            retrieved_chunks=retrieved,
        )

        # Format source chunks for response
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

        return AskResponse(
            answer=answer,
            sources=sources,
            query=request.query,
            processing_time_ms=round((time.time() - start_time) * 1000, 2),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")


@app.get("/documents", response_model=List[DocumentInfo])
async def get_documents():
    """List all uploaded documents stored in the vector database."""
    try:
        docs = list_documents()
        return [DocumentInfo(**doc) for doc in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def remove_document(document_id: str):
    """Delete a document and its embeddings from the vector store."""
    try:
        deleted_count = delete_document(document_id)
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