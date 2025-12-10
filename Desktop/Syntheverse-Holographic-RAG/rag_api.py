"""
RAG API Server
Provides REST API access to the RAG system using local embeddings.
No API calls required for processing - uses local embeddings and optional local LLM.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
import time


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    min_score: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    query: str
    processing_time: float


class RAGEngine:
    """
    RAG Engine using local embeddings.
    No API calls required - all processing is local.
    """
    
    def __init__(self,
                 embeddings_dir: str = "./vectorized/embeddings",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG engine.
        
        Args:
            embeddings_dir: Directory containing vectorized embedding JSON files
            embedding_model: HuggingFace embedding model name
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.embedding_model = embedding_model
        
        # Load local embedding model
        print(f"Loading local embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✓ Embedding model loaded")
        
        # Load all vectorized chunks
        print("Loading vectorized embeddings...")
        self.chunks = self._load_all_chunks()
        print(f"✓ Loaded {len(self.chunks)} chunks from {len(self.chunks_by_pdf)} PDFs")
    
    def _load_all_chunks(self) -> List[Dict]:
        """Load all vectorized chunks from JSON files."""
        chunks = []
        self.chunks_by_pdf = {}
        
        if not self.embeddings_dir.exists():
            raise ValueError(f"Embeddings directory not found: {self.embeddings_dir}")
        
        json_files = list(self.embeddings_dir.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    file_chunks = json.load(f)
                
                pdf_name = json_file.stem
                self.chunks_by_pdf[pdf_name] = file_chunks
                
                for chunk in file_chunks:
                    chunk['pdf_filename'] = pdf_name
                    chunks.append(chunk)
            except Exception as e:
                print(f"Warning: Error loading {json_file}: {e}")
                continue
        
        return chunks
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict]:
        """
        Search for relevant chunks using local embeddings.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
        
        Returns:
            List of relevant chunks with similarity scores
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        query_vec = np.array(query_embedding)
        
        # Calculate similarities
        results = []
        for chunk in self.chunks:
            chunk_embedding = np.array(chunk['embedding'])
            similarity = self._cosine_similarity(query_vec, chunk_embedding)
            
            if similarity >= min_score:
                results.append({
                    'text': chunk['text'],
                    'score': float(similarity),
                    'metadata': chunk.get('metadata', {}),
                    'pdf_filename': chunk.get('pdf_filename', 'Unknown'),
                    'chunk_index': chunk.get('chunk_index', 0)
                })
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def generate_answer(self, query: str, relevant_chunks: List[Dict]) -> str:
        """
        Generate answer from relevant chunks.
        Uses simple template-based generation (no LLM API calls).
        
        Args:
            query: Original query
            relevant_chunks: List of relevant chunks with scores
        
        Returns:
            Generated answer
        """
        if not relevant_chunks:
            return "I couldn't find any relevant information to answer your question."
        
        # Combine top chunks
        context = "\n\n".join([
            f"[Source: {chunk['pdf_filename']}]\n{chunk['text']}"
            for chunk in relevant_chunks[:3]  # Use top 3 chunks
        ])
        
        # Simple template-based answer generation
        answer = f"""Based on the available documents, here's what I found:

{context}

This information is derived from the following sources:
{', '.join(set(chunk['pdf_filename'] for chunk in relevant_chunks[:3]))}"""
        
        return answer
    
    def query(self, query: str, top_k: int = 5, min_score: float = 0.0) -> Dict:
        """
        Complete RAG query: search + generate answer.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            min_score: Minimum similarity score
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = time.time()
        
        # Search for relevant chunks
        relevant_chunks = self.search(query, top_k=top_k, min_score=min_score)
        
        # Generate answer
        answer = self.generate_answer(query, relevant_chunks)
        
        processing_time = time.time() - start_time
        
        return {
            'answer': answer,
            'sources': relevant_chunks,
            'query': query,
            'processing_time': processing_time,
            'num_sources': len(relevant_chunks)
        }


# Initialize FastAPI app
app = FastAPI(
    title="Syntheverse RAG API",
    description="RAG API using local embeddings - no API calls required",
    version="1.0.0"
)

# Enable CORS for UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your UI domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (UI)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize RAG engine
rag_engine = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup."""
    global rag_engine
    try:
        rag_engine = RAGEngine(
            embeddings_dir="./vectorized/embeddings",
            embedding_model="all-MiniLM-L6-v2"
        )
        print("RAG Engine initialized successfully")
    except Exception as e:
        print(f"Error initializing RAG engine: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint - serve UI."""
    ui_path = static_dir / "index.html"
    if ui_path.exists():
        return FileResponse(str(ui_path))
    return {
        "message": "Syntheverse RAG API",
        "status": "running",
        "endpoints": {
            "/query": "POST - Query the RAG system",
            "/health": "GET - Health check",
            "/stats": "GET - System statistics",
            "/ui": "GET - Web UI"
        }
    }


@app.get("/ui")
async def ui():
    """Serve the web UI."""
    ui_path = static_dir / "index.html"
    if ui_path.exists():
        return FileResponse(str(ui_path))
    raise HTTPException(status_code=404, detail="UI not found")


@app.get("/health")
async def health():
    """Health check endpoint."""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    return {
        "status": "healthy",
        "chunks_loaded": len(rag_engine.chunks),
        "pdfs_loaded": len(rag_engine.chunks_by_pdf)
    }


@app.get("/stats")
async def stats():
    """Get system statistics."""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    return {
        "total_chunks": len(rag_engine.chunks),
        "total_pdfs": len(rag_engine.chunks_by_pdf),
        "embedding_model": rag_engine.embedding_model,
        "pdfs": list(rag_engine.chunks_by_pdf.keys())
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system.
    
    Args:
        request: Query request with query text and parameters
    
    Returns:
        Query response with answer and sources
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        result = rag_engine.query(
            query=request.query,
            top_k=request.top_k,
            min_score=request.min_score
        )
        
        return QueryResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/search")
async def search(request: QueryRequest):
    """
    Search for relevant chunks (without answer generation).
    
    Args:
        request: Search request with query text and parameters
    
    Returns:
        List of relevant chunks with scores
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        results = rag_engine.search(
            query=request.query,
            top_k=request.top_k,
            min_score=request.min_score
        )
        
        return {
            "query": request.query,
            "results": results,
            "count": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 80)
    print("Starting Syntheverse RAG API Server")
    print("=" * 80)
    print("API will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("=" * 80)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

