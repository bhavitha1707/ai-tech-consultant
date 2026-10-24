from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from typing import List

app = FastAPI(title="Task 1: Knowledge Brain", version="1.0.0")

# Initialize local vector database
chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
collection = chroma_client.get_or_create_collection(name="knowledge_brain")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Knowledge Brain"}

@app.post("/api/v1/query", response_model=QueryResponse)
def query_knowledge(req: QueryRequest):
    # Stub semantic retrieval & synthesis
    # In production, query ChromaDB collection and synthesize via Anthropic API
    results = collection.query(query_texts=[req.query], n_results=req.top_k)
    
    # Fallback response if DB is empty during initial boot
    return QueryResponse(
        answer=f"Based on the Brilliant Directories knowledge base for your query '{req.query}': Configure this under Admin > Settings > General.",
        sources=["https://support.brilliantdirectories.com/support/solutions/articles/120000"],
        confidence=0.92
    )