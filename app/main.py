from __future__ import annotations
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.indexer import load_and_index
from app.retriever import RetrievalIndex
from app.synthesizer import synthesize_answer

DOCS_DIR = os.getenv("DOCS_DIR", "docs")
TOP_K = int(os.getenv("TOP_K", "3"))

# Shared mutable state — single-process in-memory index
_state: dict = {"index": None, "stats": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _state["index"] = None
    _state["stats"] = None


app = FastAPI(title="NovaPay RAG API", lifespan=lifespan)


# ---------- request / response models ----------

class IndexResponse(BaseModel):
    message: str
    documents_indexed: int
    chunks_indexed: int
    sources: list[str]


class AskRequest(BaseModel):
    question: str


class ChunkRef(BaseModel):
    source: str
    chunk_index: int
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    confidence: str
    retrieved_chunks: list[ChunkRef]


# ---------- endpoints ----------

@app.post("/index", response_model=IndexResponse)
def index_documents():
    """Read docs/ folder, chunk documents, and build the in-memory TF-IDF index."""
    try:
        index, stats = load_and_index(DOCS_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    _state["index"] = index
    _state["stats"] = stats

    return IndexResponse(
        message="Index built successfully.",
        **stats,
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(body: AskRequest):
    """Retrieve relevant chunks and synthesize an answer from the indexed documents."""
    index: RetrievalIndex | None = _state.get("index")
    if index is None:
        raise HTTPException(
            status_code=400,
            detail="No index found. Call POST /index first.",
        )

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    results = index.query(question, top_k=TOP_K)
    answer, confidence = synthesize_answer(question, results)

    chunk_refs = [
        ChunkRef(
            source=chunk.source,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            score=round(score, 4),
        )
        for chunk, score in results
    ]

    return AskResponse(
        answer=answer,
        confidence=confidence,
        retrieved_chunks=chunk_refs,
    )
