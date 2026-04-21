"""
FastAPI application — exposes the RAG pipeline as REST endpoints.

Endpoints:
  POST /query        — single-turn RAG query (returns full answer)
  POST /query/stream — streaming RAG query (SSE token stream)
  POST /ingest       — trigger document ingestion
  GET  /health       — health check
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from src.pipeline.rag_pipeline import RAGPipeline   # noqa: E402
from src.ingestion.pipeline import run_ingestion     # noqa: E402

app = FastAPI(
    title="Self-RAG API",
    description="Retrieval-Augmented Generation with self-reflection (Phase 2: basic RAG)",
    version="0.2.0",
)

# Single shared pipeline instance (lazy-initialized on first request)
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


# ── Request / Response schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    num_sources: int
    sources: list[str]

class IngestRequest(BaseModel):
    data_dir: str | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "phase": "2 — basic RAG"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        result = get_pipeline().run(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sources = list({
        doc.metadata.get("source", "unknown")
        for doc in result.source_documents
    })
    return QueryResponse(
        query=result.query,
        answer=result.answer,
        num_sources=len(result.source_documents),
        sources=sorted(sources),
    )


@app.post("/query/stream")
def query_stream(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    def token_generator():
        try:
            for token in get_pipeline().stream(request.query):
                yield token
        except Exception as e:
            yield f"\n[ERROR] {e}"

    return StreamingResponse(token_generator(), media_type="text/plain")


@app.post("/ingest")
def ingest(request: IngestRequest = IngestRequest()):
    try:
        run_ingestion(request.data_dir)
        # Reset pipeline so it picks up the new vector store
        global _pipeline
        _pipeline = None
        return {"status": "ingestion complete"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
