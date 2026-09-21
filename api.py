from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graph import format_sources
from main import run_rag


app = FastAPI(title="Agentic RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(request: AskRequest):
    result = run_rag(request.question)

    sources = []
    if result["documents_relevant"]:
        sources = format_sources(result["documents"])

    return {
        "answer": result["answer"],
        "sources": sources,
        "trace": {
            "original_question": result["original_question"],
            "final_search_query": result["search_query"],
            "retrieved_count": 4,
            "reranked_count": len(result["documents"]),
            "documents_relevant": result["documents_relevant"],
            "answer_supported": result["answer_supported"],
            "retrieval_retry_count": result["retrieval_retry_count"],
            "generation_retry_count": result["generation_retry_count"],
        },
    }
