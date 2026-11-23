from pathlib import Path
from typing import Any, Literal
import logging
import time
import sqlite3

from fastapi import FastAPI, HTTPException
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import CHECKPOINT_DB_PATH
from app.config import DEFAULT_CHROMA_DIR
from app.schemas import IngestRequest, SearchRequest, ChatRequest, ResumeRequest
from app.graph import create_graph
from app.ingestion import chunk_documents, load_documents
from app.retrieval import create_bm25_retriever, lexical_search, rrf, semantic_search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Roads and Transport Assistant")
app.state.documents = []
app.state.vector_store = None
app.state.graph = None
app.state.checkpoint_connection = sqlite3.connect(
    CHECKPOINT_DB_PATH, check_same_thread=False
)
app.state.checkpointer = SqliteSaver(app.state.checkpoint_connection)
app.state.checkpointer.setup()


def _serialize_document(document):
    return {"content": document.page_content, "metadata": document.metadata}


def _require_index():
    if not app.state.documents or app.state.vector_store is None:
        raise HTTPException(
            status_code=409, detail="No corpus is indexed. Call /ingest first."
        )


def _search(request: SearchRequest):
    _require_index()
    candidate_k = max(request.top_k * 3, 10)

    lexical_results = []
    semantic_results = []
    if request.mode in {"lexical", "hybrid"}:
        retriever = create_bm25_retriever(app.state.documents, k=candidate_k)
        lexical_results = lexical_search(
            request.query,
            retriever,
            k=candidate_k,
            filters=request.filters,
        )
    if request.mode in {"semantic", "hybrid"}:
        semantic_results = semantic_search(
            request.query,
            app.state.vector_store,
            k=candidate_k,
            filters=request.filters,
        )

    if request.mode == "hybrid":
        results, _ = rrf(lexical_results, semantic_results)
    else:
        results = lexical_results or semantic_results
    return results[: request.top_k]


@app.get("/health")  # report app readiness & index availability
def health():
    return {"status": "ok", "indexed_chunks": len(app.state.documents)}


@app.post("/ingest")  # ingest corpus, return indexed, skipped, and failed counts
def ingest(request: IngestRequest):
    manifest_path = Path(__file__).resolve().parents[1] / "data" / "corpus_manifest.csv"
    loaded = load_documents(
        request.source_folder,
        request.processed_folder,
        manifest_path=str(manifest_path),
    )
    chunks = chunk_documents(loaded, max_tokens=request.max_tokens)
    if not chunks:
        raise HTTPException(
            status_code=400, detail="No chunks were created from the supplied corpus."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 8},
    )
    vector_store = Chroma(
        collection_name="road_and_transport_collection",
        embedding_function=embeddings,
        persist_directory=str(DEFAULT_CHROMA_DIR),
    )

    # Chroma persists across application runs. Reuse its stored chunk IDs so
    # repeated ingestion only embeds and inserts chunks that are not indexed yet.
    existing_ids = set(vector_store._collection.get(include=["metadatas"])["ids"])
    new_chunks = [
        chunk for chunk in chunks if chunk.metadata["chunk_id"] not in existing_ids
    ]

    if new_chunks:
        vector_store.add_documents(
            documents=new_chunks,
            ids=[chunk.metadata["chunk_id"] for chunk in new_chunks],
        )

    app.state.documents = chunks
    app.state.vector_store = vector_store
    app.state.graph = create_graph(chunks, vector_store, app.state.checkpointer)
    return {
        "indexed_documents": len(loaded),
        "indexed_chunks": len(chunks),
        "added_chunks": len(new_chunks),
        "skipped_chunks": len(chunks) - len(new_chunks),
    }


@app.post("/search")
# accept query, mode, optional metadata filters, top_k. return ranked chunks + source metadata
def search(request: SearchRequest):
    return {"results": [_serialize_document(doc) for doc in _search(request)]}


@app.post("/chat")
# accept thread_id, message, search mode and optional filters. return answer
def chat(request: ChatRequest):
    logger.info("Received /chat request")
    _require_index()
    state = app.state.graph.invoke(
        {
            "question": request.message,
            "search_mode": request.mode,
            "filters": request.filters,
            "candidate_k": max(request.top_k * 3, 10),
            "top_k": request.top_k,
            "messages": [HumanMessage(content=request.message)],
        },
        config={"configurable": {"thread_id": request.thread_id}},
    )

    evidence = state.get("reranked_results", [])

    interrupts = state.get("__interrupt__", [])
    clarification_question = interrupts[0].value if interrupts else None

    return {
        "thread_id": request.thread_id,
        "answer": state.get("answer"),
        "clarification_question": clarification_question,
        "sources": [_serialize_document(doc) for doc in evidence],
    }


@app.post("/chat/resume")
# accept thread_id, interrupt identified (if required), clarification text, resume pending paragraph execution, reject invalid/non-pending resumes
def resume_chat(request: ResumeRequest):
    _require_index()
    result = app.state.graph.invoke(
        Command(resume=request.clarification),
        config={"configurable": {"thread_id": request.thread_id}},
    )


    evidence = result.get("reranked_results", [])
    
    interrupts = result.get("__interrupt__", [])
    clarification_question = interrupts[0].value if interrupts else None
    return {
            "thread_id": request.thread_id,
            "answer": result.get("answer"),
            "clarification_question": clarification_question,
            "sources": [_serialize_document(doc) for doc in evidence],
        }
