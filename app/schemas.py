from typing import Annotated, Any, Literal

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.config import DEFAULT_SOURCE_DIR, DEFAULT_PROCESSED_DIR

# --- API RELATED ---
class IngestRequest(BaseModel):
    source_folder: str = str(DEFAULT_SOURCE_DIR)
    processed_folder: str = str(DEFAULT_PROCESSED_DIR)
    max_tokens: int = Field(default=512, ge=1)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: Literal["lexical", "semantic", "hybrid"] = "hybrid"
    filters: dict[str, Any] | None = None
    top_k: int = Field(default=5, ge=1, le=50)


class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    mode: Literal["lexical", "semantic", "hybrid"] = "hybrid"
    filters: dict[str, Any] | None = None
    top_k: int = Field(default=5, ge=1, le=50)


class ResumeRequest(BaseModel):
    thread_id: str
    clarification: str = Field(min_length=1)


# --- GRAPH RELATED ---
class GraphState(BaseModel):
    question: str

    search_mode: Literal["lexical", "semantic", "hybrid"] = "hybrid"
    filters: dict[str, Any] | None = None
    candidate_k: int = 10 # Before reranking
    top_k: int = 5        # After reranking

    bm25_results: list[Document] = Field(default_factory=list)
    vector_results: list[Document] = Field(default_factory=list)

    fused_results: list[Document] = Field(default_factory=list)

    reranked_results: list[Document] = Field(default_factory=list)

    # Evidence actually sent to LLM
    context: str = ""

    answer: str | None = None

    # The reducer appends new turns while preserving prior turns for this thread.
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)


# --- LLM RESPONSE RELATED ---
class RerankedItem(BaseModel):
    chunk_id: str
    relevance_score: float = Field(ge=0, le=10)


class RerankResult(BaseModel):
    results: list[RerankedItem]

class ClassificationResult(BaseModel):
    classification: Literal["clear", "requires_clarification"] = "clear"