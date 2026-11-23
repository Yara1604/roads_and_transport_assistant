from typing import Annotated, Literal

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator

from app.config import DEFAULT_SOURCE_DIR, DEFAULT_PROCESSED_DIR

MetadataFilters = dict[str, str | int | float | bool] | None

# --- API RELATED ---
class IngestRequest(BaseModel):
    source_folder: str = str(DEFAULT_SOURCE_DIR)
    processed_folder: str = str(DEFAULT_PROCESSED_DIR)
    max_tokens: int = Field(default=512, ge=1)

class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: Literal["lexical", "semantic", "hybrid"] = "hybrid"
    filters: MetadataFilters = None
    top_k: int = Field(default=5, ge=1, le=20)


class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    mode: Literal["lexical", "semantic", "hybrid"] = "hybrid"
    filters: MetadataFilters = None
    top_k: int = Field(default=5, ge=1, le=50)


class ResumeRequest(BaseModel):
    thread_id: str
    clarification: str = Field(min_length=1)


class DocumentExcerptToolInput(BaseModel):
    locator: str = Field(
        min_length=1,
        description="Use chunk:<exact_chunk_id> or page:<page_number>.",
    )
    document_id: str | None = Field(
        default=None,
        description="Required only with a page locator; omit it for a chunk locator.",
    )


# --- GRAPH RELATED ---
class GraphState(BaseModel):
    question: str

    search_mode: Literal["lexical", "semantic", "hybrid"] = "hybrid"
    filters: MetadataFilters = None
    candidate_k: int = 10 # Before reranking
    top_k: int = 5        # After reranking

    bm25_results: list[Document] = Field(default_factory=list)
    vector_results: list[Document] = Field(default_factory=list)

    fused_results: list[Document] = Field(default_factory=list)

    reranked_results: list[Document] = Field(default_factory=list)

    classification: Literal["clear", "requires_clarification"] | None = None
    clarification_question: str | None = None
    clarification_questions_asked: int = 0
    clarification_limit_reached: bool = False

    # Evidence actually sent to LLM
    context: str = ""

    answer: str | None = None

    @field_validator("answer", mode="before")
    @classmethod
    def normalize_answer_content(cls, value):
        # Older checkpoints may contain Gemini's list of content blocks here.
        if isinstance(value, list):
            text_parts = []
            for block in value:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
            return "".join(text_parts)
        return value

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
    clarification_question: str | None = None
