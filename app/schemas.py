from typing import Annotated, Any, Literal

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# Shared base model

class GraphState(BaseModel):
    question: str

    search_mode: Literal["lexical", "semantic", "hybrid"] = "hybrid"
    filters: dict[str, Any] | None = None
    candidate_k: int = 10
    top_k: int = 5

    bm25_results: list[Document] = Field(default_factory=list)
    vector_results: list[Document] = Field(default_factory=list)

    fused_results: list[Document] = Field(default_factory=list)

    reranked_results: list[Document] = Field(default_factory=list)

    # Evidence actually sent to LLM
    context: str = ""

    answer: str | None = None

    # The reducer appends new turns while preserving prior turns for this thread.
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
