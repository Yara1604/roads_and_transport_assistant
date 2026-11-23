from langchain_core.documents import Document
from pydantic import BaseModel, Field
from app.models import co, llm
from langchain_community.retrievers import BM25Retriever
from app.schemas import RerankedItem, RerankResult
import logging
from cohere.core.api_error import ApiError

def _matches_metadata_filters(metadata: dict, filters: dict | None) -> bool:
    if not filters:
        return True

    # The app-facing format is {field: exact_value}; multiple fields are ANDed.
    _validate_simple_filters(filters)
    return all(metadata.get(key) == value for key, value in filters.items())


def _validate_simple_filters(filters: dict) -> None:
    for key, value in filters.items():
        if key.startswith("$") or isinstance(value, (dict, list)) or value is None:
            raise ValueError(
                "Filters must use simple field:value equality, "
                "for example {'year': 2023, 'type': 'Law'}"
            )


def _normalize_chroma_filter(filters: dict | None) -> dict | None:
    if not filters:
        return None

    _validate_simple_filters(filters)

    # Chroma expects multiple metadata conditions to be joined explicitly.
    if len(filters) > 1:
        return {"$and": [{key: value} for key, value in filters.items()]}
    return filters


def create_bm25_retriever(documents: list[Document], k: int = 10) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(documents=documents, k=k)
    return retriever


def lexical_search(
    query: str,
    bm25_retriever,
    k: int = 10,
    filters: dict | None = None,
) -> list[Document]:
    # Rank the full corpus first so filtered-out high-ranking documents do not
    # prevent eligible documents from filling the requested top-k.
    ranked_retriever = bm25_retriever.model_copy(update={"k": len(bm25_retriever.docs)})
    ranked_results = ranked_retriever.invoke(query)

    if filters:
        ranked_results = [
            doc
            for doc in ranked_results
            if _matches_metadata_filters(doc.metadata, filters)
        ]

    return ranked_results[:k]


def semantic_search(
    query: str, vector_store, k: int = 10, filters: dict | None = None
) -> list[Document]:

    chroma_filter = _normalize_chroma_filter(filters)
    if chroma_filter:
        results = vector_store.similarity_search(query=query, k=k, filter=chroma_filter)
    else:
        results = vector_store.similarity_search(query=query, k=k)

    return results


# RRF (Reciprocal Rank Fusion)


def rrf(
    bm25_results: list[Document], vector_results: list[Document], rrf_k: int = 60
) -> tuple[list[Document], dict[str, float]]:

    rrf_scores = {}

    # to convert chunk ids to Document objects later
    doc_lookup = {}

    for rank, doc in enumerate(bm25_results, start=1):
        chunk_id = doc.metadata["chunk_id"]

        doc_lookup[chunk_id] = doc

        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0

        rrf_scores[chunk_id] += 1 / (rrf_k + rank)

    for rank, doc in enumerate(vector_results, start=1):
        chunk_id = doc.metadata["chunk_id"]

        doc_lookup[chunk_id] = doc

        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0

        rrf_scores[chunk_id] += 1 / (rrf_k + rank)

    sorted_scores = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    fused_documents = [doc_lookup[chunk_id] for chunk_id, score in sorted_scores]

    # Make scores a sorted dictionary
    sorted_rrf_scores = dict(sorted_scores)

    return fused_documents, sorted_rrf_scores


def rerank_documents(
    query: str, documents: list[Document], top_k: int = 5
) -> tuple[list[Document], object]:
    if not documents:
        return [], None

    limit = min(top_k, len(documents))
    logger = logging.getLogger(__name__)

    candidate_docs = [
        f"Chunk ID: {doc.metadata.get('chunk_id', '')}\n"
        f"Headings: {doc.metadata.get('headings') or 'Not available'}\n"
        f"Content:\n{doc.page_content}"
        for doc in documents
    ]

    try:
        result = co.rerank(
            model="rerank-v5.0-pro",
            query=query,
            documents=candidate_docs,
            top_n=min(top_k, len(documents)),
        )
    except ApiError as exc:
        # Catches HTTP 429 rate limits, 4xx/5xx API errors, and quota errors
        logger.warning(
            "Cohere rerank API error (status=%s): %s. Falling back to un-reranked order.",
            getattr(exc, "status_code", "unknown"),
            exc,
        )
        return documents[:limit], None
    except Exception as exc:
        # Catches timeouts, connection drops, or unexpected SDK errors
        logger.warning(
            "Cohere rerank failed unexpectedly (%s): %s. Falling back to un-reranked order.",
            type(exc).__name__,
            exc,
        )
        return documents[:limit], None
    
    ranked_items = sorted(
        result.results, key=lambda item: item.relevance_score, reverse=True
    )

    reranked_docs = [
        documents[item.index]
        for item in ranked_items
        if 0 <= item.index < len(documents)
    ]

    # In case result.results ever comes back empty
    if not reranked_docs:
        logger.warning("Results came back empty")
        return documents[:limit], result

    return reranked_docs[:limit], result