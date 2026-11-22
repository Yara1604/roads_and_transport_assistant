from langchain_core.documents import Document
from pydantic import BaseModel, Field
from app.models import llm
from langchain_community.retrievers import BM25Retriever
from app.schemas import RerankedItem, RerankResult


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
    ranked_retriever = bm25_retriever.model_copy(
        update={"k": len(bm25_retriever.docs)}
    )
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
        results = vector_store.similarity_search(
            query=query, k=k, filter=chroma_filter
        )
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


reranker_llm = llm.with_structured_output(RerankResult)

def rerank_documents(
    query: str, documents: list[Document], top_k: int = 5
) -> tuple[list[Document], RerankResult]:

    doc_lookup = {doc.metadata["chunk_id"]: doc for doc in documents}

    candidates = "\n\n".join(f"""
    Chunk ID: {doc.metadata["chunk_id"]}
    Content:
    {doc.page_content}
    """ for doc in documents)

    prompt = f"""
    You are a relevance reranker for a Roads and Transport
    legislation retrieval system.
    Query:
    {query}
    Candidate chunks:
    {candidates}
    Rank every candidate according to its relevance to the query.
    Rules:

        Do not answer the question.
        Return every candidate exactly once.
        Do not invent chunk IDs.
        Score relevance from 0 to 10.
        Higher means more relevant.
        """

    result = reranker_llm.invoke(prompt)

    ranked_items = sorted(
        result.results, key=lambda item: item.relevance_score, reverse=True
    )

    reranked_docs = [
        doc_lookup[item.chunk_id]
        for item in ranked_items
        if item.chunk_id in doc_lookup
    ]

    return reranked_docs[:top_k], result
