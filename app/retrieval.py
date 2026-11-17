from langchain_core.documents import Document
from pydantic import BaseModel, Field
from app.models import llm
from langchain_community.retrievers import BM25Retriever
from app.schemas import RerankedItem, RerankResult


def create_bm25_retriever(documents: list[Document], k: int = 10) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(documents=documents, k=k)
    return retriever


def lexical_search(query: str, bm25_retriever, k: int = 10) -> list[Document]:
    bm25_retriever.k = k

    results = bm25_retriever.invoke(query)

    return results


def semantic_search(
    query: str, vector_store, k: int = 10, filters: dict | None = None
) -> list[Document]:

    if filters:
        results = vector_store.similarity_search(query=query, k=k, filter=filters)
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