from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage
from langgraph.types import interrupt, Command

from app.schemas import GraphState, ClassificationResult
from app.retrieval import (
    create_bm25_retriever,
    lexical_search,
    semantic_search,
    rrf,
    rerank_documents,
)
from app.models import llm
from app.prompts import answer_prompt, interruption_prompt

def create_graph(
        langchain_docs,
        vector_store,
        checkpointer=None,
):
    bm25_retriever = create_bm25_retriever(
        documents= langchain_docs,
        k= 10
    )

    # Nodes
    def bm25_node(state: GraphState):
        if state.search_mode == "semantic":
            return {"bm25_results": []}
        results = lexical_search(
            query= state.question,
            bm25_retriever= bm25_retriever,
            k=state.candidate_k,
        )

        return {
            "bm25_results": results
        }

    def vector_node(state: GraphState):
        if state.search_mode == "lexical":
            return {"vector_results": []}
        results = semantic_search(
            query = state.question,
            vector_store = vector_store,
            k=state.candidate_k,
            filters=state.filters,
        )
        return {
            "vector_results": results
        }

    def fusion_node(state: GraphState):
        if state.search_mode == "lexical":
            return {"fused_results": state.bm25_results}
        if state.search_mode == "semantic":
            return {"fused_results": state.vector_results}
        fused_docs, rrf_scores = rrf(
            state.bm25_results,
            state.vector_results,
            rrf_k=60
        )
        return {
            "fused_results": fused_docs
        }


    def rerank_node(state: GraphState):
        if not state.fused_results:
            return {"reranked_results": []}
        reranked_docs, _ = rerank_documents(
            query=state.question,
            documents=state.fused_results[:15],
            top_k=state.top_k,
        )
        return {
            "reranked_results": reranked_docs
        }

    def query_classification_node(state: GraphState):
        query = state.question
        context = "\n\n".join(doc.page_content for doc in state.reranked_results)
                # if not context:
                #     answer = "There is insufficient evidence to answer this question."
                #     return {"context": "", "answer": answer, "messages": [AIMessage(content=answer)]}

        classifier_llm = llm.with_structured_output(ClassificationResult)

        classification = (interruption_prompt | classifier_llm).invoke(
            {
                "context": context,
                "question": state.question
            })

        

    def answer_node(state: GraphState):
        context = "\n\n".join(doc.page_content for doc in state.reranked_results)
        if not context:
            answer = "There is insufficient evidence to answer this question."
            return {"context": "", "answer": answer, "messages": [AIMessage(content=answer)]}

        history = "\n".join(
            f"{message.type}: {message.content}"
            for message in state.messages[-10:-1] # Last 10 messages
        ) or "No previous conversation."

        response = (answer_prompt | llm).invoke(
            {
                "context": context,
                "question": state.question,
                "chat_history": history,
            }
        )
        return {
            "context": context,
            "answer": response.content,
            "messages": [AIMessage(content=response.content)],
        }


    # Graph
    builder = StateGraph(GraphState)

    builder.add_node("bm25", bm25_node)
    builder.add_node("vector", vector_node)
    builder.add_node("fusion", fusion_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("answer", answer_node)

    builder.add_edge(START, "bm25") # In parallel
    builder.add_edge(START, "vector")

    builder.add_edge("bm25", "fusion")
    builder.add_edge("vector", "fusion")

    builder.add_edge("fusion", "rerank")
    builder.add_edge("rerank", "answer")
    builder.add_edge("answer", END)

    return builder.compile(checkpointer=checkpointer)
