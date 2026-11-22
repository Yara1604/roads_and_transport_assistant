from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.types import interrupt, Command

from app.config import MAX_CLARIFICATION_QUESTIONS
from app.schemas import GraphState, ClassificationResult, DocumentExcerptToolInput
from app.retrieval import (
    create_bm25_retriever,
    lexical_search,
    semantic_search,
    rrf,
    rerank_documents,
)
from app.models import llm
from app.prompts import answer_prompt, interruption_prompt


def create_graph(langchain_docs, vector_store, checkpointer=None):
    bm25_retriever = create_bm25_retriever(documents=langchain_docs, k=10)

    documents_by_id = {}
    for document in langchain_docs:
        document_id = str(document.metadata.get("document_id", ""))
        documents_by_id.setdefault(document_id, []).append(document)

    @tool(args_schema=DocumentExcerptToolInput)
    def get_document_excerpt(document_id: str, locator: str) -> str:
        """Return an excerpt from a page or chunk in the indexed corpus."""
        document_chunks = documents_by_id.get(document_id, [])
        if not document_chunks:
            return "No indexed document matches that document_id."

        if locator.startswith("chunk:"):
            chunk_id = locator.removeprefix("chunk:")
            matches = [
                document
                for document in document_chunks
                if document.metadata.get("chunk_id") == chunk_id
            ]
        elif locator.startswith("page:"):
            page_number = locator.removeprefix("page:")
            if not page_number.isdigit():
                return "Invalid page locator. Use page:<page_number>."
            matches = [
                document
                for document in document_chunks
                if str(document.metadata.get("page")) == page_number
            ][:3]
        else:
            return "Invalid locator. Use chunk:<chunk_id> or page:<page_number>."

        if not matches:
            return "No indexed excerpt matches that document_id and locator."

        excerpts = []
        for document in matches:
            metadata = document.metadata
            headings = metadata.get("headings") or "Not available"
            excerpts.append(
                f"Document: {metadata.get('filename', 'Unknown title')}\n"
                f"Document ID: {document_id}\n"
                f"Chunk ID: {metadata.get('chunk_id', 'Unknown chunk')}\n"
                f"Page: {metadata.get('page', 'Not available')}\n"
                f"Headings: {headings}\n"
                f"Excerpt:\n{document.page_content[:3000]}"
            )
        return "\n\n---\n\n".join(excerpts)

    answer_agent = create_agent(
        model=llm,
        tools=[get_document_excerpt],
        name="legislation_answer_agent",
    )

    # Nodes
    def query_classification_node(state: GraphState):
        context = "\n\n".join(
            "\n".join(
                [
                    f"Document: {doc.metadata.get('filename', 'Unknown title')}",
                    f"Document ID: {doc.metadata.get('document_id', 'Unknown')}",
                    f"Chunk ID: {doc.metadata.get('chunk_id', 'Unknown')}",
                    f"Page: {doc.metadata.get('page', 'Not available')}",
                    f"Headings: {doc.metadata.get('headings') or 'Not available'}",
                    f"Text: {doc.page_content}",
                ]
            )
            for doc in state.reranked_results
        )

        classifier_llm = llm.with_structured_output(ClassificationResult)

        result = (interruption_prompt | classifier_llm).invoke(
            {
                "context": context,
                "question": state.question,
                "history": state.messages[-10:-1] if state.messages else [],
            }
        )
        
        return {"classification": result.classification,
                "clarification_question": result.clarification_question}


    def classification_router_node(state: GraphState):
        return state.classification


    def ask_for_clarification(state: GraphState):
        if state.clarification_questions_asked >= MAX_CLARIFICATION_QUESTIONS:
            answer = "Maximum number of clarification questions reached."
            return {
                        "answer": answer,
                        "messages": [AIMessage(content=answer)],
                        "clarification_limit_reached": True,
                    }
        else: 
            clarification_reply = interrupt(state.clarification_question)
            return {
                        "question": f"{state.question}\nAdditional detail:{clarification_reply}",
                        "messages": [
                            AIMessage(content=state.clarification_question),
                            HumanMessage(content=clarification_reply),
                        ],
                        "clarification_questions_asked": state.clarification_questions_asked + 1,
                        "clarification_limit_reached": False,
                    }

    def after_clarification(state: GraphState):
        if state.clarification_limit_reached:
            return END
        return ["bm25", "vector"]

    def bm25_node(state: GraphState):
        if state.search_mode == "semantic":
            return {"bm25_results": []}
        results = lexical_search(
            query=state.question,
            bm25_retriever=bm25_retriever,
            k=state.candidate_k,
            filters=state.filters,
        )

        return {"bm25_results": results}

    def vector_node(state: GraphState):
        if state.search_mode == "lexical":
            return {"vector_results": []}
        results = semantic_search(
            query=state.question,
            vector_store=vector_store,
            k=state.candidate_k,
            filters=state.filters,
        )
        return {"vector_results": results}

    def fusion_node(state: GraphState):
        if state.search_mode == "lexical":
            return {"fused_results": state.bm25_results}
        if state.search_mode == "semantic":
            return {"fused_results": state.vector_results}
        fused_docs, rrf_scores = rrf(state.bm25_results, state.vector_results, rrf_k=60)
        return {"fused_results": fused_docs}

    def rerank_node(state: GraphState):
        if not state.fused_results:
            return {"reranked_results": []}
        reranked_docs, _ = rerank_documents(
            query=state.question,
            documents=state.fused_results[:15],
            top_k=state.top_k,
        )
        return {"reranked_results": reranked_docs}

    def answer_node(state: GraphState):
        evidence = state.reranked_results
        if not evidence:
            answer = "There is insufficient evidence to answer this question."
            return {
                "context": "",
                "answer": answer,
                "messages": [AIMessage(content=answer)],
            }

        context = "\n\n".join(
            "\n".join(
                [
                    f"Document ID: {doc.metadata.get('document_id', 'Unknown')}",
                    f"Title: {doc.metadata.get('filename', 'Unknown title')}",
                    f"Chunk ID: {doc.metadata.get('chunk_id', 'Unknown')}",
                    f"Page: {doc.metadata.get('page', 'Not available')}",
                    f"Headings: {doc.metadata.get('headings') or 'Not available'}",
                    f"Text: {doc.page_content}",
                ]
            )
            for doc in evidence
        )

        history = (
            "\n".join(
                f"{message.type}: {message.content}"
                for message in state.messages[-10:-1]  # Last 10 messages
            )
            or "No previous conversation."
        )

        prompt_messages = answer_prompt.invoke(
            {
                "context": context,
                "question": state.question,
                "history": history,
            }
        ).to_messages()
        agent_result = answer_agent.invoke({"messages": prompt_messages})
        response = agent_result["messages"][-1]
        return {
            "context": context,
            "answer": response.content,
            "messages": [AIMessage(content=response.content)],
            # "history": [AIMessage(content=response.content)],
        }

    # Graph
    builder = StateGraph(GraphState)

    builder.add_node("bm25", bm25_node)
    builder.add_node("vector", vector_node)
    builder.add_node("fusion", fusion_node)
    builder.add_node("rerank", rerank_node)
    

    builder.add_edge(START, "bm25")  # In parallel
    builder.add_edge(START, "vector")

    builder.add_edge("bm25", "fusion")
    builder.add_edge("vector", "fusion")
    builder.add_edge("fusion", "rerank")

    builder.add_edge("rerank", "query_classification")
    builder.add_node("query_classification", query_classification_node)
    builder.add_node("ask_for_clarification", ask_for_clarification)
    
    builder.add_node("answer", answer_node)

    builder.add_conditional_edges(
        "query_classification",
        classification_router_node,
        {
            "clear": "answer",
            "requires_clarification": "ask_for_clarification",
        },
    )

    builder.add_conditional_edges(
        "ask_for_clarification",
        after_clarification
    )


    builder.add_edge("answer", END)

    return builder.compile(checkpointer=checkpointer)
