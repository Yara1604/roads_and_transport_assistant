# LangGraph Knowledge Assessment: 3-Hour Learning and Implementation Guide

## Objective

Build a small, understandable vertical slice of the assessment project. Use a **learn for 10–15 minutes, then implement immediately** cycle rather than studying every topic in advance.

The recommended progression is:

```text
FastAPI → PDF ingestion → chunking → vector search → BM25 → RRF
→ LangGraph orchestration → grounded answer with citations
→ persistence → interrupt/resume → tool → middleware
```

---

## 1. Understand LangGraph

**Suggested time: 15 minutes**

### Resources

1. [LangGraph Quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart)
2. [LangGraph overview](https://www.langchain.com/langgraph)

### Focus on

- **State:** shared data that moves through the workflow
- **Node:** a Python function that performs one step
- **Edge:** defines which node runs next
- **Graph:** the complete collection of nodes and edges

### Learning checkpoint

Before continuing, you should be able to explain:

```text
State = shared data
Node = Python function
Edge = where execution goes next
Graph = nodes + edges
```

---

## 2. Build the FastAPI Skeleton

**Suggested time: 15–20 minutes**

### Resources

1. [FastAPI response models](https://fastapi.tiangolo.com/tutorial/response-model/)
2. [FastAPI extra models](https://fastapi.tiangolo.com/tutorial/extra-models/)

### First target

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}
```

Then create placeholder routes for:

```text
POST /ingest
POST /search
POST /chat
POST /chat/resume
```

Do not implement all route internals yet. Confirm that the application starts and `/docs` opens.

---

## 3. Learn PDF → Chunks → Vectors → Search

**Suggested time: 25–30 minutes**

### Resource

1. [LangChain retrieval guide](https://docs.langchain.com/oss/python/deepagents/retrieval)

### Mental model

```text
PDF
 ↓
Document loader
 ↓
Documents
 ↓
Text splitter
 ↓
Chunks
 ↓
Embeddings
 ↓
Vector store
 ↓
Retriever
```

### Preserve citation metadata

Each chunk should retain data such as:

```python
{
    "document_id": "...",
    "chunk_id": "...",
    "title": "...",
    "page": 7,
    "source_url": "..."
}
```

This metadata will allow the final answer to cite the correct source and location.

---

## 4. Learn Hybrid Retrieval

**Suggested time: 20 minutes**

### Resource

1. [Hybrid Search: BM25 + Dense Vectors tutorial](https://rag-visualization.vercel.app/projects/hybrid-rag)

### Architecture

```text
              ┌── BM25 ──────┐
Query ────────┤              ├→ RRF → ranked chunks
              └── Vectors ───┘
```

### Concepts

- **BM25:** finds chunks containing matching words or identifiers.
- **Vector search:** finds chunks with similar meaning.
- **Reciprocal Rank Fusion (RRF):** combines the two ranked result lists.
- **Reranker:** reassesses the fused candidates and places the strongest evidence first.

Do not spend tonight studying the mathematics deeply. Focus on implementing and testing the pipeline.

---

## 5. Turn Retrieval into a LangGraph Workflow

**Suggested time: 25–30 minutes**

### Resource

1. [LangGraph Quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart)

### Target workflow

```text
START
  ↓
analyze_query
  ↓
 ┌──────────────────┐
 ↓                  ↓
bm25_search    vector_search
 ↓                  ↓
 └───────┬──────────┘
         ↓
      fusion
         ↓
      rerank
         ↓
 generate_answer
         ↓
        END
```

Keep BM25 and vector search as separate graph branches. This demonstrates that LangGraph is orchestrating the workflow rather than wrapping one large retrieval function.

---

## 6. Add Persistence, `thread_id`, Interrupt, and Resume

**Suggested time: 20 minutes**

### Resource

1. [LangGraph interrupts documentation](https://docs.langchain.com/oss/python/langgraph/interrupts)

### Focus on

```python
interrupt(...)
```

and:

```python
Command(resume=...)
```

### Example flow

```text
User: "Tell me about licensing"
        ↓
Graph determines that the question is ambiguous
        ↓
interrupt()
        ↓
Application asks: "Which type of licensing?"
        ↓
User: "Vehicle licensing"
        ↓
Command(resume="vehicle licensing")
        ↓
Graph continues using the same thread_id
```

Ignore advanced human-in-the-loop patterns until this basic flow works.

---

## 7. Add One Model-Selected Tool

**Suggested time: 15 minutes**

### Resource

1. [LangChain agents documentation](https://docs.langchain.com/oss/python/langchain/agents)

### Suggested tool

```python
from langchain.tools import tool

@tool
def get_document_excerpt(document_id: str, page: int) -> str:
    """Retrieve text from a specific document page."""
    # Look up and return the relevant indexed text.
```

### Important distinction

A normal function call is selected directly by your code:

```text
code → function()
```

A model-selected tool is chosen by the model:

```text
LLM → selects the tool → supplies arguments → receives result
```

Make sure you can demonstrate at least one actual model-to-tool invocation.

---

## 8. Add LangChain Middleware

**Suggested time: 10 minutes**

### Resources

1. [LangChain middleware overview](https://docs.langchain.com/oss/python/langchain/middleware/overview)
2. [LangChain middleware reference](https://reference.langchain.com/python/langchain/middleware)

### Keep it simple

Use one concrete middleware capability, such as:

- `ModelCallLimitMiddleware`
- `ToolCallLimitMiddleware`
- `ToolRetryMiddleware`
- lightweight model/tool timing or logging

Conceptually:

```text
request
   ↓
middleware
   ↓
model or tool
   ↓
middleware
   ↓
response
```

---

## Recommended 3-Hour Sequence

| Time | Activity | Minimum outcome |
|---|---|---|
| 0:00–0:15 | LangGraph basics | Understand state, nodes, edges, and compilation |
| 0:15–0:35 | FastAPI skeleton | `/health` and placeholder routes work |
| 0:35–1:05 | PDF ingestion and semantic retrieval | A real PDF produces searchable chunks |
| 1:05–1:25 | BM25 and RRF | Keyword and vector results can be fused |
| 1:25–1:55 | LangGraph orchestration | Query travels through the graph to an answer |
| 1:55–2:15 | Persistence and interrupt/resume | One thread can pause and resume |
| 2:15–2:30 | Model-selected tool | One real tool invocation works |
| 2:30–2:40 | Middleware | One middleware feature executes |
| 2:40–3:00 | Testing and documentation | Save evidence and list remaining gaps |

---

## Practical Rules for Tonight

1. **Build one small vertical slice.** Start with one or two documents, not the full corpus.
2. **Test after every step.** Do not build multiple untested components at once.
3. **Keep metadata from ingestion onward.** Missing page and source metadata will make citations harder later.
4. **Do not hide retrieval inside one function.** Keep lexical and semantic retrieval visibly separate in the graph.
5. **Avoid copying an entire repository.** Small examples are easier to understand and debug.
6. **Do not polish early.** Prioritize a working path over architecture cleanup, prompt tuning, or a frontend.
7. **Commit once the vertical slice works.** Preserve a recoverable version before adding optional features.

At each stage, ask:

> Can I explain why these 10–20 lines exist?

If yes, continue. If not, pause and reduce the example until it becomes understandable.

---

## Realistic Finish Line

A strong result for tonight is:

```text
One real document
→ ingest
→ deterministic chunks with metadata
→ BM25 and vector retrieval
→ RRF/reranking
→ LangGraph workflow
→ grounded answer with citation
→ persisted thread
→ one interrupt/resume example
→ one model-selected tool
→ one middleware feature
→ FastAPI response
```

If time runs short, prioritize through **grounded answer with citation**. Persistence, interrupts, tools, middleware, evaluation, and additional documents can be completed afterward.
