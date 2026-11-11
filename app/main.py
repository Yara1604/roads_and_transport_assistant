from fastapi import FastAPI

app = FastAPI()


@app.get("/health") # report app readiness & index availability
def health():
    return {"status": "ok"}


@app.post("/ingest") # ingest the approved local corpus, return indexed, skipped, and failed docs counts & errors
def ingest(): # This will be async
    pass


@app.post("/search") # accept query, mode, optional metadata filters, top_k. return ranked chunks + source metadata
def search():
    pass


@app.post("/chat") # accept thread_id, message, search mode and optional filters. return answer
def chat():
    pass


@app.post("/chat/resume") # accept thread_id, interrupt identified (if required), clarification text, resume pending paragraph execution, reject invalid/non-pending resumes
def resume_chat():
    pass
