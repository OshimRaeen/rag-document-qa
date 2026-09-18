from fastapi import FastAPI

app = FastAPI(
    title="Document RAG API",
    description="Question answering grounded in uploaded documents",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }