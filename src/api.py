from fastapi import FastAPI
from pydantic import BaseModel

from src.rag import answer_question


app = FastAPI(title="Esoteric AI Teacher")


class AskRequest(BaseModel):
    question: str
    mode: str = "general"
    rag_config: dict | None = None


@app.get("/")
def health_check():
    return {"status": "ok", "app": "Esoteric AI Teacher"}


@app.post("/ask")
def ask(request: AskRequest):
    result = answer_question(
        question=request.question,
        mode=request.mode,
        rag_config=request.rag_config,
    )

    return {"result": result}
