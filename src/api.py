import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.rag import answer_question


app = FastAPI(title="Esoteric AI Teacher")

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    mode: str = "general"
    rag_config: dict | None = None
    filters: dict | None = None


@app.get("/")
def health_check():
    return {"status": "ok", "app": "Esoteric AI Teacher"}


@app.post("/ask")
def ask(request: AskRequest):
    result = answer_question(
        question=request.question,
        mode=request.mode,
        rag_config=request.rag_config,
        filters=request.filters,
    )

    return {"result": result}
