# Esoteric AI Teacher Backend

FastAPI backend for the Esoteric AI Teacher app. It queries a Chroma Cloud
collection and uses local Ollama models for embeddings and answer generation.

## Requirements

- Python 3.11+
- Ollama running locally
- Ollama models:
  - `bge-m3`
  - `gemma3:4b`
- Chroma Cloud database with collection `esoteric_books`

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Fill `.env` with your own Chroma Cloud credentials. Do not commit `.env`.

## Run

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

Health check:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/
```

Ask endpoint:

```powershell
Invoke-WebRequest `
  -Method POST `
  -Uri http://127.0.0.1:8000/ask `
  -ContentType "application/json" `
  -Body '{"question":"test","mode":"extract"}'
```

## Environment Variables

```env
CHROMA_API_KEY=
CHROMA_TENANT=
CHROMA_DATABASE=esoter
OLLAMA_HOST=http://127.0.0.1:11434
```

## Local Ingestion

`src/ingest.py` can build a local Chroma database from PDFs placed in `books/`.
The `books/` folder and `chroma_db/` are intentionally ignored and are not part
of this repository.
