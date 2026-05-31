# Esoteric AI Teacher Backend

FastAPI backend for the Esoteric AI Teacher app. It queries a Chroma Cloud
collection and uses Google AI Studio Gemini models for embeddings and answer
generation.

## Requirements

- Python 3.11+
- Google AI Studio API key
- Chroma Cloud database with collection `esoteric_books`

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Fill `.env` with your own Chroma Cloud and Google AI Studio credentials. Do not
commit `.env`.

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
GOOGLE_API_KEY=
GOOGLE_LLM_MODEL=gemini-2.5-flash
GOOGLE_EMBEDDING_MODEL=models/gemini-embedding-001
```

## Chroma Reindexing

If your Chroma Cloud collection was previously indexed with Ollama `bge-m3`, you
must rebuild it with Google embeddings. Put the licensed PDFs in `books/`, set
the environment variables, then run:

```powershell
python -m src.reindex_cloud
```

This deletes and recreates the Chroma Cloud collection `esoteric_books` with
Gemini embeddings.

`src/ingest.py` can still build a local Chroma database from PDFs placed in
`books/`. The `books/` folder and `chroma_db/` are intentionally ignored and are
not part of this repository.
