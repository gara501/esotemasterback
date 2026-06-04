# Esoteric AI Teacher Backend

FastAPI backend for the Esoteric AI Teacher app. It indexes licensed PDF books
into Qdrant Cloud and uses Google AI Studio Gemini models for embeddings and
answer generation.

## Requirements

- Python 3.11+
- Qdrant Cloud URL and API key
- Google AI Studio API key
- Licensed PDFs in `books/`

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Fill `.env` with your Qdrant Cloud and Google AI Studio credentials. Do not
commit `.env`.

## Environment Variables

```env
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION=esoteric_books
BOOKS_DIR=./books
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=20
FINAL_TOP_N=8
USE_RERANKER=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
INDEX_BATCH_SIZE=16
EMBEDDING_BATCH_SIZE=4
EMBEDDING_MAX_RETRIES=6
EMBEDDING_RETRY_SLEEP_SECONDS=20
GOOGLE_API_KEY=
GOOGLE_LLM_MODEL=gemini-2.5-flash
GOOGLE_EMBEDDING_MODEL=models/gemini-embedding-001
ALLOWED_ORIGINS=*
API_URL=http://127.0.0.1:8000/ask
```

## Index Books

Put PDFs in `books/`, then run:

```powershell
python index_books.py
```

The indexer creates the Qdrant Cloud collection automatically if it does not
exist, validates vector dimensions, stores stable chunk IDs, and skips chunks
that are already indexed.

If Google returns `429 RESOURCE_EXHAUSTED`, lower the throughput in `.env`:

```env
INDEX_BATCH_SIZE=4
EMBEDDING_BATCH_SIZE=1
EMBEDDING_RETRY_SLEEP_SECONDS=60
```

The indexer skips already indexed chunks, so rerunning the command continues
from the chunks that still need embeddings.

## Ask From Console

```powershell
python query_rag.py "Explica qué dicen los libros sobre la piedra filosofal" --mode alquimia
```

Optional filters:

```powershell
python query_rag.py "Busca una cita sobre el árbol de la vida" --mode extract --filter language=es --filter page=7
```

## Run API

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

## Reset Collection

This deletes and recreates the Qdrant collection only after explicit
confirmation:

```powershell
python reset_collection.py
```

Type `RESET esoteric_books` when prompted, or replace `esoteric_books` with your
configured `QDRANT_COLLECTION`.

## Streamlit App

Start the API first, then run:

```powershell
streamlit run app.py
```

`app.py` calls the FastAPI `/ask` endpoint through `API_URL`. The backend returns
the generated answer, grouped sources, and optional debug chunks when
`debug_chunks` is enabled in `rag_config`.
