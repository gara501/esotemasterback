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
CHROMA_COLLECTION=esoteric_books_google
GOOGLE_API_KEY=
GOOGLE_LLM_MODEL=gemini-2.5-flash
GOOGLE_EMBEDDING_MODEL=models/gemini-embedding-001
ALLOWED_ORIGINS=*
REINDEX_BATCH_SIZE=5
REINDEX_SLEEP_SECONDS=2
```

For production, replace `ALLOWED_ORIGINS=*` with your Netlify domain:

```env
ALLOWED_ORIGINS=https://your-site.netlify.app
```

## Chroma Reindexing

If your Chroma Cloud collection was previously indexed with Ollama `bge-m3`, you
must rebuild it with Google embeddings. Put the licensed PDFs in `books/`, set
the environment variables, then run:

```powershell
python -m src.reindex_cloud
```

By default this creates `esoteric_books_google` with Gemini embeddings and leaves
the old collection untouched. Set `CHROMA_COLLECTION=esoteric_books_google` in
the deployed backend after the reindex completes.

If Google AI Studio returns quota errors, rerun the same command later. The
script resumes from the current collection count.

`src/ingest.py` can still build a local Chroma database from PDFs placed in
`books/`. The `books/` folder and `chroma_db/` are intentionally ignored and are
not part of this repository.
