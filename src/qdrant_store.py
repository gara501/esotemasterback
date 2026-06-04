import hashlib
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.cache import get_cached_embedding, set_cached_embedding


logger = logging.getLogger(__name__)

load_dotenv()


@dataclass(frozen=True)
class RAGSettings:
    qdrant_url: str
    qdrant_api_key: str
    collection: str
    books_dir: Path
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    final_top_n: int
    use_reranker: bool
    reranker_model: str
    google_llm_model: str
    google_embedding_model: str
    google_api_key: str


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def get_settings() -> RAGSettings:
    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip()
    google_api_key = (
        os.getenv("GOOGLE_API_KEY", "").strip()
        or os.getenv("GEMINI_API_KEY", "").strip()
    )

    missing = []
    if not qdrant_url:
        missing.append("QDRANT_URL")
    if not qdrant_api_key:
        missing.append("QDRANT_API_KEY")
    if not google_api_key:
        missing.append("GOOGLE_API_KEY or GEMINI_API_KEY")

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Add them to .env. The Qdrant API key is never logged."
        )

    return RAGSettings(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection=os.getenv("QDRANT_COLLECTION", "esoteric_books").strip()
        or "esoteric_books",
        books_dir=Path(os.getenv("BOOKS_DIR", "./books")),
        chunk_size=_env_int("CHUNK_SIZE", 1000),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 200),
        retrieval_k=_env_int("RETRIEVAL_K", 20),
        final_top_n=_env_int("FINAL_TOP_N", 8),
        use_reranker=_env_bool(
            "USE_RERANKER",
            _env_bool("ENABLE_RERANK", False),
        ),
        reranker_model=os.getenv(
            "RERANKER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        ),
        google_llm_model=os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite"),
        google_embedding_model=os.getenv(
            "GOOGLE_EMBEDDING_MODEL",
            "models/gemini-embedding-001",
        ),
        google_api_key=google_api_key,
    )


def get_qdrant_client(settings: RAGSettings | None = None) -> QdrantClient:
    settings = settings or get_settings()
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )


def get_embeddings(
    settings: RAGSettings | None = None,
) -> GoogleGenerativeAIEmbeddings:
    settings = settings or get_settings()
    return GoogleGenerativeAIEmbeddings(
        model=settings.google_embedding_model,
        google_api_key=settings.google_api_key,
    )


def detect_embedding_dimension(embeddings: GoogleGenerativeAIEmbeddings) -> int:
    probe = embeddings.embed_query("dimension probe")
    if not probe:
        raise RuntimeError("The embedding provider returned an empty vector.")
    return len(probe)


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    try:
        return bool(client.collection_exists(collection_name))
    except AttributeError:
        collections = client.get_collections().collections
        return any(collection.name == collection_name for collection in collections)


def _extract_vector_size(collection_info: Any) -> int | None:
    vectors = collection_info.config.params.vectors
    if isinstance(vectors, dict):
        if len(vectors) != 1:
            return None
        vectors = next(iter(vectors.values()))
    return getattr(vectors, "size", None)


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    if not collection_exists(client, collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info(
            "Created Qdrant collection %s with vector size %s.",
            collection_name,
            vector_size,
        )
        return

    collection_info = client.get_collection(collection_name)
    existing_size = _extract_vector_size(collection_info)
    if existing_size != vector_size:
        raise RuntimeError(
            f"Qdrant collection '{collection_name}' has vector size "
            f"{existing_size}, but the active embedding model produces "
            f"{vector_size}. Change QDRANT_COLLECTION or recreate the "
            "collection with reset_collection.py."
        )


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "book"


def infer_title_author(pdf_path: Path) -> tuple[str, str | None]:
    stem = re.sub(r"[_]+", " ", pdf_path.stem).strip()
    stem = re.sub(r"\s+", " ", stem)

    for separator in (" - ", " -- ", " — "):
        if separator in stem:
            left, right = [part.strip() for part in stem.split(separator, 1)]
            if left and right:
                return left, right

    return stem, None


def stable_book_id(pdf_path: Path) -> str:
    digest = hashlib.sha1(pdf_path.name.lower().encode("utf-8")).hexdigest()[:10]
    return f"{slugify(pdf_path.stem)}-{digest}"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_point_id(
    book_id: str,
    page: int,
    chunk_index: int,
    text_hash: str,
) -> str:
    raw = f"{book_id}:{page}:{chunk_index}:{text_hash}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def detect_language(text: str) -> str:
    lowered = f" {text[:2000].lower()} "
    spanish_markers = (" el ", " la ", " de ", " que ", " los ", " una ", " para ")
    if any(marker in lowered for marker in spanish_markers) or re.search(
        r"[áéíóúñü]",
        lowered,
    ):
        return "es"
    return "unknown"


def build_metadata_filter(filters: dict[str, Any] | None) -> Filter | None:
    if not filters:
        return None

    allowed = {"book_id", "title", "author", "page", "language"}
    conditions = []
    for key, value in filters.items():
        if key not in allowed or value in (None, ""):
            continue
        conditions.append(
            FieldCondition(
                key=key,
                match=MatchValue(value=value),
            )
        )

    if not conditions:
        return None

    return Filter(must=conditions)


def iter_pdf_chunk_payloads(
    pdf_path: Path,
    settings: RAGSettings,
) -> Iterable[dict[str, Any]]:
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", ";", ",", " ", ""],
    )

    book_id = stable_book_id(pdf_path)
    title, author = infer_title_author(pdf_path)
    indexed_at = datetime.now(timezone.utc).isoformat()
    chunk_index = 0

    for page_doc in pages:
        page_text = clean_text(page_doc.page_content)
        if not page_text:
            continue

        page = int(page_doc.metadata.get("page", 0)) + 1
        page_chunks = splitter.split_text(page_text)

        for chunk_text in page_chunks:
            chunk_text = clean_text(chunk_text)
            if not chunk_text:
                continue

            text_hash = content_hash(chunk_text)
            point_id = stable_point_id(book_id, page, chunk_index, text_hash)
            metadata = {
                "book_id": book_id,
                "title": title,
                "author": author,
                "source": str(pdf_path),
                "file_name": pdf_path.name,
                "page": page,
                "chunk_index": chunk_index,
                "chunk_id": point_id,
                "content": chunk_text,
                "content_hash": text_hash,
                "indexed_at": indexed_at,
                "document_type": "book",
                "language": detect_language(chunk_text),
            }
            payload = {
                **metadata,
                "metadata": {k: v for k, v in metadata.items() if k != "content"},
            }
            yield payload
            chunk_index += 1


def existing_point_ids(
    client: QdrantClient,
    collection_name: str,
    point_ids: list[str],
) -> set[str]:
    if not point_ids:
        return set()

    records = client.retrieve(
        collection_name=collection_name,
        ids=point_ids,
        with_payload=False,
        with_vectors=False,
    )
    return {str(record.id) for record in records}


def _is_resource_exhausted(error: Exception) -> bool:
    text = str(error).upper()
    return "RESOURCE_EXHAUSTED" in text or " 429" in text or "429 " in text


def _embed_with_retry(
    embeddings: GoogleGenerativeAIEmbeddings,
    texts: list[str],
    max_retries: int,
    initial_sleep_seconds: float,
) -> list[list[float]]:
    for attempt in range(max_retries + 1):
        try:
            return embeddings.embed_documents(texts)
        except Exception as exc:
            if not _is_resource_exhausted(exc) or attempt >= max_retries:
                raise

            sleep_seconds = initial_sleep_seconds * (2**attempt)
            logger.warning(
                "Embedding quota exhausted. Retrying in %.1f seconds "
                "(attempt %s/%s).",
                sleep_seconds,
                attempt + 1,
                max_retries,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError("Embedding retry loop exited unexpectedly.")


def embed_documents_resilient(
    embeddings: GoogleGenerativeAIEmbeddings,
    texts: list[str],
) -> list[list[float]]:
    max_retries = _env_int("EMBEDDING_MAX_RETRIES", 6)
    initial_sleep_seconds = _env_float("EMBEDDING_RETRY_SLEEP_SECONDS", 20.0)
    batch_size = max(1, _env_int("EMBEDDING_BATCH_SIZE", 4))
    vectors: list[list[float]] = []

    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        try:
            vectors.extend(
                _embed_with_retry(
                    embeddings=embeddings,
                    texts=batch,
                    max_retries=max_retries,
                    initial_sleep_seconds=initial_sleep_seconds,
                )
            )
        except Exception as exc:
            if len(batch) == 1 or not _is_resource_exhausted(exc):
                raise

            logger.warning(
                "Embedding batch of %s failed after retries. Falling back "
                "to one document at a time.",
                len(batch),
            )
            for text in batch:
                vectors.extend(
                    _embed_with_retry(
                        embeddings=embeddings,
                        texts=[text],
                        max_retries=max_retries,
                        initial_sleep_seconds=initial_sleep_seconds,
                    )
                )

    return vectors


def upsert_payloads(
    client: QdrantClient,
    collection_name: str,
    embeddings: GoogleGenerativeAIEmbeddings,
    payloads: list[dict[str, Any]],
) -> int:
    if not payloads:
        return 0

    ids = [payload["chunk_id"] for payload in payloads]
    existing_ids = existing_point_ids(client, collection_name, ids)
    new_payloads = [
        payload for payload in payloads if payload["chunk_id"] not in existing_ids
    ]

    if not new_payloads:
        return 0

    embedding_model = getattr(
        embeddings,
        "model",
        os.getenv("GOOGLE_EMBEDDING_MODEL", "unknown"),
    )
    vectors_by_hash: dict[str, list[float]] = {}
    missing_payloads = []

    for payload in new_payloads:
        text_hash = payload["content_hash"]
        cached_vector = get_cached_embedding(embedding_model, text_hash)
        if cached_vector is None:
            missing_payloads.append(payload)
            continue
        vectors_by_hash[text_hash] = cached_vector

    if missing_payloads:
        embedded_vectors = embed_documents_resilient(
            embeddings,
            [payload["content"] for payload in missing_payloads],
        )
        for payload, vector in zip(missing_payloads, embedded_vectors):
            text_hash = payload["content_hash"]
            vectors_by_hash[text_hash] = vector
            set_cached_embedding(embedding_model, text_hash, vector)

    vectors = [vectors_by_hash[payload["content_hash"]] for payload in new_payloads]
    points = [
        PointStruct(
            id=payload["chunk_id"],
            vector=vector,
            payload=payload,
        )
        for payload, vector in zip(new_payloads, vectors)
    ]

    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True,
    )
    return len(points)
