import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DB = PROJECT_ROOT / "data" / "rag_cache.sqlite"


def env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def get_cache_db_path() -> Path:
    raw_path = os.getenv("RAG_CACHE_DB", "").strip()
    if not raw_path:
        return DEFAULT_CACHE_DB

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def stable_cache_key(namespace: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


def _connect() -> sqlite3.Connection:
    db_path = get_cache_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS embedding_cache (
            cache_key TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            vector_json TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS response_cache (
            cache_key TEXT PRIMARY KEY,
            response_json TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            expires_at INTEGER
        )
        """
    )
    connection.commit()
    return connection


def get_cached_embedding(model: str, content_hash: str) -> list[float] | None:
    if not env_bool("ENABLE_LOCAL_CACHE", True) or not env_bool(
        "ENABLE_EMBEDDING_CACHE",
        True,
    ):
        return None

    cache_key = stable_cache_key(
        "embedding",
        {"model": model, "content_hash": content_hash},
    )
    with _connect() as connection:
        row = connection.execute(
            "SELECT vector_json FROM embedding_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()

    if not row:
        return None
    return json.loads(row[0])


def set_cached_embedding(
    model: str,
    content_hash: str,
    vector: list[float],
) -> None:
    if not env_bool("ENABLE_LOCAL_CACHE", True) or not env_bool(
        "ENABLE_EMBEDDING_CACHE",
        True,
    ):
        return

    cache_key = stable_cache_key(
        "embedding",
        {"model": model, "content_hash": content_hash},
    )
    with _connect() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO embedding_cache
            (cache_key, model, content_hash, vector_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                model,
                content_hash,
                json.dumps(vector, separators=(",", ":")),
                int(time.time()),
            ),
        )


def get_cached_response(cache_key: str) -> str | None:
    if not env_bool("ENABLE_LOCAL_CACHE", True) or not env_bool(
        "ENABLE_RESPONSE_CACHE",
        True,
    ):
        return None

    now = int(time.time())
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT response_json, expires_at
            FROM response_cache
            WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()

        if not row:
            return None

        response_json, expires_at = row
        if expires_at is not None and expires_at < now:
            connection.execute(
                "DELETE FROM response_cache WHERE cache_key = ?",
                (cache_key,),
            )
            return None

    return response_json


def set_cached_response(cache_key: str, response_json: str) -> None:
    if not env_bool("ENABLE_LOCAL_CACHE", True) or not env_bool(
        "ENABLE_RESPONSE_CACHE",
        True,
    ):
        return

    ttl_seconds = env_int("RESPONSE_CACHE_TTL_SECONDS", 86400)
    now = int(time.time())
    expires_at = now + ttl_seconds if ttl_seconds > 0 else None
    with _connect() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO response_cache
            (cache_key, response_json, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (cache_key, response_json, now, expires_at),
        )
