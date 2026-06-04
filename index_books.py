import logging
import os
from pathlib import Path

from src.qdrant_store import (
    detect_embedding_dimension,
    ensure_collection,
    get_embeddings,
    get_qdrant_client,
    get_settings,
    iter_pdf_chunk_payloads,
    upsert_payloads,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _batch(items, size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def resolve_books_dir(raw_books_dir: Path) -> Path:
    if raw_books_dir.is_absolute():
        return raw_books_dir
    return Path.cwd() / raw_books_dir


def main() -> None:
    settings = get_settings()
    books_dir = resolve_books_dir(settings.books_dir)
    if not books_dir.exists():
        raise RuntimeError(f"Books directory does not exist: {books_dir}")

    pdfs = sorted(books_dir.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDF files found in {books_dir}")

    embeddings = get_embeddings(settings)
    vector_size = detect_embedding_dimension(embeddings)
    client = get_qdrant_client(settings)
    ensure_collection(client, settings.collection, vector_size)

    batch_size = int(os.getenv("INDEX_BATCH_SIZE", "16"))
    total_seen = 0
    total_inserted = 0

    logger.info(
        "Indexing %s PDF files into Qdrant collection '%s'.",
        len(pdfs),
        settings.collection,
    )

    for pdf_path in pdfs:
        logger.info("Processing %s", pdf_path.name)
        payloads = iter_pdf_chunk_payloads(pdf_path, settings)

        for payload_batch in _batch(payloads, batch_size):
            total_seen += len(payload_batch)
            inserted = upsert_payloads(
                client=client,
                collection_name=settings.collection,
                embeddings=embeddings,
                payloads=payload_batch,
            )
            total_inserted += inserted
            skipped = len(payload_batch) - inserted
            logger.info(
                "Processed %s chunks, inserted %s, skipped duplicates %s.",
                total_seen,
                total_inserted,
                skipped,
            )

    logger.info(
        "Finished. Seen chunks: %s. Inserted new chunks: %s.",
        total_seen,
        total_inserted,
    )


if __name__ == "__main__":
    main()
