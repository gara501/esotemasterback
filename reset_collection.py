import logging

from qdrant_client.models import Distance, VectorParams

from src.qdrant_store import (
    collection_exists,
    detect_embedding_dimension,
    get_embeddings,
    get_qdrant_client,
    get_settings,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    embeddings = get_embeddings(settings)
    vector_size = detect_embedding_dimension(embeddings)
    client = get_qdrant_client(settings)

    expected = f"RESET {settings.collection}"
    confirmation = input(
        f"This will delete and recreate Qdrant collection "
        f"'{settings.collection}'. Type '{expected}' to continue: "
    )
    if confirmation != expected:
        logger.info("Confirmation did not match. Nothing was changed.")
        return

    if collection_exists(client, settings.collection):
        client.delete_collection(settings.collection)
        logger.info("Deleted collection %s.", settings.collection)

    client.create_collection(
        collection_name=settings.collection,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )
    logger.info(
        "Created collection %s with vector size %s.",
        settings.collection,
        vector_size,
    )


if __name__ == "__main__":
    main()
