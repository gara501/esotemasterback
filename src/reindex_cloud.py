import os
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


BOOKS_DIR = Path("books")


load_dotenv()

COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "esoteric_books_google")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
BATCH_SIZE = int(os.getenv("REINDEX_BATCH_SIZE", "5"))
RESET_COLLECTION = os.getenv("RESET_COLLECTION", "false").lower() == "true"
REINDEX_SLEEP_SECONDS = float(os.getenv("REINDEX_SLEEP_SECONDS", "2"))


def load_documents():
    docs = []

    for pdf in BOOKS_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = pdf.stem
            page.metadata["file_name"] = pdf.name

        docs.extend(pages)

    if not docs:
        raise RuntimeError("No PDFs found in books/.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
    )

    return splitter.split_documents(docs)


def main():
    chunks = load_documents()
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    client = chromadb.CloudClient(
        api_key=os.environ["CHROMA_API_KEY"],
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
    )

    if RESET_COLLECTION:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing cloud collection: {COLLECTION_NAME}")
        except Exception:
            pass

    collection = client.get_or_create_collection(COLLECTION_NAME)
    start_offset = collection.count()
    if start_offset:
        print(
            f"Collection {COLLECTION_NAME} already has {start_offset} documents. "
            f"Resuming from offset {start_offset}."
        )

    for offset in range(start_offset, len(chunks), BATCH_SIZE):
        batch = chunks[offset:offset + BATCH_SIZE]
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        ids = [f"{doc.metadata.get('source', 'source')}-{offset + index}" for index, doc in enumerate(batch)]
        vectors = embeddings.embed_documents(texts)

        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=vectors,
        )

        print(f"Uploaded {min(offset + len(batch), len(chunks))} / {len(chunks)}")
        if REINDEX_SLEEP_SECONDS > 0:
            time.sleep(REINDEX_SLEEP_SECONDS)

    print(f"Cloud documents: {collection.count()}")


if __name__ == "__main__":
    main()
