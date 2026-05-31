import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


BOOKS_DIR = Path("books")
COLLECTION_NAME = "esoteric_books"
BATCH_SIZE = 100


load_dotenv()

EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")


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

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing cloud collection: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.get_or_create_collection(COLLECTION_NAME)

    for offset in range(0, len(chunks), BATCH_SIZE):
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

    print(f"Cloud documents: {collection.count()}")


if __name__ == "__main__":
    main()
