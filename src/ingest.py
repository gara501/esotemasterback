from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


BOOKS_DIR = Path("books")
DB_DIR = "chroma_db"
COLLECTION_NAME = "esoteric_books"


def ingest():
    docs = []

    for pdf in BOOKS_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = pdf.stem
            page.metadata["file_name"] = pdf.name

        docs.extend(pages)

    if not docs:
        print("No se encontraron PDFs en la carpeta books/")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
    )

    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="bge-m3")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name=COLLECTION_NAME,
    )

    print(f"Indexed {len(chunks)} chunks from {len(docs)} pages.")


if __name__ == "__main__":
    ingest()