import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# =========================
# CONFIG
# =========================

PDF_DIR = "data/pdfs"          # folder with PDFs
CHROMA_DIR = "data/chroma"     # must match rag_pipeline.py
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# =========================
# LOAD EMBEDDINGS
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL
)


# =========================
# LOAD / CREATE CHROMA
# =========================

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)


# =========================
# INGEST PDFs
# =========================

def ingest_pdfs():
    pdf_path = Path(PDF_DIR)
    pdf_files = list(pdf_path.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDFs found in {PDF_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_chunks = []

    for pdf in pdf_files:
        print(f"📄 Loading {pdf.name}")
        loader = PyPDFLoader(str(pdf))
        docs = loader.load()

        chunks = splitter.split_documents(docs)

        # add metadata for traceability
        for c in chunks:
            c.metadata["source"] = pdf.name

        all_chunks.extend(chunks)

    print(f"🧩 Total chunks: {len(all_chunks)}")

    vectorstore.add_documents(all_chunks)
    # vectorstore.persist()

    print("✅ PDF ingestion complete")


if __name__ == "__main__":
    ingest_pdfs()
