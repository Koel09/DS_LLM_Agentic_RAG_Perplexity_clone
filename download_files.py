import os
from pathlib import Path
import requests


# GitHub raw directory containing PDFs
GITHUB_PDF_BASE_URL = (
    "https://raw.githubusercontent.com/Koel09/DS_LLM_Agentic_RAG_Perplexity_clone/main/crew_data"
)

# List PDF filenames to download
PDF_FILES = [
    "doc.pdf",
    # add more here
]

PDF_DIR = Path("data/pdfs")          # folder with PDFs


def download_rag_pipeline():
    rag_pipeline_url = "https://raw.githubusercontent.com/Koel09/DS_LLM_Agentic_RAG_Perplexity_clone/main/rag_pipeline.py"
    response = requests.get(rag_pipeline_url)
    with open("rag_pipeline.py", "w", encoding="utf-8") as f:
        f.write(response.text)

def download_pdf_ingest():
    pdf_ingest_url = "https://raw.githubusercontent.com/Koel09/DS_LLM_Agentic_RAG_Perplexity_clone/main/pdf_ingest.py"
    response = requests.get(pdf_ingest_url)
    with open("pdf_ingest.py", "w", encoding="utf-8") as f:
        f.write(response.text)

def download_pdfs():
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    for pdf in PDF_FILES:
        local_path = PDF_DIR / pdf

        if local_path.exists():
            print(f"✓ {pdf} already exists — skipping")
            continue

        url = f"{GITHUB_PDF_BASE_URL}/{pdf}"
        print(f"⬇ Downloading {pdf}")

        r = requests.get(url)
        r.raise_for_status()

        with open(local_path, "wb") as f:
            f.write(r.content)

        print(f"✓ Saved {pdf}")


download_pdf_ingest()
download_rag_pipeline()
download_pdfs()
