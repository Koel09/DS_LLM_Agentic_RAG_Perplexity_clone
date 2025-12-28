# -------------------------
# Base image
# -------------------------
FROM python:3.11-slim

# -------------------------
# Environment variables
# -------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -------------------------
# Set work directory
# -------------------------
WORKDIR /app

# -------------------------
# Install system dependencies
# -------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Copy dependency file
# -------------------------
COPY requirements.txt .

# -------------------------
# Install Python dependencies
# -------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# Copy ALL project files
# -------------------------
COPY . .

# -------------------------
# Create required directories
# -------------------------
RUN mkdir -p data/pdfs data/chroma template

# -------------------------
# Build vector store at image build time (optional but recommended)
# Comment this out if PDFs change frequently
# -------------------------
RUN python download_files.py || true
RUN python pdf_ingest.py || true

# -------------------------
# Expose Flask port
# -------------------------
EXPOSE 8000

# -------------------------
# Run Flask app
# -------------------------
CMD ["python", "app.py"]
