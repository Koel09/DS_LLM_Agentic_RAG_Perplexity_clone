# -------------------------
# Base image
# -------------------------
FROM python:3.11-slim

# -------------------------
# Environment settings
# -------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -------------------------
# System dependencies
# -------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Working directory
# -------------------------
WORKDIR /app

# -------------------------
# Install Python dependencies
# -------------------------
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -------------------------
# Copy application code
# -------------------------
COPY . .

# -------------------------
# Expose Flask port
# -------------------------
EXPOSE 5000

# -------------------------
# Environment variables (set at runtime)
# -------------------------
# GROQ_API_KEY
# TAVILY_API_KEY

# -------------------------
# Run the app
# -------------------------
CMD ["python", "app.py"]
