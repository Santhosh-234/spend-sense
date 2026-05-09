FROM python:3.11-slim

# System deps for EasyOCR (libGL) and Tesseract fallback
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create runtime directories
RUN mkdir -p data/raw data/processed data/vector_store data/models logs

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Default: run the directory watcher
ENTRYPOINT ["python", "main.py"]
CMD ["watch"]
