FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        postgresql-client curl gnupg ffmpeg tesseract-ocr chromium && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    # Install Python dependencies declared in pyproject.toml
    pip install --no-cache-dir -e .

CMD ["clipped-master"]

