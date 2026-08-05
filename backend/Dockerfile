# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

# ffmpeg: browser audio arrives as WebM/Opus and Azure's recognizer cannot read
#         it, so transcoding is a hard requirement, not an optimisation.
# libgomp1: OpenMP runtime that torch links against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first, as its own layer — it is ~200 MB and rarely changes, so
# keeping it separate means an app-code edit doesn't re-download it.
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the model weights into the image so the first request doesn't pay for a
# ~500 MB download. Adds build time; removes cold-start latency in production.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
CrossEncoder('BAAI/bge-reranker-base'); \
print('models cached')"

COPY . .

RUN mkdir -p data/uploads data/tts_audio && \
    chmod +x bin/rhubarb 2>/dev/null || true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/live || exit 1

# Single worker: each worker loads its own copy of the embedder and both
# cross-encoders (~1.2 GB resident). Scale with replicas, not workers.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
