# The HR policy assistant API. Built and run by docker compose (see docker-compose.yml).
#
# CPU-only: PyTorch comes from its CPU wheel index (see [tool.uv.sources] in pyproject.toml),
# which keeps the image about 3 GB smaller than the default CUDA build. Models (bge-m3, the
# reranker, Docling's layout models) are downloaded on first start into the hf_cache volume,
# not baked into the image.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/models/huggingface

# libgl and libglib are needed by Docling's image processing (opencv).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY config.yaml ./
COPY data/pdfs ./data/pdfs
COPY scripts ./scripts

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /models /app/logs /app/.data /app/.cache \
    && chown -R app:app /models /app/logs /app/.data /app/.cache
USER app

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Keep-alive 30 s: model loading (e.g. Docling on the first upload) can briefly starve the event
# loop, and the default 5 s then drops idle client connections mid-poll.
CMD ["uvicorn", "--factory", "hybrid_rag.api.app:create_app", "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "30"]
