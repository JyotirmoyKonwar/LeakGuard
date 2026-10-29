# syntax=docker/dockerfile:1.7
# Multi-target image.
#   --target api   FastAPI guard (+ GLiNER + Laya on CPU)
#   --target ui    Streamlit dashboard
#   --target space API + UI in one container (Hugging Face Spaces, port 7860)
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    HF_HOME=/models USE_TF=0 PATH="/app/.venv/bin:$PATH"
COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /usr/local/bin/uv
WORKDIR /app
RUN useradd -m -u 1000 app && mkdir -p /models && chown app /models

FROM base AS deps-api
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project --extra ml

FROM base AS deps-ui
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project --extra ui

FROM deps-api AS api
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra ml
# bake model weights into the image so cold starts don't download ~1.5 GB (skip with --build-arg PRELOAD=0)
ARG PRELOAD=1
RUN if [ "$PRELOAD" = "1" ]; then python -c "from gliner import GLiNER; GLiNER.from_pretrained('urchade/gliner_multi_pii-v1'); import laya; laya.load('convaiinnovations/laya-multilingual', device='cpu')"; fi
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "leakguard.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

FROM deps-ui AS ui
COPY ui ./ui
COPY eval/results ./eval/results
USER app
EXPOSE 8501
CMD ["streamlit", "run", "ui/Playground.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]

FROM api AS space
USER root
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra ml --extra ui
COPY ui ./ui
COPY eval/results ./eval/results
COPY deploy/hf-space/start.sh /start.sh
RUN chmod +x /start.sh && chown -R app /app
USER app
ENV LEAKGUARD_API=http://127.0.0.1:8000
EXPOSE 7860
CMD ["/start.sh"]
