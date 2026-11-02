# syntax=docker/dockerfile:1.7
# API image: stores / searches face embeddings computed on the client devices (no torch, no weights).
#   docker build -t frs-api .
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
COPY requirements-api.txt .
RUN --mount=type=cache,target=/root/.cache/pip /opt/venv/bin/pip install -r requirements-api.txt

FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    QDRANT_HOST=qdrant \
    QDRANT_PORT=6333 \
    FRS_WORKERS=2
RUN useradd --system --uid 10001 --no-create-home frs
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY main.py ./
COPY src ./src
USER frs
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4).status != 200)"
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers ${FRS_WORKERS} --proxy-headers"]
