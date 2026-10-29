# Lightweight image: embedding store / search only (clients compute the embeddings).
FROM python:3.12-slim

WORKDIR /app
COPY requirements-api.txt /app/requirements-api.txt
RUN pip install --no-cache-dir -r /app/requirements-api.txt

COPY src /app/src

ENV PYTHONUNBUFFERED=1 \
    FRS_ENABLE_SERVER_INFERENCE=false

WORKDIR /app/src
EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
