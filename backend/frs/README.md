# FRS API (backend)

FastAPI service that **stores and searches face embeddings** in Qdrant. The web and mobile clients compute
the 512-d embedding on the device (MobileFaceNet ArcFace `w600k_mbf`), so this service needs no GPU, no
PyTorch and no model weights.

> The previous server-side inference stack (RetinaFace + ArcFace r100 on PyTorch, Flask app, image
> endpoints) lives on the [`stable-v01`](https://github.com/rabbicse/face-app/tree/stable-v01) branch.

## Run

```bash
# Docker (whole stack, from the repo root)
docker compose up -d

# Local, with a throw-away in-memory database
pip install -r requirements-api.txt        # or: uv sync
QDRANT_LOCATION=:memory: uvicorn main:app --reload --port 5000

# Local, with a persistent Qdrant
docker run -d -p 6333:6333 -v frs_qdrant:/qdrant/storage qdrant/qdrant:v1.20.0
uvicorn main:app --reload --port 5000
```

Interactive docs: `http://localhost:5000/docs`. Deploying for free (Vercel + Qdrant Cloud):
[../../docs/deploy-free.md](../../docs/deploy-free.md).

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /api/v1/embeddings/register` | store a person's guided registration samples plus a fused template, replacing older ones |
| `POST /api/v1/embeddings/identify` | best matching person (per-person max score, threshold + margin) |
| `DELETE /api/v1/embeddings/persons/{id}?model=` | forget a person |
| `GET /api/v1/health` | status and accepted embedding models |

Each embedding model gets its own Qdrant collection (`face_embeddings_<model>`, cosine distance), so vectors
of different models are never compared. Embeddings are a JSON array of 512 floats or base64 of
little-endian float32. Request/response examples: [../../docs/guide.md](../../docs/guide.md#5-use-the-api).

## Configuration

| Environment variable | Default | |
| --- | --- | --- |
| `QDRANT_LOCATION` | - | `:memory:` or a directory for embedded Qdrant (tests and single-process demos only) |
| `QDRANT_URL` / `QDRANT_API_KEY` | - | remote Qdrant, e.g. Qdrant Cloud (takes precedence over host / port) |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | Qdrant server |
| `FRS_EMBEDDING_MODELS` | `w600k_mbf` | accepted model ids (comma separated) |
| `FRS_MATCH_THRESHOLD` | `0.45` | minimum cosine similarity for a match |
| `FRS_MATCH_MARGIN` | `0.05` | minimum gap to the second best person |
| `FRS_CORS_ORIGINS` | `*` | allowed origins (comma separated) |
| `FRS_LOG_LEVEL` | `INFO` | log level |
| `FRS_WORKERS` | `2` | uvicorn worker processes (Docker image only) |

## Layout

* `main.py` - ASGI entrypoint (`uvicorn main:app`, Docker, Vercel)
* `src/api` - FastAPI app, routes, the embedding service and config
* `src/vector_db/embedding_store.py` - Qdrant collections, upsert / search / delete
* `src/face_pipeline` - the client pipeline in Python (alignment, pose, preprocessing, embedding math). It is
  the reference for `../../shared/test-vectors/face-pipeline.json` (`python tools/generate_test_vectors.py`).
* `examples/client_embedding_demo.py` - embed / compare / register / identify / benchmark photos exactly like
  the clients do
* `tools/` - golden vector generator, ONNX fp16 compression

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```
