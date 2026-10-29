# Face Recognition System Backend (FRS)
Welcome to the Face Recognition System (FRS) Backend, a robust and efficient face detection and recognition service powered by cutting-edge technologies.

## 🚀 Features
- Face Detection: Detects and localizes faces within images.
- Face Recognition: Identifies individuals using face embeddings.
- Keypoint Extraction: Extracts facial landmarks for additional processing.
- Fast and Scalable: Designed to handle large volumes of data with low latency.
- Modular Design: Easily extensible for future enhancements.

## 🛠️ Technology Stack
- Framework: FastAPI (Python)
- Machine Learning: PyTorch
- RDBMS: PostgreSQL
- Vector Database: Qdrant
- Containerization: Docker

## 🧰 Prerequisites
Ensure you have the following installed on your system:
- Python 3.10+
- Docker & Docker Compose
- NVIDIA CUDA Toolkit (for GPU support)

## 🏗️ Installation
1. Clone the Repository
  
  ```
  git clone https://github.com/rabbicse/face-app.git  
  cd face-app/backend/frs  
  ```
2. Set Up Environment Variables. Create a `.env` file in the project root with the following:

  ```
  DATABASE_URL=postgresql://user:password@localhost:5432/frs_db  
  SECRET_KEY=your-secret-key  
  ALGORITHM=HS256  
  ACCESS_TOKEN_EXPIRE_MINUTES=30  
  ```
3. Install Dependencies

```
pip install -r requirements.txt  
```
4. Run the Application. Using Docker:
```
docker-compose up --build
```
Local Development:
```bash
uvicorn main:app --reload
```
The API will be accessible at: `http://localhost:5000`

## ⚡ Client-side embeddings (recommended)

The web and mobile clients compute the 512-d face embedding on the device; this service only stores and
searches vectors, so it needs neither a GPU nor PyTorch:

```bash
pip install -r requirements-api.txt
cd src
QDRANT_LOCATION=:memory: FRS_ENABLE_SERVER_INFERENCE=false uvicorn api.server:app --port 5000
# or build the slim image: docker build -f Dockerfile.api -t frs-api .
```

| Endpoint | Purpose |
| --- | --- |
| `POST /api/v1/embeddings/register` | store the guided registration samples (+ a fused template) of a person, replacing older ones |
| `POST /api/v1/embeddings/identify` | best matching person (per-person max score, threshold + margin) |
| `DELETE /api/v1/embeddings/persons/{id}?model=` | forget a person |
| `GET /api/v1/health` | enabled embedding models and modes |

Each embedding model gets its own Qdrant collection (`face_embeddings_<model>`, cosine distance): vectors of
different models are never compared.

| Environment variable | Default | |
| --- | --- | --- |
| `FRS_ENABLE_SERVER_INFERENCE` | `true` | also mount the legacy image endpoints (needs torch + weights; skipped automatically if missing) |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | Qdrant server |
| `QDRANT_LOCATION` | - | `:memory:` or a directory for embedded Qdrant (tests, demos) |
| `FRS_EMBEDDING_MODELS` | `w600k_mbf` | accepted model ids (comma separated) |
| `FRS_MATCH_THRESHOLD` | `0.45` | minimum cosine similarity for a match |
| `FRS_MATCH_MARGIN` | `0.05` | minimum gap to the second best person |
| `FRS_CORS_ORIGINS` | `*` | allowed origins |

### Reference pipeline, examples and tools

* `src/face_pipeline` - the client pipeline in Python (alignment, pose, preprocessing, embedding math); the
  source of `../../shared/test-vectors/face-pipeline.json` (`python tools/generate_test_vectors.py`).
* `examples/client_embedding_demo.py` - embed / compare / register / identify / benchmark photos exactly like
  the clients do (YuNet landmarks + ONNX Runtime).
* `tools/export_onnx.py` - export a backbone of `src/recognition/arc_face` to ONNX for the clients.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## 📄 API Documentation
Once the server is running, you can access the interactive API docs at:

Swagger UI: `http://localhost:8000/docs`
ReDoc: `http://localhost:8000/redoc`
