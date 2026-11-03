# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Face registration / login where the **client device computes the 512-d face embedding** (MobileFaceNet ArcFace `w600k_mbf`) and the server only stores and searches vectors in Qdrant. It's a monorepo with three active parts. Old prototypes and the PyTorch server-side inference are only on the `stable-v01` branch. Design rationale, measurements and security notes are in `docs/client-side-embeddings.md`. User guide: `docs/guide.md`. Free deployment (Vercel + Qdrant Cloud): `docs/deploy-free.md`. Docker: `docker-compose.yml`, `docs/docker.md`.

| Path | Role |
| --- | --- |
| `frontend/frs_react` | Next.js 15 web app. The on-device pipeline is in `src/face` |
| `mobile/face_app` | Flutter (Android & iOS): ML Kit landmarks + ONNX Runtime Mobile |
| `backend/frs` | FastAPI embedding store/search, the Python **reference** pipeline |
| `shared/test-vectors/face-pipeline.json` | golden values that all three implementations are tested against |

## Commands

Backend (`backend/frs`; pytest config adds `src` and `tools` to `pythonpath`):
```bash
pip install -r requirements-dev.txt
pytest                                         # all tests (in-memory Qdrant, no model weights)
pytest tests/test_embedding_api.py -k identify # single file / test
QDRANT_LOCATION=:memory: uvicorn main:app --reload --port 5000
python tools/generate_test_vectors.py          # regenerate shared golden vectors
```
Only `test_*.py` files are collected.

Web (`frontend/frs_react`):
```bash
npm install
npm run dev               # predev downloads models/WASM into public/ (scripts/setup-face-assets.mjs)
npm test                  # vitest (src/**/*.test.ts, scripts/**/*.test.mjs)
npx vitest run src/face/core/pose.test.ts   # single file
npm run lint && npx tsc --noEmit
npm run build:static      # client-only static export in out/ (no Node server; see docs/guide.md)
```
Env vars are listed in `.env.example` (`NEXT_PUBLIC_API_BASE_URL`, `NEXT_PUBLIC_REQUIRE_BLINK`, `NEXT_PUBLIC_EMBEDDING_WEBGPU`). E2E with a fake webcam: `e2e/face-flow.mjs`. Setup instructions are in the file header, and it needs playwright.

Mobile (`mobile/face_app`):
```bash
tool/setup_models.sh      # fetches assets/models/w600k_mbf.onnx
flutter analyze && flutter test
flutter test test/liveness_test.dart
flutter run --dart-define=FRS_API=http://<lan-ip>:5000   # also REQUIRE_BLINK=false
```

CI (`.github/workflows/ci.yml`) runs exactly these: `pytest -q`; `npm test`, lint, `tsc`; `flutter analyze`, `flutter test`. `.github/workflows/docker.yml` builds the `frs-api` and `frs-web` images and pushes them to Docker Hub.

Full stack in one command: `./run.sh` (no Docker: venv + npm, Qdrant local mode in `.run/qdrant`, or remote via `QDRANT_URL`; `--prod`, `--memory`, `--no-blink`, `test`, `mobile`, `reset`) or `./docker.sh` (Compose: `start|stop|status|logs|reset`). Both pick free ports (macOS AirPlay holds 5000) and must stay bash 3.2 compatible. Plain `docker compose up -d` (http://localhost:3000) also works. The web image proxies `/api/v1/*` to the `api` service (`FRS_API_PROXY_URL`, baked at build time).

## Architecture

**One pipeline, three implementations that must stay identical.** The pipeline goes landmarks → 5 points → Umeyama similarity transform onto the ArcFace 112×112 template → quality gate → embedding (+ flip TTA) → L2 normalize. It exists in:
- Python `backend/frs/src/face_pipeline/` (reference)
- TypeScript `frontend/frs_react/src/face/core/`
- Dart `mobile/face_app/lib/core/`

Any change to alignment, preprocessing, pose, embedding fusion/encoding, liveness or registration/identification logic must be mirrored in all three. Then regenerate `shared/test-vectors/face-pipeline.json` and keep `conftest.py`, `src/face/core/testVectors.ts` and `test/golden_vectors_test.dart` passing. Web, mobile and backend share one vector space, so the model id (`w600k_mbf`) must match across `src/face/config.ts`, `lib/config.dart` and backend `FRS_EMBEDDING_MODELS`.

**Web `src/face` layering:**
- `core/` is pure, framework-free logic and is unit tested
- `runtime/` is browser-only: the MediaPipe landmarker, the canvas warp, and the ONNX Runtime Web embedder running in a Web Worker (`embedding.worker.ts`, `embedderClient.ts`), all wired by `engine.ts`
- `react/` holds hooks and components (`useFaceEngine`, `FaceCamera`, `FacePreloader`)

There is one `FaceEngine` per browser session, preloaded on the forms. The page must stay cross-origin isolated (headers in `next.config.ts`) so multithreaded WASM works. Frame handling is performance-sensitive (`requestVideoFrameCallback`, drop frames rather than queue them, ≤8 embeddings/s with one in flight, imperative canvas overlay, minimal React state). Read the performance section of the design doc before changing it.

**Mobile** mirrors the web: `lib/vision/` has the camera adapter, ML Kit detector, ONNX embedder and `face_engine.dart`; `lib/core/` has the logic; `lib/api/` has the client.

**Flows:**
- Registration is guided by pose, relative to the user's own frontal pose: 3 frontal, 2 per side, 1 up, 1 down. That stores 9 sample points plus 1 fused template.
- Identification ignores pose. It fuses 3 consistent frames and needs a blink within 5 s. A match must reach `FRS_MATCH_THRESHOLD` (0.45) and beat the runner-up person by `FRS_MATCH_MARGIN` (0.05).
- Head-pose convention everywhere (ML Kit's): degrees on the non-mirrored frame. Yaw > 0 means the face is turned to the image's right, pitch > 0 means up, roll > 0 means counter-clockwise.

**Backend:**
- `main.py` is the ASGI entrypoint and is also Vercel's (via `[tool.vercel]` in `pyproject.toml`). It puts `src/` on the path and imports `api.server:app`.
- The embedding routes (`/api/v1/embeddings/{register,identify,persons/{id}}`) use `api/services/embedding_service.py` and `vector_db/embedding_store.py`. Each model gets its own collection: `face_embeddings_<model>`, cosine distance.
- Embeddings arrive as a JSON float array or as base64 little-endian float32.
- `api/config.py` is all env-driven. Qdrant selection follows `QDRANT_LOCATION` (`:memory:`/path) → `QDRANT_URL`+`QDRANT_API_KEY` → `QDRANT_HOST`/`PORT`.
- Dependencies are lightweight and pinned twice: `pyproject.toml`/`uv.lock` (Vercel) and `requirements-api.txt` (Docker, CI). Keep them in sync, and don't add heavy imports.

**Models:** the browser loads `w600k_mbf.fp16.onnx`, produced by `scripts/onnx-fp16.mjs` / `tools/compress_onnx.py`. Mobile uses the fp32 `w600k_mbf.onnx`. Model files are downloaded by the setup scripts and are not committed.
