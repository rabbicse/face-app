# Face Recognition - web app (Next.js)

Face registration and login where the **browser** computes the 512-d face embedding. Only the vector is sent
to the backend (`/api/v1/embeddings/*`).

```
webcam (requestVideoFrameCallback)
  -> MediaPipe FaceLandmarker (GPU, VIDEO mode with tracking): 478 landmarks + facial transformation matrix
  -> 5 alignment points + head pose (yaw / pitch / roll)
  -> 112x112 aligned crop: a single canvas drawImage with the similarity transform
  -> quality gate: eye distance, sharpness (Laplacian variance), brightness, framing
  -> Web Worker: ONNX Runtime Web (WebGPU, else multi-threaded WASM), w600k_mbf.onnx + flip TTA
  -> L2 normalised 512-d embedding -> backend
```

## Run

```bash
npm install
npm run dev        # predev copies the WASM runtimes and downloads the models into public/ (once)
```

Configuration (`.env.local`): `NEXT_PUBLIC_API_HOST` (default `http://localhost`), `NEXT_PUBLIC_API_PORT`
(`5000`) or `NEXT_PUBLIC_API_BASE_URL`; `NEXT_PUBLIC_EMBEDDING_MODEL_URL` / `NEXT_PUBLIC_EMBEDDING_MODEL_ID` to
use your own model (see `backend/frs/tools/export_onnx.py`).

Pages: `/registration` -> `/face/registration` (guided multi-pose capture), `/login` -> `/face/login`
(identification, no pose requirement), `/examples` (pipeline inspector, photo comparison, quick registration).

The app sends `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` headers (see `next.config.ts`) so
ONNX Runtime can use threads; resources from other origins need CORS/CORP headers.

## Code

| Path | |
| --- | --- |
| `src/face/core` | framework-free, unit tested: alignment, pose, quality, embedding math, registration session, identification aggregator |
| `src/face/runtime` | browser runtime: camera loop, MediaPipe landmarker, canvas warp, ONNX embedder (worker), `FaceEngine` |
| `src/face/react` | hooks (`useFaceEngine` with preloading, `useCamera`, `useVideoFrames`), overlay, camera view |
| `src/face/api.ts` | backend client |
| `src/components/face` | registration and login screens |
| `src/app/examples` | examples |

## Tests

```bash
npm test               # vitest: core + api client, checked against ../../shared/test-vectors
npm run lint
npx tsc --noEmit
```

`e2e/face-flow.mjs` runs registration and login in Chromium with a fake webcam video (see the header of the
file for the steps).
