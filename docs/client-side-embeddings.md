# Client-side face embeddings

The browser and the mobile app now produce the 512-d face embedding **on the device**. The server only
stores and searches vectors; it never receives a photo. This document explains why the old flow was slow and
inaccurate, how the new pipeline works, and what to keep in mind.

## 1. What made the old webcam flow slow

| Where | Problem | Effect |
| --- | --- | --- |
| Login loop | Every animation frame: detect, crop, JPEG-encode, upload. The worker's "queue" never limited concurrent requests | up to one upload per processed frame (tens per second), server backlog, growing latency |
| Server | RetinaFace-ResNet50 + ArcFace-r100 on CPU for *each* uploaded crop, after re-detecting the face in an already tight crop | estimated several hundred ms to over a second per request on CPU (not measured here: the weights are not in the repo) |
| Registration | `await registerFace()` (upload + server inference) inside the frame loop | camera froze while the server worked |
| React | `setIsProcessing(true/false)` on every frame, canvas resized every frame, rAF callbacks capturing stale state (`frsStatus`, model) | 2 re-renders per frame, flicker, loops that never stopped |
| Model loading | BlazeFace (TF.js) loaded per page, WebGPU/WebGL backend init each time | slow page start |

## 2. Why accuracy suffered

* **No alignment.** ArcFace models are trained on faces warped onto a fixed 5-point template. The client
  sent a loose, unaligned box and the server re-detected landmarks on a small JPEG.
* **No quality gate.** Blurry, dark, tiny or half visible faces were embedded and compared.
* **Single frame decisions** - one noisy frame decided login.
* **Single sample enrolment** with no pose control.
* **Every registration overwrote the previous one**: the form used a hard-coded `personId: 123` as the
  Qdrant point id.

## 3. New pipeline

```
camera frame (640x480, front camera)
  -> landmarks + head pose        web: MediaPipe FaceLandmarker (GPU, tracking)   mobile: ML Kit
  -> 5 alignment points            eyes, nose tip, mouth corners (ordered left -> right in the image)
  -> similarity transform          least squares (Umeyama) onto the ArcFace 112x112 template
  -> aligned 112x112 crop          web: one canvas drawImage with setTransform   mobile: sampled from raw NV21/BGRA
  -> quality gate                  eye distance, Laplacian sharpness, brightness, framing (+ eyes open on mobile)
  -> embedding                     w600k_mbf.onnx (MobileFaceNet, 13.6 MB) + horizontal flip TTA, L2 normalised
                                   web: ONNX Runtime Web in a Web Worker (WebGPU -> multi-threaded WASM)
                                   mobile: ONNX Runtime Mobile (XNNPACK -> CPU)
  -> server                        POST /api/v1/embeddings/{register|identify} (base64 float32, ~2.7 KB)
```

The same model file is used everywhere, and every step is implemented identically in Python
(`backend/frs/src/face_pipeline`, the reference), TypeScript (`frontend/frs_react/src/face/core`) and Dart
(`mobile/face_app/lib/core`). All three are tested against the same golden values in
`shared/test-vectors/face-pipeline.json` (regenerate with `python tools/generate_test_vectors.py`).

Measured parity: a face registered in the browser (MediaPipe landmarks, canvas warp, ORT Web) and identified
from Python (YuNet landmarks, OpenCV warp, ORT) scores **0.98** cosine similarity; the Dart warp from a rotated
NV21 buffer scores **0.998** against the Python reference. A different person scores around 0.

### Registration vs identification

| | Registration | Identification |
| --- | --- | --- |
| Head pose (roll, pitch, yaw) | required - guided capture: 3 frontal, 2 per side, 1 up, 1 down | **not used** |
| Pose reference | non-frontal targets are relative to the user's own frontal pose (cancels camera placement) | - |
| Consistency | each sample must match the samples collected so far (no person swap) | 3 consecutive frames must agree, then they are fused |
| Sent | 9 embeddings + pose metadata | 1 fused embedding |
| Stored (Qdrant) | 9 sample points + 1 fused template per person, cosine distance, one collection per model | - |
| Decision | - | best person score >= `FRS_MATCH_THRESHOLD` (0.45) and >= `FRS_MATCH_MARGIN` (0.05) above the runner-up |

Head pose convention (ML Kit's, used everywhere): degrees on the non-mirrored frame, yaw > 0 = face turned to
the image right (the user's own left), pitch > 0 = looking up, roll > 0 = counter-clockwise. The web client
reads it from MediaPipe's facial transformation matrix; the landmark depth based estimate had a ~12 degree
pitch bias on a frontal photo.

### Webcam performance techniques used

* `requestVideoFrameCallback` - work only when a new camera frame exists; frames arriving while one is being
  processed are dropped instead of queued.
* MediaPipe in VIDEO mode tracks the face, so the detector rarely runs; landmarks take a few ms on a GPU.
* The aligned crop is produced by a single GPU-accelerated `drawImage` into a 112x112 canvas - no full frame
  readback.
* Embeddings are computed in a Web Worker (inference never blocks rendering), at most 8 per second, one in
  flight, only for frames that pass the pose/quality gates. Pixel buffers are transferred, not copied.
* One `FaceEngine` per browser session, preloaded while the user is still typing on the login/registration
  form; the network and WASM files are cached (`Cache-Control`), and the page is cross-origin isolated so ONNX
  Runtime can use multi-threaded WASM when WebGPU is not available.
* React state is only updated with values that change the UI (React bails out on identical strings); the
  overlay is drawn imperatively on a canvas. The pipeline inspector refreshes its table at ~8 Hz.
* The server does no image work anymore: a request is a vector search (milliseconds).

Use `/examples/pipeline` to see the timings, pose, quality metrics and the aligned crop live.

## 4. Security considerations (read before production)

Moving inference to the client changes the trust model: **the server can no longer tell whether a vector came
from a live camera.** Anyone who obtains a stored or transmitted embedding can replay it, so treat embeddings
like credentials and biometric data:

* Use face login as a second factor, or add **liveness** - e.g. a random challenge (blink / turn to a side the
  server picks) verified on the device, or a passive anti-spoofing model.
* On mobile, bind requests to a genuine app with **Play Integrity / App Attest**, and sign requests.
* Serve the API over **https**, rate limit `/identify`, and don't return candidate scores to untrusted clients.
* Encrypt embeddings at rest and let users delete them (`DELETE /api/v1/embeddings/persons/{id}`).
* `personIdFromEmail` is a demo helper; use the id of your user database.

## 5. Model licence

`w600k_mbf.onnx` comes from InsightFace's `buffalo_s` pack, released for non-commercial research use. For a
commercial product train / license your own model, export it with `backend/frs/tools/export_onnx.py` and
register its id in `FRS_EMBEDDING_MODELS`.
