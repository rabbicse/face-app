# Face Recognition - Flutter (Android & iOS)

Registration and identification **entirely on the device**: the phone detects the face, estimates the head
pose, aligns the face and computes the 512-d ArcFace embedding. Only that vector (~2.7 KB, base64 float32)
is sent to the backend, which stores / searches it in Qdrant. It is the same pipeline and the **same model
file** as the web app, so a face registered on a phone can log in from a browser and vice versa.

```
CameraImage (NV21 / BGRA)
  -> Google ML Kit face detector (on-device): 5 landmarks, yaw/pitch/roll, eyes-open probability
  -> similarity transform to the ArcFace 5-point template
  -> 112x112 RGB crop sampled straight from the raw YUV/BGRA buffer (rotation handled in the mapping)
  -> quality gate (face size, sharpness, brightness, framing, eyes open)
  -> ONNX Runtime Mobile: w600k_mbf.onnx (MobileFaceNet) + flip TTA -> L2 normalised 512-d embedding
  -> POST /api/v1/embeddings/register | identify
```

| Flow | Head pose | What is sent |
| --- | --- | --- |
| Registration | Guided: 3x frontal, 2x each side, 1x up, 1x down. Measured *relative to the user's own frontal pose*; side and up/down samples are accepted in either order | 9 embeddings + pose metadata |
| Identification | none - any good-quality frame | 1 embedding fused from 3 consistent frames |

## Setup

```bash
cd mobile/face_app
tool/setup_models.sh          # copies / downloads assets/models/w600k_mbf.onnx (13.6 MB)
flutter pub get
```

Start the backend (no GPU / torch needed for embedding-only mode):

```bash
cd backend/frs/src
FRS_ENABLE_SERVER_INFERENCE=false uvicorn api.server:app --host 0.0.0.0 --port 5000
```

Run on a device or emulator (the default API URL `http://10.0.2.2:5000` is the host machine seen from the
Android emulator; use your computer's LAN IP for a real phone):

```bash
flutter run --dart-define=FRS_API=http://192.168.1.10:5000
```

### Platform notes

* **Android**: minSdk 24, camera + internet permissions and the ONNX Runtime ProGuard rule
  (`android/app/proguard-rules.pro`) are configured. `usesCleartextTraffic` is enabled for a local
  `http://` backend - use https in production.
* **iOS**: iOS 16+, static linkage (`ios/Podfile`), camera usage description and local networking ATS
  exception are configured. ML Kit does not run on the arm64 iOS simulator; use a real device.
* Frames are dropped while the previous one is processed, so latency stays at one frame and the UI never
  queues work. ML Kit runs in *accurate* mode for registration (needed for a reliable yaw) and in *fast*
  mode for identification.

## Tests

```bash
flutter analyze
flutter test
```

* `golden_vectors_test.dart` - alignment, pose, preprocessing and embedding math against
  `shared/test-vectors/face-pipeline.json`, the values the Python reference and the web app are tested with.
* `camera_frame_test.dart` - sampling the aligned crop from NV21, YUV420 and BGRA buffers for every sensor
  rotation.
* `sessions_test.dart` - registration plan (order, baseline-relative poses, consistency checks) and the
  multi-frame identification aggregator.
* `api_client_test.dart` - request / response format.

The NV21 path was additionally checked end to end against the real model: the crop sampled by
`warpFaceFromFrame` from a rotated NV21 buffer gives an embedding with cosine 0.998 to the Python reference.

## Model

`w600k_mbf.onnx` is InsightFace's MobileFaceNet trained on WebFace600K (from the `buffalo_s` pack). The
InsightFace pretrained models are licensed for **non-commercial research use only**. To use your own
weights, export them with `backend/frs/tools/export_onnx.py`, replace the file in the web app and here, and
register the new model id in the backend (`FRS_EMBEDDING_MODELS`) - embeddings of different models are
never comparable.
