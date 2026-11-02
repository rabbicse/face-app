# Face sign-in that keeps faces on the device

Register and sign in with a face from the browser or a phone. The device does the recognition work. Photos and
video never leave it: the server only stores and searches numeric face signatures.

<p align="center">
  <img src="docs/images/08-signin-blink.webp" alt="Sign-in screen with a live face guide asking the user to blink" width="520">
</p>

**Try it in one command** (you need [Docker](https://docs.docker.com/get-docker/) and a webcam):

```bash
git clone https://github.com/rabbicse/face-app.git && cd face-app
docker compose up -d        # then open http://localhost:3000
```

On a Mac, if port 5000 is taken by AirPlay, run `FRS_API_PORT=5001 docker compose up -d` instead.

## Why it's different

| | |
| --- | --- |
| **Private by design** | Photos and video stay on the device, and the server never needs to process an image. |
| **Fast** | 30 FPS live tracking and about 1.1 seconds from camera on to recognised, on a laptop. |
| **Cheap to run** | The user's device does the AI work, so the server needs no GPU and only runs vector search. |
| **Web, Android and iOS** | One shared recognition pipeline, so a face registered on one can sign in on another. |
| **Users stay in control** | The camera stays off until they agree, and they can delete their face data themselves. |

## Performance

Measured on an Apple M1 laptop with the production build. The full method and limits are in
[the benchmarks](docs/benchmarks.md).

| On the device (Chrome, 640×480 camera) | Median | 95th percentile |
| --- | --- | --- |
| Live tracking | **30 FPS** (the camera's limit) | |
| Finding the face landmarks | **15.9 ms** | 17.4 ms |
| Alignment and quality check | **2.2 ms** | 2.5 ms |
| Face signature (registration quality, 2 passes) | **70.7 ms** | 78.6 ms |

| User experience | Median |
| --- | --- |
| Camera on to recognised (blink step excluded) | **1.14 s** |
| Quick registration, 3 samples | **1.64 s** |
| Page open to first result, first visit (18.2 MB download) | **2.2 s** |
| Page open to first result, repeat visit (nothing downloaded) | **1.5 s** |

| Server (10,000 people, 100,000 vectors) | Median | 95th percentile |
| --- | --- | --- |
| Find a match | **13.4 ms** | 37.9 ms |
| Register a person | **101.7 ms** | 245.9 ms |

These numbers show speed, not recognition accuracy, which has not been benchmarked here. Phone performance has
not been measured yet either.

## See it in action

### 1. Register

| Enter your details | Agree before the camera starts |
| --- | --- |
| ![Registration form with name and email](docs/images/02-register-details.webp) | ![Consent screen listing what stays on the device and what is sent](docs/images/03-register-consent.webp) |

| Follow the head-pose guide | Done, with no shutter button |
| --- | --- |
| ![Live scan with a progress ring and pose steps](docs/images/05-scan-turn.webp) | ![Scan complete confirmation](docs/images/06-scan-done.webp) |

### 2. Sign in

| Turn on the camera | Blink to prove you're present |
| --- | --- |
| ![Sign-in screen before the camera starts](docs/images/07-signin-consent.webp) | ![Live sign-in asking for a blink](docs/images/08-signin-blink.webp) |

| Recognised | Your page, including deleting your data |
| --- | --- |
| ![Welcome message after a match](docs/images/09-signin-welcome.webp) | ![Signed-in page with sign out and delete options](docs/images/10-signed-in.webp) |

### 3. See under the hood

![Pipeline inspector with live timings, head pose, quality and the aligned face](docs/images/12-pipeline-inspector.webp)

The [step-by-step guide](docs/guide.md) shows every screen, including deletion, the photo comparison, camera
errors and the phone layout.

*Screenshots are real captures of the running app. The faces are AI-generated people, not real individuals (see
[credits](#credits)).*

## How it works

```
 Browser / phone                                     Server
 ┌───────────────────────────────────────┐          ┌───────────────────┐
 │ camera → find face → align            │ numbers  │ FastAPI           │
 │        → quality check → face model   │ ───────▶ │ Qdrant search     │
 │ photos and video stay here            │ only     │ no photos, no GPU │
 └───────────────────────────────────────┘          └───────────────────┘
```

1. **Registration** captures 9 samples across 5 head poses, relative to the person's own straight-ahead pose.
2. **Sign-in** combines 3 consistent frames and needs a recent blink. A match must score at least 0.45 and beat
   the next person by 0.05.
3. **Every platform shares one vector space**, kept identical by tests against shared golden values.

## Documentation

| Guide | For |
| --- | --- |
| [Step-by-step guide](docs/guide.md) | Running it, using it, the API, and troubleshooting |
| [Benchmarks](docs/benchmarks.md) | How the numbers were measured, and how to repeat them |
| [Deploy for free](docs/deploy-free.md) | Putting it online with Vercel and Qdrant Cloud |
| [Docker images](docs/docker.md) | Image settings and tags |
| [Design and security notes](docs/client-side-embeddings.md) | Technical decisions and threat model |
| [Mobile app](mobile/face_app/README.md) | Building the Flutter app for Android and iOS |

## Before commercial use

This is a working demo. A production or commercial deployment needs these first:

- **Replace the face model.** The bundled InsightFace `w600k_mbf` weights are licensed for **non-commercial
  research only**. Train or license your own; the [design notes](docs/client-side-embeddings.md) explain how to
  swap it.
- **Check the code licence.** The code is under the [GPLv3](LICENSE), which has conditions for anyone who
  distributes it.
- **Add authentication to the API.** Registering and deleting faces are currently open to anyone who can reach
  the API, and an email address isn't verified.
- **Add stronger liveness if needed.** A blink stops printed photos and still images, but not a replayed video.
- **Take legal advice on biometric data.** Rules such as GDPR and BIPA can apply, and the consent wording in this
  demo has not been legally reviewed.

## Built with

Next.js, React, MediaPipe and ONNX Runtime Web · Flutter, Google ML Kit and ONNX Runtime Mobile · FastAPI and
Qdrant · Docker and GitHub Actions. The face model is MobileFaceNet ArcFace (512 dimensions).

| Path | What |
| --- | --- |
| `frontend/frs_react` | Web app, with the on-device pipeline in `src/face` |
| `mobile/face_app` | Flutter app for Android and iOS |
| `backend/frs` | API and the Python reference pipeline |
| `shared/test-vectors` | Golden values all three implementations are tested against |

The [API reference](docs/guide.md#5-use-the-api) has worked examples. To contribute, fork the repository, run the
[tests](docs/guide.md#6-local-development) and open a pull request.

## Credits

- Demo faces: [SFHQ synthetic face dataset](https://github.com/SelfishGene/SFHQ-dataset) by David Beniaguev,
  [MIT licence](docs/images/SFHQ-LICENSE.txt).
- [InsightFace](https://github.com/deepinsight/insightface), [MediaPipe](https://ai.google.dev/edge/mediapipe)
  and [Qdrant](https://qdrant.tech).

---

Made with ❤️ by [Mehedi Hasan](https://github.com/rabbicse)
