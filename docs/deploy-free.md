# Deploying for free: Vercel + Qdrant Cloud

Because the faces are embedded on the devices, the backend only stores and searches 512-d vectors. That
fits free tiers:

| Part | Where | Root directory |
| --- | --- | --- |
| Web app (Next.js) | Vercel project #1 | `frontend/frs_react` |
| API (FastAPI, embedding store) | Vercel project #2 (Python function) | `backend/frs` |
| Vector database | Qdrant Cloud free cluster | - |

> Plans and limits change - check Vercel's and Qdrant's pricing pages. Vercel's free **Hobby** plan is for
> personal, non-commercial projects. Free Qdrant Cloud clusters are suspended after a period of inactivity.
> The legacy server-side PyTorch endpoints are too big for this setup and are not deployed.

## 1. Qdrant Cloud

1. Create an account at cloud.qdrant.io and a **free cluster**.
2. Copy the cluster URL (`https://<id>.<region>.cloud.qdrant.io:6333`) and create an **API key**.

The collection (`face_embeddings_w600k_mbf`, cosine, 512-d) and its `person_id` index are created
automatically on the first request.

## 2. API on Vercel

1. Vercel dashboard -> **Add New... -> Project** -> import this repository.
2. **Root Directory**: `backend/frs`. Leave the framework preset as detected (FastAPI / Other) and the build
   settings empty.
3. **Environment Variables**:

   | Name | Value |
   | --- | --- |
   | `QDRANT_URL` | the cluster URL from step 1 |
   | `QDRANT_API_KEY` | the API key from step 1 |
   | `FRS_ENABLE_SERVER_INFERENCE` | `false` |
   | `FRS_CORS_ORIGINS` | `*` for now; set to the web app URL in step 4 |

4. **Deploy**, then open `https://<api-project>.vercel.app/api/v1/health` - it should return
   `{"status": "ok", ...}`.

How it is wired: `backend/frs/pyproject.toml` lists only the light dependencies (FastAPI, Pydantic,
qdrant-client, NumPy - about 100 MB installed) and sets `[tool.vercel] entrypoint = "main:app"`, so Vercel
serves `backend/frs/main.py` (and never the legacy Flask `src/app.py`). The heavy `requirements.txt` is not
used. `.vercelignore` keeps the docs, sample photos and tests out of the function. The first request after
a while is a cold start (a second or two).

## 3. Web app on Vercel

1. **Add New... -> Project** -> import the same repository again.
2. **Root Directory**: `frontend/frs_react` (framework: Next.js).
3. **Environment Variables**: `NEXT_PUBLIC_API_BASE_URL` = `https://<api-project>.vercel.app`
   (optional: `NEXT_PUBLIC_REQUIRE_BLINK`, `NEXT_PUBLIC_EMBEDDING_WEBGPU`, see `.env.example`).
4. **Deploy.** The build runs `scripts/setup-face-assets.mjs`, which downloads the models and, because
   `VERCEL=1`, only puts what the site loads into `public/` (~40 MB: fp16 model, landmarker, WASM runtimes).

`NEXT_PUBLIC_*` values are baked in at build time: **redeploy** the web app after changing them. Vercel
serves https, which browsers require for camera access, and applies the cross-origin isolation headers from
`next.config.ts`.

## 4. Lock CORS down

Set `FRS_CORS_ORIGINS` of the API project to the web app's URL (comma separate several, e.g. a custom
domain) and redeploy the API.

## Mobile app

Point the Flutter app at the deployed API:

```bash
flutter run --dart-define=FRS_API=https://<api-project>.vercel.app
```

## Troubleshooting

* **Web app says "Server error" / CORS errors in the console**: `NEXT_PUBLIC_API_BASE_URL` must be the https
  API URL without a trailing slash, and `FRS_CORS_ORIGINS` must contain the exact web origin.
* **API returns 500 on register / identify**: check `QDRANT_URL` / `QDRANT_API_KEY` in the function logs,
  and that the free cluster is not suspended.
* **Console errors about blocked cross-origin resources on preview deployments**: the page is cross-origin
  isolated (needed for multi-threaded WASM), which blocks third-party scripts that do not send CORP headers,
  such as an injected preview toolbar. The face pipeline itself is unaffected.

## Verified locally

The API was installed the way Vercel's Python runtime does it (`uv sync --no-dev` from `pyproject.toml`, files
filtered by `.vercelignore`) and served via `main:app`: health, register, identify and CORS work. The web app
was built from a clean checkout with `VERCEL=1`. A real deployment could not be made from the development
environment (no Vercel access).
