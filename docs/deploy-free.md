# Deploy on the web for free: Vercel + Qdrant Cloud

The faces are embedded on the user's device, so the backend only stores and searches 512-d vectors. That
fits free tiers end to end:

| Part | Free service | Root directory |
| --- | --- | --- |
| Web app (Next.js) | Vercel project #1 (Hobby) | `frontend/frs_react` |
| API (FastAPI) | Vercel project #2 (Python function) | `backend/frs` |
| Vector database | Qdrant Cloud free cluster | - |

```
 browser ──https──▶ web.vercel.app          (static pages + models, embeddings computed in the browser)
    │
    └──https──▶ api.vercel.app/api/v1/...   (FastAPI function) ──▶ Qdrant Cloud (free cluster)
```

> **Limits.** Plans change, so check the current pricing pages. Vercel Hobby is for personal,
> non-commercial use. Free Qdrant Cloud clusters are small (about 1 GB RAM, enough for tens of thousands of
> people) and are suspended after a period of inactivity. The InsightFace model weights are licensed for
> non-commercial research only.

Time needed: about 15 minutes. You need a GitHub account with this repository (or your fork), a Vercel
account (sign in with GitHub) and a Qdrant Cloud account.

## Step 1 - Create the database (Qdrant Cloud)

1. Go to <https://cloud.qdrant.io>, sign up and choose **Create cluster -> Free**. Pick the region closest
   to your Vercel region (Vercel's default is Washington D.C., `iad1`, so pick a US-East region).
2. When the cluster is running, copy its **URL**, e.g. `https://1a2b3c.us-east4-0.gcp.cloud.qdrant.io:6333`.
3. Open **API Keys -> Create** and copy the key. It is shown once.

You don't need to create a collection. The API creates `face_embeddings_w600k_mbf` (cosine, 512-d) and its
`person_id` index on the first request.

## Step 2 - Deploy the API (Vercel project #1)

1. <https://vercel.com/new> -> **Import** the repository.
2. **Root Directory**: `backend/frs`. Keep the detected framework preset and leave the build settings empty.
3. **Environment Variables**:

   | Name | Value |
   | --- | --- |
   | `QDRANT_URL` | the cluster URL from step 1 |
   | `QDRANT_API_KEY` | the API key from step 1 |
   | `FRS_CORS_ORIGINS` | `*` for now. Step 4 locks it down. |

4. Click **Deploy**. Then open `https://<api-project>.vercel.app/api/v1/health`. You should see
   `{"status":"ok","embedding_models":["w600k_mbf"]}`.

How this works: `pyproject.toml` lists only the light dependencies (FastAPI, Pydantic, qdrant-client,
NumPy) and sets `[tool.vercel] entrypoint = "main:app"`. `uv.lock` pins the exact versions, and
`.vercelignore` keeps tests, samples and tools out of the function. The first request after an idle period
is a cold start that takes a second or two.

## Step 3 - Deploy the web app (Vercel project #2)

1. <https://vercel.com/new> -> **Import** the same repository again.
2. **Root Directory**: `frontend/frs_react` (framework: Next.js).
3. **Environment Variables**:

   | Name | Value |
   | --- | --- |
   | `NEXT_PUBLIC_API_BASE_URL` | `https://<api-project>.vercel.app` (no trailing slash) |
   | `NEXT_PUBLIC_REQUIRE_BLINK` | optional, `true` by default |

4. Click **Deploy**. The build runs `scripts/setup-face-assets.mjs`, which downloads the face models. Because
   `VERCEL=1`, it puts only what the site loads into `public/`: the fp16 embedding model, the landmarker and
   the WASM runtimes.

`NEXT_PUBLIC_*` values are compiled into the bundle, so **redeploy** the web app whenever you change them.
Vercel serves https, which browsers require for camera access. It also sends the cross-origin isolation
headers from `next.config.ts`, which multi-threaded WASM needs.

## Step 4 - Lock CORS down

In the API project, set `FRS_CORS_ORIGINS` to the web app's origin, e.g. `https://<web-project>.vercel.app`.
Separate several origins with commas, for example to add a custom domain. Then **Redeploy** the API.

## Step 5 - Try it

1. Open `https://<web-project>.vercel.app/registration`, fill in the form and follow the head-pose guide.
2. Open `/face/login`, look at the camera and blink. You are recognised and taken to the dashboard.
3. `/examples/pipeline` shows live timings, pose, quality and the aligned crop.

Point the mobile app at the same API:

```bash
cd mobile/face_app
flutter run --dart-define=FRS_API=https://<api-project>.vercel.app
```

## Optional - custom domain

Vercel -> project -> **Settings -> Domains**. After adding a domain to the web project, add it to
`FRS_CORS_ORIGINS` (step 4).

## Alternative - the Docker images on a free container host

The images published by `.github/workflows/docker.yml` (`<namespace>/frs-api`, `<namespace>/frs-web`) run on
any container host. On a free host that sleeps when idle (Render, Koyeb, Fly.io trial, ...):

* run `frs-api` with `QDRANT_URL` / `QDRANT_API_KEY` pointing at the Qdrant Cloud cluster from step 1
  (container port 8000);
* the `frs-web` image proxies `/api/v1/*` to `http://api:8000`, a name that only exists inside Docker Compose.
  On a separate host, deploy the web app to Vercel (step 3) instead, or rebuild the image with
  `--build-arg FRS_API_PROXY_URL=https://<your-api-host>`.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| "Server error" in the web app, CORS errors in the console | `NEXT_PUBLIC_API_BASE_URL` must be the https API URL without a trailing slash, and `FRS_CORS_ORIGINS` must contain the exact web origin. Redeploy after changing either. |
| API returns 500 on register / identify | Check `QDRANT_URL` / `QDRANT_API_KEY` in the function logs, and that the free cluster is not suspended (resume it in the Qdrant console). |
| Camera does not start | The page must be https (or localhost), and the browser must have camera permission. |
| Blocked cross-origin resources on preview deployments | The page is cross-origin isolated, which blocks third-party scripts without CORP headers, such as an injected preview toolbar. The face pipeline is unaffected. |
| First login after a while is slow | Cold start of the free function plus a resumed Qdrant cluster. It is fast after that. |
