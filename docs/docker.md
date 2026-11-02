# FRS - face login with on-device embeddings (Docker images)

Face registration and login where the **browser or phone computes the face embedding** and the server only
stores and searches 512-d vectors in [Qdrant](https://qdrant.tech). No photos leave the device.

| Image | What | Port |
| --- | --- | --- |
| `frs-web` | Next.js web app with the face models (MediaPipe landmarks + MobileFaceNet ONNX, run in the browser) | 3000 |
| `frs-api` | FastAPI embedding store / search (no GPU, no PyTorch) | 8000 |

Both images are multi-arch (`linux/amd64`, `linux/arm64`) and run as non-root users. Source:
<https://github.com/rabbicse/face-app>.

## Run the whole stack with one command

```bash
git clone https://github.com/rabbicse/face-app.git && cd face-app
cp .env.example .env          # set FRS_IMAGE_NAMESPACE to the Docker Hub user that publishes the images
docker compose up -d
```

Open <http://localhost:3000>. Registration is at `/registration`, face login at `/face/login`.
`docker compose down` stops the stack, and the data stays in the `qdrant_data` volume. Add `-v` to delete it.

Without cloning, save the compose file next to an `.env` file and run the same command:

```bash
curl -fsSLO https://raw.githubusercontent.com/rabbicse/face-app/master/docker-compose.yml
echo "FRS_IMAGE_NAMESPACE=<dockerhub-user>" > .env
docker compose up -d --no-build
```

## Tags

`latest` is the latest `master` build. Releases are tagged `X.Y.Z` and `X.Y`. Every master commit is also
tagged `sha-<short>`.

## Configuration

`frs-api`:

| Variable | Default | |
| --- | --- | --- |
| `QDRANT_HOST` / `QDRANT_PORT` | `qdrant` / `6333` | Qdrant server |
| `QDRANT_URL` / `QDRANT_API_KEY` | - | remote Qdrant, e.g. Qdrant Cloud |
| `FRS_MATCH_THRESHOLD` / `FRS_MATCH_MARGIN` | `0.45` / `0.05` | match acceptance |
| `FRS_CORS_ORIGINS` | `*` | allowed browser origins |
| `FRS_WORKERS` | `2` | uvicorn worker processes |

`frs-web` is configured at build time (`docker build --build-arg ...`):

| Build arg | Default | |
| --- | --- | --- |
| `FRS_API_PROXY_URL` | `http://api:8000` | where the web server proxies `/api/v1/*` |
| `NEXT_PUBLIC_API_BASE_URL` | empty (same origin) | set to call an API on another origin directly |
| `NEXT_PUBLIC_REQUIRE_BLINK` | `true` | blink liveness check at login |

The camera needs a secure context. `http://localhost` works. On another machine, put the web container
behind https, e.g. with a reverse proxy such as Caddy or Traefik.
