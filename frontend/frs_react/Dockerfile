# syntax=docker/dockerfile:1.7
# Web app image: Next.js standalone server + the on-device face models (fp16) and WASM runtimes.
#   docker build -t frs-web .
# The browser calls /api/v1/* on the same origin; the server proxies it to FRS_API_PROXY_URL.

ARG NODE_VERSION=24

FROM node:${NODE_VERSION}-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --no-audit --no-fund

FROM node:${NODE_VERSION}-alpine AS build
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1 \
    NEXT_OUTPUT=standalone \
    FACE_ASSETS_DEPLOY=1
# Baked into the bundle / server config at build time
ARG FRS_API_PROXY_URL=http://api:8000
ARG NEXT_PUBLIC_API_BASE_URL=""
ARG NEXT_PUBLIC_REQUIRE_BLINK=true
ARG NEXT_PUBLIC_EMBEDDING_WEBGPU=false
ENV FRS_API_PROXY_URL=${FRS_API_PROXY_URL} \
    NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL} \
    NEXT_PUBLIC_REQUIRE_BLINK=${NEXT_PUBLIC_REQUIRE_BLINK} \
    NEXT_PUBLIC_EMBEDDING_WEBGPU=${NEXT_PUBLIC_EMBEDDING_WEBGPU}
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# prebuild downloads the face models + copies the WASM runtimes into public/
RUN npm run build

FROM node:${NODE_VERSION}-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    HOSTNAME=0.0.0.0 \
    PORT=3000
COPY --from=build --chown=node:node /app/.next/standalone ./
COPY --from=build --chown=node:node /app/.next/static ./.next/static
COPY --from=build --chown=node:node /app/public ./public
USER node
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD wget -qO- http://127.0.0.1:3000/ >/dev/null || exit 1
CMD ["node", "server.js"]
