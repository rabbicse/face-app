import type { NextConfig } from "next";

// Every page is client-rendered on top of a static HTML shell built once at build time (no per-request
// server rendering): the face pipeline runs in the browser and data comes from the API.
//
// Build-time switches:
//  - NEXT_OUTPUT=export: plain static files in out/ for any CDN or static host, no Node.js server at all.
//    The host must send the cross-origin isolation headers (public/_headers covers Cloudflare Pages and
//    Netlify) and the browser calls the API directly, so set NEXT_PUBLIC_API_BASE_URL and the API's
//    FRS_CORS_ORIGINS.
//  - NEXT_OUTPUT=standalone (Docker image): self-contained server.js that serves the same static pages.
//  - FRS_API_PROXY_URL: proxy /api/v1/* to the API, so the browser talks to one origin
//    (no CORS, no mixed content, and cross-origin isolation keeps working). Not available with export.
const outputMode = process.env.NEXT_OUTPUT;
const staticExport = outputMode === "export";
const apiProxy = process.env.FRS_API_PROXY_URL?.replace(/\/+$/, "");

const nextConfig: NextConfig = {
  output: outputMode === "standalone" || staticExport ? outputMode : undefined,
  poweredByHeader: false,
  // static hosts serve /face/login/index.html for /face/login
  trailingSlash: staticExport ? true : undefined,
  // rewrites and headers need a Next.js server, so a static export leaves them to the host
  ...(staticExport ? {} : { rewrites, headers }),
};

async function rewrites() {
    return apiProxy ? [{ source: "/api/v1/:path*", destination: `${apiProxy}/api/v1/:path*` }] : [];
}

async function headers() {
    return [
      {
        // Cross-origin isolation enables SharedArrayBuffer => multi-threaded ONNX Runtime WASM.
        // Every sub-resource must then be same-origin or CORS/CORP enabled (the API uses CORS).
        source: "/:path*",
        headers: [
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
        ],
      },
      {
        // model + wasm files are versioned by the setup script; let the browser keep them
        source: "/:dir(models|ort|mediapipe)/:path*",
        headers: [{ key: "Cache-Control", value: "public, max-age=604800, stale-while-revalidate=86400" }],
      },
    ];
}

export default nextConfig;
