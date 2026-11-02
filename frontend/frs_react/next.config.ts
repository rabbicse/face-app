import type { NextConfig } from "next";

// Docker image only (both are read at build time, see Dockerfile):
//  - NEXT_OUTPUT=standalone: self-contained server.js with only the node_modules it needs
//  - FRS_API_PROXY_URL: proxy /api/v1/* to the API, so the browser talks to one origin
//    (no CORS, no mixed content, and cross-origin isolation keeps working)
const standalone = process.env.NEXT_OUTPUT === "standalone";
const apiProxy = process.env.FRS_API_PROXY_URL?.replace(/\/+$/, "");

const nextConfig: NextConfig = {
  output: standalone ? "standalone" : undefined,
  poweredByHeader: false,
  async rewrites() {
    return apiProxy ? [{ source: "/api/v1/:path*", destination: `${apiProxy}/api/v1/:path*` }] : [];
  },
  async headers() {
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
  },
};

export default nextConfig;
