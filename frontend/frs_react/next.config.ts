import type { NextConfig } from "next";

const nextConfig: NextConfig = {
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
