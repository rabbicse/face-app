import type { NextConfig } from "next";

/**
 * Cross-origin isolation (COOP + COEP) unlocks SharedArrayBuffer, which onnxruntime-web
 * needs for multi-threaded WASM inference (2-4x faster on CPU-only devices). All runtime
 * assets are self-hosted (scripts/setup-assets.mjs), so `require-corp` doesn't block them.
 * Disable with FACE_CROSS_ORIGIN_ISOLATION=0 if you embed third-party content.
 */
const isolate = process.env.FACE_CROSS_ORIGIN_ISOLATION !== "0";

const nextConfig: NextConfig = {
    reactStrictMode: true,
    async headers() {
        return [
            ...(isolate ? [{
                source: "/:path*",
                headers: [
                    { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
                    { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
                ],
            }] : []),
            {
                // models + wasm are versioned by filename; let browsers keep them
                source: "/:dir(models|ort|mediapipe)/:path*",
                headers: [{ key: "Cache-Control", value: "public, max-age=604800, immutable" }],
            },
        ];
    },
};

export default nextConfig;
