/** Runtime configuration (override with NEXT_PUBLIC_* env vars). */
export const FACE_CONFIG = {
    /** "server": vectors go to the FastAPI v2 API.  "local": everything stays in this browser (IndexedDB). */
    mode: (process.env.NEXT_PUBLIC_FACE_MODE ?? "server") as "server" | "local",
    apiUrl: process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000",
    models: {
        detector: process.env.NEXT_PUBLIC_FACE_DET_MODEL ?? "/models/det_500m.onnx",
        recognizer: process.env.NEXT_PUBLIC_FACE_REC_MODEL ?? "/models/w600k_mbf.onnx",
        landmarker: process.env.NEXT_PUBLIC_FACE_LANDMARKER_MODEL ?? "/models/face_landmarker.task",
        landmarkerFallback:
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    },
    wasm: {
        ort: "/ort/",
        mediapipe: "/mediapipe/wasm",
    },
    /** side (px) of the square face crop sent to the worker; multiple of 32 */
    roiSize: 192,
    /** ROI = tracker face box * this factor (context for SCRFD) */
    roiScale: 1.6,
    /** minimum time between embedding jobs while identifying (ms) */
    identifyIntervalMs: 120,
    /** try WebGPU before WASM for ONNX inference */
    preferWebGPU: process.env.NEXT_PUBLIC_FACE_WEBGPU !== "0",
} as const;
