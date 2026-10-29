const apiHost = process.env.NEXT_PUBLIC_API_HOST ?? "http://localhost";
const apiPort = process.env.NEXT_PUBLIC_API_PORT ?? "5000";

export const FACE_CONFIG = {
    apiBaseUrl: process.env.NEXT_PUBLIC_API_BASE_URL ?? `${apiHost}:${apiPort}`,
    embedding: {
        /** id sent to the backend; embeddings of different models are never compared */
        modelId: process.env.NEXT_PUBLIC_EMBEDDING_MODEL_ID ?? "w600k_mbf",
        modelUrl: process.env.NEXT_PUBLIC_EMBEDDING_MODEL_URL ?? "/models/arcface/w600k_mbf.onnx",
        inputSize: 112,
        /** average the embedding of the face and its mirror image (+accuracy, 2x compute) */
        flipTTA: true,
    },
    landmarker: {
        modelUrl: "/models/mediapipe/face_landmarker.task",
        wasmBase: "/mediapipe/wasm",
    },
    ortWasmBase: "/ort/",
    camera: { width: 640, height: 480, frameRate: 30 },
    /** max embeddings per second computed while waiting for a match / pose */
    maxEmbeddingsPerSecond: 8,
} as const;
