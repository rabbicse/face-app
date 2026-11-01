const apiHost = process.env.NEXT_PUBLIC_API_HOST ?? "http://localhost";
const apiPort = process.env.NEXT_PUBLIC_API_PORT ?? "5000";

export const FACE_CONFIG = {
    apiBaseUrl: process.env.NEXT_PUBLIC_API_BASE_URL ?? `${apiHost}:${apiPort}`,
    embedding: {
        /** id sent to the backend; embeddings of different models are never compared */
        modelId: process.env.NEXT_PUBLIC_EMBEDDING_MODEL_ID ?? "w600k_mbf",
        // float16 weight storage of w600k_mbf: half the download, same embeddings (see scripts/onnx-fp16.mjs)
        modelUrl: process.env.NEXT_PUBLIC_EMBEDDING_MODEL_URL ?? "/models/arcface/w600k_mbf.fp16.onnx",
        inputSize: 112,
        /** average the embedding of the face and its mirror image (+accuracy, 2x compute) - used for registration */
        flipTTA: true,
        /**
         * Identification already fuses 3 frames, so it uses a single pass: half the compute per frame.
         * Measured on 42 faces: cosine(single pass, TTA) 0.986 on average, similarity to TTA templates
         * changes by 0.0075 on average.
         */
        identificationFlipTTA: false,
        /** opt in to ONNX Runtime's WebGPU backend (bigger runtime download, see arcfaceEmbedder.ts) */
        webgpu: process.env.NEXT_PUBLIC_EMBEDDING_WEBGPU === "true",
    },
    landmarker: {
        modelUrl: "/models/mediapipe/face_landmarker.task",
        wasmBase: "/mediapipe/wasm",
    },
    ortWasmBase: "/ort/",
    camera: { width: 640, height: 480, frameRate: 30 },
    /** max embeddings per second computed while waiting for a match / pose */
    maxEmbeddingsPerSecond: 8,
    liveness: {
        /** identification needs a blink first: stops printed photos / still images (set to "false" for demos) */
        requireBlink: process.env.NEXT_PUBLIC_REQUIRE_BLINK !== "false",
        /** how long a blink keeps the session "live" */
        blinkWindowMs: 5000,
    },
} as const;
