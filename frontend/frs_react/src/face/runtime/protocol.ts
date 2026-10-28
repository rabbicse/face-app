/** Messages between the main thread and embedder.worker.ts */
import type { Point } from "../core/geometry";
import type { FaceQuality } from "../core/quality";

export interface InitRequest {
    type: "init";
    id: number;
    detectorUrl: string;
    recognizerUrl: string;
    ortWasmPath: string;
    preferWebGPU: boolean;
}

export interface EmbedRoiRequest {
    type: "embedRoi";
    id: number;
    /** square crop (roiSize x roiSize) around the tracked face */
    bitmap: ImageBitmap;
    /** ROI pixel -> original video pixel */
    roiToSourceScale: number;
    /** tracker keypoints in ROI coordinates, used if SCRFD misses */
    fallbackKps?: Point[];
    flipAugment?: boolean;
    /** return the 112x112 aligned crop (for debug / UI thumbnails) */
    wantAligned?: boolean;
}

export interface EmbedImageRequest {
    type: "embedImage";
    id: number;
    bitmap: ImageBitmap;
    maxFaces: number;
    flipAugment?: boolean;
}

export interface BenchmarkRequest { type: "benchmark"; id: number; iterations: number }

export type WorkerRequest = InitRequest | EmbedRoiRequest | EmbedImageRequest | BenchmarkRequest;

export interface WorkerEmbedding {
    embedding: Float32Array;
    kps: Point[];
    detScore: number;
    quality: FaceQuality;
    aligned?: ImageData;
    timings: { detectMs: number; embedMs: number; totalMs: number };
}

export interface InitResult { backend: "webgpu" | "wasm"; threads: number; loadMs: number }
export interface BenchmarkResult { detectMs: number; embedMs: number; backend: string }

export type WorkerResponse =
    | { id: number; ok: true; type: "init"; result: InitResult }
    | { id: number; ok: true; type: "embedRoi"; result: WorkerEmbedding | null }
    | { id: number; ok: true; type: "embedImage"; result: WorkerEmbedding[] }
    | { id: number; ok: true; type: "benchmark"; result: BenchmarkResult }
    | { id: number; ok: true; type: "progress"; result: { file: string; loaded: number; total: number } }
    | { id: number; ok: false; error: string };
