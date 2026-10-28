/**
 * Main-thread handle to the embedding worker. One worker per tab (module singleton), so
 * moving between the registration and login pages doesn't reload 16 MB of models.
 */
import { FACE_CONFIG } from "./config";
import type { BenchmarkResult, InitResult, WorkerEmbedding, WorkerRequest, WorkerResponse } from "./protocol";
import type { Point } from "../core/geometry";

type Pending = { resolve: (v: unknown) => void; reject: (e: Error) => void };
type DistributiveOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;

export type LoadProgress = { file: string; loaded: number; total: number };

export class EmbedderClient {
    private worker: Worker;
    private nextId = 1;
    private pending = new Map<number, Pending>();
    private busy = false;
    readonly ready: Promise<InitResult>;
    info: InitResult | null = null;

    private constructor(onProgress?: (p: LoadProgress) => void) {
        this.worker = new Worker(new URL("./embedder.worker.ts", import.meta.url), { type: "module" });
        this.worker.onmessage = (ev: MessageEvent<WorkerResponse>) => {
            const msg = ev.data;
            if (msg.ok && msg.type === "progress") { onProgress?.(msg.result); return; }
            const p = this.pending.get(msg.id);
            if (!p) return;
            this.pending.delete(msg.id);
            if (msg.ok) p.resolve(msg.result); else p.reject(new Error(msg.error));
        };
        this.worker.onerror = (e) => {
            const err = new Error(e.message || "embedding worker crashed");
            this.pending.forEach(p => p.reject(err));
            this.pending.clear();
        };
        this.ready = this.call<InitResult>({
            type: "init",
            detectorUrl: FACE_CONFIG.models.detector,
            recognizerUrl: FACE_CONFIG.models.recognizer,
            ortWasmPath: new URL(FACE_CONFIG.wasm.ort, location.href).toString(),
            preferWebGPU: FACE_CONFIG.preferWebGPU,
        }).then(info => (this.info = info));
    }

    private static instance: EmbedderClient | null = null;
    private static listeners = new Set<(p: LoadProgress) => void>();

    static get(onProgress?: (p: LoadProgress) => void): EmbedderClient {
        if (onProgress) EmbedderClient.listeners.add(onProgress);
        if (!EmbedderClient.instance) {
            EmbedderClient.instance = new EmbedderClient(p => EmbedderClient.listeners.forEach(l => l(p)));
        }
        return EmbedderClient.instance;
    }

    static removeProgressListener(l: (p: LoadProgress) => void) { EmbedderClient.listeners.delete(l); }

    private call<T>(req: DistributiveOmit<WorkerRequest, "id">, transfer: Transferable[] = []): Promise<T> {
        const id = this.nextId++;
        return new Promise<T>((resolve, reject) => {
            this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
            this.worker.postMessage({ ...req, id }, transfer);
        });
    }

    /** True while a frame is being embedded. Callers skip frames instead of queueing (backpressure). */
    get isBusy(): boolean { return this.busy; }

    /**
     * Embed a square ROI bitmap. Single-flight: returns null immediately if a job is
     * already running, so slow devices drop frames instead of building a backlog.
     */
    async embedRoi(bitmap: ImageBitmap, roiToSourceScale: number, fallbackKps?: Point[], wantAligned = false): Promise<WorkerEmbedding | null | "busy"> {
        if (this.busy) { bitmap.close(); return "busy"; }
        this.busy = true;
        try {
            await this.ready;
            return await this.call<WorkerEmbedding | null>({ type: "embedRoi", bitmap, roiToSourceScale, fallbackKps, wantAligned }, [bitmap]);
        } finally {
            this.busy = false;
        }
    }

    async embedImage(bitmap: ImageBitmap, maxFaces = 5): Promise<WorkerEmbedding[]> {
        await this.ready;
        return this.call<WorkerEmbedding[]>({ type: "embedImage", bitmap, maxFaces }, [bitmap]);
    }

    async benchmark(iterations = 20): Promise<BenchmarkResult> {
        await this.ready;
        return this.call<BenchmarkResult>({ type: "benchmark", iterations });
    }
}
