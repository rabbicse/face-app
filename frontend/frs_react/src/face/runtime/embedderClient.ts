import type { EmbedderOptions } from "./arcfaceEmbedder";
import type { EmbeddingWorkerRequest, EmbeddingWorkerResponse } from "./embedding.worker";

export interface EmbeddingResult {
    embedding: Float32Array;
    ms: number;
}

/** Runs the embedding model in a dedicated Web Worker so inference never blocks rendering. */
export class EmbedderClient {
    private nextId = 1;
    private readonly pending = new Map<number, { resolve: (r: EmbeddingResult) => void; reject: (e: Error) => void }>();

    private constructor(private readonly worker: Worker, readonly backend: string) {
        worker.onmessage = (e: MessageEvent<EmbeddingWorkerResponse>) => {
            const msg = e.data;
            if (msg.type === "result") {
                this.pending.get(msg.id)?.resolve({ embedding: msg.embedding, ms: msg.ms });
                this.pending.delete(msg.id);
            } else if (msg.type === "error" && msg.id !== undefined) {
                this.pending.get(msg.id)?.reject(new Error(msg.message));
                this.pending.delete(msg.id);
            }
        };
    }

    static create(options: EmbedderOptions): Promise<EmbedderClient> {
        const worker = new Worker(new URL("./embedding.worker.ts", import.meta.url), { type: "module" });
        return new Promise((resolve, reject) => {
            worker.onmessage = (e: MessageEvent<EmbeddingWorkerResponse>) => {
                if (e.data.type === "ready") {
                    resolve(new EmbedderClient(worker, e.data.backend));
                } else if (e.data.type === "error") {
                    worker.terminate();
                    reject(new Error(e.data.message));
                }
            };
            worker.onerror = e => {
                worker.terminate();
                reject(new Error(e.message || "embedding worker failed to start"));
            };
            worker.postMessage({ type: "init", options } satisfies EmbeddingWorkerRequest);
        });
    }

    get busy(): boolean {
        return this.pending.size > 0;
    }

    /** The pixel buffer is transferred (zero copy) to the worker. */
    embed(face: ImageData, flip: boolean): Promise<EmbeddingResult> {
        const id = this.nextId++;
        return new Promise((resolve, reject) => {
            this.pending.set(id, { resolve, reject });
            const rgba = face.data.buffer as ArrayBuffer;
            this.worker.postMessage({ type: "embed", id, rgba, flip } satisfies EmbeddingWorkerRequest, [rgba]);
        });
    }

    dispose(): void {
        this.worker.terminate();
        this.pending.forEach(p => p.reject(new Error("embedder disposed")));
        this.pending.clear();
    }
}
