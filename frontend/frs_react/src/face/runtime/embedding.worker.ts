/// <reference lib="webworker" />
import { ArcFaceEmbedder, EmbedderOptions } from "./arcfaceEmbedder";

export type EmbeddingWorkerRequest =
    | { type: "init"; options: EmbedderOptions }
    | { type: "embed"; id: number; rgba: ArrayBuffer; flip: boolean };

export type EmbeddingWorkerResponse =
    | { type: "ready"; backend: string }
    | { type: "result"; id: number; embedding: Float32Array; ms: number }
    | { type: "error"; id?: number; message: string };

let embedder: ArcFaceEmbedder | null = null;
const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.onmessage = async (event: MessageEvent<EmbeddingWorkerRequest>) => {
    const msg = event.data;
    try {
        if (msg.type === "init") {
            embedder = await ArcFaceEmbedder.create(msg.options);
            ctx.postMessage({ type: "ready", backend: embedder.backend } satisfies EmbeddingWorkerResponse);
        } else if (msg.type === "embed") {
            if (!embedder) throw new Error("embedder is not initialised");
            const start = performance.now();
            const embedding = await embedder.embed(new Uint8ClampedArray(msg.rgba), msg.flip);
            const response: EmbeddingWorkerResponse = { type: "result", id: msg.id, embedding, ms: performance.now() - start };
            ctx.postMessage(response, [embedding.buffer]);
        }
    } catch (e) {
        ctx.postMessage({
            type: "error",
            id: msg.type === "embed" ? msg.id : undefined,
            message: e instanceof Error ? e.message : String(e),
        } satisfies EmbeddingWorkerResponse);
    }
};
