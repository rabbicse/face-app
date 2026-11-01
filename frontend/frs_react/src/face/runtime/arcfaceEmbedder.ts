import type * as Ort from "onnxruntime-web";
import { l2Normalize } from "../core/embedding";
import { rgbaToTensor } from "../core/preprocess";

export interface EmbedderOptions {
    modelUrl: string;
    /** cache key: change it whenever the file behind modelUrl changes */
    modelId: string;
    wasmBase: string;
    inputSize: number;
    /**
     * Try WebGPU first. Off by default: the WASM-only runtime is half the download (2.7 vs 4.9 MB gzip),
     * compiles faster, does not compete with MediaPipe for the GPU, and this small model at <= 8 embeddings
     * per second does not need a GPU.
     */
    webgpu?: boolean;
}

/** The runtime is loaded lazily; only the chosen build (WASM or WASM+WebGPU) is downloaded. */
let ort: typeof Ort;

/** ArcFace style model (InsightFace w600k_mbf by default) running on ONNX Runtime Web. */
export class ArcFaceEmbedder {
    private batchSupported = true;

    private constructor(private readonly session: Ort.InferenceSession, readonly backend: string,
        readonly inputSize: number) { }

    static async create(opts: EmbedderOptions): Promise<ArcFaceEmbedder> {
        ort = opts.webgpu ? await import("onnxruntime-web") : await import("onnxruntime-web/wasm");
        ort.env.wasm.wasmPaths = opts.wasmBase;
        // multi threaded wasm needs a cross origin isolated page (COOP/COEP headers, see next.config.ts)
        const isolated = typeof self !== "undefined" && (self as { crossOriginIsolated?: boolean }).crossOriginIsolated;
        // leave one core to the main thread (camera, MediaPipe, rendering)
        ort.env.wasm.numThreads = isolated ? Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1)) : 1;

        const providers: string[] = [];
        if (opts.webgpu && typeof navigator !== "undefined" && "gpu" in navigator) {
            providers.push("webgpu");
        }
        providers.push("wasm");

        // download once, re-use the bytes for every backend we try
        const model = await loadModelBytes(opts.modelUrl, opts.modelId);
        let lastError: unknown;
        for (const ep of providers) {
            try {
                const session = await ort.InferenceSession.create(model, {
                    executionProviders: [ep],
                    graphOptimizationLevel: "all",
                    // silences "VerifyOutputSizes" warnings: the model declares batch 1 but accepts 2 (flip TTA)
                    logSeverityLevel: 3,
                });
                const embedder = new ArcFaceEmbedder(session, ep, opts.inputSize);
                // warm up: first run compiles shaders / allocates buffers
                await embedder.embed(new Uint8ClampedArray(opts.inputSize * opts.inputSize * 4), true);
                return embedder;
            } catch (e) {
                lastError = e;
                console.warn(`onnxruntime ${ep} backend unavailable`, e);
            }
        }
        throw lastError;
    }

    /** @param rgba aligned face crop (inputSize x inputSize RGBA) @return L2 normalised embedding */
    async embed(rgba: ArrayLike<number>, flip: boolean): Promise<Float32Array> {
        const s = this.inputSize;
        const plane = 3 * s * s;
        if (!flip) {
            return l2Normalize(await this.run(rgbaToTensor(rgba, s, s), 1));
        }
        if (this.batchSupported) {
            try {
                const data = new Float32Array(2 * plane);
                rgbaToTensor(rgba, s, s, data, 0, false);
                rgbaToTensor(rgba, s, s, data, plane, true);
                const out = await this.run(data, 2);
                return sumHalves(out);
            } catch (e) {
                console.warn("batched inference failed, falling back to sequential runs", e);
                this.batchSupported = false;
            }
        }
        const a = await this.run(rgbaToTensor(rgba, s, s), 1);
        const b = await this.run(rgbaToTensor(rgba, s, s, undefined, 0, true), 1);
        const both = new Float32Array(a.length * 2);
        both.set(a);
        both.set(b, a.length);
        return sumHalves(both);
    }

    private async run(data: Float32Array, batch: number): Promise<Float32Array> {
        const s = this.inputSize;
        const input = new ort.Tensor("float32", data, [batch, 3, s, s]);
        const out = await this.session.run({ [this.session.inputNames[0]]: input });
        const tensor = out[this.session.outputNames[0]];
        const result = (await tensor.getData()) as Float32Array;
        input.dispose();
        tensor.dispose();
        return result;
    }

    dispose(): void {
        void this.session.release();
    }
}

const MODEL_CACHE = "face-models-v1";

/**
 * Model bytes from Cache Storage, downloading them on the first visit. Large files (13.6 MB here) are often
 * evicted from, or never stored in, the browser's HTTP cache; Cache Storage keeps them until the site data is
 * cleared, so later visits start without any download.
 */
async function loadModelBytes(url: string, modelId: string): Promise<Uint8Array> {
    const key = new URL(`${url}${url.includes("?") ? "&" : "?"}model=${encodeURIComponent(modelId)}`, self.location.href).href;
    let cache: Cache | null = null;
    try {
        cache = await caches.open(MODEL_CACHE);
        const hit = await cache.match(key);
        if (hit) return new Uint8Array(await hit.arrayBuffer());
    } catch {
        cache = null; // Cache Storage unavailable (e.g. some private modes): plain download
    }
    const res = await fetch(url);
    if (!res.ok) throw new Error(`failed to download ${url}: ${res.status}`);
    const bytes = new Uint8Array(await res.arrayBuffer());
    if (cache) {
        try {
            await cache.put(key, new Response(bytes, { headers: { "Content-Type": "application/octet-stream" } }));
            // drop entries of other models / versions
            for (const req of await cache.keys()) {
                if (req.url !== key) await cache.delete(req);
            }
        } catch (e) {
            console.warn("could not cache the face model", e);
        }
    }
    return bytes;
}

/** InsightFace flip TTA: add the features of the image and its mirror, then normalise. */
function sumHalves(batched: Float32Array): Float32Array {
    const dim = batched.length / 2;
    const sum = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
        sum[i] = batched[i] + batched[i + dim];
    }
    return l2Normalize(sum);
}
