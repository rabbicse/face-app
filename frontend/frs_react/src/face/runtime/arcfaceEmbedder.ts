import * as ort from "onnxruntime-web";
import { l2Normalize } from "../core/embedding";
import { rgbaToTensor } from "../core/preprocess";

export interface EmbedderOptions {
    modelUrl: string;
    wasmBase: string;
    inputSize: number;
    preferWebGpu?: boolean;
}

/** ArcFace style model (InsightFace w600k_mbf by default) running on ONNX Runtime Web. */
export class ArcFaceEmbedder {
    private batchSupported = true;

    private constructor(private readonly session: ort.InferenceSession, readonly backend: string,
        readonly inputSize: number) { }

    static async create(opts: EmbedderOptions): Promise<ArcFaceEmbedder> {
        ort.env.wasm.wasmPaths = opts.wasmBase;
        // multi threaded wasm needs a cross origin isolated page (COOP/COEP headers, see next.config.ts)
        const isolated = typeof self !== "undefined" && (self as { crossOriginIsolated?: boolean }).crossOriginIsolated;
        ort.env.wasm.numThreads = isolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

        const providers: string[] = [];
        if (opts.preferWebGpu !== false && typeof navigator !== "undefined" && "gpu" in navigator) {
            providers.push("webgpu");
        }
        providers.push("wasm");

        let lastError: unknown;
        for (const ep of providers) {
            try {
                const session = await ort.InferenceSession.create(opts.modelUrl, {
                    executionProviders: [ep],
                    graphOptimizationLevel: "all",
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

/** InsightFace flip TTA: add the features of the image and its mirror, then normalise. */
function sumHalves(batched: Float32Array): Float32Array {
    const dim = batched.length / 2;
    const sum = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
        sum[i] = batched[i] + batched[i + dim];
    }
    return l2Normalize(sum);
}
