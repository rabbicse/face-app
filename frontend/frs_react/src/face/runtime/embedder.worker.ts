/// <reference lib="webworker" />
/**
 * Embedding worker: keeps ONNX inference (SCRFD keypoints + ArcFace 512-d) off the UI
 * thread. Receives small ImageBitmaps (transferred, zero-copy), returns Float32Arrays
 * (transferred). WebGPU when available, multi-threaded SIMD WASM otherwise.
 */
import type * as OrtTypes from "onnxruntime-web";
import { FacePipeline, ModelRunner } from "../core/pipeline";
import { RgbaImage } from "../core/image";
import { rescaleFaces } from "../core/scrfd";
import { fetchModel } from "./modelCache";
import type { EmbedImageRequest, EmbedRoiRequest, InitRequest, WorkerEmbedding, WorkerRequest, WorkerResponse } from "./protocol";

declare const self: DedicatedWorkerGlobalScope;

type Ort = typeof OrtTypes;
/**
 * The runtime is chosen at init: the WebGPU build (7 MB gzipped wasm) only when the device
 * has a WebGPU adapter, otherwise the plain SIMD WASM build (3.8 MB gzipped).
 */
let ort: Ort;

class OrtRunner implements ModelRunner {
    constructor(private session: OrtTypes.InferenceSession) { }
    async run(input: Float32Array, dims: readonly number[]): Promise<Float32Array[]> {
        const feeds = { [this.session.inputNames[0]]: new ort.Tensor("float32", input, dims as number[]) };
        const out = await this.session.run(feeds);
        const res = this.session.outputNames.map(n => out[n].data as Float32Array);
        for (const n of this.session.outputNames) out[n].dispose?.();
        return res;
    }
}

let pipeline: FacePipeline | null = null;
let backend: "webgpu" | "wasm" = "wasm";
let scratch: OffscreenCanvas | null = null;

const post = (msg: WorkerResponse, transfer: Transferable[] = []) => self.postMessage(msg, transfer);
const now = () => performance.now();

function bitmapToRgba(bitmap: ImageBitmap, maxSide = Infinity): { img: RgbaImage; scale: number } {
    const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
    const w = Math.round(bitmap.width * scale), h = Math.round(bitmap.height * scale);
    if (!scratch || scratch.width !== w || scratch.height !== h) scratch = new OffscreenCanvas(w, h);
    const ctx = scratch.getContext("2d", { willReadFrequently: true })!;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, w, h);
    ctx.drawImage(bitmap, 0, 0, w, h);
    const d = ctx.getImageData(0, 0, w, h);
    return { img: { data: d.data, width: w, height: h }, scale };
}

function letterbox(src: RgbaImage, size: number): { img: RgbaImage; scale: number } {
    const c = new OffscreenCanvas(src.width, src.height);
    c.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(src.data), src.width, src.height), 0, 0);
    const scale = size / Math.max(src.width, src.height);
    const out = new OffscreenCanvas(size, size);
    const ctx = out.getContext("2d", { willReadFrequently: true })!;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, size, size);
    ctx.drawImage(c, 0, 0, Math.round(src.width * scale), Math.round(src.height * scale));
    return { img: { data: ctx.getImageData(0, 0, size, size).data, width: size, height: size }, scale };
}

async function createSession(buf: ArrayBuffer, eps: string[]) {
    return ort.InferenceSession.create(new Uint8Array(buf), {
        executionProviders: eps,
        graphOptimizationLevel: "all",
        logSeverityLevel: 3,
    });
}

async function init(req: InitRequest) {
    const t0 = now();
    const gpu = (navigator as unknown as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
    const hasWebGPU = req.preferWebGPU && !!gpu && !!(await gpu.requestAdapter().catch(() => null));
    ort = (hasWebGPU ? await import("onnxruntime-web/webgpu") : await import("onnxruntime-web/wasm")) as unknown as Ort;
    ort.env.wasm.wasmPaths = req.ortWasmPath;
    const isolated = (self as unknown as { crossOriginIsolated?: boolean }).crossOriginIsolated === true;
    ort.env.wasm.numThreads = isolated ? Math.min(4, Math.max(1, (navigator.hardwareConcurrency ?? 2) - 1)) : 1;
    ort.env.logLevel = "error";

    const progress = (file: string) => (loaded: number, total: number) =>
        post({ id: req.id, ok: true, type: "progress", result: { file, loaded, total } });
    const [detBuf, recBuf] = await Promise.all([
        fetchModel(req.detectorUrl, progress("detector")),
        fetchModel(req.recognizerUrl, progress("recognizer")),
    ]);

    let det: OrtTypes.InferenceSession, rec: OrtTypes.InferenceSession;
    try {
        if (!hasWebGPU) throw new Error("no webgpu");
        [det, rec] = await Promise.all([createSession(detBuf, ["webgpu"]), createSession(recBuf, ["webgpu"])]);
        backend = "webgpu";
    } catch {
        [det, rec] = await Promise.all([createSession(detBuf, ["wasm"]), createSession(recBuf, ["wasm"])]);
        backend = "wasm";
    }
    pipeline = new FacePipeline(new OrtRunner(det), new OrtRunner(rec));

    // warm-up: first run compiles shaders / allocates; do it now, not on the user's first frame
    const warm: RgbaImage = { data: new Uint8ClampedArray(192 * 192 * 4).fill(128), width: 192, height: 192 };
    await pipeline.detect(warm);
    await pipeline.embedAligned({ data: new Uint8ClampedArray(112 * 112 * 4).fill(128), width: 112, height: 112 });

    return { backend, threads: ort.env.wasm.numThreads as number, loadMs: now() - t0 };
}

function pack(r: Awaited<ReturnType<FacePipeline["embedRoi"]>>, t0: number, wantAligned?: boolean): WorkerEmbedding | null {
    if (!r) return null;
    return {
        embedding: r.embedding,
        kps: r.kps,
        detScore: r.detScore,
        quality: r.quality,
        aligned: wantAligned ? new ImageData(new Uint8ClampedArray(r.aligned.data), 112, 112) : undefined,
        timings: { ...r.timings, totalMs: now() - t0 },
    };
}

async function embedRoi(req: EmbedRoiRequest): Promise<WorkerEmbedding | null> {
    const t0 = now();
    const { img } = bitmapToRgba(req.bitmap);
    req.bitmap.close();
    const r = await pipeline!.embedRoi(img, {
        fallbackKps: req.fallbackKps,
        roiToSourceScale: req.roiToSourceScale,
        flipAugment: req.flipAugment,
    });
    return pack(r, t0, req.wantAligned);
}

async function embedImage(req: EmbedImageRequest): Promise<WorkerEmbedding[]> {
    const t0 = now();
    const { img, scale: downscale } = bitmapToRgba(req.bitmap, 1600);
    req.bitmap.close();
    const { img: boxed, scale } = letterbox(img, 640);
    const faces = rescaleFaces(await pipeline!.detect(boxed, 0.5), 1 / scale)
        .sort((a, b) => (b.box.x2 - b.box.x1) * (b.box.y2 - b.box.y1) - (a.box.x2 - a.box.x1) * (a.box.y2 - a.box.y1))
        .slice(0, req.maxFaces);
    const out: WorkerEmbedding[] = [];
    for (const f of faces) {
        const r = await pipeline!.embedWithKeypoints(img, f.kps, f.score, { flipAugment: req.flipAugment }, 1 / downscale);
        const e = pack(r, t0, true)!;
        e.kps = f.kps.map(([x, y]) => [x / downscale, y / downscale]); // original image coordinates
        out.push(e);
    }
    return out;
}

async function benchmark(iterations: number) {
    const roi: RgbaImage = { data: new Uint8ClampedArray(192 * 192 * 4).map((_, i) => (i * 37) & 255), width: 192, height: 192 };
    const al: RgbaImage = { data: new Uint8ClampedArray(112 * 112 * 4).map((_, i) => (i * 13) & 255), width: 112, height: 112 };
    let t = now();
    for (let i = 0; i < iterations; i++) await pipeline!.detect(roi);
    const detectMs = (now() - t) / iterations;
    t = now();
    for (let i = 0; i < iterations; i++) await pipeline!.embedAligned(al);
    return { detectMs, embedMs: (now() - t) / iterations, backend };
}

self.onmessage = async (ev: MessageEvent<WorkerRequest>) => {
    const req = ev.data;
    try {
        if (req.type === "init") {
            post({ id: req.id, ok: true, type: "init", result: await init(req) });
            return;
        }
        if (!pipeline) throw new Error("worker not initialised");
        if (req.type === "embedRoi") {
            const result = await embedRoi(req);
            const transfer: Transferable[] = result ? [result.embedding.buffer as ArrayBuffer] : [];
            if (result?.aligned) transfer.push(result.aligned.data.buffer as ArrayBuffer);
            post({ id: req.id, ok: true, type: "embedRoi", result }, transfer);
        } else if (req.type === "embedImage") {
            post({ id: req.id, ok: true, type: "embedImage", result: await embedImage(req) });
        } else if (req.type === "benchmark") {
            post({ id: req.id, ok: true, type: "benchmark", result: await benchmark(req.iterations) });
        }
    } catch (e) {
        if ("bitmap" in req) try { req.bitmap.close(); } catch { /* already closed */ }
        post({ id: req.id, ok: false, error: e instanceof Error ? e.message : String(e) });
    }
};
