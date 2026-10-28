import { readFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { createCanvas, loadImage } from "@napi-rs/canvas";
import * as ort from "onnxruntime-node";
import type { ModelRunner } from "@/face/core/pipeline";
import type { RgbaImage } from "@/face/core/image";

export const REPO_ROOT = resolve(__dirname, "../../..");
export const MODELS_DIR = process.env.FACE_MODELS_DIR ?? resolve(REPO_ROOT, "models");

export interface ReferenceFixture {
    template: number[][];
    similarityCases: { src: number[][]; dst: number[][]; m: number[] }[];
    images: {
        path: string; width: number; height: number;
        faces: { bbox: number[]; kps: [number, number][]; score: number; embedding: string; alignedRgb?: string }[];
    }[];
    base64Probe: { values: number[]; b64: string };
}

export function loadFixture(): ReferenceFixture {
    return JSON.parse(readFileSync(resolve(__dirname, "fixtures/reference.json"), "utf8"));
}

export function modelsAvailable(): boolean {
    return existsSync(resolve(MODELS_DIR, "w600k_mbf.onnx")) && existsSync(resolve(MODELS_DIR, "det_500m.onnx"));
}

export async function loadRgba(path: string): Promise<RgbaImage> {
    const img = await loadImage(readFileSync(path));
    const c = createCanvas(img.width, img.height);
    const ctx = c.getContext("2d");
    ctx.drawImage(img, 0, 0);
    const d = ctx.getImageData(0, 0, img.width, img.height);
    return { data: new Uint8ClampedArray(d.data), width: d.width, height: d.height };
}

/** Letterbox (top-left aligned, like SCRFD's reference code) into a square. */
export async function letterbox(path: string, size: number): Promise<{ img: RgbaImage; scale: number }> {
    const img = await loadImage(readFileSync(path));
    const scale = size / Math.max(img.width, img.height);
    const c = createCanvas(size, size);
    const ctx = c.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, size, size);
    ctx.drawImage(img, 0, 0, Math.round(img.width * scale), Math.round(img.height * scale));
    const d = ctx.getImageData(0, 0, size, size);
    return { img: { data: new Uint8ClampedArray(d.data), width: size, height: size }, scale };
}

export class NodeRunner implements ModelRunner {
    private constructor(private session: ort.InferenceSession) { }

    static async create(file: string): Promise<NodeRunner> {
        const s = await ort.InferenceSession.create(resolve(MODELS_DIR, file), { logSeverityLevel: 3 });
        return new NodeRunner(s);
    }

    async run(input: Float32Array, dims: readonly number[]): Promise<Float32Array[]> {
        const feeds = { [this.session.inputNames[0]]: new ort.Tensor("float32", input, dims as number[]) };
        const out = await this.session.run(feeds);
        return this.session.outputNames.map(n => out[n].data as Float32Array);
    }
}
