/**
 * Runtime-agnostic embedding pipeline: SCRFD keypoints -> similarity alignment ->
 * ArcFace (MobileFaceNet) 512-d embedding. Used by the browser worker (onnxruntime-web)
 * and by Node tests / examples (onnxruntime-node) so both run byte-identical logic.
 */
import { Affine, alignmentMatrix, ALIGNED_SIZE, interOcular, Point } from "./geometry";
import { arcfaceInput, flipHorizontal, RgbaImage, warpAffineRgba } from "./image";
import { FaceQuality, measureQuality } from "./quality";
import { decodeScrfd, largestFace, scrfdInput, ScrfdFace } from "./scrfd";
import { l2Normalize } from "./vector";

/** Minimal interface over an ONNX Runtime session (web or node). */
export interface ModelRunner {
    /** returns outputs in the model's declared output order */
    run(input: Float32Array, dims: readonly number[]): Promise<Float32Array[]>;
}

export interface EmbedResult {
    embedding: Float32Array;
    /** keypoints in the coordinate space of the image passed in */
    kps: Point[];
    detScore: number;
    quality: FaceQuality;
    alignMatrix: Affine;
    aligned: RgbaImage;
    timings: { detectMs: number; embedMs: number };
}

export interface PipelineOptions {
    /** also embed the mirrored crop and average (≈+1–2% accuracy, 2x embed cost) */
    flipAugment?: boolean;
    detThreshold?: number;
}

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

export class FacePipeline {
    constructor(private detector: ModelRunner, private recognizer: ModelRunner) { }

    /** Detect faces on a SQUARE rgba image (side must be a multiple of 32). */
    async detect(img: RgbaImage, threshold = 0.5): Promise<ScrfdFace[]> {
        if (img.width !== img.height || img.width % 32 !== 0) {
            throw new Error(`detector input must be square and a multiple of 32, got ${img.width}x${img.height}`);
        }
        const outputs = await this.detector.run(scrfdInput(img), [1, 3, img.height, img.width]);
        return decodeScrfd(outputs, img.width, threshold);
    }

    async embedAligned(aligned: RgbaImage, flipAugment = false): Promise<Float32Array> {
        const dims = [1, 3, ALIGNED_SIZE, ALIGNED_SIZE];
        const [out] = await this.recognizer.run(arcfaceInput(aligned), dims);
        if (!flipAugment) return l2Normalize(out);
        const [flip] = await this.recognizer.run(arcfaceInput(flipHorizontal(aligned)), dims);
        const sum = new Float32Array(out.length);
        for (let i = 0; i < out.length; i++) sum[i] = out[i] + flip[i];
        return l2Normalize(sum);
    }

    /** Embed a face given keypoints in `src` coordinates. */
    async embedWithKeypoints(src: RgbaImage, kps: Point[], detScore: number, opts: PipelineOptions = {}, srcToOriginalScale = 1): Promise<EmbedResult> {
        const t0 = now();
        const m = alignmentMatrix(kps);
        const aligned = warpAffineRgba(src, m, ALIGNED_SIZE, ALIGNED_SIZE);
        const embedding = await this.embedAligned(aligned, opts.flipAugment);
        return {
            embedding, kps, detScore, alignMatrix: m, aligned,
            quality: measureQuality(aligned, interOcular(kps) * srcToOriginalScale, detScore),
            timings: { detectMs: 0, embedMs: now() - t0 },
        };
    }

    /**
     * Main entry for live video: `roi` is a square crop around the tracked face.
     * Re-detects SCRFD keypoints inside it (canonical alignment) and embeds.
     * `fallbackKps` (tracker points in ROI space) are used if SCRFD misses.
     * `roiToSourceScale` converts ROI pixels to original-frame pixels (for the size gate).
     */
    async embedRoi(roi: RgbaImage, opts: PipelineOptions & { fallbackKps?: Point[]; roiToSourceScale?: number } = {}): Promise<EmbedResult | null> {
        const t0 = now();
        const faces = await this.detect(roi, opts.detThreshold ?? 0.45);
        const face = largestFace(faces);
        const detectMs = now() - t0;
        let kps: Point[], score: number;
        if (face) { kps = face.kps; score = face.score; }
        else if (opts.fallbackKps) { kps = opts.fallbackKps; score = 0; }
        else return null;
        const r = await this.embedWithKeypoints(roi, kps, score, opts, opts.roiToSourceScale ?? 1);
        r.timings.detectMs = detectMs;
        return r;
    }
}
