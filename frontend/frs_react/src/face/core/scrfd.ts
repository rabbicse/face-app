/**
 * SCRFD (InsightFace det_500m.onnx) pre/post-processing.
 * The model returns boxes + 5 keypoints that match what ArcFace was trained with,
 * which gives better alignment (and cross-platform parity) than mapping tracker meshes.
 */
import { Point } from "./geometry";
import { RgbaImage } from "./image";

export interface ScrfdFace {
    box: { x1: number; y1: number; x2: number; y2: number };
    kps: Point[];
    score: number;
}

const STRIDES = [8, 16, 32];
const NUM_ANCHORS = 2;

/** RGBA (size x size) -> Float32 NCHW, RGB, (x - 127.5) / 128 */
export function scrfdInput(img: RgbaImage, out?: Float32Array): Float32Array {
    const n = img.width * img.height;
    const t = out ?? new Float32Array(3 * n);
    const d = img.data;
    for (let i = 0, p = 0; i < n; i++, p += 4) {
        t[i] = (d[p] - 127.5) / 128;
        t[n + i] = (d[p + 1] - 127.5) / 128;
        t[2 * n + i] = (d[p + 2] - 127.5) / 128;
    }
    return t;
}

/**
 * @param outputs the 9 model outputs in model order:
 *   [score_s8, score_s16, score_s32, bbox_s8, bbox_s16, bbox_s32, kps_s8, kps_s16, kps_s32]
 * @param inputSize the square side the image was fed at
 */
export function decodeScrfd(outputs: readonly Float32Array[], inputSize: number, scoreThresh = 0.5, nmsThresh = 0.4): ScrfdFace[] {
    if (outputs.length !== 9) throw new Error(`SCRFD expects 9 outputs, got ${outputs.length}`);
    const faces: ScrfdFace[] = [];
    STRIDES.forEach((stride, si) => {
        const scores = outputs[si], boxes = outputs[si + 3], kps = outputs[si + 6];
        const fw = Math.ceil(inputSize / stride);
        const count = scores.length;
        for (let i = 0; i < count; i++) {
            const score = scores[i];
            if (score < scoreThresh) continue;
            const cell = Math.floor(i / NUM_ANCHORS);
            const cx = (cell % fw) * stride, cy = Math.floor(cell / fw) * stride;
            const b = i * 4, k = i * 10;
            const pts: Point[] = [];
            for (let j = 0; j < 5; j++) pts.push([cx + kps[k + 2 * j] * stride, cy + kps[k + 2 * j + 1] * stride]);
            faces.push({
                box: {
                    x1: cx - boxes[b] * stride, y1: cy - boxes[b + 1] * stride,
                    x2: cx + boxes[b + 2] * stride, y2: cy + boxes[b + 3] * stride,
                },
                kps: pts,
                score,
            });
        }
    });
    return nms(faces, nmsThresh);
}

function iou(a: ScrfdFace["box"], b: ScrfdFace["box"]): number {
    const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
    const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
    const inter = ix * iy;
    const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return ua <= 0 ? 0 : inter / ua;
}

export function nms(faces: ScrfdFace[], thresh: number): ScrfdFace[] {
    const sorted = [...faces].sort((a, b) => b.score - a.score);
    const keep: ScrfdFace[] = [];
    for (const f of sorted) {
        if (keep.every(k => iou(k.box, f.box) <= thresh)) keep.push(f);
    }
    return keep;
}

/** Map detections from model-input space back to source space (scale then offset). */
export function rescaleFaces(faces: ScrfdFace[], scale: number, offsetX = 0, offsetY = 0): ScrfdFace[] {
    return faces.map(f => ({
        score: f.score,
        box: {
            x1: f.box.x1 * scale + offsetX, y1: f.box.y1 * scale + offsetY,
            x2: f.box.x2 * scale + offsetX, y2: f.box.y2 * scale + offsetY,
        },
        kps: f.kps.map(([x, y]) => [x * scale + offsetX, y * scale + offsetY] as Point),
    }));
}

/** Largest face first (the person in front of the camera). */
export function largestFace(faces: ScrfdFace[]): ScrfdFace | null {
    let best: ScrfdFace | null = null, area = -1;
    for (const f of faces) {
        const a = (f.box.x2 - f.box.x1) * (f.box.y2 - f.box.y1);
        if (a > area) { area = a; best = f; }
    }
    return best;
}
