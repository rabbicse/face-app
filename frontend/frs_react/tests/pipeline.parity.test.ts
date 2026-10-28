/**
 * End-to-end parity: the TypeScript pipeline (decode -> SCRFD -> align -> ArcFace) must
 * produce the same 512-d vectors as the Python reference (examples/python/face_ref.py).
 * Requires the ONNX models in <repo>/models (python scripts/download_models.py).
 */
import { beforeAll, describe, expect, it } from "vitest";
import { resolve } from "node:path";
import { FacePipeline } from "@/face/core/pipeline";
import { rescaleFaces, ScrfdFace } from "@/face/core/scrfd";
import { base64ToEmbedding, dot, isValidEmbedding } from "@/face/core/vector";
import { warpAffineRgba } from "@/face/core/image";
import { alignmentMatrix, interOcular, Point, squareRoi } from "@/face/core/geometry";
import { createCanvas, loadImage } from "@napi-rs/canvas";
import { readFileSync } from "node:fs";
import { letterbox, loadFixture, loadRgba, modelsAvailable, NodeRunner, REPO_ROOT } from "./helpers";

const fx = loadFixture();
const run = modelsAvailable() ? describe : describe.skip;

run("pipeline parity with Python reference", () => {
    let pipe: FacePipeline;
    beforeAll(async () => {
        pipe = new FacePipeline(await NodeRunner.create("det_500m.onnx"), await NodeRunner.create("w600k_mbf.onnx"));
    });

    it("warpAffineRgba matches cv2.warpAffine on the reference crops", async () => {
        let checked = 0;
        for (const im of fx.images) {
            const face = im.faces[0];
            if (!face?.alignedRgb) continue;
            const src = await loadRgba(resolve(REPO_ROOT, im.path));
            const ours = warpAffineRgba(src, alignmentMatrix(face.kps as Point[]), 112, 112);
            const ref = Buffer.from(face.alignedRgb, "base64");
            let diff = 0;
            for (let i = 0, p = 0; i < 112 * 112; i++, p += 4) {
                diff += Math.abs(ours.data[p] - ref[i * 3]) + Math.abs(ours.data[p + 1] - ref[i * 3 + 1]) + Math.abs(ours.data[p + 2] - ref[i * 3 + 2]);
            }
            expect(diff / (112 * 112 * 3)).toBeLessThan(1.5); // mean abs error in 0..255 levels
            checked++;
        }
        expect(checked).toBeGreaterThan(0);
    });

    it("given the same keypoints, TS embeddings equal Python embeddings (cos > 0.995)", async () => {
        let n = 0, worst = 1;
        for (const im of fx.images) {
            const src = await loadRgba(resolve(REPO_ROOT, im.path));
            for (const f of im.faces) {
                const r = await pipe.embedWithKeypoints(src, f.kps as Point[], f.score);
                expect(isValidEmbedding(r.embedding)).toBe(true);
                const c = dot(r.embedding, base64ToEmbedding(f.embedding));
                worst = Math.min(worst, c);
                n++;
            }
        }
        expect(n).toBeGreaterThan(10);
        expect(worst).toBeGreaterThan(0.995);
    });

    it("full-image SCRFD detection finds the same faces and keypoints", async () => {
        for (const im of fx.images) {
            const { img, scale } = await letterbox(resolve(REPO_ROOT, im.path), 640);
            const faces = rescaleFaces(await pipe.detect(img, 0.5), 1 / scale);
            for (const ref of im.faces) {
                const iod = Math.hypot(ref.kps[0][0] - ref.kps[1][0], ref.kps[0][1] - ref.kps[1][1]);
                const match = faces.find(f => Math.hypot(f.kps[2][0] - ref.kps[2][0], f.kps[2][1] - ref.kps[2][1]) < iod * 0.5);
                expect(match, `${im.path}: face missing`).toBeDefined();
                const err = Math.max(...match!.kps.map((p, k) => Math.hypot(p[0] - ref.kps[k][0], p[1] - ref.kps[k][1])));
                expect(err / iod).toBeLessThan(0.08);
            }
        }
    });

    it("ROI path (as used for live video) stays within cos > 0.95 of the full-image reference", async () => {
        let worst = 1, n = 0;
        for (const im of fx.images) {
            const image = await loadImage(readFileSync(resolve(REPO_ROOT, im.path)));
            for (const ref of im.faces) {
                const [x1, y1, x2, y2] = ref.bbox;
                const roi = squareRoi({ x: x1, y: y1, width: x2 - x1, height: y2 - y1 }, 1.6);
                const S = 192;
                // what the browser does with createImageBitmap(video, sx, sy, sw, sh, {resizeWidth: S})
                const c = createCanvas(S, S);
                const ctx = c.getContext("2d");
                ctx.drawImage(image, roi.x, roi.y, roi.side, roi.side, 0, 0, S, S);
                const d = ctx.getImageData(0, 0, S, S);
                const r = await pipe.embedRoi({ data: new Uint8ClampedArray(d.data), width: S, height: S }, { roiToSourceScale: roi.side / S });
                expect(r).not.toBeNull();
                worst = Math.min(worst, dot(r!.embedding, base64ToEmbedding(ref.embedding)));
                expect(r!.quality.interOcularPx).toBeCloseTo(interOcular(ref.kps as Point[]), -1);
                n++;
            }
        }
        expect(n).toBeGreaterThan(10);
        expect(worst).toBeGreaterThan(0.93);
    });

    it("separates identities: same-person pairs across photos score higher than different people", async () => {
        // bbt3 and bbt4 share cast members; reference (Python) scores >0.55 for them
        const all: { img: string; e: Float32Array }[] = [];
        for (const im of fx.images) for (const f of im.faces) all.push({ img: im.path, e: base64ToEmbedding(f.embedding) });
        const cross = [];
        for (let i = 0; i < all.length; i++) for (let j = i + 1; j < all.length; j++) {
            if (all[i].img !== all[j].img) cross.push(dot(all[i].e, all[j].e));
        }
        cross.sort((a, b) => b - a);
        // top cross-photo pairs are genuine matches, the bulk are impostors
        expect(cross[0]).toBeGreaterThan(0.6);
        const median = cross[Math.floor(cross.length / 2)];
        expect(median).toBeLessThan(0.15);
    });
});

describe("decode helpers", () => {
    it("ScrfdFace rescale keeps keypoints consistent", () => {
        const f: ScrfdFace = { box: { x1: 10, y1: 10, x2: 20, y2: 20 }, kps: [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], score: 0.9 };
        const [r] = rescaleFaces([f], 2, 5, 5);
        expect(r.kps[4]).toEqual([23, 25]);
        expect(r.box.x2).toBe(45);
    });
});
