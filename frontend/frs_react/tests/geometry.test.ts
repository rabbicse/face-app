import { describe, expect, it } from "vitest";
import {
    ARCFACE_TEMPLATE_112, applyAffine, composeAffine, estimateSimilarity, invertAffine, Point, squareRoi,
} from "@/face/core/geometry";
import { loadFixture } from "./helpers";

const fx = loadFixture();

describe("estimateSimilarity (Umeyama)", () => {
    it("matches the Python / scikit-image reference on random point sets", () => {
        for (const c of fx.similarityCases) {
            const m = estimateSimilarity(c.src as unknown as Point[], c.dst as unknown as Point[]);
            m.forEach((v, i) => expect(v).toBeCloseTo(c.m[i], 6));
        }
    });

    it("recovers an exact similarity transform", () => {
        const s = 1.7, th = 0.3, tx = 12, ty = -4;
        const src: Point[] = [[10, 20], [50, 22], [30, 40], [15, 60], [45, 61]];
        const dst = src.map(([x, y]) => [s * (Math.cos(th) * x - Math.sin(th) * y) + tx, s * (Math.sin(th) * x + Math.cos(th) * y) + ty] as Point);
        const m = estimateSimilarity(src, dst);
        src.forEach((p, i) => {
            const q = applyAffine(m, p);
            expect(q[0]).toBeCloseTo(dst[i][0], 9);
            expect(q[1]).toBeCloseTo(dst[i][1], 9);
        });
    });

    it("maps the template onto itself with the identity", () => {
        const m = estimateSimilarity(ARCFACE_TEMPLATE_112, ARCFACE_TEMPLATE_112);
        [1, 0, 0, 0, 1, 0].forEach((v, i) => expect(m[i]).toBeCloseTo(v, 9));
    });

    it("rejects degenerate input", () => {
        expect(() => estimateSimilarity([[1, 1], [1, 1]], [[0, 0], [1, 1]])).toThrow();
        expect(() => estimateSimilarity([[1, 1]], [[0, 0]])).toThrow();
    });
});

describe("affine helpers", () => {
    it("invert ∘ m = identity", () => {
        const m = estimateSimilarity(fx.similarityCases[0].src as unknown as Point[], fx.similarityCases[0].dst as unknown as Point[]);
        const id = composeAffine(invertAffine(m), m);
        [1, 0, 0, 0, 1, 0].forEach((v, i) => expect(id[i]).toBeCloseTo(v, 9));
    });

    it("squareRoi is centred on the box and scaled", () => {
        const r = squareRoi({ x: 100, y: 50, width: 80, height: 100 }, 1.5);
        expect(r.side).toBe(150);
        expect(r.x + r.side / 2).toBe(140);
        expect(r.y + r.side / 2).toBe(100);
    });
});
