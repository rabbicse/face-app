import { describe, expect, it } from "vitest";
import { alignmentMatrix, ARCFACE_TEMPLATE_112 } from "./alignment";
import { applyAffine, estimateSimilarityTransform } from "./geometry";
import { loadTestVectors } from "./testVectors";

const vectors = loadTestVectors();

describe("similarity transform", () => {
    it("is the identity for the template itself", () => {
        const m = alignmentMatrix(ARCFACE_TEMPLATE_112);
        [1, 0, 0, 0, 1, 0].forEach((v, i) => expect(m[i]).toBeCloseTo(v, 9));
    });

    it("recovers a known rotation, scale and translation", () => {
        const angle = (17 * Math.PI) / 180;
        const src = ARCFACE_TEMPLATE_112.map(p => ({
            x: 2.5 * (p.x * Math.cos(angle) - p.y * Math.sin(angle)) + 120,
            y: 2.5 * (p.x * Math.sin(angle) + p.y * Math.cos(angle)) + 80,
        }));
        const m = alignmentMatrix(src);
        src.forEach((p, i) => {
            const q = applyAffine(m, p);
            expect(q.x).toBeCloseTo(ARCFACE_TEMPLATE_112[i].x, 6);
            expect(q.y).toBeCloseTo(ARCFACE_TEMPLATE_112[i].y, 6);
        });
    });

    it("scales the template for other crop sizes", () => {
        const m = alignmentMatrix(ARCFACE_TEMPLATE_112, 224);
        expect(m[0]).toBeCloseTo(2, 9);
        expect(m[4]).toBeCloseTo(2, 9);
    });

    it("rejects degenerate input", () => {
        expect(() => estimateSimilarityTransform([{ x: 1, y: 1 }, { x: 1, y: 1 }], [{ x: 0, y: 0 }, { x: 1, y: 1 }])).toThrow();
        expect(() => alignmentMatrix([{ x: 0, y: 0 }])).toThrow();
    });

    it.each(vectors.alignment.map((c, i) => [i, c] as const))("matches the Python reference (case %i)", (_, c) => {
        const m = alignmentMatrix(c.landmarks.map(([x, y]) => ({ x, y })));
        const expected = c.matrix.flat();
        expected.forEach((v, i) => expect(m[i]).toBeCloseTo(v, 4));
    });
});
