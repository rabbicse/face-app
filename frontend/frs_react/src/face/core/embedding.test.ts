import { describe, expect, it } from "vitest";
import { cosineSimilarity, decodeEmbedding, dot, encodeEmbedding, fuseEmbeddings, l2Normalize } from "./embedding";
import { laplacianVariance, meanValue, rgbaToTensor, toGray } from "./preprocess";
import { loadTestVectors } from "./testVectors";

const { embedding: e, preprocess: pre } = loadTestVectors();

describe("embedding math", () => {
    it("normalises like the reference", () => {
        Array.from(l2Normalize(e.a)).forEach((v, i) => expect(v).toBeCloseTo(e.a_normalized[i], 5));
    });

    it("computes cosine similarity and fusion like the reference", () => {
        expect(cosineSimilarity(e.a, e.b)).toBeCloseTo(e.cosine_ab, 5);
        Array.from(fuseEmbeddings([e.a, e.b, e.c])).forEach((v, i) => expect(v).toBeCloseTo(e.fused_abc[i], 5));
    });

    it("uses the backend wire format", () => {
        expect(encodeEmbedding([1, -2, 3.5])).toBe(e.base64_of_1_2_3);
        const v = Float32Array.from({ length: 512 }, (_, i) => Math.sin(i));
        expect(Array.from(decodeEmbedding(encodeEmbedding(v)))).toEqual(Array.from(v));
    });

    it("rejects zero vectors and dimension mismatches", () => {
        expect(() => l2Normalize([0, 0, 0])).toThrow();
        expect(() => dot([1], [1, 2])).toThrow();
    });
});

describe("preprocessing", () => {
    it("matches the reference NCHW tensor", () => {
        const rgba: number[] = [];
        for (let i = 0; i < pre.rgb.length; i += 3) rgba.push(pre.rgb[i], pre.rgb[i + 1], pre.rgb[i + 2], 255);
        Array.from(rgbaToTensor(rgba, pre.width, pre.height)).forEach((v, i) => expect(v).toBeCloseTo(pre.tensor[i], 6));
    });

    it("mirrors horizontally and writes at an offset (batched TTA input)", () => {
        const rgba = [255, 0, 0, 255, 0, 0, 255, 255]; // red, blue (1x2)
        const out = new Float32Array(12);
        rgbaToTensor(rgba, 2, 1, out, 0, false);
        rgbaToTensor(rgba, 2, 1, out, 6, true);
        expect(Array.from(out.subarray(0, 2))).toEqual([1, -1]); // R plane
        expect(Array.from(out.subarray(6, 8))).toEqual([-1, 1]); // R plane mirrored
    });

    it("measures sharpness and brightness", () => {
        const w = 16, h = 16;
        const flat = new Float32Array(w * h).fill(128);
        const checker = Float32Array.from({ length: w * h }, (_, i) => ((i % w) + Math.floor(i / w)) % 2 ? 255 : 0);
        expect(laplacianVariance(flat, w, h)).toBe(0);
        expect(laplacianVariance(checker, w, h)).toBeGreaterThan(1000);
        expect(meanValue(toGray([255, 255, 255, 255, 0, 0, 0, 255], 2, 1))).toBeCloseTo(127.5, 3);
    });
});
