import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { convertWeightsToFp16, parseFields, toHalf } from "./onnx-fp16.mjs";

const fromHalf = h => {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e === 0) return s * m * 2 ** -24;
    if (e === 31) return m ? NaN : s * Infinity;
    return s * (1 + m / 1024) * 2 ** (e - 15);
};

describe("float16 conversion", () => {
    it.each([[0, 0], [1, 0x3c00], [-2, 0xc000], [0.5, 0x3800], [65504, 0x7bff], [1e6, 0x7c00], [5.960464477539063e-8, 1]])(
        "%f -> %i", (v, h) => expect(toHalf(v)).toBe(h));

    it("rounds to nearest with a relative error below 2^-11", () => {
        for (const v of [0.1234567, -3.14159, 1e-3, 123.456, 7e-5]) {
            expect(Math.abs(fromHalf(toHalf(v)) - v) / Math.abs(v)).toBeLessThan(2 ** -11);
        }
    });

    it("rounds ties to even", () => {
        expect(toHalf(1 + 2 ** -11)).toBe(0x3c00); // exactly between 1 and 1 + 2^-10 -> even mantissa
        expect(toHalf(1 + 3 * 2 ** -11)).toBe(0x3c02);
    });
});

const MODEL = path.resolve(__dirname, "../public/models/arcface/w600k_mbf.onnx");

describe.runIf(existsSync(MODEL))("ONNX weight conversion (needs npm run setup:face)", () => {
    it("halves w600k_mbf and keeps a valid graph", () => {
        const input = new Uint8Array(readFileSync(MODEL));
        const { model, converted } = convertWeightsToFp16(input);
        expect(converted).toBe(50);
        expect(model.length).toBeLessThan(input.length * 0.52);
        const graph = parseFields(parseFields(model).find(f => f.field === 7).value);
        const casts = graph.filter(f => f.field === 1 && new TextDecoder().decode(f.value).includes("Cast"));
        expect(casts.length).toBe(50);
    });
});
