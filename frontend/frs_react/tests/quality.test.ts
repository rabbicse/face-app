import { describe, expect, it } from "vitest";
import { ENROLL_QUALITY, IDENTIFY_QUALITY, laplacianVariance, measureQuality, qualityIssues, qualityScore } from "@/face/core/quality";
import { RgbaImage } from "@/face/core/image";

function img(fn: (x: number, y: number) => number): RgbaImage {
    const data = new Uint8ClampedArray(112 * 112 * 4);
    for (let y = 0; y < 112; y++) for (let x = 0; x < 112; x++) {
        const v = fn(x, y), p = (y * 112 + x) * 4;
        data[p] = data[p + 1] = data[p + 2] = v; data[p + 3] = 255;
    }
    return { data, width: 112, height: 112 };
}

describe("quality", () => {
    it("laplacian variance is high for texture and ~0 for flat images", () => {
        const flat = img(() => 128), checker = img((x, y) => ((x + y) % 2 ? 200 : 60));
        const g = (i: RgbaImage) => Float32Array.from({ length: 112 * 112 }, (_, k) => i.data[k * 4]);
        expect(laplacianVariance(g(flat), 112, 112)).toBe(0);
        expect(laplacianVariance(g(checker), 112, 112)).toBeGreaterThan(1000);
    });

    it("flags dark, blurry and small faces", () => {
        const dark = measureQuality(img(() => 20), 80, 0.9);
        const issues = qualityIssues(dark, ENROLL_QUALITY);
        expect(issues).toContain("too_dark");
        expect(issues).toContain("blurry");
        expect(qualityIssues({ ...dark, brightness: 130, contrast: 40, sharpness: 300, interOcularPx: 20 }, IDENTIFY_QUALITY)).toEqual(["too_far"]);
    });

    it("passes a good face and ranks it above a poor one", () => {
        const good = { sharpness: 400, brightness: 130, contrast: 45, interOcularPx: 90, detScore: 0.95 };
        expect(qualityIssues(good, ENROLL_QUALITY)).toEqual([]);
        expect(qualityScore(good)).toBeGreaterThan(qualityScore({ ...good, sharpness: 30, interOcularPx: 35 }));
    });
});
